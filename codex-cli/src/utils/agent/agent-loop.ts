import type { ReviewDecision } from "./review.js";
import type { ApplyPatchCommand, ApprovalPolicy } from "../../approvals.js";
import type { AppConfig } from "../config.js";
import type {
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseItem,
} from "openai/resources/responses/responses.mjs";
import type { Reasoning } from "openai/resources.mjs";

import { log, isLoggingEnabled } from "./log.js";
import { OPENAI_BASE_URL, OPENAI_TIMEOUT_MS } from "../config.js";
import { parseToolCallArguments } from "../parsers.js";
import {
  ORIGIN,
  CLI_VERSION,
  getSessionId,
  setCurrentModel,
  setSessionId,
} from "../session.js";
import { handleExecCommand } from "./handle-exec-command.js";
import { randomUUID } from "node:crypto";
import OpenAI, { APIConnectionTimeoutError } from "openai";
import { initBraintrustLogger, wrapOpenAIWithBraintrust } from "../braintrust-logger.js";

// Wait time before retrying after rate limit errors (ms).
const RATE_LIMIT_RETRY_WAIT_MS = parseInt(
  process.env["OPENAI_RATE_LIMIT_RETRY_WAIT_MS"] || "2500",
  10,
);

export type CommandConfirmation = {
  review: ReviewDecision;
  applyPatch?: ApplyPatchCommand | undefined;
  customDenyMessage?: string;
};

const alreadyProcessedResponses = new Set();

type AgentLoopParams = {
  model: string;
  config?: AppConfig;
  instructions?: string;
  approvalPolicy: ApprovalPolicy;
  onItem: (item: ResponseItem) => void;
  onLoading: (loading: boolean) => void;

  /** Called when the command is not auto-approved to request explicit user review. */
  getCommandConfirmation: (
    command: Array<string>,
    applyPatch: ApplyPatchCommand | undefined,
  ) => Promise<CommandConfirmation>;
  onLastResponseId: (lastResponseId: string) => void;
};

export class AgentLoop {
  private model: string;
  private instructions?: string;
  private approvalPolicy: ApprovalPolicy;
  private config: AppConfig;

  // Using `InstanceType<typeof OpenAI>` sidesteps typing issues with the OpenAI package under
  // the TS 5+ `moduleResolution=bundler` setup. OpenAI client instance. We keep the concrete
  // type to avoid sprinkling `any` across the implementation while still allowing paths where
  // the OpenAI SDK types may not perfectly match. The `typeof OpenAI` pattern captures the
  // instance shape without resorting to `any`.
  private oai: OpenAI;

  private onItem: (item: ResponseItem) => void;
  private onLoading: (loading: boolean) => void;
  private getCommandConfirmation: (
    command: Array<string>,
    applyPatch: ApplyPatchCommand | undefined,
  ) => Promise<CommandConfirmation>;
  private onLastResponseId: (lastResponseId: string) => void;

  /**
   * A reference to the currently active stream returned from the OpenAI
   * client. We keep this so that we can abort the request if the user decides
   * to interrupt the current task (e.g. via the escape hot‑key).
   */
  private currentStream: unknown | null = null;
  /** Incremented with every call to `run()`. Allows us to ignore stray events
   * from streams that belong to a previous run which might still be emitting
   * after the user has canceled and issued a new command. */
  private generation = 0;
  /** AbortController for in‑progress tool calls (e.g. shell commands). */
  private execAbortController: AbortController | null = null;
  /** Set to true when `cancel()` is called so `run()` can exit early. */
  private canceled = false;
  /** Function calls that were emitted by the model but never answered because
   *  the user cancelled the run.  We keep the `call_id`s around so the *next*
   *  request can send a dummy `function_call_output` that satisfies the
   *  contract and prevents the
   *    400 | No tool output found for function call …
   *  error from OpenAI. */
  private pendingAborts: Set<string> = new Set();
  /** Set to true by `terminate()` – prevents any further use of the instance. */
  private terminated = false;
  /** Master abort controller – fires when terminate() is invoked. */
  private readonly hardAbort = new AbortController();

  /**
   * Abort the ongoing request/stream, if any. This allows callers (typically
   * the UI layer) to interrupt the current agent step so the user can issue
   * new instructions without waiting for the model to finish.
   */
  public cancel(): void {
    if (this.terminated) {
      return;
    }
    if (isLoggingEnabled()) {
      log(
        `AgentLoop.cancel() invoked – currentStream=${Boolean(
          this.currentStream,
        )} execAbortController=${Boolean(
          this.execAbortController,
        )} generation=${this.generation}`,
      );
    }
    (
      this.currentStream as { controller?: { abort?: () => void } } | null
    )?.controller?.abort?.();

    this.canceled = true;
    this.execAbortController?.abort();
    if (isLoggingEnabled()) {
      log("AgentLoop.cancel(): execAbortController.abort() called");
    }

    // If we have *not* seen any function_call IDs yet there is nothing that
    // needs to be satisfied in a follow‑up request.  In that case we clear
    // the stored lastResponseId so a subsequent run starts a clean turn.
    if (this.pendingAborts.size === 0) {
      try {
        this.onLastResponseId("");
      } catch {
        /* ignore */
      }
    }

    // NOTE: We intentionally do *not* clear `lastResponseId` here.  If the
    // stream produced a `function_call` before the user cancelled, OpenAI now
    // expects a corresponding `function_call_output` that must reference that
    // very same response ID.  We therefore keep the ID around so the
    // follow‑up request can still satisfy the contract.

    // If we have *not* seen any function_call IDs yet there is nothing that
    // needs to be satisfied in a follow‑up request.  In that case we clear
    // the stored lastResponseId so a subsequent run starts a clean turn.
    if (this.pendingAborts.size === 0) {
      try {
        this.onLastResponseId("");
      } catch {
        /* ignore */
      }
    }

    // NOTE: We intentionally do *not* clear `lastResponseId` here.  If the
    // stream produced a `function_call` before the user cancelled, OpenAI now
    // expects a corresponding `function_call_output` that must reference that
    // very same response ID.  We therefore keep the ID around so the
    // follow‑up request can still satisfy the contract.
    this.onLoading(false);

    /* Inform the UI that the run was aborted by the user. */
    // const cancelNotice: ResponseItem = {
    //   id: `cancel-${Date.now()}`,
    //   type: "message",
    //   role: "system",
    //   content: [
    //     {
    //       type: "input_text",
    //       text: "⏹️  Execution canceled by user.",
    //     },
    //   ],
    // };
    // this.onItem(cancelNotice);

    this.generation += 1;
    if (isLoggingEnabled()) {
      log(`AgentLoop.cancel(): generation bumped to ${this.generation}`);
    }
  }

  /**
   * Hard‑stop the agent loop. After calling this method the instance becomes
   * unusable: any in‑flight operations are aborted and subsequent invocations
   * of `run()` will throw.
   */
  public terminate(): void {
    if (this.terminated) {
      return;
    }
    this.terminated = true;

    this.hardAbort.abort();

    this.cancel();
  }

  public sessionId: string;
  /*
   * Cumulative thinking time across this AgentLoop instance (ms).
   * Currently not used anywhere – comment out to keep the strict compiler
   * happy under `noUnusedLocals`.  Restore when telemetry support lands.
   */
  // private cumulativeThinkingMs = 0;
  constructor({
    model,
    instructions,
    approvalPolicy,
    // `config` used to be required.  Some unit‑tests (and potentially other
    // callers) instantiate `AgentLoop` without passing it, so we make it
    // optional and fall back to sensible defaults.  This keeps the public
    // surface backwards‑compatible and prevents runtime errors like
    // "Cannot read properties of undefined (reading 'apiKey')" when accessing
    // `config.apiKey` below.
    config,
    onItem,
    onLoading,
    getCommandConfirmation,
    onLastResponseId,
  }: AgentLoopParams & { config?: AppConfig }) {
    this.model = model;
    this.instructions = instructions;
    this.approvalPolicy = approvalPolicy;

    // If no `config` has been provided we derive a minimal stub so that the
    // rest of the implementation can rely on `this.config` always being a
    // defined object.  We purposefully copy over the `model` and
    // `instructions` that have already been passed explicitly so that
    // downstream consumers (e.g. telemetry) still observe the correct values.
    this.config =
      config ??
      ({
        model,
        instructions: instructions ?? "",
      } as AppConfig);
    this.onItem = onItem;
    this.onLoading = onLoading;
    this.getCommandConfirmation = getCommandConfirmation;
    this.onLastResponseId = onLastResponseId;
    this.sessionId = getSessionId() || randomUUID().replaceAll("-", "");
    // Configure OpenAI client with optional timeout (ms) from environment
    const timeoutMs = OPENAI_TIMEOUT_MS;
    const apiKey = this.config.apiKey ?? process.env["OPENAI_API_KEY"] ?? "";
    this.oai = new OpenAI({
      // The OpenAI JS SDK only requires `apiKey` when making requests against
      // the official API.  When running unit‑tests we stub out all network
      // calls so an undefined key is perfectly fine.  We therefore only set
      // the property if we actually have a value to avoid triggering runtime
      // errors inside the SDK (it validates that `apiKey` is a non‑empty
      // string when the field is present).
      ...(apiKey ? { apiKey } : {}),
      baseURL: OPENAI_BASE_URL,
      defaultHeaders: {
        originator: ORIGIN,
        version: CLI_VERSION,
        session_id: this.sessionId,
      },
      ...(timeoutMs !== undefined ? { timeout: timeoutMs } : {}),
    });

    // Initialize Braintrust logging if enabled
    if (this.config.braintrustEnabled && this.config.braintrustApiKey) {
      (async () => {
        try {
          await initBraintrustLogger(
            this.config.braintrustProjectName,
            this.config.braintrustApiKey
          );
          this.oai = wrapOpenAIWithBraintrust(this.oai);
          if (isLoggingEnabled()) {
            log("Braintrust logging initialized and OpenAI client wrapped");
          }
        } catch (error) {
          if (isLoggingEnabled()) {
            log(`Failed to initialize Braintrust logging: ${error}`);
          }
        }
      })();
    }

    setSessionId(this.sessionId);
    setCurrentModel(this.model);

    this.hardAbort = new AbortController();

    this.hardAbort.signal.addEventListener(
      "abort",
      () => {
        if (isLoggingEnabled()) {
          log("AgentLoop.hardAbort signal fired");
        }
        this.cancel();
      },
      { once: true },
    );
  }
