import * as braintrust from "braintrust";
import OpenAI from "openai";
import { CLI_VERSION, ORIGIN, getSessionId } from "./session.js";

let logger: braintrust.Logger | null = null;

/**
 * Initialize the Braintrust logger
 * @param projectName The name of the project in Braintrust
 * @param apiKey The Braintrust API key
 */
export async function initBraintrustLogger(
  projectName: string = "Codex",
  apiKey?: string
): Promise<braintrust.Logger> {
  if (logger) {
    return logger;
  }

  logger = await braintrust.initLogger(projectName, {
    apiKey,
    asyncFlush: true,
    metadata: {
      cli_version: CLI_VERSION,
      origin: ORIGIN,
      session_id: getSessionId(),
    },
  });

  return logger;
}

/**
 * Wrap an OpenAI client instance with Braintrust logging
 * @param openaiClient The OpenAI client instance to wrap
 * @returns The wrapped OpenAI client
 */
export function wrapOpenAIWithBraintrust(openaiClient: OpenAI): OpenAI {
  if (!logger) {
    throw new Error("Braintrust logger not initialized. Call initBraintrustLogger first.");
  }

  return braintrust.wrapOpenAI(openaiClient);
}

/**
 * Flush any pending logs to Braintrust
 */
export async function flushBraintrustLogs(): Promise<void> {
  if (logger) {
    await logger.flush();
  }
}

/**
 * Log feedback to Braintrust
 * @param spanId The ID of the span to log feedback for
 * @param feedback The feedback to log
 */
export async function logFeedback(
  spanId: string,
  feedback: {
    score?: number;
    expected?: any;
    comment?: string;
    metadata?: Record<string, any>;
  }
): Promise<void> {
  if (!logger) {
    throw new Error("Braintrust logger not initialized. Call initBraintrustLogger first.");
  }

  await logger.logFeedback(spanId, feedback);
}
