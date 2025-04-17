import { initLogger, wrapOpenAI } from '@braintrust/core';
import OpenAI from 'openai';
import { isLoggingEnabled } from '../agent/log.js';

/**
 * Initialize the Braintrust logger.
 * This should be called at the start of the application.
 * 
 * @param projectName Optional project name to use in Braintrust
 * @returns A function to flush logs
 */
export function initBraintrustLogger(projectName?: string) {
  // Only initialize if BRAINTRUST_API_KEY is set
  const apiKey = process.env.BRAINTRUST_API_KEY;
  if (!apiKey) {
    return { flush: () => Promise.resolve() };
  }

  return initLogger({
    apiKey,
    project: projectName || 'Codex CLI',
    asyncFlush: true,
  });
}

/**
 * Wraps an OpenAI client instance with Braintrust logging.
 * If BRAINTRUST_API_KEY is not set, returns the original client.
 * 
 * @param client The OpenAI client to wrap
 * @returns The wrapped OpenAI client or the original if BRAINTRUST_API_KEY is not set
 */
export function wrapWithBraintrust(client: OpenAI): OpenAI {
  const apiKey = process.env.BRAINTRUST_API_KEY;
  
  if (!apiKey || !isLoggingEnabled()) {
    return client;
  }
  
  try {
    return wrapOpenAI(client);
  } catch (error) {
    console.error('Failed to wrap OpenAI client with Braintrust:', error);
    return client;
  }
}
