import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it } from 'vitest';
import { OpenAIProvider } from '../src/core/llm/providers/openai.js';

describe('OpenAI-compatible provider request lifecycle', () => {
  const servers: ReturnType<typeof createServer>[] = [];

  afterEach(async () => {
    await Promise.all(servers.splice(0).map(server => new Promise<void>(resolve => {
      server.closeAllConnections();
      server.close(() => resolve());
    })));
  });

  it('aborts a response that keeps the connection alive without completing its body', async () => {
    const server = createServer((_request, response) => {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.write('{"id":"never-completes"');
      const keepalive = setInterval(() => response.write(' '), 10);
      response.on('close', () => clearInterval(keepalive));
    });
    servers.push(server);
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
    const { port } = server.address() as AddressInfo;
    const provider = new OpenAIProvider({
      apiKey: 'test-key',
      baseUrl: `http://127.0.0.1:${port}`,
      model: 'test-model',
      timeoutMs: 100,
      maxRetries: 0,
    });

    const startedAt = Date.now();
    await expect(provider.complete([
      { role: 'user', content: 'Return one bounded response.' },
    ])).rejects.toThrow();

    expect(Date.now() - startedAt).toBeLessThan(2_000);
  });

  it('uses reasoning_content when a compatible non-streaming response has no content', async () => {
    const server = createServer((_request, response) => {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        id: 'reasoning-only-completion',
        object: 'chat.completion',
        created: 1,
        model: 'test-model',
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: null,
            reasoning_content: 'compatible fallback output',
          },
          finish_reason: 'stop',
        }],
        usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
      }));
    });
    servers.push(server);
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
    const { port } = server.address() as AddressInfo;
    const provider = new OpenAIProvider({
      apiKey: 'test-key',
      baseUrl: `http://127.0.0.1:${port}`,
      model: 'test-model',
      timeoutMs: 1_000,
      maxRetries: 0,
    });

    const completion = await provider.complete([
      { role: 'user', content: 'Return one response.' },
    ]);

    expect(completion.content).toBe('compatible fallback output');
    expect(completion.usage?.totalTokens).toBe(7);
  });

  it('uses streamed reasoning_content only when no visible content is emitted', async () => {
    const server = createServer((_request, response) => {
      response.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });
      response.write(`data: ${JSON.stringify({
        id: 'reasoning-only-stream',
        object: 'chat.completion.chunk',
        created: 1,
        model: 'test-model',
        choices: [{
          index: 0,
          delta: { reasoning_content: 'streamed fallback output' },
          finish_reason: null,
        }],
      })}\n\n`);
      response.write(`data: ${JSON.stringify({
        id: 'reasoning-only-stream',
        object: 'chat.completion.chunk',
        created: 1,
        model: 'test-model',
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
        usage: { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 },
      })}\n\n`);
      response.end('data: [DONE]\n\n');
    });
    servers.push(server);
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
    const { port } = server.address() as AddressInfo;
    const provider = new OpenAIProvider({
      apiKey: 'test-key',
      baseUrl: `http://127.0.0.1:${port}`,
      model: 'test-model',
      timeoutMs: 1_000,
      maxRetries: 0,
    });

    const chunks = [];
    for await (const chunk of provider.stream([
      { role: 'user', content: 'Return one streamed response.' },
    ])) {
      chunks.push(chunk);
    }

    expect(chunks.map(chunk => chunk.content).join('')).toBe('streamed fallback output');
    expect(chunks.at(-1)?.done).toBe(true);
    expect(chunks.at(-1)?.usage?.totalTokens).toBe(8);
  });
});
