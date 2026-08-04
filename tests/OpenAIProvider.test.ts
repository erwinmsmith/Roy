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
});
