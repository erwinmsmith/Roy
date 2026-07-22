import { afterEach, describe, expect, it, vi } from 'vitest';
import { WebFetchTool, WebSearchTool } from '../src/core/tools/webTools.js';

describe('web tools', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('parses real-provider Bing RSS results into structured search evidence', async () => {
    const rss = `<?xml version="1.0"?><rss><channel>
      <item><title>Node.js Fetch</title><link>https://nodejs.org/api/globals.html#fetch</link><description>Official fetch API documentation.</description></item>
      <item><title>Unsafe</title><link>http://127.0.0.1/private</link><description>Must be excluded.</description></item>
    </channel></rss>`;
    vi.stubGlobal('fetch', vi.fn(async () => new Response(rss, {
      status: 200,
      headers: { 'content-type': 'application/rss+xml' },
    })));

    const result = await new WebSearchTool({ searchProvider: 'bing' }).execute({
      query: 'Node.js fetch official documentation',
      maxResults: 5,
    });

    expect(result.success).toBe(true);
    expect(result.result).toEqual(expect.objectContaining({
      provider: 'bing',
      results: [expect.objectContaining({
        title: 'Node.js Fetch',
        url: 'https://nodejs.org/api/globals.html#fetch',
        source: 'nodejs.org',
      })],
    }));
  });

  it('rejects localhost and cloud metadata URLs before fetching', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const tool = new WebFetchTool();

    const localhost = await tool.execute({ url: 'https://localhost/admin' });
    const metadata = await tool.execute({ url: 'http://169.254.169.254/latest/meta-data' });

    expect(localhost.success).toBe(false);
    expect(metadata.success).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects private IPv4 addresses encoded as IPv4-mapped IPv6', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const tool = new WebFetchTool();

    const result = await tool.execute({ url: 'https://[::ffff:192.168.1.1]/private' });

    expect(result.success).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('extracts readable HTML and strips executable page content', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(`
      <html><head><title>Example Docs</title><script>secret()</script></head>
      <body><nav>Navigation</nav><main><h1>Fetch API</h1><p>AbortSignal can stop a request.</p><a href="/details">Details</a></main></body></html>
    `, { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' } })));

    // Use a public IP literal so the test does not depend on DNS.
    const result = await new WebFetchTool().execute({ url: 'https://93.184.216.34/docs', maxChars: 2000 });

    expect(result.success).toBe(true);
    expect(result.result).toEqual(expect.objectContaining({
      title: 'Example Docs',
      text: 'Fetch API AbortSignal can stop a request. Details',
      links: [{ text: 'Details', url: 'https://93.184.216.34/details' }],
    }));
    expect(JSON.stringify(result.result)).not.toContain('secret()');
  });

  it('extracts only the requested HTML fragment section while preserving page links', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(`
      <html><head><title>Runtime Docs</title></head><body><main>
        <section id="intro"><h2>Introduction</h2><p>Unrelated overview.</p></section>
        <section id="fetch"><h2>fetch</h2><p>Fetch is a global API.</p><a href="/abort">AbortSignal</a></section>
      </main></body></html>
    `, { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' } })));

    const result = await new WebFetchTool().execute({ url: 'https://93.184.216.34/docs#fetch' });

    expect(result.success).toBe(true);
    expect(result.result).toEqual(expect.objectContaining({
      finalUrl: 'https://93.184.216.34/docs#fetch',
      title: 'Runtime Docs - fetch',
      text: 'fetch Fetch is a global API. AbortSignal',
      links: [{ text: 'AbortSignal', url: 'https://93.184.216.34/abort' }],
    }));
    expect(JSON.stringify(result.result)).not.toContain('Unrelated overview');
  });

  it('reads enough raw HTML to reach main content after a large navigation shell', async () => {
    const navigation = `NAVIGATION_NOISE_${'x'.repeat(90_000)}`;
    vi.stubGlobal('fetch', vi.fn(async () => new Response(`
      <html><head><title>Large Docs</title></head><body>
        <nav>${navigation}</nav>
        <main><h1>AbortSignal.timeout</h1><p>The signal aborts after the requested delay.</p></main>
      </body></html>
    `, { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' } })));

    const result = await new WebFetchTool().execute({ url: 'https://93.184.216.34/large-docs', maxChars: 2000 });

    expect(result.success).toBe(true);
    expect(result.result).toEqual(expect.objectContaining({
      text: 'AbortSignal.timeout The signal aborts after the requested delay.',
    }));
    expect(JSON.stringify(result.result)).not.toContain('NAVIGATION_NOISE');
  });
});
