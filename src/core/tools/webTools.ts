import { lookup } from 'node:dns/promises';
import { isIP } from 'node:net';
import { load } from 'cheerio';
import type { Tool, ToolResult } from './types.js';

export type WebSearchProviderName = 'auto' | 'brave' | 'bing';

export interface WebToolConfig {
  enabled: boolean;
  searchProvider: WebSearchProviderName;
  braveApiKeyEnv: string;
  timeoutMs: number;
  maxResults: number;
  maxContentChars: number;
  allowHttp: boolean;
  userAgent: string;
}

export interface WebSearchResultItem {
  title: string;
  url: string;
  snippet: string;
  source: string;
}

export interface WebSearchResult {
  query: string;
  provider: Exclude<WebSearchProviderName, 'auto'>;
  results: WebSearchResultItem[];
  fetchedAt: string;
}

export interface WebFetchResult {
  url: string;
  finalUrl: string;
  title?: string;
  contentType: string;
  text: string;
  links: Array<{ text: string; url: string }>;
  truncated: boolean;
  fetchedAt: string;
}

const DEFAULT_CONFIG: WebToolConfig = {
  enabled: true,
  searchProvider: 'auto',
  braveApiKeyEnv: 'BRAVE_SEARCH_API_KEY',
  timeoutMs: 15_000,
  maxResults: 5,
  maxContentChars: 20_000,
  allowHttp: false,
  userAgent: 'RoyRuntime/0.1 (+https://github.com/erwinmsmith/Roy)',
};

export class WebSearchTool implements Tool {
  readonly name = 'web.search';
  readonly description = 'Search the public web and return titles, URLs, and snippets from real search results.';
  readonly version = '0.1.0';
  readonly parameters = {
    query: { type: 'string' as const, required: true, description: 'Search query.' },
    maxResults: { type: 'number' as const, required: false, description: 'Maximum result count.' },
    domain: { type: 'string' as const, required: false, description: 'Optional domain restriction.' },
  };

  private readonly config: WebToolConfig;

  constructor(config: Partial<WebToolConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  validate(params: Record<string, unknown>): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];
    if (typeof params.query !== 'string' || params.query.trim().length === 0) {
      errors.push('query must be a non-empty string');
    }
    if (params.maxResults !== undefined
      && (typeof params.maxResults !== 'number' || !Number.isInteger(params.maxResults) || params.maxResults <= 0)) {
      errors.push('maxResults must be a positive integer when provided');
    }
    if (params.domain !== undefined && (typeof params.domain !== 'string' || !isValidDomain(params.domain))) {
      errors.push('domain must be a valid hostname when provided');
    }
    return { valid: errors.length === 0, errors: errors.length > 0 ? errors : undefined };
  }

  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    if (!this.config.enabled) return { success: false, error: 'web.search is disabled by workspace configuration' };
    const rawQuery = String(params.query).trim().slice(0, 800);
    const domain = typeof params.domain === 'string' ? params.domain.trim().toLowerCase() : undefined;
    const query = domain ? `${rawQuery} site:${domain}` : rawQuery;
    const maxResults = Math.min(Math.max(1, Number(params.maxResults ?? this.config.maxResults)), 10);
    const configuredKey = process.env[this.config.braveApiKeyEnv]?.trim();
    const provider = this.config.searchProvider === 'auto'
      ? configuredKey ? 'brave' : 'bing'
      : this.config.searchProvider;

    try {
      const results = provider === 'brave'
        ? await this.searchBrave(query, maxResults, configuredKey)
        : await this.searchBing(query, maxResults);
      return {
        success: true,
        result: {
          query: rawQuery,
          provider,
          results,
          fetchedAt: new Date().toISOString(),
        } satisfies WebSearchResult,
        metadata: { provider, resultCount: results.length, domain },
      };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error), metadata: { provider } };
    }
  }

  private async searchBrave(query: string, maxResults: number, apiKey?: string): Promise<WebSearchResultItem[]> {
    if (!apiKey) throw new Error(`${this.config.braveApiKeyEnv} is required when the Brave search provider is selected`);
    const url = new URL('https://api.search.brave.com/res/v1/web/search');
    url.searchParams.set('q', query);
    url.searchParams.set('count', String(maxResults));
    const response = await fetchWithTimeout(url, {
      timeoutMs: this.config.timeoutMs,
      headers: {
        Accept: 'application/json',
        'Accept-Encoding': 'gzip',
        'User-Agent': this.config.userAgent,
        'X-Subscription-Token': apiKey,
      },
    });
    if (!response.ok) throw new Error(`Brave Search returned HTTP ${response.status}`);
    const payload = await response.json() as {
      web?: { results?: Array<{ title?: unknown; url?: unknown; description?: unknown }> };
    };
    return (payload.web?.results ?? [])
      .filter(item => typeof item.url === 'string' && isPublicHttpUrl(item.url, this.config.allowHttp))
      .slice(0, maxResults)
      .map(item => ({
        title: cleanText(String(item.title ?? 'Untitled result')),
        url: String(item.url),
        snippet: cleanText(String(item.description ?? '')),
        source: new URL(String(item.url)).hostname,
      }));
  }

  private async searchBing(query: string, maxResults: number): Promise<WebSearchResultItem[]> {
    const url = new URL('https://www.bing.com/search');
    url.searchParams.set('format', 'rss');
    url.searchParams.set('q', query);
    const response = await fetchWithTimeout(url, {
      timeoutMs: this.config.timeoutMs,
      headers: { Accept: 'application/rss+xml, application/xml;q=0.9', 'User-Agent': this.config.userAgent },
    });
    if (!response.ok) throw new Error(`Bing Search returned HTTP ${response.status}`);
    const xml = await response.text();
    const $ = load(xml, { xmlMode: true });
    return $('item').toArray()
      .map(item => {
        const title = cleanText($(item).find('title').first().text());
        const resultUrl = $(item).find('link').first().text().trim();
        const snippet = cleanText($(item).find('description').first().text());
        if (!resultUrl || !isPublicHttpUrl(resultUrl, this.config.allowHttp)) return undefined;
        return { title, url: resultUrl, snippet, source: new URL(resultUrl).hostname } satisfies WebSearchResultItem;
      })
      .filter((item): item is WebSearchResultItem => Boolean(item))
      .slice(0, maxResults);
  }
}

export class WebFetchTool implements Tool {
  readonly name = 'web.fetch';
  readonly description = 'Fetch readable text from a public HTTP(S) page with private-network and redirect protection.';
  readonly version = '0.1.0';
  readonly parameters = {
    url: { type: 'string' as const, required: true, description: 'Public HTTP(S) URL.' },
    maxChars: { type: 'number' as const, required: false, description: 'Maximum extracted text characters.' },
  };

  private readonly config: WebToolConfig;

  constructor(config: Partial<WebToolConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  validate(params: Record<string, unknown>): { valid: boolean; errors?: string[] } {
    const errors: string[] = [];
    if (typeof params.url !== 'string' || !isPublicHttpUrl(params.url, this.config.allowHttp)) {
      errors.push(`url must be a valid public ${this.config.allowHttp ? 'HTTP(S)' : 'HTTPS'} URL`);
    }
    if (params.maxChars !== undefined
      && (typeof params.maxChars !== 'number' || !Number.isInteger(params.maxChars) || params.maxChars <= 0)) {
      errors.push('maxChars must be a positive integer when provided');
    }
    return { valid: errors.length === 0, errors: errors.length > 0 ? errors : undefined };
  }

  async execute(params: Record<string, unknown>): Promise<ToolResult> {
    if (!this.config.enabled) return { success: false, error: 'web.fetch is disabled by workspace configuration' };
    const maxChars = Math.min(Math.max(1, Number(params.maxChars ?? this.config.maxContentChars)), 100_000);
    try {
      const result = await this.fetchPage(String(params.url), maxChars);
      return {
        success: true,
        result,
        metadata: { finalUrl: result.finalUrl, contentType: result.contentType, truncated: result.truncated },
      };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  private async fetchPage(input: string, maxChars: number): Promise<WebFetchResult> {
    let current = new URL(input);
    for (let redirect = 0; redirect <= 5; redirect += 1) {
      await assertPublicUrl(current, this.config.allowHttp);
      const response = await fetchWithTimeout(current, {
        timeoutMs: this.config.timeoutMs,
        redirect: 'manual',
        headers: {
          Accept: 'text/html, text/plain, application/json, application/xml;q=0.8',
          'User-Agent': this.config.userAgent,
        },
      });
      if (response.status >= 300 && response.status < 400) {
        const location = response.headers.get('location');
        if (!location) throw new Error(`Redirect response ${response.status} did not include a Location header`);
        current = new URL(location, current);
        continue;
      }
      if (!response.ok) throw new Error(`Web fetch returned HTTP ${response.status}`);
      const contentType = response.headers.get('content-type')?.toLowerCase() ?? 'application/octet-stream';
      if (!isReadableContentType(contentType)) throw new Error(`Unsupported web content type: ${contentType}`);
      const rawByteLimit = Math.min(Math.max(maxChars * 8, 256_000), 1_000_000);
      const raw = await readResponseWithLimit(response, rawByteLimit);
      const parsed = extractReadableText(raw.text, contentType, current);
      const text = parsed.text.slice(0, maxChars);
      return {
        url: input,
        finalUrl: current.toString(),
        title: parsed.title,
        contentType,
        text,
        links: parsed.links,
        truncated: raw.truncated || parsed.text.length > maxChars,
        fetchedAt: new Date().toISOString(),
      };
    }
    throw new Error('Too many redirects while fetching URL');
  }
}

async function fetchWithTimeout(
  url: URL,
  options: { timeoutMs: number; headers: Record<string, string>; redirect?: 'error' | 'follow' | 'manual' }
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs);
  try {
    return await fetch(url, {
      method: 'GET',
      headers: options.headers,
      redirect: options.redirect ?? 'follow',
      signal: controller.signal,
    });
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(`Web request timed out after ${options.timeoutMs}ms`, { cause: error });
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function assertPublicUrl(url: URL, allowHttp: boolean): Promise<void> {
  if (!isPublicHttpUrl(url.toString(), allowHttp)) throw new Error('URL is not an allowed public HTTP(S) URL');
  const addresses = await lookup(url.hostname, { all: true, verbatim: true });
  if (addresses.length === 0) throw new Error('URL hostname did not resolve');
  for (const address of addresses) {
    if (isPrivateAddress(address.address)) throw new Error(`URL resolves to a non-public address: ${address.address}`);
  }
}

function isPublicHttpUrl(input: string, allowHttp: boolean): boolean {
  try {
    const url = new URL(input);
    if (url.username || url.password) return false;
    if (url.protocol !== 'https:' && !(allowHttp && url.protocol === 'http:')) return false;
    if (url.port && !['80', '443'].includes(url.port)) return false;
    const hostname = url.hostname.toLowerCase().replace(/\.$/, '');
    if (!hostname || hostname === 'localhost' || hostname.endsWith('.localhost') || hostname.endsWith('.local')) return false;
    return !isPrivateAddress(hostname);
  } catch {
    return false;
  }
}

function isPrivateAddress(input: string): boolean {
  const address = input.toLowerCase().replace(/^\[|\]$/g, '');
  const version = isIP(address);
  if (version === 4) {
    const [a, b] = address.split('.').map(Number);
    return a === 0 || a === 10 || a === 127 || a >= 224
      || (a === 169 && b === 254)
      || (a === 172 && b >= 16 && b <= 31)
      || (a === 192 && b === 168)
      || (a === 100 && b >= 64 && b <= 127);
  }
  if (version === 6) {
    const mappedV4 = extractMappedIpv4(address);
    if (mappedV4) return isPrivateAddress(mappedV4);
    return address === '::' || address === '::1'
      || address.startsWith('fc') || address.startsWith('fd')
      || /^fe[89abcdef]/.test(address)
      || address.startsWith('ff');
  }
  return false;
}

function extractMappedIpv4(address: string): string | undefined {
  const dotted = address.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/);
  if (dotted) return dotted[1];
  const hexadecimal = address.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (!hexadecimal) return undefined;
  const high = Number.parseInt(hexadecimal[1], 16);
  const low = Number.parseInt(hexadecimal[2], 16);
  return `${high >> 8}.${high & 0xff}.${low >> 8}.${low & 0xff}`;
}

function isValidDomain(input: string): boolean {
  const normalized = input.trim().toLowerCase();
  return normalized.length <= 253
    && !normalized.includes('/')
    && /^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$/.test(normalized)
    && normalized.includes('.');
}

function isReadableContentType(contentType: string): boolean {
  return contentType.includes('text/html') || contentType.includes('text/plain')
    || contentType.includes('application/json') || contentType.includes('application/xml')
    || contentType.includes('text/xml') || contentType.includes('application/rss+xml');
}

async function readResponseWithLimit(response: Response, maxBytes: number): Promise<{ text: string; truncated: boolean }> {
  if (!response.body) return { text: '', truncated: false };
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  let truncated = false;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    const remaining = maxBytes - bytes;
    if (remaining <= 0) {
      truncated = true;
      await reader.cancel();
      break;
    }
    chunks.push(value.subarray(0, remaining));
    bytes += Math.min(value.byteLength, remaining);
    if (value.byteLength > remaining) {
      truncated = true;
      await reader.cancel();
      break;
    }
  }
  return { text: new TextDecoder().decode(Buffer.concat(chunks)), truncated };
}

function extractReadableText(
  raw: string,
  contentType: string,
  baseUrl: URL
): { title?: string; text: string; links: Array<{ text: string; url: string }> } {
  if (contentType.includes('html')) {
    const $ = load(raw);
    const title = cleanText($('title').first().text()) || undefined;
    $('script, style, noscript, svg, template, iframe').remove();
    $('address, article, aside, blockquote, div, footer, h1, h2, h3, h4, h5, h6, header, li, main, nav, p, section, td, th, br')
      .each((_, element) => { $(element).after(' '); });
    const article = $('article').first();
    const main = $('main').first();
    const source = article.length > 0 ? article : main.length > 0 ? main : $('body');
    const fragment = decodeFragment(baseUrl.hash);
    let sectionText: string | undefined;
    let sectionTitle = '';
    if (fragment) {
      const target = $('[id]').filter((_, element) => $(element).attr('id') === fragment).first();
      if (target.length > 0) {
        const section = target.is('section') ? target : target.closest('section');
        const selected = section.length > 0 ? section : target;
        sectionText = selected.text();
        sectionTitle = cleanText(selected.find('h1, h2, h3, h4').first().text());
      }
    }
    const links = $('body').find('a[href]').toArray()
      .map(element => {
        const href = $(element).attr('href');
        if (!href) return undefined;
        try {
          const url = new URL(href, baseUrl).toString();
          if (!isPublicHttpUrl(url, false)) return undefined;
          return { text: cleanText($(element).text()).slice(0, 240), url };
        } catch {
          return undefined;
        }
      })
      .filter((item): item is { text: string; url: string } => Boolean(item))
      .filter((item, index, values) => values.findIndex(candidate => candidate.url === item.url) === index)
      .slice(0, 100);
    return {
      title: sectionTitle ? `${title ?? baseUrl.hostname} - ${sectionTitle}` : title,
      text: cleanText(sectionText ?? source.text()),
      links,
    };
  }
  if (contentType.includes('json')) {
    try {
      return { text: JSON.stringify(JSON.parse(raw), null, 2), links: [] };
    } catch {
      return { text: raw.trim(), links: [] };
    }
  }
  return { text: cleanText(raw), links: [] };
}

function decodeFragment(hash: string): string | undefined {
  if (!hash || hash === '#') return undefined;
  try {
    return decodeURIComponent(hash.slice(1));
  } catch {
    return hash.slice(1);
  }
}

function cleanText(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

export const defaultWebToolConfig = (): WebToolConfig => ({ ...DEFAULT_CONFIG });
