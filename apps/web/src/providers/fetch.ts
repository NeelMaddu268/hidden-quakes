/** Minimal fetch shape so tests can pass a fake without a full `Response`. */
export interface FetchResponseLike {
  ok: boolean;
  status: number;
  json(): Promise<unknown>;
}
export type FetchLike = (url: string) => Promise<FetchResponseLike>;

export class BundleFetchError extends Error {
  readonly url: string;
  readonly status: number | null;
  constructor(url: string, status: number | null, detail?: string) {
    super(`${url}: ${status === null ? "request failed" : `HTTP ${status}`}${detail ? ` (${detail})` : ""}`);
    this.name = "BundleFetchError";
    this.url = url;
    this.status = status;
  }
}

export class SchemaVersionError extends Error {
  constructor(
    readonly expected: string,
    readonly actual: string,
    readonly url: string,
  ) {
    super(
      `${url} has schemaVersion ${actual}; this build expects ${expected}. ` +
        "Regenerate the bundle with the current exporter, or rebuild the app.",
    );
    this.name = "SchemaVersionError";
  }
}

export const defaultFetch: FetchLike = (url) => fetch(url);

/** GET + JSON, with a typed error on a non-2xx status. `notFoundAsNull` turns 404 into `null`. */
export async function fetchJson<T>(
  fetchImpl: FetchLike,
  url: string,
  options: { notFoundAsNull?: boolean } = {},
): Promise<T | null> {
  let response: FetchResponseLike;
  try {
    response = await fetchImpl(url);
  } catch (error) {
    throw new BundleFetchError(url, null, error instanceof Error ? error.message : String(error));
  }
  if (response.status === 404 && options.notFoundAsNull) return null;
  if (!response.ok) throw new BundleFetchError(url, response.status);
  return (await response.json()) as T;
}
