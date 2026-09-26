/**
 * Data-mode selection (docs/01 → Web app → Modes). Mode comes from `?mode=`; the default is
 * `showcase`. `mock` is a development mode: production builds only allow it when the
 * `NEXT_PUBLIC_ALLOW_MOCK=1` build flag is set (Gate M deploys the mock bundle that way).
 */
import type { DataMode } from "@hq/contracts";

export const DEFAULT_MODE: DataMode = "showcase";
export const DATA_MODES: readonly DataMode[] = ["mock", "showcase", "live", "snapshot"];

export function isDataMode(value: unknown): value is DataMode {
  return typeof value === "string" && (DATA_MODES as readonly string[]).includes(value);
}

/** `?mode=` from a query string; unknown or missing values fall back to the default. */
export function parseMode(search: string): DataMode {
  const value = new URLSearchParams(search).get("mode");
  return isDataMode(value) ? value : DEFAULT_MODE;
}

export function mockAllowed(): boolean {
  return process.env.NODE_ENV !== "production" || process.env.NEXT_PUBLIC_ALLOW_MOCK === "1";
}
