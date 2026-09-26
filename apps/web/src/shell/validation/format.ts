/**
 * Number formatting for the panels. Every value comes from provider data; these helpers only
 * decide digit grouping and precision, never the number itself.
 */

const formatters = new Map<number, Intl.NumberFormat>();

/** Digit-grouped, at most `maxFractionDigits` decimals, trailing zeros dropped. */
export function formatNumber(value: number, maxFractionDigits: number): string {
  let formatter = formatters.get(maxFractionDigits);
  if (!formatter) {
    formatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: maxFractionDigits });
    formatters.set(maxFractionDigits, formatter);
  }
  return formatter.format(value);
}

/** A number the panel may show: present, numeric and finite (never NaN, never null). */
export function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
