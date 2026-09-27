// Where the Tour's caption sits (WEB-09): a lower third, centered on the viewport, that keeps clear of
// every overlay measured on screen (the shell's blocks, H3's panels, the open drawer). Pure, so the
// component and the tests agree. The band starts at the bottom edge; when the overlays leave too little
// room there (the time scrubber, REVEAL), it steps up to sit just above the overlay under the center.

export const CAPTION_LAYOUT = Object.freeze({
  /** The shell's --shell-inset: the caption's distance from the viewport edges. */
  inset: 24,
  /** Clearance kept from every overlay. */
  gap: 14,
  /** The caption never grows wider than this (a readable line length at 18 px). */
  maxWidth: 620,
  /** Below this free width the band steps up instead of squeezing the caption. */
  minWidth: 300,
  /** How many times the band may step up before giving up and pinning to the top. */
  maxSteps: 8,
});

/** A measured overlay, in viewport CSS px. */
export interface ObstacleRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface CaptionBox {
  left: number;
  /** The caption's bottom edge, as a CSS `bottom` (distance from the viewport's bottom). */
  bottom: number;
  width: number;
}

interface Interval {
  l: number;
  r: number;
}

/** [lo, hi] minus every blocked interval (sorted, clipped), as free intervals left to right. */
function freeIntervals(lo: number, hi: number, blocked: Interval[]): Interval[] {
  blocked.sort((a, b) => a.l - b.l);
  const free: Interval[] = [];
  let x = lo;
  for (const b of blocked) {
    if (b.r <= x) continue;
    if (b.l >= hi) break;
    if (b.l > x) free.push({ l: x, r: b.l });
    x = Math.max(x, b.r);
  }
  if (x < hi) free.push({ l: x, r: hi });
  return free;
}

/**
 * The caption's box in a W × H viewport, given its current height `h` and the overlays on screen.
 * The caption is centered on the viewport when the free span allows it, else shifted as little as the
 * span permits; its width is the free span, capped at `maxWidth`.
 */
export function captionBox(W: number, H: number, h: number, obstacles: readonly ObstacleRect[]): CaptionBox {
  const L = CAPTION_LAYOUT;
  const cx = W / 2;
  const lo = L.inset;
  const hi = W - L.inset;
  let bandBottom = H - L.inset;
  for (let step = 0; step <= L.maxSteps && bandBottom - h >= L.inset; step++) {
    const top = bandBottom - h;
    const blockers = obstacles.filter(
      (o) => o.width > 0 && o.height > 0 && o.top < bandBottom + L.gap && o.top + o.height > top - L.gap,
    );
    const free = freeIntervals(
      lo,
      hi,
      blockers.map((o) => ({ l: o.left - L.gap, r: o.left + o.width + L.gap })),
    );
    // The span under the center, else the widest.
    let span = free.find((f) => f.l <= cx && cx <= f.r && f.r - f.l >= L.minWidth) ?? null;
    if (span === null) {
      const widest = free.reduce<Interval | null>((a, f) => (a === null || f.r - f.l > a.r - a.l ? f : a), null);
      if (widest !== null && widest.r - widest.l >= L.minWidth) span = widest;
    }
    if (span !== null) {
      const width = Math.min(L.maxWidth, span.r - span.l);
      const left = Math.min(Math.max(cx - width / 2, span.l), span.r - width);
      return { left: Math.round(left), bottom: Math.round(H - bandBottom), width: Math.floor(width) };
    }
    // Step up above the overlay under the center (or, failing that, the lowest-reaching blocker).
    const under = blockers.filter((o) => o.left - L.gap <= cx && cx <= o.left + o.width + L.gap);
    const pool = under.length > 0 ? under : blockers;
    const nextBottom = Math.min(...pool.map((o) => o.top)) - L.gap;
    if (!(nextBottom < bandBottom)) break;
    bandBottom = nextBottom;
  }
  // Nowhere below: pin to the top center (never observed at the demo's sizes; kept so it's never lost).
  const width = Math.min(L.maxWidth, W - 2 * L.inset);
  return { left: Math.round((W - width) / 2), bottom: Math.round(H - L.inset - h), width: Math.floor(width) };
}
