// Hidden Quakes design tokens (docs/02 §6, values from docs/lanes/H3-visualization.md).
// The scene (H3) and the shell (H4) both read these, so the canvas and the overlay stay one palette.
// Colors carry meaning: `recovered` is the only accent, `alert` is for the SYNTHETIC banner and errors only.

export type ColorToken =
  | "bg"
  | "surface"
  | "terrain"
  | "contour"
  | "text"
  | "textDim"
  | "public"
  | "recovered"
  | "strictHalo"
  | "geo"
  | "pickP"
  | "pickS"
  | "station"
  | "alert";

export const colors: Readonly<Record<ColorToken, string>> = {
  bg: "#07090C", // page and scene background
  surface: "#0E1217", // drawer, panels
  terrain: "#1A2027", // terrain base
  contour: "#2A333D", // contours, slices, ruler ticks
  text: "#E6E9ED", // primary text, counters
  textDim: "#8A94A0", // labels, secondary
  public: "#DCE6F2", // public-catalog events
  recovered: "#FFB547", // recovered events: the only accent
  strictHalo: "#FFD08A", // Tier A halos
  geo: "#7FE0CF", // geothermal reference
  pickP: "#5AA9FF", // P picks
  pickS: "#FF8A4C", // S picks
  station: "#7C8B99", // station glyphs
  alert: "#FF4D4D", // SYNTHETIC banner and errors only
};

// Inter for UI, JetBrains Mono (tabular figures) for numbers. The shell loads the faces with next/font;
// if it exposes them as CSS variables `--font-inter` / `--font-jetbrains-mono` they win, otherwise the
// named families and then system fallbacks apply. Pair `fonts.mono` with `numeric` for counters.
export const fonts: Readonly<{ ui: string; mono: string }> = {
  ui: "var(--font-inter, Inter), ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
  mono: "var(--font-jetbrains-mono, 'JetBrains Mono'), ui-monospace, SFMono-Regular, Menlo, monospace",
};

/** CSS for numbers that must not jitter while they count (counters, drawer stats). */
export const numeric = { fontVariantNumeric: "tabular-nums" } as const;

// Motion: 150 ms micro, 600 ms state, 1,200 ms scene moves, cubic-out. Nothing loops except the Live pulse.
export const motion: Readonly<{ micro: 150; state: 600; scene: 1200; ease: string }> = {
  micro: 150,
  state: 600,
  scene: 1200,
  ease: "cubic-bezier(0.33, 1, 0.68, 1)", // easeOutCubic; `easeOutCubic` below is the same curve for JS
};

/** The JS twin of `motion.ease` for animations driven per frame (camera, uniforms). */
export function easeOutCubic(t: number): number {
  const c = t <= 0 ? 0 : t >= 1 ? 1 : t;
  const inv = 1 - c;
  return 1 - inv * inv * inv;
}

export type TierToken = "A" | "B" | "C";

// Tier is carried by opacity and size, never by color alone (docs/lanes/H3 → Scene composition).
export const tierStyle: Readonly<Record<TierToken, { opacity: number; size: number }>> = {
  A: { opacity: 1.0, size: 1.0 },
  B: { opacity: 0.6, size: 0.8 },
  C: { opacity: 0.3, size: 0.65 },
};

/** Opacity of Tier B and C under STRICT (they fade over `motion.state`, never vanish outright). */
export const strictFadeOpacity = 0.05;

/** Every color as a CSS custom property (`--hq-bg`, `--hq-text-dim`, …) for `:root` or a wrapper style. */
export function cssVariables(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const key of Object.keys(colors) as ColorToken[]) {
    out[`--hq-${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`] = colors[key];
  }
  out["--hq-font-ui"] = fonts.ui;
  out["--hq-font-mono"] = fonts.mono;
  out["--hq-ease"] = motion.ease;
  out["--hq-motion-micro"] = `${motion.micro}ms`;
  out["--hq-motion-state"] = `${motion.state}ms`;
  out["--hq-motion-scene"] = `${motion.scene}ms`;
  return out;
}
