// Where the depth-section panel sits in plan view, as a pure function of the viewport (CSS px) and the
// shell's measured edges, so the panel, the plan camera's framing and the scene labels all agree. The
// shell (H4) owns the overlay: the title block top-left (title, subtitle, mode line, buttons, download
// row), counters and filter pills top-right, the validation card and mode pills bottom-left; H3's legend
// and corner notes sit bottom-right, and the evidence drawer is the right edge at clamp(420px, 40vw,
// 720px). The panel docks in the left column between the title block and the validation card; when
// that column is too short (the card grew, the header grew: 1280×720 on the final build), it docks in
// the right column, under the counters and above the legend, instead of disappearing.

export const SECTION_LAYOUT = Object.freeze({
  /** The shell's --shell-inset. */
  inset: 24,
  /** The top edge when the shell's blocks can't be measured (the title block without its later rows). */
  top: 136,
  /**
   * Clears the bottom-left stack when the card has never been measured at this viewport size: mode pills
   * plus the validation card (shell/validation: bottom 24px + 2.75rem; with every row and its notes at
   * 13 px it is about 330 px tall at 1280 wide) and a gap. Once measured, the card's real top is used.
   */
  bottomReserve: 420,
  /** Gap kept from every measured shell edge (title block, card, counters, legend). */
  cardGap: 16,
  /** Gap kept between the panel and the evidence drawer. */
  drawerGap: 24,
  /**
   * The validation card's column (inset + at most 22rem + a gap): with the panel docked right, the plan
   * camera still keeps the data clear of the card on the left.
   */
  cardColumn: 24 + 352 + 24,
  widthFrac: 0.34,
  minWidth: 300,
  maxWidth: 760,
  heightFrac: 0.4,
  minHeight: 180,
  maxHeight: 480,
  /** Horizontal half-width of the REVEAL button region the panel must not cover (fraction of vw). */
  revealHalfWidthFrac: 0.2,
});

export interface PanelRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** The evidence drawer's width at this viewport width (drawer/styles.ts: clamp(420px, 40vw, 720px)). */
export function drawerWidthPx(viewportWidth: number): number {
  return Math.min(720, Math.max(420, 0.4 * viewportWidth));
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** The shell's measured edges (viewport CSS px); null when that block isn't on screen. */
export interface ShellEdges {
  /** Bottom of the top-left title block (title, subtitle, mode line, buttons, download row). */
  headerBottom: number | null;
  /** Top of the validation card (bottom-left; after the reveal). */
  cardTop: number | null;
  /** Bottom of the top-right block (counters and filter pills). */
  topRightBottom: number | null;
  /** Top of H3's bottom-right stack (the legend and the corner notes above it). */
  cornerTop: number | null;
}

export const NO_EDGES: Readonly<ShellEdges> = Object.freeze({
  headerBottom: null,
  cardTop: null,
  topRightBottom: null,
  cornerTop: null,
});

export interface DockedPanel extends PanelRect {
  side: "left" | "right";
}

const num = (v: number | null): v is number => v !== null && Number.isFinite(v);

/**
 * The panel's rect and side for a viewport and the shell's edges, or null when neither column has room
 * (the panel is then not shown; the plan map still works). Left column first: under the title block,
 * above the validation card (or, before the card exists, above where it will be). Else the right
 * column: under the counters and pills, above the legend. The width never reaches under the drawer's
 * width nor the centered REVEAL button, so the rect doesn't depend on whether the drawer is open (the
 * plan camera frames around it once).
 */
export function sectionDock(viewportWidth: number, viewportHeight: number, edges: ShellEdges = NO_EDGES): DockedPanel | null {
  return dockIn("left", viewportWidth, viewportHeight, edges) ?? dockIn("right", viewportWidth, viewportHeight, edges);
}

/**
 * The left-column dock alone (the panel's only place before WEB-13), with the validation card's top when
 * known; null when the left column is too short. Kept for the scrubber's and corner notes' overlap tests.
 */
export function sectionPanelRect(viewportWidth: number, viewportHeight: number, cardTopPx?: number | null): PanelRect | null {
  const d = dockIn("left", viewportWidth, viewportHeight, { ...NO_EDGES, cardTop: cardTopPx ?? null });
  return d && { left: d.left, top: d.top, width: d.width, height: d.height };
}

function dockIn(side: "left" | "right", W: number, H: number, edges: ShellEdges): DockedPanel | null {
  const L = SECTION_LAYOUT;
  if (!(W > 0) || !(H > 0)) return null;
  const maxByDrawer = W - drawerWidthPx(W) - L.drawerGap - L.inset;
  const maxByReveal = W * (0.5 - L.revealHalfWidthFrac) - L.inset;
  const width = Math.floor(Math.min(clamp(W * L.widthFrac, L.minWidth, L.maxWidth), maxByDrawer, maxByReveal));
  if (width < L.minWidth) return null;
  const wanted = clamp(H * L.heightFrac, L.minHeight, L.maxHeight);
  let top: number;
  let bottom: number;
  if (side === "left") {
    // Under the title block; above the validation card (or where it will be).
    top = num(edges.headerBottom) ? Math.max(L.inset, edges.headerBottom + L.cardGap) : L.top;
    bottom = num(edges.cardTop) ? edges.cardTop - L.cardGap : H - L.bottomReserve;
  } else {
    // Under the counters and filter pills; above the legend and corner notes.
    top = num(edges.topRightBottom) ? Math.max(L.inset, edges.topRightBottom + L.cardGap) : L.top;
    bottom = num(edges.cornerTop) ? edges.cornerTop - L.cardGap : H - L.bottomReserve;
  }
  const height = Math.min(wanted, bottom - top);
  if (!(height >= L.minHeight)) return null;
  return {
    side,
    left: side === "left" ? L.inset : W - L.inset - width,
    top: Math.round(top),
    width,
    height: Math.floor(height),
  };
}

/**
 * CSS px the plan camera keeps clear of data on each side: the panel plus a gap on its side, and with a
 * right-docked panel the validation card's column on the left (the card and the panel both stay up).
 * 0 on both sides without a panel.
 */
export function planReservePx(dock: DockedPanel | null, viewportWidth: number): { left: number; right: number } {
  if (!dock) return { left: 0, right: 0 };
  return dock.side === "left"
    ? { left: dock.left + dock.width + SECTION_LAYOUT.inset, right: 0 }
    : { left: SECTION_LAYOUT.cardColumn, right: viewportWidth - dock.left + SECTION_LAYOUT.inset };
}
