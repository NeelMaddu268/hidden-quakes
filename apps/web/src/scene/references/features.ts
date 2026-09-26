// Geo features from the bundle's features.json (wells, facilities, boundaries), all in `colors.geo`.
// Honesty rules (docs/00, CLAUDE.md): a feature whose location isn't verified from a cited source is
// drawn dashed and labelled "approximate"; labels use the feature's own `name` and nothing else.

import { writeEnuToScene } from "../coords";
import type { GeoFeature } from "../types";

export const APPROXIMATE = "approximate";

export interface FeatureStyle {
  /** Unverified → dashed line / dashed ring; verified → solid. */
  dashed: boolean;
  approximate: boolean;
  /** Wells and facilities are the geothermal reference and glow a little; boundaries don't. */
  glow: boolean;
  /** Label text: the feature's own name, plus "approximate" when unverified. */
  label: string;
}

export function featureStyle(f: Pick<GeoFeature, "kind" | "name" | "source">): FeatureStyle {
  const approximate = f.source.verified !== true;
  return {
    dashed: approximate,
    approximate,
    glow: f.kind !== "boundary",
    label: approximate ? `${f.name} (${APPROXIMATE})` : f.name,
  };
}

type Vec3 = [number, number, number];

export type FeatureGeometry =
  | { type: "point"; at: Vec3 }
  | { type: "line"; points: Vec3[]; closed: boolean };

/**
 * Scene geometry for a feature: facilities (and any one-point path) are a point at path[0]; wells are
 * a polyline through their trajectory; boundaries are closed polylines. Null when there's nothing to
 * draw (empty path, or a boundary with fewer than 3 vertices).
 */
export function featureGeometry(f: Pick<GeoFeature, "kind" | "path">, verticalExaggeration: number): FeatureGeometry | null {
  const pts: Vec3[] = f.path.map((p) => {
    const out: Vec3 = [0, 0, 0];
    writeEnuToScene(p, verticalExaggeration, out, 0);
    return out;
  });
  if (pts.length === 0) return null;
  if (f.kind === "facility" || pts.length === 1) return { type: "point", at: pts[0] };
  if (f.kind === "boundary") {
    if (pts.length < 3) return null;
    return { type: "line", points: [...pts, pts[0]], closed: true };
  }
  return { type: "line", points: pts, closed: false };
}

/** Where a feature's label is pinned: the top of the feature (wellhead, facility, highest vertex). */
export function featureLabelAnchor(geom: FeatureGeometry): Vec3 {
  if (geom.type === "point") return geom.at;
  let best = geom.points[0];
  for (const p of geom.points) if (p[1] > best[1]) best = p;
  return best;
}

/** Why a feature can't be drawn (for a loud console message), or null. */
export function featureIssue(f: Pick<GeoFeature, "id" | "kind" | "path">): string | null {
  if (f.path.length === 0) return `feature ${f.id} has an empty path`;
  if (f.kind === "boundary" && f.path.length < 3) return `boundary ${f.id} has ${f.path.length} vertices (< 3)`;
  return null;
}
