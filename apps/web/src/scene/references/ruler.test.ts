import { Matrix4, PerspectiveCamera, Vector3, Vector4 } from "three";
import { describe, expect, it } from "vitest";
import { computeBounds } from "../camera/bounds";
import { CAMERA_FOV_DEG, presetPose } from "../camera/presets";
import { depthKmToSceneY } from "../coords";
import {
  declutterLabels,
  RULER_MAX_DEPTH_KM,
  rulerAnchor,
  rulerDepthsKm,
  rulerLayout,
  rulerRevealOpacity,
  SLICE_DEPTHS_KM,
  sliceSegments,
  stickyTitleT,
  tickLabel,
} from "./ruler";

const scene = {
  depthLabel: "Depth below site surface (ref 1628 m ASL)",
  originElevM: 1627.7,
  refSurfaceElevM: 1627.7,
  verticalExaggeration: 1,
};

describe("rulerLayout (acceptance: the ruler label reads SceneMeta.depthLabel)", () => {
  it("titles the ruler with SceneMeta.depthLabel verbatim, whatever it says", () => {
    const anchor = { x: -7, z: 0 };
    expect(rulerLayout(scene, anchor).title).toBe(scene.depthLabel);
    const other = { ...scene, depthLabel: "Tiefe unter Gelände (Bezug 1500 m ü. NN) — test ✓" };
    expect(rulerLayout(other, anchor).title).toBe(other.depthLabel);
  });

  it("ticks every 1 km from 0 to 6 km below the site surface, placed with depthKmToSceneY", () => {
    const layout = rulerLayout(scene, { x: -7, z: 1 });
    expect(layout.ticks.map((t) => t.depthKm)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    for (const t of layout.ticks) {
      expect(t.y).toBe(depthKmToSceneY(t.depthKm, scene));
      expect(t.label).toBe(tickLabel(t.depthKm));
    }
    expect(layout.ticks.map((t) => t.label)).toEqual(["0 km", "1 km", "2 km", "3 km", "4 km", "5 km", "6 km"]);
    expect(layout.titleAt).toEqual([-7, layout.ticks[0].y, 1]);
  });

  it("measures depth from refSurfaceElevM, not the origin, and honours vertical exaggeration", () => {
    const s = { ...scene, originElevM: 1500, refSurfaceElevM: 1627.7, verticalExaggeration: 2 };
    const layout = rulerLayout(s, { x: 0, z: 0 });
    expect(layout.ticks[0].y).toBeCloseTo(((1627.7 - 1500) / 1000) * 2, 9);
    expect(layout.ticks[6].y - layout.ticks[0].y).toBeCloseTo(-12, 9);
  });

  it("draws the spine plus one segment per tick, pointing west", () => {
    const layout = rulerLayout(scene, { x: -7, z: 0 });
    expect(layout.segments).toHaveLength(2 + 2 * layout.ticks.length);
    expect(layout.segments[0][1]).toBe(layout.ticks[0].y);
    expect(layout.segments[1][1]).toBe(layout.ticks[6].y);
    expect(layout.segments[3][0]).toBeLessThan(layout.segments[2][0]);
  });

  it("rulerDepthsKm validates its range", () => {
    expect(rulerDepthsKm()).toHaveLength(RULER_MAX_DEPTH_KM + 1);
    expect(rulerDepthsKm(3, 0.5)).toEqual([0, 0.5, 1, 1.5, 2, 2.5, 3]);
    expect(() => rulerDepthsKm(6, 0)).toThrow();
  });
});

describe("ruler placement is visible in every camera preset", () => {
  // Two test-local clouds: the dev harness's (12 km wide, 0.5–6 km deep) and a compact one
  // (4 km wide, 2–5 km deep, like the camera preset tests), whose side view doesn't reach the surface.
  function cloud(halfWidthKm: number, topKm: number, bottomKm: number): Float32Array {
    const pts: number[] = [];
    for (let i = 0; i < 400; i++) {
      const a = (i / 400) * Math.PI * 2;
      const r = halfWidthKm * ((i % 7) / 6);
      pts.push(Math.cos(a) * r, depthKmToSceneY(topKm + ((bottomKm - topKm) * (i % 11)) / 10, scene), Math.sin(a) * r);
    }
    return new Float32Array(pts);
  }

  function cameraFor(view: "oblique" | "side" | "plan", bounds: ReturnType<typeof computeBounds>, aspect: number) {
    const pose = presetPose(view, bounds, aspect);
    const cam = new PerspectiveCamera(CAMERA_FOV_DEG, aspect, 0.02, 1000);
    cam.position.set(...pose.position);
    cam.lookAt(new Vector3(...pose.target));
    cam.updateMatrixWorld(true);
    return cam;
  }

  const onScreen = (p: Vector3) => Math.abs(p.x) < 1 && Math.abs(p.y) < 1;
  const clouds = { dev: cloud(6, 0.5, 6), compact: cloud(2, 2, 5) };
  const HEIGHT_PX = 720;
  const TITLE_MARGIN_PX = 34;

  for (const [name, pts] of Object.entries(clouds)) {
    const bounds = computeBounds([pts], depthKmToSceneY(0, scene));
    for (const view of ["oblique", "side", "plan"] as const) {
      for (const aspect of [16 / 9, 4 / 3]) {
        it(`${name} cloud, ${view} @ ${aspect.toFixed(2)}: the title and the ruler are on screen`, () => {
          const cam = cameraFor(view, bounds, aspect);
          const layout = rulerLayout(scene, rulerAnchor(bounds));
          // The sticky title (as DepthRuler places it) is always inside the viewport.
          const viewProj = new Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
          const [top, bottom] = layout.segments;
          const c0 = new Vector4(...top, 1).applyMatrix4(viewProj);
          const c1 = new Vector4(...bottom, 1).applyMatrix4(viewProj);
          const t = stickyTitleT(c0.y, c0.w, c1.y, c1.w, 1 - (2 * TITLE_MARGIN_PX) / HEIGHT_PX);
          const titleAt = new Vector3(top[0], top[1] + t * (bottom[1] - top[1]), top[2]).project(cam);
          expect(onScreen(titleAt), `title at ${titleAt.toArray()}`).toBe(true);
          expect(titleAt.y).toBeLessThanOrEqual(1 - (2 * TITLE_MARGIN_PX) / HEIGHT_PX + 1e-9);
          // Enough of the ruler is on screen to read it: at least 3 of its 7 ticks.
          const visible = layout.ticks.filter((tk) => onScreen(new Vector3(...tk.labelAt).project(cam)));
          expect(visible.length, visible.map((tk) => tk.label).join()).toBeGreaterThanOrEqual(3);
          // With the dev cloud, the whole ruler fits in every preset.
          if (name === "dev") expect(visible).toHaveLength(layout.ticks.length);
        });
      }
    }
  }

  it("stands just west of the framed data, level with its centre", () => {
    const bounds = computeBounds([clouds.dev], 0);
    const a = rulerAnchor(bounds);
    expect(a.x).toBeLessThan(bounds.min[0]);
    expect(a.z).toBe(bounds.center[2]);
  });
});

describe("stickyTitleT", () => {
  it("stays at the surface while it's on screen", () => {
    expect(stickyTitleT(0.5, 1, -0.5, 1, 0.9)).toBe(0);
  });

  it("slides down the spine to where it enters the viewport, exactly (perspective-correct)", () => {
    // Top at NDC 1.5 (w = 2 → y = 3), bottom at NDC −0.5 (w = 4 → y = −2). Solve for NDC 0.9.
    const t = stickyTitleT(3, 2, -2, 4, 0.9);
    const y = 3 + t * (-2 - 3);
    const w = 2 + t * (4 - 2);
    expect(y / w).toBeCloseTo(0.9, 12);
  });

  it("clamps to the bottom when the whole ruler is above the viewport", () => {
    expect(stickyTitleT(3, 1, 2, 1, 0.9)).toBe(1);
  });
});

describe("declutterLabels", () => {
  const out = new Uint8Array(7);

  it("keeps every label when they're spaced out (side view)", () => {
    const ys = [100, 150, 200, 250, 300, 350, 400];
    expect([...declutterLabels(ys.map(() => 50), ys, 40, 15, out)]).toEqual([1, 1, 1, 1, 1, 1, 1]);
  });

  it("hides overlapping labels but keeps both ends (plan view, ruler seen end-on)", () => {
    const xs = [200, 186, 173, 161, 150, 140, 131]; // bunched horizontally, same height
    const ys = xs.map(() => 300);
    const mask = [...declutterLabels(xs, ys, 40, 15, out)];
    expect(mask[0]).toBe(1);
    expect(mask[6]).toBe(1);
    const shown = mask.flatMap((m, i) => (m ? [i] : []));
    for (let k = 1; k < shown.length; k++) {
      expect(Math.abs(xs[shown[k]] - xs[shown[k - 1]])).toBeGreaterThanOrEqual(40);
    }
  });

  it("shows only the top label when everything collapses to one point", () => {
    const xs = new Array(7).fill(10);
    expect([...declutterLabels(xs, xs, 40, 15, out)]).toEqual([1, 0, 0, 0, 0, 0, 0]);
  });

  it("writes into the caller's buffer (no per-frame allocation)", () => {
    expect(declutterLabels([0, 100], [0, 100], 40, 15, out)).toBe(out);
  });
});

describe("sliceSegments", () => {
  const extent = { eMin: -8000, eMax: 8000, nMin: -6000, nMax: 7000 };

  it("draws one square outline per km of depth, 1–6 km, over the surface extent", () => {
    expect(SLICE_DEPTHS_KM).toEqual([1, 2, 3, 4, 5, 6]);
    const segs = sliceSegments(extent, scene);
    expect(segs).toHaveLength(6 * 8);
    SLICE_DEPTHS_KM.forEach((d, k) => {
      for (const p of segs.slice(k * 8, k * 8 + 8)) expect(p[1]).toBe(depthKmToSceneY(d, scene));
    });
    const xs = segs.map((p) => p[0]);
    const zs = segs.map((p) => p[2]);
    expect([Math.min(...xs), Math.max(...xs)]).toEqual([-8, 8]);
    expect([Math.min(...zs), Math.max(...zs)]).toEqual([-7, 6]); // z = −n
  });
});

describe("rulerRevealOpacity", () => {
  it("hides the ruler on the pre-reveal frame and shows it once the terrain has faded", () => {
    expect(rulerRevealOpacity(1, 0.12)).toBe(0);
    expect(rulerRevealOpacity(0.12, 0.12)).toBe(1);
    expect(rulerRevealOpacity(0.56, 0.12)).toBeCloseTo(0.5, 12);
  });

  it("clamps outside the fade and never divides by zero", () => {
    expect(rulerRevealOpacity(1.2, 0.12)).toBe(0);
    expect(rulerRevealOpacity(0, 0.12)).toBe(1);
    expect(rulerRevealOpacity(0.5, 1)).toBe(1);
  });
});
