import { Matrix4, OrthographicCamera, PerspectiveCamera } from "three";
import { describe, expect, it } from "vitest";
import { betterHit, isClick, pickNearest, PICK_THRESHOLD_PX, type PickQuery } from "./pick";

// Identity view-projection: NDC = world xyz, so screen x = (x + 1) / 2 × width, y = (1 − y) / 2 × height.
const IDENTITY = new Matrix4().elements;
const W = 800;
const H = 600;

function query(x: number, y: number, over: Partial<PickQuery> = {}): PickQuery {
  return { viewProj: IDENTITY, width: W, height: H, x, y, thresholdPx: PICK_THRESHOLD_PX, ...over };
}

/** Screen px of an NDC point under the identity matrix. */
const sx = (ndcX: number) => ((ndcX + 1) / 2) * W;
const sy = (ndcY: number) => ((1 - ndcY) / 2) * H;

describe("pickNearest", () => {
  const positions = new Float32Array([
    0, 0, 0, // 0: screen center (400, 300)
    0.5, 0.5, 0, // 1: (600, 150)
    -0.5, -0.5, 0, // 2: (200, 450)
  ]);

  it("returns the instance under the pointer with its screen distance", () => {
    const hit = pickNearest(positions, query(sx(0.5) + 3, sy(0.5) - 4));
    expect(hit?.index).toBe(1);
    expect(hit?.distPx).toBeCloseTo(5, 4);
  });

  it("returns null outside the threshold (clicking empty space)", () => {
    expect(pickNearest(positions, query(sx(0) + 13, sy(0)))).toBeNull();
    expect(pickNearest(positions, query(sx(0) + 11.9, sy(0)))?.index).toBe(0);
    expect(pickNearest(positions, query(sx(0) + 20, sy(0), { thresholdPx: 21 }))?.index).toBe(0);
  });

  it("prefers the nearest on screen", () => {
    const close = new Float32Array([0, 0, 0, 0.02, 0, 0]); // 8 px apart on screen
    expect(pickNearest(close, query(sx(0.02) - 1, sy(0)))?.index).toBe(1);
    expect(pickNearest(close, query(sx(0) + 1, sy(0)))?.index).toBe(0);
  });

  it("skips invisible instances", () => {
    const hit = pickNearest(positions, query(sx(0), sy(0), { visible: (i) => i !== 0 }));
    expect(hit).toBeNull();
    const hit2 = pickNearest(positions, query(sx(0.5), sy(0.5), { visible: (i) => i !== 0 }));
    expect(hit2?.index).toBe(1);
  });

  it("skips points clipped by the near/far planes", () => {
    const outside = new Float32Array([0, 0, 1.5, 0, 0, -1.5]);
    expect(pickNearest(outside, query(sx(0), sy(0)))).toBeNull();
  });

  describe("with a real perspective camera", () => {
    const cam = new PerspectiveCamera(40, W / H, 0.02, 1000);
    cam.position.set(0, 0, 10);
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld();
    const vp = new Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse).elements;
    const focalPx = (cam.projectionMatrix.elements[5] * H) / 2;

    it("projects like three.js Vector3.project", () => {
      const p = new Float32Array([1.2, -0.7, -3]);
      const v = cam.position.clone().set(1.2, -0.7, -3).project(cam);
      const hit = pickNearest(p, query(((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H, { viewProj: vp }));
      expect(hit?.index).toBe(0);
      expect(hit?.distPx).toBeLessThan(1e-3);
      expect(hit?.depth).toBeCloseTo(13, 4); // clip w = distance along the view axis
    });

    it("breaks screen ties by camera distance: the nearer of two stacked events wins", () => {
      // Both on the view axis: same screen point, 5 km apart in depth. Listed far-first.
      const stacked = new Float32Array([0, 0, -5, 0, 0, 0]);
      const hit = pickNearest(stacked, query(W / 2 + 2, H / 2, { viewProj: vp }));
      expect(hit?.index).toBe(1);
    });

    it("never picks a point behind the camera", () => {
      const behind = new Float32Array([0, 0, 20]);
      expect(pickNearest(behind, query(W / 2, H / 2, { viewProj: vp }))).toBeNull();
    });

    it("widens the hit area to a glyph drawn larger than the threshold", () => {
      const p = new Float32Array([0, 0, 9.5]); // 0.5 km from the camera
      const radiusKm = 0.06;
      const glyphPx = (radiusKm * focalPx) / 0.5; // ~98 px at this distance
      expect(glyphPx).toBeGreaterThan(40);
      const q = query(W / 2 + 40, H / 2, { viewProj: vp });
      expect(pickNearest(p, q)).toBeNull();
      expect(pickNearest(p, { ...q, focalPx, radius: () => radiusKm })?.index).toBe(0);
    });
  });
});

describe("betterHit", () => {
  it("orders by screen distance, then depth, then index", () => {
    expect(betterHit({ index: 5, distPx: 3, depth: 9 }, { index: 1, distPx: 6, depth: 1 })).toBe(true);
    expect(betterHit({ index: 5, distPx: 3.2, depth: 2 }, { index: 1, distPx: 3, depth: 4 })).toBe(true);
    expect(betterHit({ index: 5, distPx: 3, depth: 4 }, { index: 1, distPx: 3, depth: 4 })).toBe(false);
    expect(betterHit({ index: 0, distPx: 3, depth: 4 }, null)).toBe(true);
  });
});

describe("isClick", () => {
  it("accepts a short, still press", () => {
    expect(isClick({ x: 10, y: 10, t: 0 }, { x: 12, y: 11, t: 120 })).toBe(true);
  });
  it("rejects an orbit drag (moved) or a long press", () => {
    expect(isClick({ x: 10, y: 10, t: 0 }, { x: 14, y: 10, t: 50 })).toBe(false);
    expect(isClick({ x: 10, y: 10, t: 0 }, { x: 10, y: 10, t: 300 })).toBe(false);
  });
});

 it("caps close-up hit areas to the rendered screen radius", () => {
   const viewProj = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
   expect(pickNearest([0,0,0], { viewProj, width:100, height:100, x:80, y:50,
     thresholdPx:12, focalPx:50, radius:()=>100, maxRadiusPx:18 })).toBeNull();
 });


describe("orthographic plan picking", () => {
  const cam = new OrthographicCamera(-4, 4, 3, -3, .1, 100);
  cam.position.set(0, 10, 0); cam.up.set(0, 0, -1); cam.lookAt(0, 0, 0); cam.updateMatrixWorld();
  const vp = new Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse).elements;

  it("picks the nearer event when vertically stacked points share constant clip w", () => {
    const stacked = new Float32Array([0, -5, 0, 0, 0, 0]); // farther event deliberately listed first
    expect(pickNearest(stacked, query(W / 2, H / 2, { viewProj: vp }))?.index).toBe(1);
  });
  it("keeps north up, east right and ignores clipped events", () => {
    const positions = new Float32Array([1, 0, -1, 1, 20, -1, 1, -200, -1]);
    expect(pickNearest(positions, query(500, 200, { viewProj: vp }))?.index).toBe(0);
  });
  it("makes a horizontal radius independent of event depth", () => {
    const q = query(W / 2 + 20, H / 2, { viewProj: vp, focalPx: 100, radius: () => .25 });
    expect(pickNearest([0, 0, 0], q)?.index).toBe(0);
    expect(pickNearest([0, -50, 0], q)?.index).toBe(0);
  });
});
