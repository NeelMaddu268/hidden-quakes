// Terrain (WEB-02): the baked DEM surface from public/terrain/, or the labelled abstract slab.
export { Terrain, type TerrainProps } from "./Terrain";
export { FORCE_ABSTRACT_SLAB, ABSTRACT_SURFACE_LABEL, useSurfaceChoice } from "./surface";
export { useTerrainAsset, loadTerrain, urlForcesSlab, TERRAIN_BASE_URL, type TerrainAsset } from "./load";
export { surfaceExtentM } from "./grid";
export { RENDER_ORDER } from "./renderOrder";
