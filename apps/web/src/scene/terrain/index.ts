// Terrain (WEB-02): the baked DEM surface from public/terrain/, or the labelled abstract slab.
export { Terrain, FORCE_ABSTRACT_SLAB, ABSTRACT_SURFACE_LABEL, type TerrainProps } from "./Terrain";
export { useTerrainAsset, loadTerrain, urlForcesSlab, type TerrainAsset } from "./load";
export { surfaceExtentM } from "./grid";
