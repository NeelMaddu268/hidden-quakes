// Placeholders for H3 components that don't exist yet. Each renders nothing; page.tsx swaps the
// import for the real one when the ticket lands, and nothing else changes.

// WEB-05 (H3): replace with `import { EvidenceDrawer } from "@/drawer";` in page.tsx.
export function EvidenceDrawer() {
  return null;
}

// WEB-06 (H3): replace with `import { TimeScrubber } from "@/scene/time";` in page.tsx.
export function TimeScrubber() {
  return null;
}
