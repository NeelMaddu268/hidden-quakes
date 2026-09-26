"use client";

import { ProviderRoot } from "@/providers";
import { Scene } from "@/scene";
import { EvidenceDrawer } from "@/drawer";
import { Shell } from "@/shell";
// WEB-06 (H3): swap for `import { TimeScrubber } from "@/scene/time";`
import { TimeScrubber } from "@/shell/placeholders";

// Page composition (docs/01 → Web app). The scene is the full-bleed canvas; the shell is the
// pointer-transparent overlay; the drawer and the scrubber are H3's and mount beside them.
export default function Page() {
  return (
    <ProviderRoot>
      <Scene />
      <Shell />
      <EvidenceDrawer />
      <TimeScrubber />
    </ProviderRoot>
  );
}
