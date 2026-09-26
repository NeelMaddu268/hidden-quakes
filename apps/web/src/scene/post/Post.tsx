"use client";

import { useFrame, useThree } from "@react-three/fiber";
import { Bloom, EffectComposer, Vignette, type BloomProps } from "@react-three/postprocessing";
import { useRef } from "react";
import { HalfFloatType } from "three";
import { sceneFx } from "../fx";
import { LOOK, msaaSamplesFor } from "../look";

type BloomEffectInstance = NonNullable<Extract<NonNullable<BloomProps["ref"]>, { current: unknown }>["current"]>;

/**
 * Post stack (lane doc → Scene composition): bloom with a luminance threshold so only events (and the
 * geothermal reference) glow, plus a light vignette. The threshold is designed against the token
 * colors in scene/look.ts. The reveal swells bloom through `sceneFx.bloomBoost`.
 */
export function Post() {
  const bloom = useRef<BloomEffectInstance>(null);
  const dpr = useThree((s) => s.viewport.dpr);
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);
  const samples = msaaSamplesFor(width * height * dpr * dpr);

  useFrame(() => {
    const b = bloom.current;
    if (b) b.intensity = LOOK.bloom.intensity * sceneFx.bloomBoost;
  });

  return (
    <EffectComposer multisampling={samples} frameBufferType={HalfFloatType}>
      <Bloom
        ref={bloom}
        mipmapBlur
        intensity={LOOK.bloom.intensity}
        luminanceThreshold={LOOK.bloom.luminanceThreshold}
        luminanceSmoothing={LOOK.bloom.luminanceSmoothing}
        radius={LOOK.bloom.radius}
      />
      <Vignette offset={LOOK.vignette.offset} darkness={LOOK.vignette.darkness} />
    </EffectComposer>
  );
}
