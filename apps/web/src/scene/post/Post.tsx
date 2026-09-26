"use client";

import { useFrame } from "@react-three/fiber";
import { Bloom, EffectComposer, Vignette, type BloomProps } from "@react-three/postprocessing";
import { useRef } from "react";
import { HalfFloatType } from "three";
import { sceneFx } from "../fx";

type BloomEffectInstance = NonNullable<Extract<NonNullable<BloomProps["ref"]>, { current: unknown }>["current"]>;

/**
 * Post stack (lane doc → Scene composition): bloom with a luminance threshold so only events glow,
 * plus a light vignette. Event cores render above 1.0 (uGlow) in a half-float buffer; terrain, lines
 * and labels stay below the threshold. The reveal swells bloom through `sceneFx.bloomBoost`.
 */
export const BLOOM = Object.freeze({
  intensity: 1.1,
  luminanceThreshold: 0.85,
  luminanceSmoothing: 0.2,
  radius: 0.7,
});

export function Post() {
  const bloom = useRef<BloomEffectInstance>(null);

  useFrame(() => {
    const b = bloom.current;
    if (b) b.intensity = BLOOM.intensity * sceneFx.bloomBoost;
  });

  return (
    <EffectComposer multisampling={4} frameBufferType={HalfFloatType}>
      <Bloom
        ref={bloom}
        mipmapBlur
        intensity={BLOOM.intensity}
        luminanceThreshold={BLOOM.luminanceThreshold}
        luminanceSmoothing={BLOOM.luminanceSmoothing}
        radius={BLOOM.radius}
      />
      <Vignette offset={0.3} darkness={0.45} />
    </EffectComposer>
  );
}
