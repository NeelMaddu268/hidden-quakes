"use client";

import { Html, type HtmlProps } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { useMemo } from "react";

/** Keep label roots attached to the canvas container when R3F connects its event target. */
export function SceneHtml(props: HtmlProps) {
  const gl = useThree((state) => state.gl);
  const portal = useMemo(() => ({ current: gl.domElement.parentElement! }), [gl]);
  return <Html {...props} portal={portal} />;
}
