"use client";

import { useFrame } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import { useDemo } from "../../state/demo";
import { sceneFx } from "../fx";
import { LOOK } from "../look";
import { REVEAL_FRAME_PRIORITY } from "../reveal/RevealDriver";
import { stepTime, TIME_ALL, timeNowRel, type TimeStep } from "./clock";
import { listenTNow } from "./listen";

/**
 * Time mode's clock (WEB-06), inside the canvas so playback advances with rendered frames. Each frame,
 * before any layer reads `sceneFx`: starts the replay when time mode takes effect after the reveal
 * (playhead at windowStart, playing), advances a playing replay at LOOK.time.playbackRate and stops it
 * at the window end, then publishes "now" (seconds since windowStart) as `sceneFx.timeNowRel`. While a
 * Listen clip plays (WEB-10, ./listen.ts) the playhead follows the audio's currentTime instead.
 * Playback is one small store `set` per frame, like the reveal's progress (docs/02 routes tNow to the
 * shell and the scrubber through the store).
 */
export function TimeDriver({ windowStart, windowEnd }: { windowStart: number; windowEnd: number }) {
  useEffect(
    () => () => {
      sceneFx.timeNowRel = TIME_ALL;
    },
    [],
  );

  const step = useRef<TimeStep>({ tNow: 0, playing: false });

  useFrame((_, delta) => {
    const s = useDemo.getState();
    const next = step.current;
    if (stepTime(s, delta, windowStart, windowEnd, LOOK.time.playbackRate, next, listenTNow())) {
      if (next.tNow !== s.tNow) s.setTNow(next.tNow);
      if (next.playing !== s.playing) s.setPlaying(next.playing);
    }
    sceneFx.timeNowRel = timeNowRel(useDemo.getState(), windowStart);
  }, REVEAL_FRAME_PRIORITY);

  return null;
}
