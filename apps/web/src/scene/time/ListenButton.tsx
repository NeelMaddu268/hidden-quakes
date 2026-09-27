"use client";

import { colors } from "@hq/visualization";
import { useEffect, useRef, useState } from "react";
import { audioToTNow, listenLabel, listenTitle, type ListenManifest } from "../../audio/manifest";
import { useDemo } from "../../state/demo";
import { listenClock, setListenClock, type ListenClock } from "./listen";

/** Speaker glyph, or a stop square while the clip plays (inline SVG, token colors). */
function SpeakerIcon({ on }: { on: boolean }) {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" style={{ flex: "none" }}>
      {on ? (
        <rect x="2.5" y="2.5" width="7" height="7" fill="currentColor" />
      ) : (
        <>
          <path d="M1.5 4.5 H3.8 L6.8 2 V10 L3.8 7.5 H1.5 Z" fill="currentColor" />
          <path d="M8.4 4 Q9.6 6 8.4 8 M9.8 2.8 Q11.6 6 9.8 9.2" stroke="currentColor" fill="none" strokeWidth="1" />
        </>
      )}
    </svg>
  );
}

/**
 * Listen (WEB-10): plays the busiest-hour clip and replays that hour in time mode, so candidate events
 * light up as their pops arrive. Starting it sets the playhead to the clip's start and hands it to the
 * audio (./listen.ts; the TimeDriver follows audio.currentTime every frame). Pressing it again, the clip
 * ending, the replay's pause, scrubbing, leaving time mode or a reset all stop it and leave the clock
 * where the audio was. A play() the browser refuses (autoplay policy, no playable source) restores the
 * clock and says so on the button.
 */
export function ListenButton({ clip }: { clip: ListenManifest }) {
  const audio = useRef<HTMLAudioElement>(null);
  /** Ends the current session (pauses, releases the playhead, unsubscribes); null when idle. */
  const stopRef = useRef<(() => void) | null>(null);
  const [listening, setListening] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  // Unmounting mid-clip (time mode off, the scrubber hidden by the drawer) stops the clip and the replay.
  useEffect(
    () => () => {
      if (!stopRef.current) return;
      stopRef.current();
      useDemo.getState().setPlaying(false);
    },
    [],
  );

  /** Leave the clock where the audio is and stop (the store subscription ends the session). */
  const finish = () => {
    const el = audio.current;
    if (!stopRef.current || !el) return;
    const s = useDemo.getState();
    s.setTNow(audioToTNow(clip, el.currentTime));
    s.setPlaying(false);
  };

  const start = () => {
    const el = audio.current;
    if (!el) return;
    const s = useDemo.getState();
    const before = s.tNow;
    const session: ListenClock = { manifest: clip, audio: el };
    let unsubscribe = () => {};
    const stop = () => {
      if (stopRef.current !== stop) return;
      stopRef.current = null;
      unsubscribe();
      if (listenClock() === session) setListenClock(null);
      el.pause();
      setListening(false);
    };

    setFailed(null);
    el.currentTime = 0;
    setListenClock(session);
    s.setTNow(clip.startUtc);
    s.setPlaying(true); // turns time mode on if needed; the TimeDriver now follows the audio
    stopRef.current = stop;
    unsubscribe = useDemo.subscribe((st) => {
      if (!st.playing || !st.timeMode || st.phase !== "revealed") stop();
    });
    setListening(true);

    const played = el.play() as Promise<void> | undefined;
    played?.catch((err: unknown) => {
      if (stopRef.current !== stop) return; // already stopped (our own pause aborts a pending play)
      stop();
      const st = useDemo.getState();
      st.setPlaying(false);
      if (before !== null) st.setTNow(before);
      const reason = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      console.warn(`[listen] the browser did not start the audio: ${reason}`);
      setFailed(reason);
    });
  };

  const label = listenLabel(clip);
  return (
    <>
      <button
        type="button"
        onClick={() => (stopRef.current ? finish() : start())}
        // Like the play button: Space / Enter stay here, every presenter key still reaches the shell.
        onKeyDown={(e) => {
          if (e.key === " " || e.key === "Enter") e.stopPropagation();
        }}
        aria-pressed={listening}
        aria-label={label}
        title={failed ? `Audio did not start (${failed}). ${listenTitle(clip)}` : listenTitle(clip)}
        data-testid="time-listen"
        data-failed={failed ? "true" : undefined}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          minWidth: 0,
          maxWidth: "45%",
          height: 22,
          padding: "0 9px",
          border: `1px solid ${listening ? colors.text : colors.contour}`,
          borderRadius: 999,
          background: "transparent",
          color: failed ? colors.textDim : colors.text,
          fontSize: 11,
          cursor: "pointer",
          whiteSpace: "nowrap",
        }}
      >
        <SpeakerIcon on={listening} />
        <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{label}</span>
      </button>
      <audio ref={audio} preload="none" onPause={finish} onEnded={finish} data-testid="time-listen-audio">
        {clip.sources.map((s) => (
          <source key={s.src} src={s.src} type={s.type} />
        ))}
      </audio>
    </>
  );
}
