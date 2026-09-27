"use client";

import { colors } from "@hq/visualization";
import { useEffect, useRef, useState } from "react";
import { LISTEN_CLIPS, listenLabel, listenTitle, useListenManifest } from "../audio/manifest";

/**
 * Listen to the hero event (WEB-10): a small play button in the evidence drawer, shown only when the
 * drawer's event is the hero clip's `heroEventId` (from public/audio/hidden-quakes-hero.json). Plays the
 * short sped-up clip centred on the event's origin; it does not move the scene clock. Label, station,
 * channel, window and speed all come from the manifest; without a valid manifest nothing renders.
 */
export function HeroListen({ eventId }: { eventId: string }) {
  const clip = useListenManifest(LISTEN_CLIPS.hero);
  const audio = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  const show = clip !== null && clip.heroEventId === eventId;

  // Another event (or a closed drawer) never keeps the clip playing. The <audio> is gone by the time its
  // pause event fires, so onPause never runs: reset the button here, or it comes back stuck on "stop".
  useEffect(() => {
    const el = audio.current;
    return () => {
      el?.pause();
      setPlaying(false);
    };
  }, [show]);

  if (!show) return null;

  const toggle = () => {
    const el = audio.current;
    if (!el) return;
    if (playing) {
      el.pause();
      return;
    }
    setFailed(null);
    el.currentTime = 0;
    setPlaying(true);
    const played = el.play() as Promise<void> | undefined;
    played?.catch((err: unknown) => {
      const reason = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      if (err instanceof Error && err.name === "AbortError") return; // our own pause beat it
      console.warn(`[listen] the browser did not start the audio: ${reason}`);
      setPlaying(false);
      setFailed(reason);
    });
  };

  const label = listenLabel(clip);
  return (
    <div className="hqd-section" data-testid="hero-listen">
      <button
        type="button"
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === " " || e.key === "Enter") e.stopPropagation();
        }}
        aria-pressed={playing}
        aria-label={label}
        title={failed ? `Audio did not start (${failed}). ${listenTitle(clip)}` : listenTitle(clip)}
        data-testid="hero-listen-button"
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 8,
          maxWidth: "100%",
          padding: "5px 12px",
          border: `1px solid ${playing ? colors.text : colors.contour}`,
          borderRadius: 999,
          background: "transparent",
          color: failed ? colors.textDim : colors.text,
          font: "inherit",
          fontSize: 12,
          cursor: "pointer",
        }}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" style={{ flex: "none" }}>
          {playing ? (
            <rect x="2.5" y="2.5" width="7" height="7" fill="currentColor" />
          ) : (
            <path d="M3 1.5 L10.5 6 L3 10.5 Z" fill="currentColor" />
          )}
        </svg>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
      </button>
      <audio
        ref={audio}
        preload="none"
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        data-testid="hero-listen-audio"
      >
        {clip.sources.map((s) => (
          <source key={s.src} src={s.src} type={s.type} />
        ))}
      </audio>
    </div>
  );
}
