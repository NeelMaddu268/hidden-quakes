"use client";

import { useEffect, useState } from "react";

/**
 * DOM label elements by index, with their border-box sizes kept current by a ResizeObserver (e.g. when
 * the web font finishes loading), so per-frame code reads sizes without forcing a layout. drei's <Html>
 * renders its children in a separate React root that commits later than the component that owns them,
 * so elements are collected through stable ref callbacks (`ref(i)`) rather than read in an effect.
 * Without ResizeObserver (jsdom) sizes are read once on attach.
 */
export class LabelElements {
  private readonly els: (HTMLElement | null)[] = [];
  private readonly w: number[] = [];
  private readonly h: number[] = [];
  private readonly callbacks: ((el: HTMLElement | null) => void)[] = [];
  private observer: ResizeObserver | null = null;
  private attachVersion = 0;

  /** The element at `i`, or null until it mounts. */
  el(i: number): HTMLElement | null {
    return this.els[i] ?? null;
  }

  /** Border-box width / height (CSS px) of the element at `i`; 0 until measured or while detached. */
  width(i: number): number {
    return this.w[i] ?? 0;
  }

  height(i: number): number {
    return this.h[i] ?? 0;
  }

  /**
   * Bumped whenever an element attaches or detaches, so a frame hook that writes style only on change
   * can re-apply it to an element that mounted after its last write.
   */
  get version(): number {
    return this.attachVersion;
  }

  /** A stable ref callback for index `i` (the same function every render). */
  ref(i: number): (el: HTMLElement | null) => void {
    let cb = this.callbacks[i];
    if (!cb) {
      cb = (el) => this.attach(i, el);
      this.callbacks[i] = cb;
    }
    return cb;
  }

  /** Starts (or restarts) observing every attached element. */
  connect(): void {
    const o = this.getObserver();
    if (o) for (const el of this.els) if (el) o.observe(el);
  }

  disconnect(): void {
    this.observer?.disconnect();
  }

  private attach(i: number, el: HTMLElement | null): void {
    const old = this.els[i] ?? null;
    if (old === el) return;
    if (old) this.observer?.unobserve(old);
    this.els[i] = el;
    this.attachVersion++;
    if (!el) {
      this.w[i] = 0;
      this.h[i] = 0;
      return;
    }
    const o = this.getObserver();
    if (o) o.observe(el);
    else this.measure(i);
  }

  private measure(i: number): void {
    const el = this.els[i];
    this.w[i] = el ? el.offsetWidth : 0;
    this.h[i] = el ? el.offsetHeight : 0;
  }

  private getObserver(): ResizeObserver | null {
    if (!this.observer && typeof ResizeObserver !== "undefined") {
      this.observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const i = this.els.indexOf(entry.target as HTMLElement);
          if (i >= 0) this.measure(i);
        }
      });
    }
    return this.observer;
  }
}

/** One LabelElements for the component's lifetime, observing while mounted. */
export function useLabelElements(): LabelElements {
  const [labels] = useState(() => new LabelElements());
  useEffect(() => {
    labels.connect();
    return () => labels.disconnect();
  }, [labels]);
  return labels;
}
