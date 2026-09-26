"use client";

import type { MouseEvent } from "react";
import { useDemo } from "@/state/demo";
import styles from "./Shell.module.css";

/** The one call to action before the reveal. Mounted only while `phase === "public"`. */
export function RevealButton() {
  const onClick = (event: MouseEvent<HTMLButtonElement>) => {
    // Drop focus so the presenter's next Space goes to the keyboard map, not to a focused button.
    event.currentTarget.blur();
    useDemo.getState().reveal();
  };
  return (
    <button type="button" className={styles.reveal} onClick={onClick}>
      Reveal hidden signal
    </button>
  );
}
