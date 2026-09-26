// PUBLIC / ALL / STRICT (docs/02 EventFilter) as data: which candidate instances each filter shows,
// and the consistency check between what the scene draws and what the shell's counters print.

import { strictFadeOpacity } from "@hq/visualization";
import type { EventFilter } from "../../state/demo";
import type { AnalysisSummary, SeismicEvent } from "../types";
import { FILTER_LOOK } from "./fade";

/**
 * Does `filter` show a candidate of this tier, after the reveal? Derived from the same look table the
 * renderer draws with (FILTER_LOOK), so the count and the pixels can't drift apart. "Shown" means drawn
 * above the STRICT background weight: B and C under STRICT stay faintly on screen at
 * `strictFadeOpacity` as context, and are not counted, which is exactly what the shell's STRICT
 * counter (`summary.strictQualityCount`) reports.
 */
export function isCandidateShown(tier: SeismicEvent["tier"], filter: EventFilter): boolean {
  const look = FILTER_LOOK[filter];
  const tierOpacity = tier === "A" ? look.tierA : tier === "B" ? look.tierB : look.tierC;
  return look.candidates * tierOpacity > strictFadeOpacity;
}

/** Ids of the candidate events a filter shows, in bundle order. */
export function shownCandidateIds(events: readonly Pick<SeismicEvent, "id" | "tier">[], filter: EventFilter): string[] {
  const out: string[] = [];
  for (const ev of events) if (isCandidateShown(ev.tier, filter)) out.push(ev.id);
  return out;
}

/** Number of candidate instances a filter shows. */
export function shownCandidateCount(events: readonly Pick<SeismicEvent, "tier">[], filter: EventFilter): number {
  let n = 0;
  for (const ev of events) if (isCandidateShown(ev.tier, filter)) n++;
  return n;
}

/**
 * Mismatches between what the scene would show and what the summary (the shell's counters) says.
 * STRICT must show exactly `strictQualityCount` instances and ALL exactly `candidateCount`. A mismatch
 * means the bundle is internally inconsistent; the scene logs it loudly rather than hiding it.
 */
export function filterCountIssues(
  events: readonly Pick<SeismicEvent, "tier">[],
  summary: Pick<AnalysisSummary, "strictQualityCount" | "candidateCount">,
): string[] {
  const issues: string[] = [];
  const strict = shownCandidateCount(events, "strict");
  const all = shownCandidateCount(events, "all");
  if (strict !== summary.strictQualityCount) {
    issues.push(`STRICT would show ${strict} events but summary.strictQualityCount is ${summary.strictQualityCount}`);
  }
  if (all !== summary.candidateCount) {
    issues.push(`ALL would show ${all} events but summary.candidateCount is ${summary.candidateCount}`);
  }
  return issues;
}
