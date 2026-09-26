/**
 * The validation panel on screen: appears after the reveal, prints the mock bundle's own
 * `validation.json` + `meta.json` values (read from `apps/web/public/data/mock/`), and drops the
 * validation-sourced rows when the bundle has no `validation.json`.
 */
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ProviderRoot, StaticBundleProvider, type AnalysisSummary, type Validation } from "@/providers";
import { useDemo } from "@/state/demo";
import mockMeta from "../../../public/data/mock/meta.json";
import mockValidation from "../../../public/data/mock/validation.json";
import { Shell } from "../Shell";
import { bundleFiles, fakeFetch, type FixtureOptions } from "../test-fixture";
import { formatNumber } from "./format";
import { chanceNote } from "./rows";

const validation = mockValidation as unknown as Validation;
const summary = mockMeta.summary as unknown as AnalysisSummary;

beforeEach(() => {
  useDemo.getState().reset();
});

afterEach(() => {
  cleanup();
});

async function mountReady(options: FixtureOptions) {
  const provider = new StaticBundleProvider("mock", { fetchImpl: fakeFetch({ mock: bundleFiles("mock-run", options) }) });
  render(
    <ProviderRoot mode="mock" provider={provider}>
      <Shell />
    </ProviderRoot>,
  );
  await screen.findByTestId("counter-public");
  await act(async () => {});
}

const rowText = (id: string) => screen.getByTestId(`validation-row-${id}`).querySelector("dd")!.textContent;

describe("validation panel", () => {
  it("stays off the pre-reveal frame and appears once the reveal starts", async () => {
    await mountReady({ validation, summary });
    expect(screen.queryByTestId("validation-panel")).toBeNull();
    act(() => useDemo.getState().reveal());
    expect(screen.getByTestId("validation-panel")).toBeTruthy();
    act(() => useDemo.getState().reset());
    expect(screen.queryByTestId("validation-panel")).toBeNull();
  });

  it("prints the mock bundle's values, formatted, one row per source field", async () => {
    await mountReady({ validation, summary });
    act(() => useDemo.getState().reveal());
    // The mock fixture fills every field, so every row is present.
    const panel = screen.getByTestId("validation-panel");
    const labels = Array.from(panel.querySelectorAll("dt")).map((dt) => dt.textContent);
    expect(labels).toEqual([
      "Catalog recall",
      "Strict events",
      "Median stations",
      "Median timing misfit",
      "Depth resolution (synthetic, all stations)",
      "STA/LTA strict events",
      "PhaseNet vs STA/LTA gain",
      "Chance associations",
    ]);
    expect(rowText("recall")).toBe(`${formatNumber(summary.recoveredCatalogCount, 0)} / ${formatNumber(summary.publicCatalogCount, 0)}`);
    expect(rowText("strict")).toBe(
      `${formatNumber(summary.strictQualityCount, 0)} (${formatNumber(summary.strictAdditionalCount, 0)} not in public catalog)`,
    );
    expect(rowText("stations")).toBe(formatNumber(summary.medianStations, 1));
    expect(rowText("residual")).toBe(`${formatNumber(summary.medianRmsS, 3)} s`);
    expect(rowText("depth")).toBe(`±${formatNumber(validation.synthetic.medianVErrM, 0)} m`);
    const staltaFull = validation.baseline.find((r) => r.method === "stalta" && r.associationProfile === "full")!;
    expect(rowText("stalta")).toBe(`${formatNumber(staltaFull.tiers.A, 0)} of ${formatNumber(staltaFull.candidates, 0)} candidates`);
    expect(rowText("gain")).toBe(`${formatNumber(summary.baseline!.gain, 2)}×`);
    expect(rowText("chance")).toBe(formatNumber(validation.nullTest!.meanChanceEvents, 1));
    // Only the chance row carries a note; the other qualifiers live in the labels.
    expect(screen.queryByTestId("validation-note-stalta")).toBeNull();
    expect(screen.queryByTestId("validation-note-depth")).toBeNull();
    expect(screen.queryByTestId("validation-note-strict")).toBeNull();
    // The chance value is a mean over the null test's scrambles; its note says so from the data.
    expect(screen.getByTestId("validation-note-chance").textContent).toBe(chanceNote(validation.nullTest));
    expect(screen.getByTestId("validation-note-chance").textContent).toContain(`Mean of ${validation.nullTest!.nShuffles} timing scrambles`);
  });

  it("drops the validation-sourced rows when the bundle has no validation.json", async () => {
    await mountReady({ summary });
    act(() => useDemo.getState().reveal());
    const panel = screen.getByTestId("validation-panel");
    const ids = Array.from(panel.querySelectorAll("[data-testid^=validation-row-]")).map((el) =>
      el.getAttribute("data-testid")!.replace("validation-row-", ""),
    );
    expect(ids).toEqual(["recall", "strict", "stations", "residual"]);
  });

  it("puts values in the mono face with tabular figures", async () => {
    await mountReady({ validation, summary });
    act(() => useDemo.getState().reveal());
    const value = screen.getByTestId("validation-row-strict").querySelector("dd")!;
    expect(value.style.fontFamily).toMatch(/JetBrains Mono/);
    expect(value.style.fontVariantNumeric).toBe("tabular-nums");
  });
});
