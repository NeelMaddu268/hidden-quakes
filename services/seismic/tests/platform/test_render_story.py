"""DEMO-03: ``scripts/render-story.py`` fills the pitch and Devpost placeholders from a bundle.

Renders the synthetic mock bundle into ``tmp_path`` and checks that every placeholder in both
docs is substituted or listed as manual, that a synthetic bundle is banner-marked, that a
missing ``validation.json`` renders as "not available" rather than failing, and that a
placeholder the script does not know fails loudly."""

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "render-story.py"
MOCK_BUNDLE = REPO_ROOT / "apps" / "web" / "public" / "data" / "mock"
SHOWCASE_BUNDLE = REPO_ROOT / "apps" / "web" / "public" / "data" / "showcase"
PITCH_DOC = REPO_ROOT / "docs" / "demo" / "pitch-and-qa.md"
DEVPOST_DOC = REPO_ROOT / "docs" / "demo" / "devpost.md"
SHOTS_DOC = REPO_ROOT / "docs" / "demo" / "video-shot-list.md"
EXPO_DOC = REPO_ROOT / "docs" / "demo" / "expo-card.md"
BANNER = "SYNTHETIC BUNDLE, NOT FOR SUBMISSION"


@pytest.fixture(scope="module")
def story() -> ModuleType:
    """Import the renderer from its hyphenated path as a module."""
    spec = importlib.util.spec_from_file_location("render_story", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_story"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rendered(story: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list, str]:
    out = tmp_path_factory.mktemp("story")
    rows, report = story.render(MOCK_BUNDLE, out)
    return out, rows, report


def strip_fences(text: str) -> str:
    kept: list[str] = []
    fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if not fence:
            kept.append(line)
    return "\n".join(kept)


def test_writes_the_three_files_with_the_synthetic_banner(
    rendered: tuple[Path, list, str], story: ModuleType
) -> None:
    out, _, report = rendered
    for name in story.OUTPUT_FILES:
        text = (out / name).read_text(encoding="utf-8")
        assert text.startswith("# " + BANNER), name
    assert BANNER in report
    assert "| Placeholder |" in report  # the numbers table is printed to stdout too


def test_every_placeholder_is_substituted_or_manual(
    rendered: tuple[Path, list, str], story: ModuleType
) -> None:
    out, rows, _ = rendered
    by_doc = {
        "pitch": (PITCH_DOC, story.PITCH_FILLED_MD),
        "devpost": (DEVPOST_DOC, story.DEVPOST_FILLED_MD),
        "pitch (shot list)": (SHOTS_DOC, story.SHOTS_FILLED_MD),
        "pitch (expo card)": (EXPO_DOC, story.EXPO_FILLED_MD),
    }
    reported = {(r.doc, r.name) for r in rows}
    assert all(r.status in story.STATUSES for r in rows)
    literal_tokens = {r.name for r in rows if r.status == story.STATUS_LITERAL}
    for label, (doc, filled_name) in by_doc.items():
        kind = label.split(" ")[0]  # the token form: the shot list uses the pitch's
        source_tokens = {tok for _, tok in story.scan_tokens(doc.read_text(encoding="utf-8"), kind)}
        assert source_tokens, kind
        # Every token in the doc has a row (a context-dependent token like {n} has labelled rows).
        for tok in source_tokens:
            assert any(
                name == tok or name.startswith(tok + " ") for d, name in reported if d == doc.name
            ), tok
        filled = strip_fences((out / filled_name).read_text(encoding="utf-8"))
        leftover = {tok for _, tok in story.scan_tokens(filled, kind)} - literal_tokens
        assert leftover == set(), leftover
        assert "copy-ok" not in filled
    # The manual list is explicit: no bundle field, so the pitch says where to read it.
    manual = {r.name for r in rows if r.status == story.STATUS_MANUAL}
    assert "{depthBand}" in manual and "{latency}" in manual
    assert "<video URL>" in manual
    # The repository URL comes from the checkout's origin remote, in https form.
    repo = next(r for r in rows if r.name == "<repo URL>")
    assert repo.status in {story.STATUS_VALUE, story.STATUS_MANUAL}
    if repo.status == story.STATUS_VALUE:
        assert repo.text.startswith("https://github.com/") and not repo.text.endswith(".git")
    # The deployed URL is the public link docs/deploy.md names, never a team-internal alias.
    deployed = next(r for r in rows if r.name == "<deployed URL>")
    assert deployed.status == story.STATUS_VALUE
    assert deployed.text.startswith("https://") and ".vercel.app" in deployed.text
    assert "projects" not in deployed.text


def test_values_come_from_the_bundle(rendered: tuple[Path, list, str]) -> None:
    out, rows, _ = rendered
    meta = json.loads((MOCK_BUNDLE / "meta.json").read_text(encoding="utf-8"))
    validation = json.loads((MOCK_BUNDLE / "validation.json").read_text(encoding="utf-8"))
    value = {(r.doc, r.name): r.text for r in rows}
    assert value[("pitch-and-qa.md", "{publicCatalogCount}")] == str(
        meta["summary"]["publicCatalogCount"]
    )
    assert value[("pitch-and-qa.md", "{windowLabel}")] == meta["run"]["windowLabel"]
    assert value[("devpost.md", "<from meta.json: summary.candidateCount>")] == str(
        meta["summary"]["candidateCount"]
    )
    full = {
        row["method"]: row for row in validation["baseline"] if row["associationProfile"] == "full"
    }
    assert value[("pitch-and-qa.md", "{strictPhasenet}")] == str(full["phasenet"]["tiers"]["A"])
    assert value[("pitch-and-qa.md", "{strictStalta}")] == str(full["stalta"]["tiers"]["A"])
    hero = json.loads(
        (MOCK_BUNDLE / "evidence" / f"{meta['scene']['heroEventId']}.json").read_text()
    )
    events = json.loads((MOCK_BUNDLE / "events.json").read_text(encoding="utf-8"))
    hero_event = next(e for e in events if e["id"] == meta["scene"]["heroEventId"])
    # The drawer header prints quality.nStations; the evidence trace count is only a fallback.
    assert value[("pitch-and-qa.md", "{nStations}")] == str(hero_event["quality"]["nStations"])
    assert "traces" in hero  # the evidence file exists, and is not what the count comes from
    hidden = [e for e in events if e["tier"] == "A" and e["catalogMatch"] is None]
    best = min(
        hidden, key=lambda e: (-e["quality"]["nStations"], e["quality"]["rmsS"], e["t"], e["id"])
    )
    assert value[("pitch-and-qa.md", "{hiddenHeroId}")] == best["id"]
    pitch = (out / "pitch-filled.md").read_text(encoding="utf-8")
    assert f"lists {meta['summary']['publicCatalogCount']} events" in pitch
    # A code span wrapping two placeholders keeps its backticks.
    devpost = (out / "devpost-filled.md").read_text(encoding="utf-8")
    assert (
        f"`{meta['summary']['publicCatalogCount']} PUBLIC → {meta['summary']['candidateCount']} RECOVERED`"
        in devpost
    )


needs_showcase = pytest.mark.skipif(
    not (SHOWCASE_BUNDLE / "meta.json").is_file(), reason="no showcase bundle in this checkout"
)


def _meta_validation_bundle(tmp_path: Path, source: Path) -> Path:
    """A bundle with only ``meta.json`` and ``validation.json`` copied from ``source`` (the hero
    evidence file is left out; it renders as not available)."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("meta.json", "validation.json"):
        shutil.copy(source / name, bundle / name)
    return bundle


@needs_showcase
def test_science_qa_placeholders_come_from_the_showcase_bundle(
    story: ModuleType, tmp_path: Path
) -> None:
    """DEMO-03 science Q&A: the baseline misfit/station/candidate counts, the matched strict
    count, the held-out depth difference and the fewest stations on a recovered public event
    render from the run of record's bundle, rounded as the doc quotes them."""
    bundle = _meta_validation_bundle(tmp_path, SHOWCASE_BUNDLE)
    rows, _ = story.render(bundle, tmp_path / "out")
    value = {r.name: r for r in rows if r.doc == "pitch-and-qa.md"}
    meta = json.loads((SHOWCASE_BUNDLE / "meta.json").read_text(encoding="utf-8"))
    validation = json.loads((SHOWCASE_BUNDLE / "validation.json").read_text(encoding="utf-8"))
    full = {
        row["method"]: row for row in validation["baseline"] if row["associationProfile"] == "full"
    }
    held_out = meta["run"]["locator"]["statics"]["crossValidatedOffsets"]["after"]
    expected = {
        "{staltaCandidates}": str(full["stalta"]["candidates"]),
        "{staltaRecoveredPublic}": str(full["stalta"]["recoveredPublic"]),
        "{phasenetMedianRmsS}": f"{full['phasenet']['medianRmsS']:.3f}",
        "{staltaMedianRmsS}": f"{full['stalta']['medianRmsS']:.3f}",
        "{phasenetMedianStations}": f"{full['phasenet']['medianStations']:g}",
        "{staltaMedianStations}": f"{full['stalta']['medianStations']:g}",
        "{strictMatchedCount}": str(meta["run"]["tiering"]["counts"]["matched"]["A"]),
        "{medianStations}": f"{meta['summary']['medianStations']:g}",
        "{heldOutMedianAbsDzM}": f"{held_out['medianAbsDzM']:.0f}",
        "{medianVErrM}": f"{validation['synthetic']['medianVErrM']:.0f}",  # as the card rounds it
    }
    for token, text in expected.items():
        assert value[token].status == story.STATUS_VALUE, token
        assert value[token].text == text, token
    # The Tier B nStations bar is the fewest stations on any recovered public event: check it
    # against the events themselves.
    events = json.loads((SHOWCASE_BUNDLE / "events.json").read_text(encoding="utf-8"))
    matched = [e["quality"]["nStations"] for e in events if e["catalogMatch"] is not None]
    assert value["{minMatchedStations}"].status == story.STATUS_VALUE
    assert value["{minMatchedStations}"].text == f"{min(matched):g}"
    filled = (tmp_path / "out" / story.PITCH_FILLED_MD).read_text(encoding="utf-8")
    dz = expected["{heldOutMedianAbsDzM}"]
    assert f"the median depth difference from the catalog is {dz} m" in filled


@needs_showcase
def test_min_matched_stations_needs_the_worst_of_matched_bar(
    story: ModuleType, tmp_path: Path
) -> None:
    """A Tier B quantile other than zero makes the nStations bar something other than the worst
    recovered public event, so the clause is marked for omission instead of misquoted."""
    bundle = _meta_validation_bundle(tmp_path, SHOWCASE_BUNDLE)
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    meta["run"]["tiering"]["thresholds"]["quantiles"]["B"] = 0.1
    (bundle / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    rows, _ = story.render(bundle, tmp_path / "out")
    row = {r.name: r for r in rows if r.doc == "pitch-and-qa.md"}["{minMatchedStations}"]
    assert row.status == story.STATUS_CONDITION and "omit this clause" in row.text


@needs_showcase
def test_baseline_fields_need_both_full_rows(story: ModuleType, tmp_path: Path) -> None:
    """The STA/LTA numbers are quoted only next to PhaseNet's: with the STA/LTA full row gone,
    every baseline placeholder is marked for omission."""
    bundle = _meta_validation_bundle(tmp_path, SHOWCASE_BUNDLE)
    validation = json.loads((bundle / "validation.json").read_text(encoding="utf-8"))
    validation["baseline"] = [
        row
        for row in validation["baseline"]
        if not (row["method"] == "stalta" and row["associationProfile"] == "full")
    ]
    (bundle / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    rows, _ = story.render(bundle, tmp_path / "out")
    by_name = {r.name: r for r in rows if r.doc == "pitch-and-qa.md"}
    for token in ("{staltaMedianRmsS}", "{phasenetMedianRmsS}", "{staltaMedianStations}"):
        assert by_name[token].status == story.STATUS_CONDITION, token
    # The card's own "A of N candidates" denominator (main's resolver) needs only the STA/LTA row.
    assert by_name["{staltaCandidates}"].status == story.STATUS_NOT_AVAILABLE


def test_missing_validation_renders_not_available(story: ModuleType, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copy(MOCK_BUNDLE / "meta.json", bundle / "meta.json")
    rows, report = story.render(bundle, tmp_path / "out")
    assert "validation.json is missing" in report
    by_name = {(r.doc, r.name): r for r in rows}
    err = by_name[("pitch-and-qa.md", "{medianVErrM}")]
    assert err.status == story.STATUS_NOT_AVAILABLE and "validation.json missing" in err.text
    # Conditional sentences gated on validation.json fields are marked for omission.
    null = by_name[("devpost.md", "<from validation.json: nullTest.meanChanceEvents>")]
    assert null.status == story.STATUS_CONDITION and "omit this sentence" in null.text
    # The hero evidence file is absent too.
    assert by_name[("pitch-and-qa.md", "{nStations}")].status == story.STATUS_NOT_AVAILABLE
    filled = (tmp_path / "out" / "pitch-filled.md").read_text(encoding="utf-8")
    assert "[not available: validation.json missing]" in filled


def test_condition_not_met_when_gate_is_null(story: ModuleType, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(MOCK_BUNDLE, bundle)
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    meta["summary"]["baseline"] = None
    (bundle / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    rows, _ = story.render(bundle, tmp_path / "out")
    gain = {(r.doc, r.name): r for r in rows}[("pitch-and-qa.md", "{gain}")]
    assert gain.status == story.STATUS_CONDITION
    assert "summary.baseline is null; omit this sentence" in gain.text


def test_real_bundle_has_no_banner(story: ModuleType, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(MOCK_BUNDLE, bundle)
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    meta["run"]["isSynthetic"] = False
    meta["scene"]["isSynthetic"] = False
    (bundle / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    _, report = story.render(bundle, tmp_path / "out")
    assert BANNER not in report
    for name in story.OUTPUT_FILES:
        assert BANNER not in (tmp_path / "out" / name).read_text(encoding="utf-8")


def test_unknown_placeholder_fails_loudly(story: ModuleType, tmp_path: Path) -> None:
    pitch = tmp_path / "pitch-and-qa.md"
    pitch.write_text(
        PITCH_DOC.read_text(encoding="utf-8") + "\nSay {brandNewNumber} here.\n", encoding="utf-8"
    )
    with pytest.raises(story.UnknownPlaceholderError, match=re.escape("{brandNewNumber}")):
        story.render(MOCK_BUNDLE, tmp_path / "out", pitch_doc=pitch)
    devpost = tmp_path / "devpost.md"
    devpost.write_text(
        DEVPOST_DOC.read_text(encoding="utf-8") + "\n`<from meta.json: summary.newField>`\n",
        encoding="utf-8",
    )
    with pytest.raises(story.UnknownPlaceholderError, match=re.escape("summary.newField")):
        story.render(MOCK_BUNDLE, tmp_path / "out", devpost_doc=devpost)
    # The CLI turns it into a nonzero exit, not a traceback.
    assert (
        story.main([str(MOCK_BUNDLE), "--out", str(tmp_path / "out"), "--pitch", str(pitch)]) == 1
    )


def test_docs_in_the_repo_are_untouched(rendered: tuple[Path, list, str]) -> None:
    # The renderer reads the docs; the placeholders must still be there afterwards.
    assert "{publicCatalogCount}" in PITCH_DOC.read_text(encoding="utf-8")
    assert "<from meta.json: summary.publicCatalogCount>" in DEVPOST_DOC.read_text(encoding="utf-8")


def test_decoy_test_sentence_needs_confidence_json(story: ModuleType, tmp_path: Path) -> None:
    """ML-01 (H2's PR #107 format): the label and the held-out AUC render from
    ``confidence.json`` (``model.heldOut.rocAuc``, two decimals); without the file, or without
    the AUC, the decoy-test sentences are marked for omission."""
    bundle = tmp_path / "bundle"
    shutil.copytree(MOCK_BUNDLE, bundle)
    rows, _ = story.render(bundle, tmp_path / "absent")
    by_name = {(r.doc, r.name): r for r in rows}
    pitch = by_name[("pitch-and-qa.md", "{heldOutRocAuc}")]
    devpost = by_name[("devpost.md", "<from confidence.json: model.heldOut.rocAuc>")]
    assert pitch.status == story.STATUS_CONDITION and "omit this sentence" in pitch.text
    assert devpost.status == story.STATUS_CONDITION
    assert by_name[("pitch-and-qa.md", "{confidenceLabel}")].status == story.STATUS_CONDITION

    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    hero = meta["scene"]["heroEventId"]
    file = {
        "schema": "hq.confidence/1",
        "runId": meta["run"]["id"],
        "model": {
            "name": "gbm",
            "heldOut": {"rocAuc": 0.8737, "rocAucEqualStationCount": 0.8149, "folds": 5},
            "trainedOn": {"positives": 12, "decoys": 40, "shuffles": 20},
        },
        "label": "AI decoy test",
        "description": "How much the timing looks like a real association, not a decoy",
        "events": {hero: 0.61},
    }
    (bundle / "confidence.json").write_text(json.dumps(file), encoding="utf-8")
    rows, _ = story.render(bundle, tmp_path / "present")
    value = {(r.doc, r.name): r.text for r in rows}
    assert value[("pitch-and-qa.md", "{heldOutRocAuc}")] == "0.87"
    assert value[("pitch-and-qa.md", "{confidenceLabel}")] == "AI decoy test"
    assert value[("devpost.md", "<from confidence.json: model.heldOut.rocAuc>")] == "0.87"
    assert value[("devpost.md", "<from confidence.json: label>")] == "AI decoy test"
    assert value[("pitch-and-qa.md", "{heldOutRocAucEqualStations}")] == "0.81"
    assert value[("pitch-and-qa.md", "{confidencePositives}")] == "12"
    assert value[("devpost.md", "<from confidence.json: model.trainedOn.decoys>")] == "40"
    filled = (tmp_path / "present" / "pitch-filled.md").read_text(encoding="utf-8")
    assert (
        "the AI decoy test: a classifier trained on this run's 12 candidates and 40 scrambled-clock decoys"
        in filled
    )
    assert "held-out ROC AUC 0.87 (0.81 even at equal station count)" in filled

    file["model"] = {"name": "gbm"}
    (bundle / "confidence.json").write_text(json.dumps(file), encoding="utf-8")
    rows, _ = story.render(bundle, tmp_path / "no-auc")
    row = {(r.doc, r.name): r for r in rows}[("pitch-and-qa.md", "{heldOutRocAuc}")]
    assert row.status == story.STATUS_CONDITION


def test_window_date_and_hidden_hero_stations(story: ModuleType, tmp_path: Path) -> None:
    """The date spoken in the pitches is the window's UTC day in words (never the window label),
    only for a window inside one UTC day; the H beat's station count is the hidden hero's own
    ``quality.nStations``, the event ``{hiddenHeroId}`` names."""
    bundle = tmp_path / "bundle"
    shutil.copytree(MOCK_BUNDLE, bundle)
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    day = 86_400
    meta["run"]["windowStart"] = 1_788_998_400.0  # 2026-09-10 00:00 UTC
    meta["run"]["windowEnd"] = 1_788_998_400.0 + day  # exclusive: the next midnight
    (bundle / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    rows, _ = story.render(bundle, tmp_path / "one-day")
    by_name = {(r.doc, r.name): r for r in rows}
    for key in (
        ("pitch-and-qa.md", "{windowDate}"),
        ("devpost.md", "<from meta.json: run.windowStart, as a date>"),
    ):
        assert by_name[key].status == story.STATUS_VALUE, key
        assert by_name[key].text == "September 10", key
    events = json.loads((bundle / "events.json").read_text(encoding="utf-8"))
    hidden = [e for e in events if e["tier"] == "A" and e["catalogMatch"] is None]
    best = min(
        hidden, key=lambda e: (-e["quality"]["nStations"], e["quality"]["rmsS"], e["t"], e["id"])
    )
    stations = by_name[("pitch-and-qa.md", "{hiddenHeroStations}")]
    assert stations.status == story.STATUS_VALUE
    assert stations.text == str(best["quality"]["nStations"])
    assert by_name[("pitch-and-qa.md", "{hiddenHeroId}")].text == best["id"]

    meta["run"]["windowEnd"] = 1_788_998_400.0 + 2 * day
    (bundle / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    rows, _ = story.render(bundle, tmp_path / "two-days")
    row = {(r.doc, r.name): r for r in rows}[("pitch-and-qa.md", "{windowDate}")]
    assert row.status == story.STATUS_NOT_AVAILABLE and "more than one UTC day" in row.text


@needs_showcase
def test_devpost_joint_share_and_final_copy(story: ModuleType, tmp_path: Path) -> None:
    """The Devpost quotes the recovered public events meeting every strict bar at once from
    ``run.tiering.matchedSet``, and on the run of record everything above the fill-in checklist
    renders as final copy: no marker but the video URL."""
    out = tmp_path / "out"
    rows, _ = story.render(SHOWCASE_BUNDLE, out)
    meta = json.loads((SHOWCASE_BUNDLE / "meta.json").read_text(encoding="utf-8"))
    matched = meta["run"]["tiering"]["matchedSet"]
    value = {(r.doc, r.name): r.text for r in rows}
    assert value[
        ("devpost.md", "<from meta.json: run.tiering.matchedSet.meetingEveryBar.A>")
    ] == str(matched["meetingEveryBar"]["A"])
    assert value[("devpost.md", "<from meta.json: run.tiering.matchedSet.n>")] == str(matched["n"])
    devpost = (out / story.DEVPOST_FILLED_MD).read_text(encoding="utf-8")
    copy = devpost.split("## Fill-in checklist")[0].split("-->", 1)[1]
    markers = re.findall(r"\[(?:manual|not available|condition not met):[^\]]*\]", copy)
    assert markers == ["[manual: the uploaded video URL]"]
    assert "only if" not in copy and "run.json" not in copy
