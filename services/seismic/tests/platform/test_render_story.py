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
PITCH_DOC = REPO_ROOT / "docs" / "demo" / "pitch-and-qa.md"
DEVPOST_DOC = REPO_ROOT / "docs" / "demo" / "devpost.md"
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
    }
    reported = {(r.doc, r.name) for r in rows}
    assert all(r.status in story.STATUSES for r in rows)
    literal_tokens = {r.name for r in rows if r.status == story.STATUS_LITERAL}
    for kind, (doc, filled_name) in by_doc.items():
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
    assert {"<repo URL>", "<video URL>"} <= manual
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
    assert value[("pitch-and-qa.md", "{nStations}")] == str(len(hero["traces"]))
    pitch = (out / "pitch-filled.md").read_text(encoding="utf-8")
    assert f"lists {meta['summary']['publicCatalogCount']} events" in pitch
    # A code span wrapping two placeholders keeps its backticks.
    devpost = (out / "devpost-filled.md").read_text(encoding="utf-8")
    assert (
        f"`{meta['summary']['publicCatalogCount']} PUBLIC → {meta['summary']['candidateCount']} RECOVERED`"
        in devpost
    )


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
