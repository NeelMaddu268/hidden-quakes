#!/usr/bin/env python3
"""Fill the pitch and Devpost placeholders from an exported bundle (ticket DEMO-03).

``docs/demo/pitch-and-qa.md`` and ``docs/demo/devpost.md`` never carry a number: every count,
magnitude and date is a placeholder (``{name}`` in the pitch, ``<from FILE: field>`` in the
Devpost) that names the bundle field it renders from (CLAUDE.md rule 4). This script is the
one lookup those docs promise: it reads the bundle, resolves every placeholder with the rules
from the two docs' own tables ("Numbers to fill Saturday evening", "Fill-in checklist"), and
writes three files under ``--out`` (default ``data/story/``, gitignored):

    numbers.md          placeholder -> value -> source field, one row per placeholder
    pitch-filled.md     docs/demo/pitch-and-qa.md with the placeholders substituted
    devpost-filled.md   docs/demo/devpost.md with the placeholders substituted

It never edits the docs in the repo. Substituted text is one of:

    a value                        the field is present
    [manual: ...]                  no bundle field (depth ruler, live API, URLs); read it there
    [not available: ...]           the field or file is missing from the bundle
    [condition not met: ...; omit this sentence]
                                   a conditional sentence (gain, baseline counts, null test,
                                   magnitude, G-R) whose gate field is null; delete the sentence

A placeholder in either doc that this script does not know is an error, so a new placeholder
can never ride silently into the submission text. A synthetic bundle (``meta.run.isSynthetic``
or ``meta.scene.isSynthetic``) puts a banner at the top of every output file.

Usage (from ``services/seismic`` so ``hq`` and ``hq_contracts`` resolve)::

    uv run python ../../scripts/render-story.py <bundle dir> [--out <dir>]
    make story BUNDLE=apps/web/public/data/showcase
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hq_contracts import models as m

from hq.export.files import EVIDENCE_DIR, META_JSON, VALIDATION_JSON

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DEMO = REPO_ROOT / "docs" / "demo"
PITCH_DOC = DOCS_DEMO / "pitch-and-qa.md"
DEVPOST_DOC = DOCS_DEMO / "devpost.md"
DEFAULT_OUT = REPO_ROOT / "data" / "story"

NUMBERS_MD = "numbers.md"
PITCH_FILLED_MD = "pitch-filled.md"
DEVPOST_FILLED_MD = "devpost-filled.md"
OUTPUT_FILES = (NUMBERS_MD, PITCH_FILLED_MD, DEVPOST_FILLED_MD)

SYNTHETIC_BANNER = (
    "# SYNTHETIC BUNDLE, NOT FOR SUBMISSION\n"
    "\n"
    "**Every number below is made up** (`meta.run.isSynthetic` / `meta.scene.isSynthetic` is "
    "true). Render the run of record before any of this reaches the pitch, the Devpost or "
    "the README.\n"
    "\n"
    "---\n"
    "\n"
)
COPY_OK_RE = re.compile(r"[ \t]*<!--\s*copy-ok\s*-->")
FENCE_RE = re.compile(r"^[ \t]*```")

# Placeholder syntax per doc. An optional backtick on either side is part of the match so a
# substituted number is not left in code formatting. The Devpost form allows one nested
# `<...>` (``<from evidence/<heroEventId>.json: traces.length>``); no backtick may sit inside,
# so two placeholders in one line never merge.
TOKEN_RE = {
    "pitch": re.compile(r"`?\{[A-Za-z]+\}`?"),
    "devpost": re.compile(r"`?<[A-Za-z](?:[^<>`]|<[^<>`]*>)*>`?"),
}

STATUS_VALUE = "value"
STATUS_MANUAL = "manual"
STATUS_NOT_AVAILABLE = "not available"
STATUS_CONDITION = "condition not met"
STATUS_LITERAL = "literal"
STATUSES = (STATUS_VALUE, STATUS_MANUAL, STATUS_NOT_AVAILABLE, STATUS_CONDITION, STATUS_LITERAL)


class RenderError(ValueError):
    """A bundle or a doc the script cannot render honestly; the CLI turns it into exit 1."""


class UnknownPlaceholderError(RenderError):
    """A placeholder in a doc that no spec covers."""


# --------------------------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------------------------


@dataclass
class Bundle:
    """The bundle files this script reads, validated; ``notes`` lists what was missing."""

    path: Path
    meta: dict[str, Any]
    validation: dict[str, Any] | None
    hero_evidence: dict[str, Any] | None
    notes: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return str(self.meta["run"]["id"])

    @property
    def is_synthetic(self) -> bool:
        return bool(self.meta["run"]["isSynthetic"] or self.meta["scene"]["isSynthetic"])


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_bundle(bundle_dir: Path) -> Bundle:
    """Validate ``meta.json`` (required), ``validation.json`` and the hero event's evidence
    file (both optional: what they feed renders as "not available")."""
    meta_path = bundle_dir / META_JSON
    if not meta_path.is_file():
        raise RenderError(f"{meta_path} is missing; a bundle needs at least {META_JSON}")
    meta = m.BundleMeta.model_validate(_load_json(meta_path)).model_dump()
    notes: list[str] = []

    validation: dict[str, Any] | None = None
    validation_path = bundle_dir / VALIDATION_JSON
    if validation_path.is_file():
        validation = m.Validation.model_validate(_load_json(validation_path)).model_dump()
    else:
        notes.append(f"{VALIDATION_JSON} is missing: every field it feeds renders as not available")

    hero: dict[str, Any] | None = None
    hero_id = meta["scene"]["heroEventId"]
    if hero_id is None:
        notes.append("scene.heroEventId is null: the hero station count renders as not available")
    else:
        hero_path = bundle_dir / EVIDENCE_DIR / f"{hero_id}.json"
        if hero_path.is_file():
            hero = m.EventEvidence.model_validate(_load_json(hero_path)).model_dump()
        else:
            notes.append(
                f"{EVIDENCE_DIR}/{hero_id}.json is missing: the hero station count renders as "
                "not available"
            )
    return Bundle(
        path=bundle_dir, meta=meta, validation=validation, hero_evidence=hero, notes=notes
    )


# --------------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    text: str  # what replaces the placeholder in the filled doc
    source: str  # the bundle field (or where to read it by hand)
    status: str  # one of STATUSES


Resolver = Callable[[Bundle], Resolved]


def fmt(value: Any, decimals: int | None = None) -> str:
    """Deterministic, human-readable number formatting: integers plain, floats to at most four
    decimals with trailing zeros dropped, or to exactly ``decimals`` when a doc quotes a field
    rounded (the magnitude errors, approved at two decimals)."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if decimals is not None:
            return f"{value:.{decimals}f}"
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _lookup(bundle: Bundle, file: str, path: str) -> tuple[Any, str]:
    """Walk ``path`` ("a.b.c") in the named bundle file. Returns ``(value, source)`` with
    ``value`` None when the file or any key is missing."""
    source = f"{file} → {path}"
    root: Any
    if file == META_JSON:
        root = bundle.meta
    elif file == VALIDATION_JSON:
        root = bundle.validation
    else:
        raise RenderError(f"unknown bundle file in a spec: {file}")
    node: Any = root
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None, source
        node = node[key]
    return node, source


def _not_available(bundle: Bundle, file: str, source: str) -> Resolved:
    if file == VALIDATION_JSON and bundle.validation is None:
        reason = f"{VALIDATION_JSON} missing"
    else:
        reason = f"{source} absent or null"
    return Resolved(f"[not available: {reason}]", source, STATUS_NOT_AVAILABLE)


def _condition_not_met(source: str, gate: str) -> Resolved:
    return Resolved(
        f"[condition not met: {gate} is null; omit this sentence]", source, STATUS_CONDITION
    )


def meta_field(path: str, *, decimals: int | None = None) -> Resolver:
    def resolve(bundle: Bundle) -> Resolved:
        value, source = _lookup(bundle, META_JSON, path)
        if value is None:
            return _not_available(bundle, META_JSON, source)
        return Resolved(fmt(value, decimals), source, STATUS_VALUE)

    return resolve


def validation_field(path: str, *, decimals: int | None = None) -> Resolver:
    def resolve(bundle: Bundle) -> Resolved:
        value, source = _lookup(bundle, VALIDATION_JSON, path)
        if value is None:
            return _not_available(bundle, VALIDATION_JSON, source)
        return Resolved(fmt(value, decimals), source, STATUS_VALUE)

    return resolve


def none_strict_scrambles() -> Resolver:
    """The null test's shuffle count, for the clause "none of the chance events reached the
    strict tier in any of the N scrambles". The clause is true only when ``meanChanceStrict``
    is exactly zero (a mean of zero over N shuffles means every shuffle had zero), so any other
    value marks the clause for omission; a missing null test omits it like the sentence."""

    source = f"{VALIDATION_JSON} → nullTest.nShuffles"
    gate = f"{VALIDATION_JSON} → nullTest.meanChanceStrict == 0"
    note = f" (only with {gate}: the none-at-the-strict-tier clause)"

    def resolve(bundle: Bundle) -> Resolved:
        strict, strict_source = _lookup(bundle, VALIDATION_JSON, "nullTest.meanChanceStrict")
        n, _ = _lookup(bundle, VALIDATION_JSON, "nullTest.nShuffles")
        if strict is None or n is None:
            r = _condition_not_met(source, f"{VALIDATION_JSON} → nullTest")
            return Resolved(r.text, source + note, r.status)
        if strict != 0:
            return Resolved(
                f"[condition not met: {strict_source} is not zero; omit this clause]",
                source + note,
                STATUS_CONDITION,
            )
        return Resolved(fmt(n), source + note, STATUS_VALUE)

    return resolve


def gated(gate_file: str, gate_path: str, inner: Resolver, *, sentence: str) -> Resolver:
    """A conditional placeholder: when the gate field is null (the kill switch fired or the
    stage did not run) the sentence is to be dropped, whatever the inner field holds."""

    def resolve(bundle: Bundle) -> Resolved:
        gate_value, gate_source = _lookup(bundle, gate_file, gate_path)
        if gate_value is None:
            if gate_file == VALIDATION_JSON and bundle.validation is None:
                gate_source = f"{gate_source} ({VALIDATION_JSON} missing)"
            r = _condition_not_met(inner(bundle).source, gate_source)
            return Resolved(r.text, f"{r.source} (only with {gate_source}: {sentence})", r.status)
        r = inner(bundle)
        return Resolved(r.text, f"{r.source} (only with {gate_source}: {sentence})", r.status)

    return resolve


def manual(where: str) -> Resolver:
    def resolve(bundle: Bundle) -> Resolved:
        return Resolved(f"[manual: {where}]", where, STATUS_MANUAL)

    return resolve


def literal(what: str) -> Resolver:
    """A token the docs use to describe the placeholder form itself; left as it is."""

    def resolve(bundle: Bundle) -> Resolved:
        return Resolved("", what, STATUS_LITERAL)

    return resolve


def baseline_field(method: str, path: str, *, decimals: int | None = None) -> Resolver:
    """``validation.baseline`` row with this method and ``associationProfile: "full"`` →
    ``path``; only when both ``full`` rows exist (the Validation card's "Strict events,
    PhaseNet vs STA/LTA" row), so the two methods are only ever quoted together."""
    source = f"{VALIDATION_JSON} → baseline[method={method}, associationProfile=full].{path}"
    gate = f"{VALIDATION_JSON} → baseline[] full rows for both phasenet and stalta"

    def resolve(bundle: Bundle) -> Resolved:
        if bundle.validation is None:
            return _not_available(bundle, VALIDATION_JSON, source)
        rows = {
            row["method"]: row
            for row in bundle.validation["baseline"]
            if row["associationProfile"] == "full"
        }
        if "phasenet" not in rows or "stalta" not in rows:
            return _condition_not_met(source, gate)
        node: Any = rows[method]
        for key in path.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return _not_available(bundle, VALIDATION_JSON, source)
        return Resolved(fmt(node, decimals), f"{source} (only with {gate})", STATUS_VALUE)

    return resolve


def baseline_strict(method: str) -> Resolver:
    """The strict (Tier A) count of that method's ``full`` baseline row."""
    return baseline_field(method, "tiers.A")


def matched_min_stations() -> Resolver:
    """The fewest stations any recovered public event was located on: the Tier B
    ``nStations`` bar, which is the worst matched event's value only while
    ``tiering.thresholds.quantiles.B`` is 0 ("worst of matched"); any other quantile omits
    the clause."""
    source = f"{META_JSON} → run.tiering.thresholds.B.nStations.value"
    gate = f"{META_JSON} → run.tiering.thresholds.quantiles.B == 0"

    def resolve(bundle: Bundle) -> Resolved:
        value, _ = _lookup(bundle, META_JSON, "run.tiering.thresholds.B.nStations.value")
        quantile, _ = _lookup(bundle, META_JSON, "run.tiering.thresholds.quantiles.B")
        if value is None or quantile is None:
            return _not_available(bundle, META_JSON, source)
        if quantile != 0:
            return Resolved(
                f"[condition not met: {gate} is false; omit this clause]", source, STATUS_CONDITION
            )
        return Resolved(fmt(value, 0), f"{source} (only with {gate})", STATUS_VALUE)

    return resolve


def hero_station_count() -> Resolver:
    source = f"{EVIDENCE_DIR}/<scene.heroEventId>.json → traces.length"

    def resolve(bundle: Bundle) -> Resolved:
        if bundle.hero_evidence is None:
            return Resolved(
                f"[not available: {source} (no hero evidence file in the bundle)]",
                source,
                STATUS_NOT_AVAILABLE,
            )
        return Resolved(fmt(len(bundle.hero_evidence["traces"])), source, STATUS_VALUE)

    return resolve


def ratio(file: str, numerator: str, denominator: str, *, gate: str | None = None) -> Resolver:
    """Render "A of n" from two fields of one file; ``gate`` names the object both live in."""

    def resolve(bundle: Bundle) -> Resolved:
        num, num_source = _lookup(bundle, file, numerator)
        den, den_source = _lookup(bundle, file, denominator)
        source = f"{num_source} over {den_source}"
        if num is None or den is None:
            missing = num_source if num is None else den_source
            return _not_available(bundle, file, missing if gate is None else f"{file} → {gate}")
        return Resolved(f"{fmt(num)} of {fmt(den)}", source, STATUS_VALUE)

    return resolve


def velocity_model_name() -> Resolver:
    """``run.velocityModel`` is H2's dict; the checklist says to take the model name from it
    even if the key is not ``name``."""
    source = f"{META_JSON} → run.velocityModel.name"

    def resolve(bundle: Bundle) -> Resolved:
        model, _ = _lookup(bundle, META_JSON, "run.velocityModel")
        if not isinstance(model, dict):
            return _not_available(bundle, META_JSON, source)
        for key in ("name", "model", "id"):
            if isinstance(model.get(key), str):
                return Resolved(model[key], f"{META_JSON} → run.velocityModel.{key}", STATUS_VALUE)
        keys = ", ".join(sorted(model))
        return Resolved(
            f"[not available: run.velocityModel has no name key (keys: {keys})]",
            source,
            STATUS_NOT_AVAILABLE,
        )

    return resolve


# --------------------------------------------------------------------------------------------
# Specs: one entry per placeholder, built from the two docs' tables
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    token: str  # the placeholder as written, without surrounding backticks
    resolver: Resolver
    label: str | None = None  # report label when one token has several meanings
    context: str | None = None  # regex the line must match for this spec to apply
    companion: tuple[str, Resolver] | None = None  # token that must sit on the same line

    @property
    def name(self) -> str:
        return self.label or self.token


MAG_GATE = (VALIDATION_JSON, "magnitude")
MAG_SENTENCE = "the magnitude kill switch did not fire"

# docs/demo/pitch-and-qa.md, "Numbers to fill Saturday evening".
PITCH_SPECS: tuple[Spec, ...] = (
    Spec("{publicCatalogCount}", meta_field("summary.publicCatalogCount")),
    Spec("{candidateCount}", meta_field("summary.candidateCount")),
    Spec("{strictQualityCount}", meta_field("summary.strictQualityCount")),
    Spec("{recoveredCatalogCount}", meta_field("summary.recoveredCatalogCount")),
    Spec("{additionalCount}", meta_field("summary.additionalCount")),
    Spec("{strictAdditionalCount}", meta_field("summary.strictAdditionalCount")),
    Spec("{N}", meta_field("summary.publicCatalogCount"), label="{N} (PUBLIC counter)"),
    Spec("{nStations}", hero_station_count()),
    Spec("{medianVErrM}", validation_field("synthetic.medianVErrM", decimals=0)),
    Spec(
        "{gain}",
        gated(
            META_JSON,
            "summary.baseline",
            meta_field("summary.baseline.gain"),
            sentence="the gain row",
        ),
    ),
    Spec("{strictPhasenet}", baseline_strict("phasenet")),
    Spec("{strictStalta}", baseline_strict("stalta")),
    Spec("{staltaCandidates}", baseline_field("stalta", "candidates")),
    Spec("{staltaRecoveredPublic}", baseline_field("stalta", "recoveredPublic")),
    Spec("{phasenetMedianRmsS}", baseline_field("phasenet", "medianRmsS", decimals=3)),
    Spec("{staltaMedianRmsS}", baseline_field("stalta", "medianRmsS", decimals=3)),
    Spec("{phasenetMedianStations}", baseline_field("phasenet", "medianStations")),
    Spec("{staltaMedianStations}", baseline_field("stalta", "medianStations")),
    Spec("{strictMatchedCount}", meta_field("run.tiering.counts.matched.A")),
    Spec("{medianStations}", meta_field("summary.medianStations")),
    Spec("{minMatchedStations}", matched_min_stations()),
    Spec(
        "{heldOutMedianAbsDzM}",
        meta_field("run.locator.statics.crossValidatedOffsets.after.medianAbsDzM", decimals=0),
    ),
    Spec(
        "{meanChanceEvents}",
        gated(
            VALIDATION_JSON,
            "nullTest",
            validation_field("nullTest.meanChanceEvents"),
            sentence="the null test",
        ),
    ),
    Spec("{nShuffles}", none_strict_scrambles()),
    Spec(
        "{n}",
        gated(*MAG_GATE, validation_field("magnitude.n"), sentence=MAG_SENTENCE),
        label="{n} (calibration events, Q27)",
        context=r"calibration events",
    ),
    Spec(
        "{n}",
        manual("/api/live/status → updatedAt (Live mode label); only if LIVE is up"),
        label="{n} (minutes ago, Live)",
        context=r"minutes ago",
    ),
    Spec(
        "{looMae}",
        gated(*MAG_GATE, validation_field("magnitude.looMae", decimals=2), sentence=MAG_SENTENCE),
        companion=(
            "{nullModelMae}",
            meta_field("run.matching.magnitude.leaveOneEventOut.nullModelMae", decimals=2),
        ),
    ),
    Spec(
        "{magType}",
        gated(
            *MAG_GATE,
            meta_field("run.matching.magnitude.calibrationMagType"),
            sentence=MAG_SENTENCE,
        ),
    ),
    Spec(
        "{nullModelMae}",
        gated(
            *MAG_GATE,
            meta_field("run.matching.magnitude.leaveOneEventOut.nullModelMae", decimals=2),
            sentence=MAG_SENTENCE,
        ),
    ),
    Spec(
        "{belowCalibratedRange}",
        gated(
            *MAG_GATE,
            ratio(
                META_JSON,
                "run.matching.magnitude.magnitudes.belowCalibratedRange",
                "run.matching.magnitude.magnitudes.written",
                gate="run.matching.magnitude.magnitudes",
            ),
            sentence=MAG_SENTENCE,
        ),
    ),
    Spec(
        "{strictJointShare}",
        ratio(
            META_JSON,
            "run.tiering.matchedSet.meetingEveryBar.A",
            "run.tiering.matchedSet.n",
            gate="run.tiering.matchedSet",
        ),
    ),
    Spec(
        "{depthBand}",
        manual(
            "no bundle field; read off the depth ruler with STRICT on: a depth band, bunched or "
            "spread, never a geometry and never a mechanism"
        ),
    ),
    Spec(
        "{latency}",
        manual("/api/live/status → latencyS and /health → served.latencyS; only if measured"),
    ),
    Spec("{windowLabel}", meta_field("run.windowLabel")),
    Spec("{value}", literal("describes the placeholder form; not a placeholder")),
)

# docs/demo/devpost.md, "Fill-in checklist".
DEVPOST_SPECS: tuple[Spec, ...] = (
    Spec("<from meta.json: summary.publicCatalogCount>", meta_field("summary.publicCatalogCount")),
    Spec("<from meta.json: summary.candidateCount>", meta_field("summary.candidateCount")),
    Spec("<from meta.json: summary.strictQualityCount>", meta_field("summary.strictQualityCount")),
    Spec(
        "<from meta.json: summary.recoveredCatalogCount>",
        meta_field("summary.recoveredCatalogCount"),
    ),
    Spec("<from meta.json: summary.additionalCount>", meta_field("summary.additionalCount")),
    Spec(
        "<from meta.json: summary.strictAdditionalCount>",
        meta_field("summary.strictAdditionalCount"),
    ),
    Spec("<from meta.json: summary.medianStations>", meta_field("summary.medianStations")),
    Spec("<from meta.json: summary.medianRmsS>", meta_field("summary.medianRmsS")),
    Spec(
        "<from meta.json: summary.baseline.gain>",
        gated(
            META_JSON,
            "summary.baseline",
            meta_field("summary.baseline.gain"),
            sentence="the gain sentence",
        ),
    ),
    Spec(
        "<from validation.json: baseline[method=phasenet, associationProfile=full].tiers.A>",
        baseline_strict("phasenet"),
    ),
    Spec(
        "<from validation.json: baseline[method=stalta, associationProfile=full].tiers.A>",
        baseline_strict("stalta"),
    ),
    Spec("<from meta.json: run.windowLabel>", meta_field("run.windowLabel")),
    Spec("<from meta.json: run.pickerWeights>", meta_field("run.pickerWeights")),
    Spec("<from meta.json: run.velocityModel.name>", velocity_model_name()),
    Spec(
        "<from validation.json: synthetic.medianVErrM>", validation_field("synthetic.medianVErrM")
    ),
    Spec(
        "<from validation.json: nullTest.meanChanceEvents>",
        gated(
            VALIDATION_JSON,
            "nullTest",
            validation_field("nullTest.meanChanceEvents"),
            sentence="the null-test sentence",
        ),
    ),
    Spec("<from validation.json: nullTest.nShuffles>", none_strict_scrambles()),
    Spec(
        "<from validation.json: magnitude.n>",
        gated(*MAG_GATE, validation_field("magnitude.n"), sentence=MAG_SENTENCE),
    ),
    Spec(
        "<from validation.json: magnitude.looMae>",
        gated(*MAG_GATE, validation_field("magnitude.looMae", decimals=2), sentence=MAG_SENTENCE),
        companion=(
            "<from meta.json: run.matching.magnitude.leaveOneEventOut.nullModelMae>",
            meta_field("run.matching.magnitude.leaveOneEventOut.nullModelMae", decimals=2),
        ),
    ),
    Spec(
        "<from meta.json: run.matching.magnitude.calibrationMagType>",
        gated(
            *MAG_GATE,
            meta_field("run.matching.magnitude.calibrationMagType"),
            sentence=MAG_SENTENCE,
        ),
    ),
    Spec(
        "<from meta.json: run.matching.magnitude.leaveOneEventOut.nullModelMae>",
        gated(
            *MAG_GATE,
            meta_field("run.matching.magnitude.leaveOneEventOut.nullModelMae", decimals=2),
            sentence=MAG_SENTENCE,
        ),
    ),
    Spec(
        "<from meta.json: run.matching.magnitude.magnitudes.belowCalibratedRange>",
        gated(
            *MAG_GATE,
            meta_field("run.matching.magnitude.magnitudes.belowCalibratedRange"),
            sentence=MAG_SENTENCE,
        ),
    ),
    Spec(
        "<from meta.json: run.matching.magnitude.magnitudes.written>",
        gated(
            *MAG_GATE,
            meta_field("run.matching.magnitude.magnitudes.written"),
            sentence=MAG_SENTENCE,
        ),
    ),
    Spec("<from evidence/<heroEventId>.json: traces.length>", hero_station_count()),
    Spec("<heroEventId>", meta_field("scene.heroEventId")),
    Spec("<scene.heroEventId>", meta_field("scene.heroEventId")),
    Spec("<deployed URL>", manual("the production URL (docs/deploy.md)")),
    Spec("<repo URL>", manual("the GitHub repository URL")),
    Spec("<video URL>", manual("the uploaded video URL")),
    Spec("<from FILE: field>", literal("describes the placeholder form; not a placeholder")),
)

DOCS: tuple[tuple[str, Path, tuple[Spec, ...], str], ...] = (
    ("pitch", PITCH_DOC, PITCH_SPECS, PITCH_FILLED_MD),
    ("devpost", DEVPOST_DOC, DEVPOST_SPECS, DEVPOST_FILLED_MD),
)


# --------------------------------------------------------------------------------------------
# Substitution
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    doc: str
    name: str
    text: str
    source: str
    status: str
    count: int


def _match_spec(specs: Iterable[Spec], token: str, line: str) -> Spec | None:
    for spec in specs:
        if spec.token != token:
            continue
        if spec.context is None or re.search(spec.context, line):
            return spec
    return None


def scan_tokens(text: str, kind: str) -> list[tuple[int, str]]:
    """Every placeholder outside fenced code, as ``(line number, token)`` without backticks."""
    found: list[tuple[int, str]] = []
    fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(line):
            fence = not fence
            continue
        if fence:
            continue
        for match in TOKEN_RE[kind].finditer(line):
            found.append((lineno, match.group(0).strip("`")))
    return found


class _Filler:
    """Per-doc substitution state: resolves each spec once and counts its uses."""

    def __init__(self, kind: str, specs: tuple[Spec, ...], bundle: Bundle) -> None:
        self.kind = kind
        self.specs = specs
        self.bundle = bundle
        self.resolved: dict[str, Resolved] = {}
        self.counts: dict[str, int] = {}
        self.unknown: list[tuple[int, str]] = []

    def fill_line(self, line: str, lineno: int) -> str:
        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            token = raw.strip("`")
            spec = _match_spec(self.specs, token, line)
            if spec is None:
                self.unknown.append((lineno, token))
                return raw
            if spec.name not in self.resolved:
                self.resolved[spec.name] = spec.resolver(self.bundle)
            self.counts[spec.name] = self.counts.get(spec.name, 0) + 1
            r = self.resolved[spec.name]
            if r.status == STATUS_LITERAL:
                return raw
            text = r.text
            if (
                spec.companion is not None
                and spec.companion[0] not in line
                and r.status == STATUS_VALUE
            ):
                # looMae is quoted only next to the null-model error (FYI-H2-7).
                partner = spec.companion[1](self.bundle)
                text = f"{text} (versus {partner.text} for a no-skill baseline)"
            # A placeholder wrapped in its own backticks loses them; a backtick that opens or
            # closes a longer code span (`{a} PUBLIC → {b} RECOVERED`) stays where it was.
            lead, trail = raw.startswith("`"), raw.endswith("`")
            if lead and trail:
                return text
            return ("`" if lead else "") + text + ("`" if trail else "")

        return TOKEN_RE[self.kind].sub(replace, line)


def fill_doc(
    text: str, kind: str, specs: tuple[Spec, ...], bundle: Bundle, doc_name: str
) -> tuple[str, list[Row]]:
    """Substitute every placeholder outside fenced code and drop ``<!-- copy-ok -->`` markers.
    Raises :class:`UnknownPlaceholderError` naming every token no spec covers."""
    filler = _Filler(kind, specs, bundle)
    out_lines: list[str] = []
    fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(line):
            fence = not fence
            out_lines.append(line)
            continue
        if fence:
            out_lines.append(line)
            continue
        out_lines.append(filler.fill_line(COPY_OK_RE.sub("", line), lineno))
    if filler.unknown:
        listing = ", ".join(f"{token} (line {lineno})" for lineno, token in filler.unknown)
        raise UnknownPlaceholderError(
            f"{doc_name}: placeholder(s) this script does not know: {listing}. Add each to the "
            "specs in scripts/render-story.py (a bundle field, or manual) before rendering."
        )
    rows = [
        Row(doc_name, name, r.text, r.source, r.status, filler.counts[name])
        for name, r in sorted(filler.resolved.items())
    ]
    return "\n".join(out_lines) + "\n", rows


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------


def _md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def numbers_table(rows: list[Row]) -> str:
    lines = [
        "| Doc | Placeholder | Value | Source | Status | Uses |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        shown = r.text if r.status != STATUS_LITERAL else "(left as written)"
        lines.append(
            f"| {r.doc} | `{_md_cell(r.name)}` | {_md_cell(shown)} | {_md_cell(r.source)} | "
            f"{r.status} | {r.count} |"
        )
    return "\n".join(lines) + "\n"


def _status_counts(rows: list[Row]) -> str:
    counts = {status: sum(1 for r in rows if r.status == status) for status in STATUSES}
    return ", ".join(f"{n} {status}" for status, n in counts.items())


def _marker_lines(text: str) -> list[int]:
    return [
        i
        for i, line in enumerate(text.splitlines(), start=1)
        if "[condition not met:" in line or "[not available:" in line or "[manual:" in line
    ]


def header(bundle: Bundle, title: str) -> str:
    banner = SYNTHETIC_BANNER if bundle.is_synthetic else ""
    return (
        f"{banner}<!-- {title}: rendered by scripts/render-story.py from {bundle.path} "
        f"(runId {bundle.run_id}). Do not edit the docs in the repo; rerun the script. "
        "[manual: ...] is read off the screen or the API; [not available: ...] is a field the "
        "bundle lacks; [condition not met: ...] names a sentence to delete. -->\n\n"
    )


def render(
    bundle_dir: Path,
    out_dir: Path,
    *,
    pitch_doc: Path = PITCH_DOC,
    devpost_doc: Path = DEVPOST_DOC,
) -> tuple[list[Row], str]:
    """Render the three output files; returns the rows and the stdout report."""
    bundle = load_bundle(bundle_dir)
    docs = (
        ("pitch", pitch_doc, PITCH_SPECS, PITCH_FILLED_MD),
        ("devpost", devpost_doc, DEVPOST_SPECS, DEVPOST_FILLED_MD),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    report: list[str] = []
    if bundle.is_synthetic:
        report.append("SYNTHETIC BUNDLE, NOT FOR SUBMISSION: every rendered number is made up.")
    report.append(f"bundle: {bundle.path} (runId {bundle.run_id}, mode {bundle.meta['mode']})")
    report.extend(f"note: {note}" for note in bundle.notes)

    for kind, doc_path, specs, out_name in docs:
        if not doc_path.is_file():
            raise RenderError(f"{doc_path} is missing")
        filled, doc_rows = fill_doc(
            doc_path.read_text(encoding="utf-8"), kind, specs, bundle, doc_path.name
        )
        rows.extend(doc_rows)
        body = header(bundle, out_name) + filled
        (out_dir / out_name).write_text(body, encoding="utf-8")
        markers = _marker_lines(filled)
        where = ", ".join(str(i) for i in markers) if markers else "none"
        report.append(
            f"{out_name}: {len(doc_rows)} placeholder(s) from {doc_path.relative_to(REPO_ROOT) if doc_path.is_relative_to(REPO_ROOT) else doc_path}; "
            f"lines to resolve by hand: {where}"
        )

    table = numbers_table(rows)
    numbers_body = (
        header(bundle, NUMBERS_MD)
        + "# Story numbers\n\n"
        + f"Bundle `{bundle.path}`, run `{bundle.run_id}`. {_status_counts(rows)}.\n\n"
        + table
    )
    (out_dir / NUMBERS_MD).write_text(numbers_body, encoding="utf-8")
    report.append(f"wrote {', '.join(str(out_dir / name) for name in OUTPUT_FILES)}")
    report.append("")
    report.append(table.rstrip("\n"))
    if bundle.is_synthetic:
        report.append("")
        report.append("SYNTHETIC BUNDLE, NOT FOR SUBMISSION.")
    return rows, "\n".join(report) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "bundle", type=Path, help="bundle directory holding meta.json (and validation.json)"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})"
    )
    parser.add_argument("--pitch", type=Path, default=PITCH_DOC, help=argparse.SUPPRESS)
    parser.add_argument("--devpost", type=Path, default=DEVPOST_DOC, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        _, report = render(args.bundle, args.out, pitch_doc=args.pitch, devpost_doc=args.devpost)
    except RenderError as exc:
        print(f"render-story: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
