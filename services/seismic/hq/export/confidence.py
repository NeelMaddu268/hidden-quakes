"""``confidence.json`` (ML-01, H2 Seismology): the chance-association model's per-event scores.

The model is trained on decoy events made by scrambling station clocks; an event's score says how
much its timing looks like a real association rather than a scrambled-clock decoy. It is not a
probability that the event is an earthquake.

The file is optional. When the run directory has one, the exporter checks it against the run
(``confidence_problems``) and writes it into the bundle key-sorted; without it the bundle is
exactly what it was. ``check_bundle`` applies the same checks to a written bundle, plus the byte
cap. Format (built outside ``packages/contracts``, which is frozen)::

    {"schema": "hq.confidence/1", "runId": ..., "model": {..., "heldOut": {"rocAuc": ...}},
     "label": ..., "description": ..., "events": {"<eventId>": <score in [0, 1]>, ...}}
"""

import json
import logging
import math
from collections.abc import Collection
from pathlib import Path
from typing import Any

from hq.export.errors import ExportError
from hq.export.files import CONFIDENCE_JSON

log = logging.getLogger(__name__)

CONFIDENCE_SCHEMA = "hq.confidence/1"
MAX_CONFIDENCE_BYTES = 40 * 1024  # format cap of the written file (a few hundred scores)
OWNER = "ML-01 (H2 Seismology)"
# docs/00 language rules, applied to the two strings the UI shows verbatim.
FORBIDDEN_PHRASES = ("confirmed earthquake", "caused by", "predict", "official")
MAX_LISTED = 5


def _listed(items: Collection[str]) -> str:
    shown = ", ".join(sorted(items)[:MAX_LISTED])
    return shown + (", ..." if len(items) > MAX_LISTED else "")


def _is_unit_score(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def confidence_problems(raw: object, event_ids: Collection[str], run_id: str) -> list[str]:
    """Every way ``raw`` (parsed JSON) fails the format for a bundle of ``event_ids`` from run
    ``run_id``; empty when it passes."""
    name = CONFIDENCE_JSON
    if not isinstance(raw, dict):
        return [f"{name}: expected a JSON object"]
    problems: list[str] = []
    if raw.get("schema") != CONFIDENCE_SCHEMA:
        problems.append(f"{name}: schema {raw.get('schema')!r} != {CONFIDENCE_SCHEMA!r}")
    if raw.get("runId") != run_id:
        problems.append(f"{name}: runId {raw.get('runId')!r} != {run_id!r}")
    for key in ("label", "description"):
        text = raw.get(key)
        if text is None:
            continue
        if not isinstance(text, str):
            problems.append(f"{name}: {key} is not a string")
        elif any(phrase in text.lower() for phrase in FORBIDDEN_PHRASES):
            problems.append(f"{name}: {key} {text!r} uses a forbidden phrase (docs/00)")
    model = raw.get("model")
    held_out = model.get("heldOut") if isinstance(model, dict) else None
    roc_auc = held_out.get("rocAuc") if isinstance(held_out, dict) else None
    if roc_auc is not None and not _is_unit_score(roc_auc):
        problems.append(f"{name}: model.heldOut.rocAuc {roc_auc!r} is not a number in [0, 1]")
    events = raw.get("events")
    if not isinstance(events, dict) or not events:
        problems.append(f"{name}: events must be a non-empty object of eventId -> score")
        return problems
    known = set(event_ids)
    unknown = [eid for eid in events if eid not in known]
    if unknown:
        problems.append(
            f"{name}: {len(unknown)} event id(s) not in events.json: {_listed(unknown)}"
        )
    bad = [eid for eid, score in events.items() if not _is_unit_score(score)]
    if bad:
        problems.append(f"{name}: {len(bad)} score(s) not a number in [0, 1]: {_listed(bad)}")
    return problems


def read_confidence(
    run_dir: Path, event_ids: Collection[str], run_id: str
) -> dict[str, Any] | None:
    """``runs/<runId>/confidence.json`` checked against the run, or ``None`` when absent.
    A file that exists but fails the format is an ``ExportError`` listing every problem."""
    path = Path(run_dir) / CONFIDENCE_JSON
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExportError(f"{path} is not valid JSON ({OWNER}): {exc}") from exc
    problems = confidence_problems(raw, event_ids, run_id)
    if problems:
        lines = "\n".join(f"  - {p}" for p in problems)
        raise ExportError(f"{path} fails the {CONFIDENCE_SCHEMA} format ({OWNER}):\n{lines}")
    log.info(
        "export: %s scores %d of %d events (%s)", path, len(raw["events"]), len(event_ids), OWNER
    )
    return raw  # a dict: confidence_problems flags anything else
