"""docs/02 §5: the four H2 library functions H4's validation reruns call, pinned by signature.

A renamed or reordered positional parameter, a new required parameter or a changed result field
would break H4's reruns without failing any functional test, so this checks the signatures
themselves (Definition of done: "match docs/02 §5 exactly"). Additions must stay keyword-only
with a default (REQ-H2-8, REQ-H2-9, REQ-H1-5).
"""

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any

import pytest

import hq.associate
import hq.locate
import hq.match
import hq.tier

# docs/02 §5, written out: function -> (positional parameters, result class, result fields).
DOCS02_API: dict[str, tuple[Callable[..., Any], tuple[str, ...], type, tuple[str, ...]]] = {
    "associate": (hq.associate.associate, ("picks", "stations", "cfg", "run"),
                  hq.associate.AssocResult, ("events", "picks")),
    "locate": (hq.locate.locate, ("assoc", "picks", "stations", "cfg", "run"),
               hq.locate.LocateResult, ("events", "arrivals", "statics")),
    "match": (hq.match.match, ("events_located", "catalog", "cfg"),
              hq.match.MatchResult, ("matches", "sensitivity")),
    "assign_tiers": (hq.tier.assign_tiers, ("events_located", "matches", "cfg"),
                     hq.tier.TierResult, ("events", "tiering")),
}


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(DOCS02_API))
def test_docs02_section5_signature(name: str) -> None:
    fn, positional, result, fields = DOCS02_API[name]
    params = list(inspect.signature(fn).parameters.values())
    head = [p for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert tuple(p.name for p in head) == positional
    assert all(p.default is inspect.Parameter.empty for p in head)
    extras = params[len(head):]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty
               for p in extras), [p.name for p in extras]
    assert inspect.signature(fn).return_annotation is result
    assert dataclasses.is_dataclass(result) and result.__dataclass_params__.frozen  # type: ignore[attr-defined]
    assert tuple(f.name for f in dataclasses.fields(result)) == fields
