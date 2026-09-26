"""hq.config.seismology: the showcase seismology.yaml loads with every section present.

Guards against a bad merge of hq/config/seismology.py: when two lanes insert config classes at
the same place, a union merge can drop a nested validator's ``return self``, and pydantic then
turns that section into ``None`` without an error. Every model-typed field must hold a model.
"""

import types
import typing

import pytest
import yaml
from pydantic import BaseModel

from hq.config.seismology import SEISMIC_ROOT, SeismologyConfig

pytestmark = pytest.mark.smoke

SEISMOLOGY_YAML = SEISMIC_ROOT / "configs" / "showcase" / "seismology.yaml"


def _model_fields_hold_models(model: BaseModel, path: str) -> list[str]:
    """Paths of model-typed fields (``X`` or ``list[X]`` / ``dict[str, X]``) not holding an ``X``;
    an optional model field may be ``None``."""
    bad: list[str] = []
    for name, info in type(model).model_fields.items():
        value = getattr(model, name)
        where = f"{path}.{name}"
        annotation = info.annotation
        args = typing.get_args(annotation)
        optional = typing.get_origin(annotation) in (typing.Union, types.UnionType) and (
            type(None) in args
        )
        if optional and value is None:
            continue
        is_model = isinstance(annotation, type) and issubclass(annotation, BaseModel)
        if is_model and not isinstance(value, annotation):
            bad.append(f"{where} is {value!r}")
            continue
        if isinstance(value, BaseModel):
            bad += _model_fields_hold_models(value, where)
        elif isinstance(value, list | tuple):
            for k, item in enumerate(value):
                if isinstance(item, BaseModel):
                    bad += _model_fields_hold_models(item, f"{where}[{k}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, BaseModel):
                    bad += _model_fields_hold_models(item, f"{where}[{key!r}]")
    return bad


def test_showcase_seismology_yaml_loads_every_section_as_a_model() -> None:
    cfg = SeismologyConfig.model_validate(yaml.safe_load(SEISMOLOGY_YAML.read_text()))
    for name, info in SeismologyConfig.model_fields.items():
        assert isinstance(annotation := info.annotation, type) and issubclass(
            annotation, BaseModel
        ), f"SeismologyConfig.{name} should be a config section model"
        assert isinstance(getattr(cfg, name), annotation), f"SeismologyConfig.{name} is not loaded"
    assert _model_fields_hold_models(cfg, "seismology") == []


def test_a_section_validator_returning_none_is_caught() -> None:
    """The failure this file guards against, reproduced on a throwaway model."""
    from pydantic import ConfigDict, model_validator

    class Inner(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: int

        @model_validator(mode="after")
        def _check(self) -> "Inner":  # a merge dropped the ``return self``
            return None  # type: ignore[return-value]

    class Outer(BaseModel):
        inner: Inner

    assert _model_fields_hold_models(Outer.model_validate({"inner": {"a": 1}}), "outer") != []


def test_unmatched_reason_thresholds_follow_the_associator() -> None:
    raw = yaml.safe_load((SEISMIC_ROOT / "configs/showcase/seismology.yaml").read_text())
    cfg = SeismologyConfig.model_validate(raw)
    assert cfg.matching.reasons.minPickProb == cfg.associator.minPickProb
    assert cfg.matching.reasons.minStations <= cfg.associator.minStations
    for key, value, message in (("minPickProb", 0.2, "must equal associator.minPickProb"),
                                ("minStations", cfg.associator.minStations + 1,
                                 "must not exceed associator.minStations")):
        bad = {**raw, "matching": {**raw["matching"],
                                   "reasons": {**raw["matching"]["reasons"], key: value}}}
        with pytest.raises(ValueError, match=message):
            SeismologyConfig.model_validate(bad)
