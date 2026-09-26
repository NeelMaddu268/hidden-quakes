"""Run tables on disk (docs/02 §2).

Every lane reads and writes run tables only through these helpers, so a column rename can't
silently break a neighbor.

Flattening rule
    Nested models become prefixed columns joined by ``_`` (``enu_e``, ``quality_nStations``,
    ``catalogMatch_dtS``). Lists stay list columns. ``None`` stays null; an optional nested model
    that is ``None`` writes null into every one of its columns and reads back as ``None``.
    Times stay float64 epoch seconds. ``dict``-typed fields (``Station.staticsS``,
    ``SweepPoint.params``) are stored as one JSON text column, because parquet structs can't
    hold a row-dependent key set.

Parquet files carry ``schemaVersion`` and ``model`` in their file metadata; ``read_table``
returns them in ``DataFrame.attrs``.
"""

import json
import math
import types
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from hq_contracts.models import SCHEMA_VERSION

M = TypeVar("M", bound=BaseModel)

SEP = "_"
META_SCHEMA_VERSION = b"schemaVersion"
META_MODEL = b"model"


class TableSchemaError(ValueError):
    """A parquet file was not written by ``write_table`` or has another schema version."""


# --- annotation helpers -------------------------------------------------------------------


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner annotation, is_optional) for ``X | None`` / ``Optional[X]``; else (ann, False)."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(annotation)) == 2:
            return args[0], True
    return annotation, False


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    inner, _ = _strip_optional(annotation)
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return inner
    return None


def _is_dict_field(annotation: Any) -> bool:
    inner, _ = _strip_optional(annotation)
    return inner is dict or typing.get_origin(inner) is dict


def columns_for(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Flattened column names of ``model``, in field order."""
    cols: list[str] = []
    for name, field in model.model_fields.items():
        key = f"{prefix}{name}"
        nested = _nested_model(field.annotation)
        if nested is not None:
            cols.extend(columns_for(nested, f"{key}{SEP}"))
        else:
            cols.append(key)
    return cols


# --- model -> row ---------------------------------------------------------------------------


def _flatten(
    instance: BaseModel | None, model: type[BaseModel], prefix: str, out: dict[str, Any]
) -> None:
    for name, field in model.model_fields.items():
        key = f"{prefix}{name}"
        value = None if instance is None else getattr(instance, name)
        nested = _nested_model(field.annotation)
        if nested is not None:
            _flatten(value, nested, f"{key}{SEP}", out)
        elif _is_dict_field(field.annotation):
            out[key] = None if value is None else json.dumps(value, sort_keys=True)
        elif isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, list) and any(isinstance(v, BaseModel) for v in value):
            raise TypeError(
                f"{model.__name__}.{name}: lists of models can't be stored as a table column; "
                "this model is JSON-only (docs/02 §2)"
            )
        else:
            out[key] = value


def to_frame(models: Sequence[BaseModel], model: type[BaseModel] | None = None) -> pd.DataFrame:
    """Flatten model instances into one row each. Pass ``model`` to shape an empty frame."""
    if model is None:
        if not models:
            raise ValueError(
                "to_frame needs `model=` to build an empty frame with the right columns"
            )
        model = type(models[0])
    rows: list[dict[str, Any]] = []
    for m in models:
        if not isinstance(m, model):
            raise TypeError(f"expected {model.__name__}, got {type(m).__name__}")
        row: dict[str, Any] = {}
        _flatten(m, model, "", row)
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=columns_for(model))


# --- row -> model ---------------------------------------------------------------------------


def _py(value: Any) -> Any:
    """Convert pandas/numpy cell values to plain Python; NaN/NaT become None."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _unflatten(row: dict[str, Any], model: type[BaseModel], prefix: str) -> dict[str, Any] | None:
    """Rebuild the dict for ``model`` from flattened ``row``; None if every leaf is null."""
    out: dict[str, Any] = {}
    any_value = False
    for name, field in model.model_fields.items():
        key = f"{prefix}{name}"
        nested = _nested_model(field.annotation)
        if nested is not None:
            sub = _unflatten(row, nested, f"{key}{SEP}")
            _, optional = _strip_optional(field.annotation)
            if sub is None and not optional:
                sub = {}  # let pydantic report the missing required fields
            out[name] = sub
            any_value = any_value or sub is not None
            continue
        if key not in row:
            raise TableSchemaError(f"column {key!r} missing for {model.__name__}")
        value = _py(row[key])
        if value is not None and _is_dict_field(field.annotation):
            value = json.loads(value) if isinstance(value, str) else value
        out[name] = value
        any_value = any_value or value is not None
    return out if any_value else None


def from_frame(df: pd.DataFrame, model: type[M]) -> list[M]:
    """Inverse of ``to_frame``: one validated ``model`` per row."""
    missing = [c for c in columns_for(model) if c not in df.columns]
    if missing:
        raise TableSchemaError(f"{model.__name__} columns missing from frame: {missing}")
    out: list[M] = []
    for row in df.to_dict("records"):
        data = _unflatten(row, model, "")
        out.append(model.model_validate(data if data is not None else {}))
    return out


# --- parquet --------------------------------------------------------------------------------


def write_table(df: pd.DataFrame, path: Path | str, model_name: str) -> None:
    """Write ``df`` as parquet with ``schemaVersion`` and ``model`` in the file metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[META_SCHEMA_VERSION] = SCHEMA_VERSION.encode()
    metadata[META_MODEL] = model_name.encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def read_table(path: Path | str) -> pd.DataFrame:
    """Read a parquet written by ``write_table``; ``df.attrs`` holds ``schemaVersion`` and ``model``."""
    path = Path(path)
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if META_SCHEMA_VERSION not in metadata or META_MODEL not in metadata:
        raise TableSchemaError(f"{path} was not written by hq_contracts.io.write_table")
    version = metadata[META_SCHEMA_VERSION].decode()
    if version != SCHEMA_VERSION:
        raise TableSchemaError(
            f"{path} has schemaVersion {version}, this code expects {SCHEMA_VERSION}"
        )
    df = table.to_pandas()
    df.attrs["schemaVersion"] = version
    df.attrs["model"] = metadata[META_MODEL].decode()
    return df


def read_models(path: Path | str, model: type[M]) -> list[M]:
    """``from_frame(read_table(path), model)`` with a check that the file holds that model."""
    df = read_table(path)
    if df.attrs["model"] != model.__name__:
        raise TableSchemaError(f"{path} holds {df.attrs['model']} rows, not {model.__name__}")
    return from_frame(df, model)


def write_models(models: Sequence[M], path: Path | str, model: type[M] | None = None) -> None:
    """``write_table(to_frame(models), path, Model.__name__)``."""
    if model is None:
        if not models:
            raise ValueError("write_models needs `model=` for an empty list")
        model = type(models[0])
    write_table(to_frame(models, model), path, model.__name__)
