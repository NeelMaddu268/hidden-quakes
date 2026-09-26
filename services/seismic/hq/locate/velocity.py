"""1D layered velocity models in ``elevM`` (m above sea level, up positive).

A layer file (``configs/velocity/*.csv``) is a block of ``# key: value`` header lines followed by a
CSV table ``topElevM,vpMPerS,vsMPerS`` ordered top-down. Each row is the top of a layer with that
layer's constant velocities; the deepest layer is a half-space. The header carries the source
(citation, URLs, license), how the source's depths were converted to ``elevM``, and the datum.

Boundary convention: layer ``i`` spans ``topElevM[i+1] < elevM <= topElevM[i]``, so a point exactly
on a boundary takes the layer below it. The top of the model belongs to the first layer. Anything
above the top raises: the model is never extrapolated upward implicitly. ``with_top_extended_to``
does it explicitly and records that it did.
"""

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

log = logging.getLogger(__name__)

COLUMNS = ("topElevM", "vpMPerS", "vsMPerS")
REQUIRED_HEADER_KEYS = (
    "name",
    "citation",
    "url",
    "sourceFile",
    "license",
    "sourceDatum",
    "layerConvention",
    "conversion",
    "units",
    "datum",
    "verified",
)
OPTIONAL_HEADER_KEYS = ("model", "sourceSha256", "sourceColumns", "retrieved", "notes")
BOUNDARY_CONVENTION = (
    "layer i spans topElevM[i+1] < elevM <= topElevM[i]: a point exactly on a boundary takes the "
    "layer below; the deepest layer is a half-space; above the top of the model is an error"
)


class LayerFileError(ValueError):
    """A layer file is malformed, incomplete or physically impossible."""


@dataclass(frozen=True)
class SourceRef:
    """Where a model comes from. Same fields as ``SourceRef`` in docs/02."""

    citation: str
    url: str
    verified: bool  # values taken unchanged from the authoritative source file

    def to_record(self) -> dict[str, Any]:
        return {"citation": self.citation, "url": self.url, "verified": self.verified}


@dataclass(frozen=True)
class TopExtension:
    """An explicit upward extension of the top layer (for receivers above the model top)."""

    fromElevM: float  # the source model's own top, m ASL
    toElevM: float  # the new top, m ASL

    def to_record(self) -> dict[str, float]:
        return {"fromElevM": self.fromElevM, "toElevM": self.toElevM}


def _frozen(values: ArrayLike) -> NDArray[np.float64]:
    arr = np.array(values, dtype=np.float64)  # always a copy
    arr.setflags(write=False)
    return arr


def _check_layers(top: NDArray[np.float64], vp: NDArray[np.float64], vs: NDArray[np.float64]) -> None:
    if top.ndim != 1 or top.size == 0:
        raise LayerFileError("a layer model needs at least one layer")
    if not (top.shape == vp.shape == vs.shape):
        raise LayerFileError(f"column lengths differ: {top.shape}, {vp.shape}, {vs.shape}")
    for name, arr in zip(COLUMNS, (top, vp, vs), strict=True):
        if not np.all(np.isfinite(arr)):
            raise LayerFileError(f"{name} has non-finite values: {arr.tolist()}")
    steps = np.diff(top)
    if np.any(steps >= 0):
        bad = int(np.argmax(steps >= 0))
        raise LayerFileError(
            f"topElevM must strictly decrease top-down; row {bad + 1} ({top[bad]}) is followed "
            f"by {top[bad + 1]}"
        )
    for name, arr in (("vpMPerS", vp), ("vsMPerS", vs)):
        if np.any(arr <= 0):
            raise LayerFileError(f"{name} must be positive, got {arr.tolist()}")
    if np.any(vs >= vp):
        bad = int(np.argmax(vs >= vp))
        raise LayerFileError(f"Vs must be below Vp; layer {bad + 1} has Vp {vp[bad]}, Vs {vs[bad]}")


@dataclass(frozen=True, eq=False)
class LayerModel:
    """A 1D layered P/S model. Arrays are read-only copies, ordered top-down, in m ASL and m/s."""

    name: str
    datum: str  # one sentence: what topElevM is relative to and how the source was converted
    source: SourceRef
    top_elev_m: NDArray[np.float64]
    vp_m_per_s: NDArray[np.float64]
    vs_m_per_s: NDArray[np.float64]
    source_file: str = ""  # exact URL of the source file the values came from
    license: str = ""
    top_extension: TopExtension | None = None
    header: Mapping[str, str] = field(default_factory=dict)  # the layer file's full header

    def __post_init__(self) -> None:
        for attr in ("top_elev_m", "vp_m_per_s", "vs_m_per_s"):
            object.__setattr__(self, attr, _frozen(getattr(self, attr)))
        object.__setattr__(self, "header", dict(self.header))
        _check_layers(self.top_elev_m, self.vp_m_per_s, self.vs_m_per_s)
        if not self.name or not self.datum:
            raise LayerFileError("a layer model needs a name and a datum sentence")

    @property
    def n_layers(self) -> int:
        return int(self.top_elev_m.size)

    @property
    def top_of_model_elev_m(self) -> float:
        return float(self.top_elev_m[0])

    def layer_index(self, elev_m: ArrayLike) -> NDArray[np.intp]:
        """Index of the layer containing each elevation (m ASL), with the boundary convention above.

        Raises ``ValueError`` for non-finite elevations and for any point above the top of the model.
        """
        elev = np.asarray(elev_m, dtype=np.float64)
        if not np.all(np.isfinite(elev)):
            raise ValueError("elevations must be finite")
        top = self.top_of_model_elev_m
        if np.any(elev > top):
            raise ValueError(
                f"elevation {float(np.max(elev)):.1f} m ASL is above the top of velocity model "
                f"{self.name!r} ({top:.1f} m ASL); extend it explicitly with with_top_extended_to()"
            )
        ascending = self.top_elev_m[::-1]
        # count of layer tops >= elev, minus one: the deepest top at or above the point
        idx = self.n_layers - np.searchsorted(ascending, elev, side="left") - 1
        return np.asarray(idx, dtype=np.intp)

    def vp_at(self, elev_m: ArrayLike) -> NDArray[np.float64]:
        """P velocity (m/s) at each elevation (m ASL); same shape as the input."""
        return np.asarray(self.vp_m_per_s[self.layer_index(elev_m)])

    def vs_at(self, elev_m: ArrayLike) -> NDArray[np.float64]:
        """S velocity (m/s) at each elevation (m ASL); same shape as the input."""
        return np.asarray(self.vs_m_per_s[self.layer_index(elev_m)])

    def with_top_extended_to(self, elev_m: float) -> "LayerModel":
        """A copy whose top layer reaches up to ``elev_m`` (m ASL), with the extension recorded.

        The top layer keeps its velocities; no other layer changes. ``elev_m`` must lie above the
        current top. Extending twice keeps the source model's original top in the record.
        """
        new_top = float(elev_m)
        if not math.isfinite(new_top):
            raise ValueError("elev_m must be finite")
        current = self.top_of_model_elev_m
        if new_top <= current:
            raise ValueError(
                f"{new_top:.1f} m ASL is not above the top of {self.name!r} ({current:.1f} m ASL); "
                "there is nothing to extend"
            )
        original = self.top_extension.fromElevM if self.top_extension else current
        tops = self.top_elev_m.copy()
        tops[0] = new_top
        log.info(
            "velocity model %s: top layer extended from %.1f to %.1f m ASL (source top %.1f)",
            self.name, current, new_top, original,
        )
        return replace(
            self,
            top_elev_m=tops,
            top_extension=TopExtension(fromElevM=original, toElevM=new_top),
        )

    def to_record(self) -> dict[str, Any]:
        """The ``ProcessingRun.velocityModel`` dict: name, SourceRef, layers, datum, provenance."""
        layers = [
            {"topElevM": float(t), "vpMPerS": float(p), "vsMPerS": float(s)}
            for t, p, s in zip(self.top_elev_m, self.vp_m_per_s, self.vs_m_per_s, strict=True)
        ]
        return {
            "name": self.name,
            "source": self.source.to_record(),
            "sourceFile": self.source_file,
            "license": self.license,
            "datum": self.datum,
            "boundaryConvention": BOUNDARY_CONVENTION,
            "layers": layers,
            "topExtension": self.top_extension.to_record() if self.top_extension else None,
        }


def _parse_header(lines: list[tuple[int, str]], path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    known = set(REQUIRED_HEADER_KEYS) | set(OPTIONAL_HEADER_KEYS)
    for lineno, line in lines:
        body = line[1:].strip()
        key, sep, value = body.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or not key.isidentifier():
            raise LayerFileError(f"{path}:{lineno}: header lines must be '# key: value', got {line!r}")
        if key not in known:
            raise LayerFileError(f"{path}:{lineno}: unknown header key {key!r}")
        if key in header:
            raise LayerFileError(f"{path}:{lineno}: duplicate header key {key!r}")
        if not value:
            raise LayerFileError(f"{path}:{lineno}: header key {key!r} is empty")
        header[key] = value
    missing = [k for k in REQUIRED_HEADER_KEYS if k not in header]
    if missing:
        raise LayerFileError(f"{path}: missing header fields {missing}")
    if header["verified"] not in ("true", "false"):
        raise LayerFileError(f"{path}: verified must be 'true' or 'false', got {header['verified']!r}")
    return header


def _parse_table(lines: list[tuple[int, str]], path: Path) -> NDArray[np.float64]:
    if not lines:
        raise LayerFileError(f"{path}: no column header line after the '#' header")
    lineno, first = lines[0]
    if tuple(c.strip() for c in first.split(",")) != COLUMNS:
        raise LayerFileError(f"{path}:{lineno}: columns must be {','.join(COLUMNS)}, got {first!r}")
    rows: list[list[float]] = []
    for lineno, line in lines[1:]:
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(COLUMNS):
            raise LayerFileError(f"{path}:{lineno}: expected {len(COLUMNS)} values, got {line!r}")
        try:
            rows.append([float(c) for c in cells])
        except ValueError as err:
            raise LayerFileError(f"{path}:{lineno}: non-numeric value in {line!r}") from err
    if not rows:
        raise LayerFileError(f"{path}: no layer rows")
    return np.array(rows, dtype=np.float64)


def load_layer_model(path: Path) -> LayerModel:
    """Load and validate a layer file. Raises ``LayerFileError`` on anything malformed."""
    started = time.perf_counter()
    path = Path(path)
    header_lines: list[tuple[int, str]] = []
    table_lines: list[tuple[int, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            raise LayerFileError(f"{path}:{lineno}: blank lines are not allowed")
        if line.startswith("#"):
            if table_lines:
                raise LayerFileError(f"{path}:{lineno}: '#' lines must all come before the table")
            header_lines.append((lineno, line))
        else:
            table_lines.append((lineno, line))
    header = _parse_header(header_lines, path)
    table = _parse_table(table_lines, path)
    try:
        model = LayerModel(
            name=header["name"],
            datum=header["datum"],
            source=SourceRef(
                citation=header["citation"],
                url=header["url"],
                verified=header["verified"] == "true",
            ),
            top_elev_m=table[:, 0],
            vp_m_per_s=table[:, 1],
            vs_m_per_s=table[:, 2],
            source_file=header["sourceFile"],
            license=header["license"],
            header=header,
        )
    except LayerFileError as err:
        raise LayerFileError(f"{path}: {err}") from err
    log.info(
        "loaded velocity model %s from %s: %d layers, top %.1f m ASL, half-space from %.1f m ASL "
        "(%.3f s)",
        model.name, path, model.n_layers, model.top_of_model_elev_m, float(model.top_elev_m[-1]),
        time.perf_counter() - started,
    )
    return model


def _step_profile(
    tops: NDArray[np.float64], values: NDArray[np.float64], bottom_elev_m: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Staircase (value, elevation) vertices for a layered profile down to ``bottom_elev_m``."""
    bottoms = np.append(tops[1:], bottom_elev_m)
    elev = np.column_stack([tops, bottoms]).ravel()
    vals = np.repeat(values, 2)
    return vals, elev


def plot_profile(
    model: LayerModel,
    path: Path,
    *,
    bottom_elev_m: float,
    reference_elevations: Mapping[str, float] | None = None,
) -> Path:
    """Plot Vp, Vs and Vp/Vs against elevM with the datum stated on the figure; write a PNG.

    ``bottom_elev_m`` (m ASL, from config) is where the drawing of the half-space stops; it must
    lie below the half-space top. ``reference_elevations`` draws labelled horizontal lines.
    """
    started = time.perf_counter()
    half_space_top = float(model.top_elev_m[-1])
    if not bottom_elev_m < half_space_top:
        raise ValueError(
            f"bottom_elev_m {bottom_elev_m} must lie below the half-space top {half_space_top}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(9.0, 8.0), layout="constrained")
    FigureCanvasAgg(fig)
    ax_v, ax_r = fig.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [3, 1]})
    vp_x, elev_y = _step_profile(model.top_elev_m, model.vp_m_per_s, bottom_elev_m)
    vs_x, _ = _step_profile(model.top_elev_m, model.vs_m_per_s, bottom_elev_m)
    ratio_x, _ = _step_profile(model.top_elev_m, model.vp_m_per_s / model.vs_m_per_s, bottom_elev_m)
    ax_v.plot(vp_x, elev_y, color="tab:blue", label="Vp")
    ax_v.plot(vs_x, elev_y, color="tab:red", label="Vs")
    ax_r.plot(ratio_x, elev_y, color="black")

    for ax in (ax_v, ax_r):
        for top in model.top_elev_m[1:]:
            ax.axhline(float(top), color="0.85", linewidth=0.8, zorder=0)
        ax.axhline(half_space_top, color="0.5", linestyle=":", linewidth=1.0)
        ax.grid(axis="x", color="0.92")
    ax_v.annotate(
        "half-space", xy=(0.01, half_space_top), xycoords=("axes fraction", "data"),
        va="top", fontsize=8, color="0.4",
    )
    if model.top_extension is not None:
        ext = model.top_extension
        for ax in (ax_v, ax_r):
            ax.axhspan(ext.fromElevM, ext.toElevM, color="orange", alpha=0.15, zorder=0)
        ax_v.annotate(
            f"top layer extended from {ext.fromElevM:.0f} m ASL", xy=(0.01, ext.toElevM),
            xycoords=("axes fraction", "data"), va="top", fontsize=8, color="darkorange",
        )
    for label, elev in (reference_elevations or {}).items():
        for ax in (ax_v, ax_r):
            ax.axhline(elev, color="tab:green", linestyle="--", linewidth=1.0)
        ax_v.annotate(
            f"{label} ({elev:.1f} m ASL)", xy=(0.99, elev), xycoords=("axes fraction", "data"),
            ha="right", va="bottom", fontsize=8, color="tab:green",
        )

    ax_v.set_xlabel("velocity (m/s)")
    ax_v.set_ylabel("elevM (m above mean sea level, up positive)")
    ax_v.legend(loc="lower left")
    ax_r.set_xlabel("Vp/Vs")
    fig.suptitle(f"{model.name}: {model.n_layers} layers", fontsize=11)
    footer = f"Datum: {model.datum}\nSource: {model.source.url} ({model.source_file})"
    fig.text(0.5, -0.01, footer, ha="center", va="top", fontsize=8, wrap=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("wrote velocity profile %s (%.2f s)", path, time.perf_counter() - started)
    return path
