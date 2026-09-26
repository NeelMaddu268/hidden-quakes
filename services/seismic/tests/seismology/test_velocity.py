"""LOC-01: the committed FORGE 1D layer file, the layer-model loader and the seismology config."""

import json
import math
from pathlib import Path

import numpy as np
import pydantic
import pytest
import yaml

from hq.config.seismology import SeismologyConfig
from hq.locate.velocity import (
    COLUMNS,
    REQUIRED_HEADER_KEYS,
    LayerFileError,
    LayerModel,
    SourceRef,
    load_layer_model,
    plot_profile,
)

pytestmark = pytest.mark.smoke

SEISMIC_ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_YAML = SEISMIC_ROOT / "configs" / "showcase" / "seismology.yaml"

# GDR 1613 "Velocity Model Ver1.csv" (Inverted_1D_VelocityModel.csv, sha256 806cfad0...), as
# published: TVDSS (m), vp (m/s), vs (m/s). Pins the committed file to the source values.
PUBLISHED_TVDSS_VP_VS = [
    (-1800, 2296, 1163),
    (-1595, 2600, 1203),
    (-1227, 3097, 1808),
    (-1027, 4319, 2339),
    (-445, 5272, 2444),
    (-315, 5113, 3138),
    (217, 5615, 3374),
    (610, 5521, 3334),
]
# Physically sane Vp/Vs for crustal rock and compacted sediment: above sqrt(2) (Poisson's ratio > 0)
# and below 3 (only unconsolidated, water-saturated sediment goes higher).
VP_VS_MIN, VP_VS_MAX = math.sqrt(2.0), 3.0
SOURCEREF_FIELDS = {"citation", "url", "verified"}  # docs/02 SourceRef

VALID_HEADER = {
    "name": "test_model",
    "citation": "Test citation",
    "url": "https://example.invalid/submission",
    "sourceFile": "https://example.invalid/file.csv",
    "license": "CC BY 4.0",
    "sourceDatum": "depth below mean sea level",
    "layerConvention": "rows are layer tops",
    "conversion": "topElevM = -depth",
    "units": "m ASL; m/s",
    "datum": "topElevM is m above mean sea level.",
    "verified": "true",
}
VALID_ROWS = ["1000,3000,1700", "0,4000,2300", "-1000,5000,2900"]


def _write(tmp_path: Path, header: dict[str, str], rows: list[str], columns: str | None = None) -> Path:
    lines = [f"# {k}: {v}" for k, v in header.items()]
    lines.append(columns if columns is not None else ",".join(COLUMNS))
    lines.extend(rows)
    path = tmp_path / "model.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _toy(tops: list[float], vp: list[float], vs: list[float]) -> LayerModel:
    return LayerModel(
        name="toy",
        datum="topElevM is m above mean sea level.",
        source=SourceRef(citation="toy", url="https://example.invalid", verified=False),
        top_elev_m=np.array(tops),
        vp_m_per_s=np.array(vp),
        vs_m_per_s=np.array(vs),
    )


@pytest.fixture(scope="module")
def cfg() -> SeismologyConfig:
    return SeismologyConfig.model_validate(yaml.safe_load(SHOWCASE_YAML.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def forge(cfg: SeismologyConfig) -> LayerModel:
    return load_layer_model(SEISMIC_ROOT / cfg.velocity.layerFile)


# --- the committed FORGE model ---------------------------------------------------------------


def test_committed_file_matches_published_source(forge: LayerModel) -> None:
    tvdss, vp, vs = (np.array(col, dtype=float) for col in zip(*PUBLISHED_TVDSS_VP_VS, strict=True))
    np.testing.assert_array_equal(forge.top_elev_m, -tvdss)
    np.testing.assert_array_equal(forge.vp_m_per_s, vp)
    np.testing.assert_array_equal(forge.vs_m_per_s, vs)


def test_committed_model_is_physical(forge: LayerModel) -> None:
    assert np.all(np.diff(forge.top_elev_m) < 0)
    assert np.all(forge.vp_m_per_s > 0) and np.all(forge.vs_m_per_s > 0)
    assert np.all(forge.vs_m_per_s < forge.vp_m_per_s)
    ratio = forge.vp_m_per_s / forge.vs_m_per_s
    assert np.all((ratio > VP_VS_MIN) & (ratio < VP_VS_MAX)), ratio


def test_committed_header_states_source_and_datum(forge: LayerModel) -> None:
    for key in REQUIRED_HEADER_KEYS:
        assert forge.header[key]
    assert forge.source.url == "https://gdr.openei.org/submissions/1613"
    assert forge.source_file == "https://gdr.openei.org/files/1613/Inverted_1D_VelocityModel.csv"
    assert "10.15121/2376139" in forge.source.citation
    assert "mean sea level" in forge.datum and "half-space" in forge.datum
    assert "TVDSS" in forge.header["sourceDatum"] and "p. 4" in forge.header["sourceDatum"]
    assert "CC BY 4.0" in forge.license
    sha = forge.header["sourceSha256"]
    assert len(sha) == 64 and set(sha) <= set("0123456789abcdef")


def test_to_record_fills_sourceref_and_layers(forge: LayerModel) -> None:
    rec = forge.to_record()
    assert set(rec["source"]) == SOURCEREF_FIELDS
    assert rec["source"]["citation"] and rec["source"]["url"].startswith("https://")
    assert isinstance(rec["source"]["verified"], bool)
    assert rec["name"] == forge.name and rec["datum"] == forge.datum
    assert [row["topElevM"] for row in rec["layers"]] == forge.top_elev_m.tolist()
    assert set(rec["layers"][0]) == set(COLUMNS)
    assert rec["topExtension"] is None
    json.dumps(rec)  # ProcessingRun is JSON


# --- lookup and boundary convention ------------------------------------------------------------


def test_boundary_takes_layer_below(forge: LayerModel) -> None:
    tops = forge.top_elev_m
    for i in range(1, forge.n_layers):
        assert forge.vp_at(tops[i]) == forge.vp_m_per_s[i]
        assert forge.vs_at(tops[i]) == forge.vs_m_per_s[i]
        just_above = np.nextafter(tops[i], np.inf)
        assert forge.vp_at(just_above) == forge.vp_m_per_s[i - 1]
    assert forge.vp_at(tops[0]) == forge.vp_m_per_s[0]  # top of model belongs to layer 0
    assert forge.vs_at(tops[-1] - 5000.0) == forge.vs_m_per_s[-1]  # half-space


def test_lookup_is_vectorized_and_matches_scalar_reference(forge: LayerModel) -> None:
    rng = np.random.default_rng(1613)
    elev = rng.uniform(forge.top_elev_m[-1] - 1000.0, forge.top_elev_m[0], size=(40, 25))
    elev[0, : forge.n_layers] = forge.top_elev_m  # include every boundary exactly
    vp = forge.vp_at(elev)
    assert vp.shape == elev.shape
    for z, v in zip(elev.ravel(), vp.ravel(), strict=True):
        layer = max(i for i, t in enumerate(forge.top_elev_m) if t >= z)
        assert v == forge.vp_m_per_s[layer]
    assert forge.vs_at(float(forge.top_elev_m[2])).shape == ()


def test_above_top_raises(forge: LayerModel) -> None:
    with pytest.raises(ValueError, match="above the top"):
        forge.vp_at(forge.top_of_model_elev_m + 0.001)
    with pytest.raises(ValueError, match="above the top"):
        forge.vs_at([0.0, forge.top_of_model_elev_m + 100.0])
    with pytest.raises(ValueError, match="finite"):
        forge.vp_at([0.0, np.nan])


def test_top_extension_is_explicit_and_recorded(forge: LayerModel) -> None:
    source_top = forge.top_of_model_elev_m
    ext = forge.with_top_extended_to(2450.0)
    assert ext.top_of_model_elev_m == 2450.0
    np.testing.assert_array_equal(ext.top_elev_m[1:], forge.top_elev_m[1:])
    np.testing.assert_array_equal(ext.vp_m_per_s, forge.vp_m_per_s)
    assert ext.vp_at(2400.0) == forge.vp_m_per_s[0]
    assert ext.to_record()["topExtension"] == {"fromElevM": source_top, "toElevM": 2450.0}
    again = ext.with_top_extended_to(2600.0)
    assert again.to_record()["topExtension"] == {"fromElevM": source_top, "toElevM": 2600.0}
    assert forge.top_of_model_elev_m == source_top and forge.top_extension is None
    with pytest.raises(ValueError, match="nothing to extend"):
        forge.with_top_extended_to(source_top)
    with pytest.raises(ValueError, match="above the top"):
        forge.vp_at(2400.0)


def test_arrays_are_read_only(forge: LayerModel) -> None:
    with pytest.raises(ValueError):
        forge.vp_m_per_s[0] = 1.0


def test_single_layer_half_space() -> None:
    model = _toy([500.0], [4000.0], [2300.0])
    np.testing.assert_array_equal(model.vp_at([500.0, 0.0, -9000.0]), [4000.0] * 3)


# --- malformed files fail loudly ---------------------------------------------------------------


def test_valid_tmp_file_loads(tmp_path: Path) -> None:
    model = load_layer_model(_write(tmp_path, VALID_HEADER, VALID_ROWS))
    assert model.n_layers == 3 and model.source.verified is True


@pytest.mark.parametrize("missing", REQUIRED_HEADER_KEYS)
def test_missing_header_field_fails(tmp_path: Path, missing: str) -> None:
    header = {k: v for k, v in VALID_HEADER.items() if k != missing}
    with pytest.raises(LayerFileError, match=missing):
        load_layer_model(_write(tmp_path, header, VALID_ROWS))


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (["1000,3000,1700", "1000,4000,2300"], "strictly decrease"),
        (["0,3000,1700", "1000,4000,2300"], "strictly decrease"),
        (["1000,-3000,1700"], "positive"),
        (["1000,3000,0"], "positive"),
        (["1000,3000,3000"], "Vs must be below Vp"),
        (["1000,3000,abc"], "non-numeric"),
        (["1000,3000"], "expected 3 values"),
        (["1000,3000,1700,5"], "expected 3 values"),
        (["1000,nan,1700"], "non-finite"),
        ([], "no layer rows"),
    ],
)
def test_bad_rows_fail(tmp_path: Path, rows: list[str], match: str) -> None:
    with pytest.raises(LayerFileError, match=match):
        load_layer_model(_write(tmp_path, VALID_HEADER, rows))


def test_bad_columns_fail(tmp_path: Path) -> None:
    with pytest.raises(LayerFileError, match="columns must be"):
        load_layer_model(_write(tmp_path, VALID_HEADER, VALID_ROWS, columns="depthM,vp,vs"))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("# name: a\n# name: b\n", "duplicate"),
        ("# colour: blue\n", "unknown header key"),
        ("# just a comment\n", "key: value"),
        ("# name:\n", "empty"),
    ],
)
def test_bad_header_lines_fail(tmp_path: Path, text: str, match: str) -> None:
    path = _write(tmp_path, VALID_HEADER, VALID_ROWS)
    path.write_text(text + path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(LayerFileError, match=match):
        load_layer_model(path)


def test_bad_verified_value_fails(tmp_path: Path) -> None:
    with pytest.raises(LayerFileError, match="verified"):
        load_layer_model(_write(tmp_path, {**VALID_HEADER, "verified": "yes"}, VALID_ROWS))


def test_comment_inside_table_and_blank_lines_fail(tmp_path: Path) -> None:
    with pytest.raises(LayerFileError, match="before the table"):
        load_layer_model(_write(tmp_path, VALID_HEADER, [VALID_ROWS[0], "# late", VALID_ROWS[1]]))
    with pytest.raises(LayerFileError, match="blank"):
        load_layer_model(_write(tmp_path, VALID_HEADER, [VALID_ROWS[0], "", VALID_ROWS[1]]))


# --- plot and config ---------------------------------------------------------------------------


def test_plot_profile_writes_png(forge: LayerModel, cfg: SeismologyConfig, tmp_path: Path) -> None:
    out = plot_profile(
        forge.with_top_extended_to(2450.0),
        tmp_path / "profile.png",
        bottom_elev_m=cfg.velocity.profilePlotBottomElevM,
        reference_elevations={"reference surface": 1627.7},
    )
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="below the half-space top"):
        plot_profile(forge, tmp_path / "bad.png", bottom_elev_m=float(forge.top_elev_m[-1]))


def test_seismology_config(cfg: SeismologyConfig) -> None:
    assert not cfg.velocity.layerFile.is_absolute()
    assert cfg.velocity.model3d.url.endswith(cfg.velocity.model3d.cacheFile)
    raw = yaml.safe_load(SHOWCASE_YAML.read_text(encoding="utf-8"))
    with pytest.raises(pydantic.ValidationError):
        SeismologyConfig.model_validate({**raw, "surprise": 1})
    with pytest.raises(pydantic.ValidationError):
        SeismologyConfig.model_validate(
            {"velocity": {**raw["velocity"], "layerFile": "/abs/forge_1d.csv"}}
        )
    with pytest.raises(pydantic.ValidationError):
        cfg.velocity.layerFile = Path("other.csv")  # type: ignore[misc]
