"""LOC-01: the committed FORGE 1D layer file, the layer-model loader and the seismology config."""

import copy
import csv
import dataclasses
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import pytest
import yaml

from hq.config.run import RunSection
from hq.config.seismology import SEISMIC_ROOT, SeismologyConfig, Velocity3dConfig, VelocityConfig
from hq.locate.velocity import (
    COLUMNS,
    LayerFileError,
    LayerModel,
    SourceRef,
    check_units,
    load_configured_model,
    load_layer_model,
    plot_profile,
    profile_figure,
)

pytestmark = pytest.mark.smoke

SHOWCASE = SEISMIC_ROOT / "configs" / "showcase"
# Unmodified bytes of GDR 1613 Inverted_1D_VelocityModel.csv (CC BY 4.0; attribution in
# fixtures/README.txt): "TVDSS (m),vp (m/s),vs (m/s)," then rows with trailing commas, CRLF.
SOURCE_FIXTURE = Path(__file__).parent / "fixtures" / "gdr1613_Inverted_1D_VelocityModel.csv"
# Physically sane Vp/Vs for crustal rock and compacted sediment: above sqrt(2) (Poisson's ratio > 0)
# and below 3 (only unconsolidated, water-saturated sediment goes higher).
VP_VS_MIN, VP_VS_MAX = math.sqrt(2.0), 3.0
SOURCEREF_FIELDS = {"citation", "url", "verified"}  # docs/02 SourceRef
# Written out here rather than imported, so a key dropped from the module's own lists fails a test.
REQUIRED_HEADER = (
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
EVIDENCE_KEYS = (
    "sourceSha256",
    "sourceDatum",
    "layerConvention",
    "conversion",
    "notes",
    "verifiedBasis",
)
RECORD_KEYS = {
    "name",
    "source",
    "sourceFile",
    "license",
    "datum",
    "boundaryConvention",
    "layers",
    "topExtension",
    *EVIDENCE_KEYS,
}
# Every top-level section of seismology.yaml, so a section merged in from another ticket is covered.
YAML_SECTIONS = (
    "<root>",
    *yaml.safe_load((SHOWCASE / "seismology.yaml").read_text(encoding="utf-8")),
    "velocity.model3d",
)

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
    "verifiedBasis": "the test says so",
    "sourceSha256": hashlib.sha256(b"test source").hexdigest(),
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
        source_file="https://example.invalid/toy.csv",
        license="CC0",
    )


def _layer_rows(model: LayerModel) -> list[tuple[float, float, float]]:
    return list(zip(model.top_elev_m, model.vp_m_per_s, model.vs_m_per_s, strict=True))


def _record_rows(rec: dict) -> list[tuple[float, float, float]]:
    return [(row["topElevM"], row["vpMPerS"], row["vsMPerS"]) for row in rec["layers"]]


@pytest.fixture(scope="module")
def cfg() -> SeismologyConfig:
    raw = yaml.safe_load((SHOWCASE / "seismology.yaml").read_text(encoding="utf-8"))
    return SeismologyConfig.model_validate(raw)


@pytest.fixture(scope="module")
def run() -> RunSection:
    raw = yaml.safe_load((SHOWCASE / "run.yaml").read_text(encoding="utf-8"))
    return RunSection.model_validate(raw)


@pytest.fixture(scope="module")
def forge(cfg: SeismologyConfig) -> LayerModel:
    return load_configured_model(cfg.velocity)


def _load(path: Path, cfg: SeismologyConfig) -> LayerModel:
    return load_layer_model(
        path,
        vp_range_m_per_s=cfg.velocity.plausibleVpMPerS,
        vs_range_m_per_s=cfg.velocity.plausibleVsMPerS,
        min_layer_thickness_m=cfg.velocity.minLayerThicknessM,
    )


def _extend(model: LayerModel, elev_m: float, cfg: SeismologyConfig) -> LayerModel:
    return model.with_top_extended_to(elev_m, max_extension_m=cfg.velocity.maxTopExtensionM)


# --- the committed FORGE model ---------------------------------------------------------------


def test_committed_file_matches_source_bytes(forge: LayerModel) -> None:
    raw = SOURCE_FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == forge.header["sourceSha256"]
    rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    assert [c.strip() for c in rows[0] if c] == ["TVDSS (m)", "vp (m/s)", "vs (m/s)"]
    src = np.array([[float(c) for c in row if c] for row in rows[1:]])
    np.testing.assert_array_equal(forge.top_elev_m, -src[:, 0])  # topElevM = -TVDSS
    np.testing.assert_array_equal(forge.vp_m_per_s, src[:, 1])
    np.testing.assert_array_equal(forge.vs_m_per_s, src[:, 2])


def test_committed_model_is_physical(forge: LayerModel, cfg: SeismologyConfig) -> None:
    assert np.all(np.diff(forge.top_elev_m) < 0)
    assert np.all(forge.vp_m_per_s > 0) and np.all(forge.vs_m_per_s > 0)
    assert np.all(forge.vs_m_per_s < forge.vp_m_per_s)
    ratio = forge.vp_m_per_s / forge.vs_m_per_s
    assert np.all((ratio > VP_VS_MIN) & (ratio < VP_VS_MAX)), ratio
    for values, (low, high) in (
        (forge.vp_m_per_s, cfg.velocity.plausibleVpMPerS),
        (forge.vs_m_per_s, cfg.velocity.plausibleVsMPerS),
    ):
        assert np.all((values >= low) & (values <= high)), values


def test_committed_header_states_source_and_datum(forge: LayerModel) -> None:
    for key in (*REQUIRED_HEADER, *EVIDENCE_KEYS):
        assert forge.header[key], key
    assert forge.source.url == "https://gdr.openei.org/submissions/1613"
    assert forge.source_file == "https://gdr.openei.org/files/1613/Inverted_1D_VelocityModel.csv"
    assert "10.15121/2376139" in forge.source.citation
    assert "mean sea level" in forge.datum and "half-space" in forge.datum
    assert "NAVD88" in forge.datum and "not by the velocity CSV" in forge.datum
    assert "TVDSS" in forge.header["sourceDatum"] and "p. 4" in forge.header["sourceDatum"]
    assert "Fig. 3a" in forge.header["sourceDatum"] and "Fig. 3a" in forge.header["layerConvention"]
    assert "5 events" in forge.header["notes"]
    assert forge.header["verifiedBasis"]
    assert "CC BY 4.0" in forge.license


def test_to_record_carries_sourceref_layers_and_evidence(forge: LayerModel) -> None:
    rec = forge.to_record()
    assert set(rec) == RECORD_KEYS
    assert set(rec["source"]) == SOURCEREF_FIELDS
    assert rec["source"]["citation"] and rec["source"]["url"].startswith("https://")
    assert rec["source"]["verified"] is True
    assert rec["name"] == forge.name and rec["datum"] == forge.datum
    assert _record_rows(rec) == _layer_rows(forge)
    assert set(rec["layers"][0]) == set(COLUMNS)
    for key in EVIDENCE_KEYS:
        assert isinstance(rec[key], str) and rec[key] == forge.header[key], key
    assert "TVDSS" in rec["sourceDatum"] and "mean sea level" in rec["sourceDatum"]
    assert "Fig. 3a" in rec["layerConvention"] and "-TVDSS" in rec["conversion"]
    assert "boundary takes the layer below" in rec["boundaryConvention"]
    assert rec["topExtension"] is None
    json.dumps(rec)  # ProcessingRun is JSON


def test_verified_false_reaches_the_record(tmp_path: Path, cfg: SeismologyConfig) -> None:
    model = _load(_write(tmp_path, {**VALID_HEADER, "verified": "false"}, VALID_ROWS), cfg)
    assert model.source.verified is False
    assert model.to_record()["source"]["verified"] is False


def test_sourceref_stand_in_matches_docs02() -> None:
    assert {f.name for f in dataclasses.fields(SourceRef)} == SOURCEREF_FIELDS


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


def test_top_extension_is_explicit_and_recorded(forge: LayerModel, cfg: SeismologyConfig) -> None:
    source_top = forge.top_of_model_elev_m
    ext = _extend(forge, 2450.0, cfg)
    assert ext.top_of_model_elev_m == 2450.0
    np.testing.assert_array_equal(ext.top_elev_m[1:], forge.top_elev_m[1:])
    np.testing.assert_array_equal(ext.vp_m_per_s, forge.vp_m_per_s)
    np.testing.assert_array_equal(ext.vs_m_per_s, forge.vs_m_per_s)
    assert ext.vp_at(2400.0) == forge.vp_m_per_s[0]
    rec = ext.to_record()
    assert rec["topExtension"]["fromElevM"] == source_top
    assert rec["topExtension"]["toElevM"] == 2450.0
    assert "extended upward" in rec["topExtension"]["note"]
    assert _record_rows(rec) == _layer_rows(ext)
    assert rec["sourceSha256"] == forge.header["sourceSha256"]
    again = _extend(ext, 2600.0, cfg)
    assert again.to_record()["topExtension"]["fromElevM"] == source_top
    assert again.to_record()["topExtension"]["toElevM"] == 2600.0
    assert forge.top_of_model_elev_m == source_top and forge.top_extension is None
    with pytest.raises(ValueError, match="above the top"):
        forge.vp_at(2400.0)


def test_top_extension_is_a_no_op_when_the_model_already_reaches(
    forge: LayerModel, cfg: SeismologyConfig
) -> None:
    assert _extend(forge, forge.top_of_model_elev_m, cfg) is forge
    assert _extend(forge, 1000.0, cfg) is forge
    ext = _extend(forge, 2450.0, cfg)
    assert _extend(ext, 2450.0, cfg) is ext
    with pytest.raises(ValueError, match="finite"):
        _extend(forge, float("nan"), cfg)


def test_top_extension_is_capped_above_the_source_top(forge: LayerModel) -> None:
    top = forge.top_of_model_elev_m
    assert forge.with_top_extended_to(top + 500.0, max_extension_m=500.0).top_of_model_elev_m == (
        top + 500.0
    )
    with pytest.raises(ValueError, match="max_extension_m"):
        forge.with_top_extended_to(top + 500.5, max_extension_m=500.0)
    step = forge.with_top_extended_to(top + 300.0, max_extension_m=500.0)
    with pytest.raises(ValueError, match="source top"):  # measured from the source top, not the last
        step.with_top_extended_to(top + 600.0, max_extension_m=500.0)
    with pytest.raises(ValueError, match="finite"):
        forge.with_top_extended_to(top + 1.0, max_extension_m=float("nan"))


def _assert_read_only(model: LayerModel) -> None:
    for arr in (model.top_elev_m, model.vp_m_per_s, model.vs_m_per_s):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 1.0
    with pytest.raises(TypeError):
        model.header["verified"] = "false"  # type: ignore[index]


def test_arrays_and_header_are_read_only(forge: LayerModel) -> None:
    _assert_read_only(forge)


def test_pickle_and_copies_round_trip(forge: LayerModel, cfg: SeismologyConfig) -> None:
    for model in (forge, _extend(forge, 2450.0, cfg)):
        for twin in (pickle.loads(pickle.dumps(model)), copy.deepcopy(model), copy.copy(model)):
            assert twin.to_record() == model.to_record()
            assert dict(twin.header) == dict(model.header)
            _assert_read_only(twin)
    as_dict = dataclasses.asdict(forge)
    assert as_dict["header"]["sourceSha256"] == forge.header["sourceSha256"]
    np.testing.assert_array_equal(as_dict["vp_m_per_s"], forge.vp_m_per_s)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("verified", "false"),
        ("name", "other"),
        ("datum", "other"),
        ("citation", "other"),
        ("url", "other"),
        ("sourceFile", "other"),
        ("license", "other"),
    ],
)
def test_header_must_agree_with_fields(forge: LayerModel, key: str, value: str) -> None:
    with pytest.raises(LayerFileError, match=f"disagree on \\['{key}'\\]"):
        dataclasses.replace(forge, header={**forge.header, key: value})


def test_single_layer_half_space() -> None:
    model = _toy([500.0], [4000.0], [2300.0])
    np.testing.assert_array_equal(model.vp_at([500.0, 0.0, -9000.0]), [4000.0] * 3)


@pytest.mark.parametrize("empty", ["citation", "url", "source_file", "license", "name", "datum"])
def test_empty_provenance_fails(empty: str) -> None:
    toy = _toy([500.0], [4000.0], [2300.0])
    if empty == "citation":
        source = dataclasses.replace(toy.source, citation=" ")
    elif empty == "url":
        source = dataclasses.replace(toy.source, url=" ")
    else:
        source = toy.source
    changes: dict[str, Any] = {} if empty in ("citation", "url") else {empty: ""}
    with pytest.raises(LayerFileError, match=empty):
        dataclasses.replace(toy, source=source, **changes)


def test_check_units_catches_a_model_built_directly(cfg: SeismologyConfig) -> None:
    guards: dict[str, Any] = {
        "vp_range_m_per_s": cfg.velocity.plausibleVpMPerS,
        "vs_range_m_per_s": cfg.velocity.plausibleVsMPerS,
        "min_layer_thickness_m": cfg.velocity.minLayerThicknessM,
    }
    check_units(_toy([1000.0, 0.0], [3000.0, 4000.0], [1700.0, 2300.0]), **guards)
    with pytest.raises(LayerFileError, match="km/s"):  # e.g. a column read from a km/s NetCDF
        check_units(_toy([1000.0, 0.0], [3.0, 4.0], [1.7, 2.3]), **guards)
    with pytest.raises(LayerFileError, match="topElevM in km"):
        check_units(_toy([1.0, 0.0], [3000.0, 4000.0], [1700.0, 2300.0]), **guards)


# --- malformed files fail loudly ---------------------------------------------------------------


def test_valid_tmp_file_loads(tmp_path: Path, cfg: SeismologyConfig) -> None:
    model = _load(_write(tmp_path, VALID_HEADER, VALID_ROWS), cfg)
    assert model.n_layers == 3 and model.source.verified is True


def test_byte_order_mark_is_ignored(tmp_path: Path, cfg: SeismologyConfig) -> None:
    path = _write(tmp_path, VALID_HEADER, VALID_ROWS)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert _load(path, cfg).name == VALID_HEADER["name"]


@pytest.mark.parametrize("missing", REQUIRED_HEADER)
def test_missing_header_field_fails(tmp_path: Path, cfg: SeismologyConfig, missing: str) -> None:
    header = {k: v for k, v in VALID_HEADER.items() if k != missing}
    with pytest.raises(LayerFileError, match=missing):
        _load(_write(tmp_path, header, VALID_ROWS), cfg)


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
        (["1000,3.0,1.7", "0,4.0,2.3"], "km/s"),  # velocities in km/s under m/s column names
        (["1000,3000,50"], "vsMPerS values"),
        (["1000,12000,1700"], "vpMPerS values"),
        # the FORGE tops in km (TVDSS km, as Fig. 3a plots them) with velocities still in m/s
        (["1.8,2296,1163", "1.595,2600,1203", "-0.61,5521,3334"], "topElevM in km"),
    ],
)
def test_bad_rows_fail(tmp_path: Path, cfg: SeismologyConfig, rows: list[str], match: str) -> None:
    with pytest.raises(LayerFileError, match=match):
        _load(_write(tmp_path, VALID_HEADER, rows), cfg)


def test_bad_columns_fail(tmp_path: Path, cfg: SeismologyConfig) -> None:
    with pytest.raises(LayerFileError, match="columns must be"):
        _load(_write(tmp_path, VALID_HEADER, VALID_ROWS, columns="depthM,vp,vs"), cfg)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("# name: a\n# name: b\n", "duplicate"),
        ("# colour: blue\n", "unknown header key"),
        ("# just a comment\n", "key: value"),
        ("# name:\n", "empty"),
    ],
)
def test_bad_header_lines_fail(
    tmp_path: Path, cfg: SeismologyConfig, text: str, match: str
) -> None:
    path = _write(tmp_path, VALID_HEADER, VALID_ROWS)
    path.write_text(text + path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(LayerFileError, match=match):
        _load(path, cfg)


def test_bad_verified_value_fails(tmp_path: Path, cfg: SeismologyConfig) -> None:
    with pytest.raises(LayerFileError, match="verified"):
        _load(_write(tmp_path, {**VALID_HEADER, "verified": "yes"}, VALID_ROWS), cfg)


@pytest.mark.parametrize("missing", ["verifiedBasis", "sourceSha256"])
def test_verified_true_needs_basis_and_hash(
    tmp_path: Path, cfg: SeismologyConfig, missing: str
) -> None:
    header = {k: v for k, v in VALID_HEADER.items() if k != missing}
    with pytest.raises(LayerFileError, match=f"verified: true also needs .*{missing}"):
        _load(_write(tmp_path, header, VALID_ROWS), cfg)
    unverified = _load(_write(tmp_path, {**header, "verified": "false"}, VALID_ROWS), cfg)
    assert unverified.to_record()[missing] is None


@pytest.mark.parametrize(
    "sha", ["806cfad0", "806CFAD01F65CF9F2DAC3376A1BDC458F3BEDAF65E359C3E2815E28D249DD994", "x" * 64]
)
def test_malformed_sha256_fails(tmp_path: Path, cfg: SeismologyConfig, sha: str) -> None:
    with pytest.raises(LayerFileError, match="sourceSha256"):
        _load(_write(tmp_path, {**VALID_HEADER, "sourceSha256": sha}, VALID_ROWS), cfg)


def test_comment_inside_table_and_blank_lines_fail(tmp_path: Path, cfg: SeismologyConfig) -> None:
    with pytest.raises(LayerFileError, match="before the table"):
        _load(_write(tmp_path, VALID_HEADER, [VALID_ROWS[0], "# late", VALID_ROWS[1]]), cfg)
    with pytest.raises(LayerFileError, match="blank"):
        _load(_write(tmp_path, VALID_HEADER, [VALID_ROWS[0], "", VALID_ROWS[1]]), cfg)


# --- plot and config ---------------------------------------------------------------------------


def _figure_texts(model: LayerModel, cfg: SeismologyConfig, run: RunSection) -> list[str]:
    fig = profile_figure(
        model,
        bottom_elev_m=cfg.velocity.profilePlotBottomElevM,
        reference_elevations={"reference surface": run.refSurfaceElevM},
    )
    assert "elevM" in fig.axes[0].get_ylabel()
    return [t.get_text() for t in fig.texts] + [t.get_text() for ax in fig.axes for t in ax.texts]


def test_profile_figure_states_the_datum_source_and_extension(
    forge: LayerModel, cfg: SeismologyConfig, run: RunSection
) -> None:
    extended = f"top layer extended from {forge.top_of_model_elev_m:.0f} m ASL"
    for model, is_extended in ((_extend(forge, 2450.0, cfg), True), (forge, False)):
        texts = _figure_texts(model, cfg, run)
        assert any(forge.datum in t for t in texts)
        assert any(f"Source: {forge.source.url}" in t for t in texts)
        assert any(f"reference surface ({run.refSurfaceElevM:.1f} m ASL)" in t for t in texts)
        assert any(extended in t for t in texts) is is_extended
    with pytest.raises(ValueError, match="below the half-space top"):
        profile_figure(forge, bottom_elev_m=float(forge.top_elev_m[-1]))


def test_plot_profile_writes_png(
    forge: LayerModel, cfg: SeismologyConfig, run: RunSection, tmp_path: Path
) -> None:
    out = plot_profile(
        forge,
        tmp_path / "profile.png",
        bottom_elev_m=cfg.velocity.profilePlotBottomElevM,
        reference_elevations={"reference surface": run.refSurfaceElevM},
    )
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_layer_path_ignores_the_working_directory(
    cfg: SeismologyConfig, forge: LayerModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cfg.velocity.layer_path().is_absolute()
    assert load_configured_model(cfg.velocity).to_record() == forge.to_record()


def test_seismology_config(cfg: SeismologyConfig) -> None:
    assert not cfg.velocity.layerFile.is_absolute()
    assert cfg.velocity.layer_path().is_file()
    assert cfg.velocity.model3d.url.endswith(cfg.velocity.model3d.cacheFile)
    raw = yaml.safe_load((SHOWCASE / "seismology.yaml").read_text(encoding="utf-8"))
    rooted_paths = ("/abs/forge_1d.csv", "\\abs\\forge_1d.csv", "C:\\abs\\forge_1d.csv", "C:forge_1d.csv")
    for rooted in rooted_paths:
        with pytest.raises(pydantic.ValidationError, match="layerFile must be relative"):
            VelocityConfig.model_validate({**raw["velocity"], "layerFile": rooted})
    with pytest.raises(pydantic.ValidationError, match="plausibleVsMPerS"):
        VelocityConfig.model_validate({**raw["velocity"], "plausibleVsMPerS": [5000.0, 100.0]})
    for nested in ("a/b.nc", "a\\b.nc"):
        with pytest.raises(pydantic.ValidationError, match="cacheFile must be a bare file name"):
            Velocity3dConfig.model_validate({**raw["velocity"]["model3d"], "cacheFile": nested})
    with pytest.raises(pydantic.ValidationError):
        cfg.velocity.layerFile = Path("other.csv")  # type: ignore[misc]


@pytest.mark.parametrize("where", YAML_SECTIONS)
def test_unknown_config_keys_fail(where: str) -> None:
    raw = yaml.safe_load((SHOWCASE / "seismology.yaml").read_text(encoding="utf-8"))
    target = raw
    for key in [] if where == "<root>" else where.split("."):
        target = target[key]
    target["layerfile"] = "typo.csv"
    with pytest.raises(pydantic.ValidationError, match="layerfile"):
        SeismologyConfig.model_validate(raw)
