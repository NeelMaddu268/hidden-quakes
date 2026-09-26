"""Config loader: one YAML per lane composed into a ``RunConfig`` (docs/02 §3).

``load_config(dir)`` reads ``run.yaml`` (H2), ``signal.yaml`` (H1), ``seismology.yaml`` (H2) and
``export.yaml`` (H4) from one directory and validates each with its lane's Pydantic model.
Unknown keys are errors, never warnings.

``run`` and ``export`` are required. A lane section whose model module and YAML are both absent
loads as ``None`` with a warning naming the owner, so nobody waits on anybody; exactly one of the
two present is an error naming the owner. A stage that needs a section that is ``None`` asks for
it with ``RunConfig.section(name)`` and gets the same clear error.
"""

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from hq.config.export import ExportConfig
from hq.config.run import RunSection

log = logging.getLogger(__name__)


class ConfigError(ValueError):
    """A config directory, file or section is missing, malformed or holds unknown keys."""


@dataclass(frozen=True)
class SectionSpec:
    """One lane's config section: where its model lives, which file it reads, who owns it."""

    name: str
    module: str
    class_name: str
    filename: str
    owner: str
    required: bool


SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("run", "hq.config.run", "RunSection", "run.yaml", "H2 Seismology", True),
    SectionSpec("signal", "hq.config.signal", "SignalConfig", "signal.yaml", "H1 Signal", False),
    SectionSpec(
        "seismology",
        "hq.config.seismology",
        "SeismologyConfig",
        "seismology.yaml",
        "H2 Seismology",
        False,
    ),
    SectionSpec("export", "hq.config.export", "ExportConfig", "export.yaml", "H4 Platform", True),
)
SECTION_BY_NAME: dict[str, SectionSpec] = {spec.name: spec for spec in SECTIONS}


@dataclass(frozen=True)
class RunConfig:
    """Every lane's config for one run. ``signal`` / ``seismology`` are ``None`` until merged."""

    run: RunSection
    signal: Any  # hq.config.signal.SignalConfig (H1) once it exists, else None
    seismology: Any  # hq.config.seismology.SeismologyConfig (H2) once it exists, else None
    export: ExportConfig

    def section(self, name: str) -> Any:
        """The named section, or a ``ConfigError`` naming its owner when it isn't loaded."""
        spec = SECTION_BY_NAME.get(name)
        if spec is None:
            raise ConfigError(f"unknown config section {name!r}; sections: {list(SECTION_BY_NAME)}")
        value = getattr(self, name)
        if value is None:
            raise ConfigError(
                f"config section {name!r} is not loaded: {spec.filename} and "
                f"{spec.module}.{spec.class_name} are not merged yet (owner: {spec.owner})"
            )
        return value


def _import_model(spec: SectionSpec) -> type[BaseModel] | None:
    """The section's Pydantic model, or ``None`` when its module isn't merged yet."""
    try:
        module = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        if exc.name == spec.module:
            return None
        raise  # the module exists but one of its own imports is broken: a real error
    model = getattr(module, spec.class_name, None)
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise ConfigError(
            f"{spec.module} has no Pydantic model {spec.class_name} (owner: {spec.owner})"
        )
    if model.model_config.get("extra") != "forbid":
        raise ConfigError(
            f"{spec.module}.{spec.class_name} must set extra='forbid' so unknown keys in "
            f"{spec.filename} are errors (owner: {spec.owner})"
        )
    return model


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _load_section(config_dir: Path, spec: SectionSpec) -> BaseModel | None:
    yaml_path = config_dir / spec.filename
    has_yaml = yaml_path.is_file()
    model = _import_model(spec)
    if model is None and not has_yaml:
        if spec.required:
            raise ConfigError(
                f"required section {spec.name!r}: neither {yaml_path} nor "
                f"{spec.module}.{spec.class_name} exists (owner: {spec.owner})"
            )
        log.warning(
            "config section %r not loaded: %s and %s.%s are not merged yet (owner: %s)",
            spec.name,
            spec.filename,
            spec.module,
            spec.class_name,
            spec.owner,
        )
        return None
    if model is None:
        raise ConfigError(
            f"{yaml_path} exists but {spec.module} (owner: {spec.owner}) is not merged yet"
        )
    if not has_yaml:
        raise ConfigError(
            f"{spec.module}.{spec.class_name} exists but {yaml_path} is missing "
            f"(owner: {spec.owner})"
        )
    data = _read_yaml(yaml_path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{yaml_path} (owner: {spec.owner}):\n{exc}") from exc


def load_config(config_dir: Path) -> RunConfig:
    """Compose every lane's YAML in ``config_dir`` into one validated ``RunConfig``."""
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise ConfigError(f"config directory not found: {config_dir}")
    loaded = {spec.name: _load_section(config_dir, spec) for spec in SECTIONS}
    run, export = loaded["run"], loaded["export"]
    if not isinstance(run, RunSection) or not isinstance(export, ExportConfig):
        raise ConfigError(f"{config_dir}: run/export sections loaded as unexpected types")
    log.info(
        "loaded config from %s: %s",
        config_dir,
        ", ".join(name for name, value in loaded.items() if value is not None),
    )
    return RunConfig(
        run=run, signal=loaded["signal"], seismology=loaded["seismology"], export=export
    )


__all__ = [
    "SECTIONS",
    "SECTION_BY_NAME",
    "ConfigError",
    "ExportConfig",
    "RunConfig",
    "RunSection",
    "SectionSpec",
    "load_config",
]
