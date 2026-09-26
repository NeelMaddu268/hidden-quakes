"""The JSON sidecars of ``runs/<runId>/`` that the validate stage writes or embeds and that the
exporter reads when ``validation.json`` is not there (docs/02 §2).

Every sidecar holds one contract model (or a list of one) as JSON; ``validation_notes.json``
holds H4's own ``hq.validate.notes.ValidationNotes`` (provenance the frozen contract models
have no field for). ``Sidecar.read`` returns
``None`` when the file does not exist and raises the caller's error type when it exists but is
not valid, so a missing sidecar is a documented gap and a corrupt one fails loudly in both
stages. ``Sidecar.write`` is atomic and byte-stable: same value, same bytes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hq_contracts.models import (
    BaselineRow,
    GRCurve,
    MagCalibration,
    NullTest,
    SyntheticTest,
    Validation,
)
from pydantic import TypeAdapter, ValidationError

from hq.runs import write_text_atomic
from hq.validate.notes import ValidationNotes

H2 = "H2 Seismology"
H4 = "H4 Platform"


@dataclass(frozen=True)
class Sidecar:
    """One JSON file of the run directory: its name, its model and who writes it."""

    filename: str
    adapter: TypeAdapter[Any]
    label: str  # the model, for messages ("SyntheticTest", "list of BaselineRow")
    stage: str  # the stage that writes it (docs/01 -> Pipeline)
    owner: str

    def path(self, run_dir: Path) -> Path:
        return run_dir / self.filename

    def read(self, run_dir: Path, error: type[Exception]) -> Any:
        """The parsed file, ``None`` when absent, ``error`` when present but invalid."""
        path = self.path(run_dir)
        if not path.is_file():
            return None
        try:
            return self.adapter.validate_json(path.read_bytes())
        except (ValidationError, ValueError) as exc:  # ValueError: malformed JSON
            raise error(f"{path} is not a valid {self.label}:\n{exc}") from exc

    def write(self, run_dir: Path, value: Any) -> Path:
        path = self.path(run_dir)
        write_text_atomic(path, self.adapter.dump_json(value, indent=2).decode() + "\n")
        return path


SYNTHETIC = Sidecar("synthetic.json", TypeAdapter(SyntheticTest), "SyntheticTest", "locate", H2)
MAGNITUDE = Sidecar(
    "magnitude.json", TypeAdapter(MagCalibration), "MagCalibration", "magnitude", H2
)
NULL_TEST = Sidecar("null_test.json", TypeAdapter(NullTest), "NullTest", "validate", H4)
BASELINE = Sidecar(
    "baseline.json", TypeAdapter(list[BaselineRow]), "list of BaselineRow", "validate", H4
)
GR = Sidecar("gr.json", TypeAdapter(GRCurve), "GRCurve", "validate", H4)
VALIDATION = Sidecar("validation.json", TypeAdapter(Validation), "Validation", "validate", H4)
NOTES = Sidecar(
    "validation_notes.json", TypeAdapter(ValidationNotes), "ValidationNotes", "validate", H4
)

SYNTHETIC_JSON = SYNTHETIC.filename
MAGNITUDE_JSON = MAGNITUDE.filename
NULL_TEST_JSON = NULL_TEST.filename
BASELINE_JSON = BASELINE.filename
GR_JSON = GR.filename
VALIDATION_JSON = VALIDATION.filename
NOTES_JSON = NOTES.filename
