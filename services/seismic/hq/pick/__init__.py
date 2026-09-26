"""PhaseNet picking (H1).

``hq.pick.phasenet`` picks one station's stream gap-safely; ``hq.pick.ab`` runs the weight A/B and
Check B on the known-event windows; ``hq.pick.run`` picks the full run window (SEIS-06) and its
``run(ctx)`` is this package's stage entry point (H4's stage registry calls ``hq.pick.run(ctx)``).
Importing this package does not import torch or seisbench; ``load_model`` does that on first use.
"""

from hq.pick.phasenet import PickDiagnostics, load_model, make_pick, pick_stream
from hq.pick.run import run  # binds the stage function over the submodule attribute of that name

__all__ = ["PickDiagnostics", "load_model", "make_pick", "pick_stream", "run"]
