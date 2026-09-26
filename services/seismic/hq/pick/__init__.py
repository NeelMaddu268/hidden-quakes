"""PhaseNet picking (H1).

``hq.pick.phasenet`` picks one station's stream gap-safely; ``hq.pick.ab`` runs the weight A/B and
Check B on the known-event windows; ``hq.pick.run`` picks the full run window (SEIS-06) and its
``run(ctx)`` is this package's stage entry point (H4's stage registry calls ``hq.pick.run(ctx)``).
Importing this package does not import torch or seisbench; ``load_model`` does that on first use.

The stage function is bound over the submodule attribute of the same name, on purpose: RUN-01's
``resolve_stage`` takes ``getattr(hq.pick, "run")`` and needs a callable, and a lazy wrapper would
be replaced by the module object as soon as anything (a spawned worker, the wrapper itself)
imports ``hq.pick.run``. Two consequences: ``import hq.pick.run as m`` binds the function, so use
``from hq.pick.run import ...`` or ``importlib.import_module("hq.pick.run")`` for the module; and
``python -m hq.pick.run`` prints runpy's harmless "found in sys.modules" ``RuntimeWarning``.
``hq.pick.run`` imports ``hq.pick.ab`` only lazily, so ``python -m hq.pick.ab`` stays silent.
"""

from hq.pick.phasenet import PickDiagnostics, load_model, make_pick, pick_stream
from hq.pick.run import run  # binds the stage function over the submodule attribute of that name

__all__ = ["PickDiagnostics", "load_model", "make_pick", "pick_stream", "run"]
