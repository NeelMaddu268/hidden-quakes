"""PhaseNet picking (H1).

``hq.pick.phasenet`` picks one station's stream gap-safely; ``hq.pick.ab`` runs the weight A/B and
Check B on the known-event windows; ``hq.pick.run`` picks the full run window (SEIS-06) and its
``run(ctx)`` is the stage entry point (H4's registry, ``hq.runs.resolve_stage``, imports the
module ``hq.pick.run`` and takes its ``run``).
Importing this package does not import torch or seisbench; ``load_model`` does that on first use.

The stage function is also bound here, over the submodule attribute of the same name, so
``hq.pick.run`` is callable. The registry does not rely on it, and the rebinding is harmless
(a lazy wrapper would be replaced by the module object as soon as anything imported
``hq.pick.run``). Two consequences: ``import hq.pick.run as m`` binds the function, so use
``from hq.pick.run import ...`` or ``importlib.import_module("hq.pick.run")`` for the module; and
``python -m hq.pick.run`` prints runpy's harmless "found in sys.modules" ``RuntimeWarning``.
``hq.pick.run`` imports ``hq.pick.ab`` only lazily, so ``python -m hq.pick.ab`` stays silent.
"""

from hq.pick.phasenet import PickDiagnostics, load_model, make_pick, pick_stream
from hq.pick.run import run  # binds the stage function over the submodule attribute of that name

__all__ = ["PickDiagnostics", "load_model", "make_pick", "pick_stream", "run"]
