"""PhaseNet picking (H1).

``hq.pick.phasenet`` picks one station's stream gap-safely; ``hq.pick.ab`` runs the weight A/B and
Check B on the known-event windows. Importing this package does not import torch or seisbench;
``load_model`` does that on first use.
"""

from hq.pick.phasenet import PickDiagnostics, load_model, make_pick, pick_stream

__all__ = ["PickDiagnostics", "load_model", "make_pick", "pick_stream"]
