"""Stage ``baseline`` (SEIS-07): STA/LTA picks and threshold sweep. See ``hq.baseline.run``.

H4's registry (``hq.runs.resolve_stage``) imports the module ``hq.baseline.run`` and calls its
``run(ctx)``. The function is also bound here, which is harmless: importing the submodule before
binding the name keeps ``hq.baseline.run`` the stage function, since a later first import of the
submodule would otherwise rebind the package attribute to the module.
"""

from hq.baseline.run import run

__all__ = ["run"]
