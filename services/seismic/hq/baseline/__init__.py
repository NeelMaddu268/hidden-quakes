"""Stage ``baseline`` (SEIS-07): STA/LTA picks and threshold sweep. See ``hq.baseline.run``.

The stage registry imports ``hq.baseline`` and calls ``run(ctx)``. Importing the submodule here,
before binding the name, keeps ``hq.baseline.run`` the stage function: a later first import of the
submodule would otherwise rebind the package attribute to the module.
"""

from hq.baseline.run import run

__all__ = ["run"]
