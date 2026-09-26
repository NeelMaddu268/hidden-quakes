"""Local magnitude calibrated on matched public events (MAG-01, docs/lanes/H2 -> Magnitude).

``M_cat = a log10(A_peak) + b log10(R) + c + station term``, with ``A_peak`` the peak horizontal
Wood-Anderson amplitude (mm) in the S window after response removal and ``R`` the hypocentral
distance (km); the event magnitude (type ``ML_cal``) is the median of its station magnitudes.
The model is calibrated on one public regional catalog magnitude type only
(``magnitude.calibrationMagType``) and written into ``events.parquet`` only when its
leave-one-event-out MAE is at most ``magnitude.maxLooMae``.

- ``hq.magnitude.amplitude``: windows, station screening, response removal + Wood-Anderson,
  chunked amplitude measurement;
- ``hq.magnitude.calibrate``: robust fit with sum-to-zero station terms, event magnitudes,
  leave-one-event-out;
- ``hq.magnitude.run``: the stage. ``run`` here is that stage function, because the stage
  registry (H4's ``hq.runs.STAGES``) resolves stage ``magnitude`` as the attribute
  ``hq.magnitude.run``.
"""

from hq.magnitude.amplitude import (
    MagnitudeError,
    StationScreen,
    measure_amplitudes,
    plan_windows,
    screen_station,
)
from hq.magnitude.calibrate import (
    Calibration,
    event_magnitudes,
    fit_calibration,
    leave_one_event_out,
)

# Last, so the package attribute ``run`` is the stage function, not the submodule.
from hq.magnitude.run import run

__all__ = [
    "Calibration",
    "MagnitudeError",
    "StationScreen",
    "event_magnitudes",
    "fit_calibration",
    "leave_one_event_out",
    "measure_amplitudes",
    "plan_windows",
    "run",
    "screen_station",
]
