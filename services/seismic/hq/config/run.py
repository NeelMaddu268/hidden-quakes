"""Run section: the showcase window, region and coordinate reference (``configs/showcase/run.yaml``).

Every lane reads these values. The catalog query and the waveform window use the same bounds,
and every ENU coordinate is measured from ``origin`` (docs/01 -> Conventions).
"""

from datetime import UTC, datetime, timedelta

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NS_PER_S = 1_000_000_000
_NS_PER_US = 1_000


def epoch_s(ns: int) -> float:
    """Integer epoch nanoseconds UTC -> ``t``, float epoch seconds UTC (docs/01 -> Conventions).

    The one conversion to the canonical ``t`` form: exact integer division, so the result is the
    correctly rounded double of the instant. ``RunSection.window_start_s``/``window_end_s`` and
    H2's origin times (``epoch_s(UTCDateTime.ns)``) both go through it, so an instant exactly on a
    window edge compares equal to that edge. ObsPy's ``UTCDateTime.timestamp`` divides by the float
    ``1e9`` and lands one ulp low at many sub-second instants; don't mix it with these values.
    """
    return ns / _NS_PER_S


def epoch_ns(instant: datetime) -> int:
    """Exact epoch nanoseconds UTC of an aware datetime (microsecond resolution)."""
    return (instant - _EPOCH) // timedelta(microseconds=1) * _NS_PER_US


class Origin(BaseModel):
    """Fixed scene/ENU reference point. ``u = elevM - origin.elevM``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    elevM: float  # m above sea level


class RunSection(BaseModel):
    """Contents of ``run.yaml``. Unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    windowStart: AwareDatetime  # ISO 8601 UTC, inclusive
    windowEnd: AwareDatetime  # ISO 8601 UTC, exclusive
    bbox: tuple[float, float, float, float]  # minLon, minLat, maxLon, maxLat
    origin: Origin
    refSurfaceElevM: float  # m ASL; display depthKm = (refSurfaceElevM - elevM) / 1000

    @model_validator(mode="after")
    def _check(self) -> "RunSection":
        for label, value in (("windowStart", self.windowStart), ("windowEnd", self.windowEnd)):
            if value.utcoffset() != timedelta(0):
                raise ValueError(f"{label} must be UTC (offset +00:00 or Z), got {value.isoformat()}")
        if self.windowEnd <= self.windowStart:
            raise ValueError("windowEnd must be after windowStart")
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if not (-180.0 <= min_lon < max_lon <= 180.0 and -90.0 <= min_lat < max_lat <= 90.0):
            raise ValueError(f"bbox must be (minLon, minLat, maxLon, maxLat), got {self.bbox}")
        if not (min_lon <= self.origin.lon <= max_lon and min_lat <= self.origin.lat <= max_lat):
            raise ValueError("origin must lie inside bbox")
        return self

    @property
    def window_start_s(self) -> float:
        """``windowStart`` as epoch seconds UTC, the canonical ``t`` form (``epoch_s``)."""
        return epoch_s(epoch_ns(self.windowStart))

    @property
    def window_end_s(self) -> float:
        """``windowEnd`` as epoch seconds UTC, the canonical ``t`` form (``epoch_s``)."""
        return epoch_s(epoch_ns(self.windowEnd))
