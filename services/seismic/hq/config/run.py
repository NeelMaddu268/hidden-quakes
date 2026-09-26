"""Run section: the showcase window, region and coordinate reference (``configs/showcase/run.yaml``).

Every lane reads these values. The catalog query and the waveform window use the same bounds,
and every ENU coordinate is measured from ``origin`` (docs/01 -> Conventions).
"""

from datetime import timedelta

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


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
        """``windowStart`` as epoch seconds UTC, the canonical ``t`` form."""
        return self.windowStart.timestamp()

    @property
    def window_end_s(self) -> float:
        """``windowEnd`` as epoch seconds UTC, the canonical ``t`` form."""
        return self.windowEnd.timestamp()
