"""hq.config.run: the canonical epoch-seconds conversion shared by window bounds and origin times."""

from datetime import timedelta

import numpy as np
import pytest
from obspy import UTCDateTime

from hq.config.run import RunSection, epoch_ns, epoch_s

pytestmark = pytest.mark.smoke

US_PER_DAY = 86_400 * 1_000_000


def test_window_seconds_match_datetime_and_obspy_nanoseconds(run_section: RunSection) -> None:
    """At any microsecond instant: window_*_s == datetime.timestamp() == epoch_s(UTCDateTime.ns).

    ObsPy's own ``UTCDateTime.timestamp`` is one ulp low at some of these instants, which is why
    origin times go through ``epoch_s`` too; the sample must contain such an instant.
    """
    rng = np.random.default_rng(7)
    obspy_low = 0
    for offset_us in rng.integers(0, US_PER_DAY, size=2000):
        instant = run_section.windowStart + timedelta(microseconds=int(offset_us))
        run = RunSection.model_validate(
            {
                **run_section.model_dump(),
                "windowStart": instant,
                "windowEnd": instant + timedelta(microseconds=1),
            }
        )
        ns = UTCDateTime(instant).ns
        assert epoch_ns(instant) == ns
        assert run.window_start_s == instant.timestamp() == epoch_s(ns)
        obspy_low += UTCDateTime(instant).timestamp < run.window_start_s
    assert obspy_low > 0


def test_showcase_window_seconds(run_section: RunSection) -> None:
    assert run_section.window_start_s == run_section.windowStart.timestamp()
    assert run_section.window_end_s == run_section.windowEnd.timestamp()
    assert run_section.window_start_s == epoch_s(UTCDateTime(run_section.windowStart).ns)
