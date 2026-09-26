"""LOC-10: pick harvest at predicted arrivals (hq.locate.harvest) and its config section.

Offline. The smoke tests work on small tables built here.
"""

from typing import Any

import pytest

from hq.config.seismology import SeismologyConfig

# --- config (smoke) ----------------------------------------------------------------------------


def _with_harvest(raw: dict[str, Any], **harvest: Any) -> dict[str, Any]:
    return {**raw, "harvest": {**raw["harvest"], **harvest}}


@pytest.mark.smoke
def test_showcase_config_ships_harvest_off(seismology_config: SeismologyConfig) -> None:
    h = seismology_config.harvest
    assert h.enabled is False
    assert h.windowS.P <= seismology_config.locator.outlier.floorS
    assert h.windowS.S <= seismology_config.locator.outlier.floorS
    assert h.minProb >= seismology_config.associator.minPickProb


@pytest.mark.smoke
def test_harvest_config_rejects_bad_values(seismology_config: SeismologyConfig) -> None:
    raw = seismology_config.model_dump(mode="json")
    floor = raw["locator"]["outlier"]["floorS"]
    with pytest.raises(ValueError, match="must not exceed locator.outlier.floorS"):
        SeismologyConfig.model_validate(_with_harvest(raw, windowS={"P": 0.1, "S": floor + 0.01}))
    with pytest.raises(ValueError, match="must not exceed locator.outlier.floorS"):
        SeismologyConfig.model_validate(_with_harvest(raw, windowS={"P": floor + 0.01, "S": 0.1}))
    below = raw["associator"]["minPickProb"] - 0.05
    with pytest.raises(ValueError, match="must not be below associator.minPickProb"):
        SeismologyConfig.model_validate(_with_harvest(raw, minProb=below))
    for bad in ({"iterations": 1}, {"windowS": {"P": 0.0, "S": 0.1}}, {"minProb": 1.5}):
        with pytest.raises(ValueError):
            SeismologyConfig.model_validate(_with_harvest(raw, **bad))
    missing = {k: v for k, v in raw.items() if k != "harvest"}
    with pytest.raises(ValueError, match="harvest"):
        SeismologyConfig.model_validate(missing)
    # The boundary itself is allowed.
    ok = SeismologyConfig.model_validate(_with_harvest(raw, windowS={"P": floor, "S": floor}))
    assert ok.harvest.windowS.S == floor
