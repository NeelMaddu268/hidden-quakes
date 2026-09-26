import pytest
from pydantic import ValidationError

from hq.config.signal import SignalConfig


@pytest.mark.smoke
def test_signal_yaml_parses(signal_cfg: SignalConfig) -> None:
    assert isinstance(signal_cfg, SignalConfig)


@pytest.mark.smoke
def test_unknown_key_is_an_error(raw_signal_yaml: dict) -> None:
    raw = dict(raw_signal_yaml)
    raw["notAKnob"] = 1
    with pytest.raises(ValidationError):
        SignalConfig.model_validate(raw)
