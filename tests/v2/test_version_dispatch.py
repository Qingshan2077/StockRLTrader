import pytest


def test_environment_version_dispatch_preserves_v1_and_rejects_unknown():
    from stockrl.env import TradingEnv
    from stockrl.env_v2 import TradingEnvV2,environment_class
    assert environment_class(1) is TradingEnv
    assert environment_class(2) is TradingEnvV2
    with pytest.raises(ValueError,match='PROTOCOL_INCOMPATIBLE'):
        environment_class(3)
