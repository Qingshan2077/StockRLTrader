import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config_loader_module():
    module_path = ROOT / "utils" / "config_loader.py"
    module_name = "_config_loader_under_test"

    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = lambda stream: {}

    original_yaml = sys.modules.get("yaml")
    sys.modules["yaml"] = fake_yaml
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if original_yaml is None:
            sys.modules.pop("yaml", None)
        else:
            sys.modules["yaml"] = original_yaml


class DictConfig:
    def __init__(self, data):
        self.data = data

    def get(self, path, default=None):
        current = self.data
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


class ConfigLoaderEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_module = load_config_loader_module()
        cls.ConfigLoader = cls.config_module.ConfigLoader

    def make_loader(self):
        loader = self.ConfigLoader.__new__(self.ConfigLoader)
        loader._config = {
            "system": {"data_dir": "stock_data", "seed": 42},
            "market_data": {"provider": "yfinance", "cache_format": "csv"},
            "data_split": {"train_ratio": 0.6, "val_ratio": 0.2},
            "ticker": "AAPL",
        }
        return loader

    def test_resolve_env_path_supports_compact_and_alias_names(self):
        loader = self.make_loader()
        self.assertEqual(loader._resolve_env_path("system_data_dir"), "system.data_dir")
        self.assertEqual(loader._resolve_env_path("system__data_dir"), "system.data_dir")
        self.assertEqual(loader._resolve_env_path("data_split_train_ratio"), "data_split.train_ratio")
        self.assertEqual(loader._resolve_env_path("data_dir"), "system.data_dir")
        self.assertEqual(loader._resolve_env_path("seed"), "system.seed")
        self.assertEqual(loader._resolve_env_path("ticker"), "ticker")

    def test_apply_env_overrides_updates_nested_paths(self):
        loader = self.make_loader()
        with patch.dict(os.environ, {"QUANT_system_data_dir": "tmpdir",
                                     "QUANT_data_split_train_ratio": "0.75"},
                        clear=True):
            loader._apply_env_overrides()

        self.assertEqual(loader._config["system"]["data_dir"], "tmpdir")
        self.assertAlmostEqual(loader._config["data_split"]["train_ratio"], 0.75)

    def test_apply_env_overrides_supports_aliases(self):
        loader = self.make_loader()
        with patch.dict(os.environ, {"QUANT_DATA_DIR": "custom_stock_data"},
                        clear=True):
            loader._apply_env_overrides()

        self.assertEqual(loader._config["system"]["data_dir"], "custom_stock_data")


class RiskEngineConfigTests(unittest.TestCase):
    def test_flat_top_level_config_is_supported(self):
        from layers.risk.risk_engine import RiskEngine

        engine = RiskEngine(DictConfig({
            "position": {
                "max_gross_exposure": 0.4,
                "max_net_exposure": 0.25,
                "max_single_asset_weight": 0.15,
            },
            "volatility": {"enabled": False},
            "drawdown": {"enabled": False, "circuit_breaker": -0.10},
            "liquidity": {"enabled": False},
            "turnover": {"max_daily_turnover": 0.12},
            "stop_loss": {"threshold": -0.08, "reentry_penalty": 0.4},
            "commission": 0.002,
            "slippage": 0.003,
            "tax": 0.004,
            "market_impact": {"model": "linear", "k": 0.2},
        }))

        self.assertEqual(engine.position.max_single, 0.15)
        self.assertEqual(engine.position.max_net, 0.25)
        self.assertEqual(engine.max_turnover, 0.12)
        self.assertEqual(engine.stop_loss_threshold, -0.08)
        self.assertEqual(engine.cost_model.commission, 0.002)
        self.assertEqual(engine.cost_model.slippage, 0.003)
        self.assertEqual(engine.cost_model.tax, 0.004)
        self.assertEqual(engine.cost_model.impact_model, "linear")
        self.assertEqual(engine.cost_model.k, 0.2)

    def test_nested_risk_and_backtest_sections_still_work(self):
        from layers.risk.risk_engine import RiskEngine

        engine = RiskEngine(DictConfig({
            "risk": {
                "position": {
                    "max_gross_exposure": 0.5,
                    "max_net_exposure": 0.35,
                    "max_single_asset_weight": 0.2,
                },
                "volatility": {"enabled": True, "target_vol": 0.2},
                "drawdown": {"enabled": True, "circuit_breaker": -0.15},
                "liquidity": {"enabled": True, "minimum_volume": 20000},
                "turnover": {"max_daily_turnover": 0.08},
                "stop_loss": {"threshold": -0.06, "reentry_penalty": 0.3},
            },
            "backtest": {
                "commission": 0.0015,
                "slippage": 0.0007,
                "tax": 0.0005,
                "market_impact": {"model": "sqrt", "k": 0.15},
            },
        }))

        self.assertEqual(engine.position.max_gross, 0.5)
        self.assertEqual(engine.drawdown.circuit_breaker, -0.15)
        self.assertEqual(engine.liquidity.minimum_volume, 20000)
        self.assertEqual(engine.stop_loss_threshold, -0.06)
        self.assertEqual(engine.cost_model.commission, 0.0015)
        self.assertEqual(engine.cost_model.slippage, 0.0007)
        self.assertEqual(engine.cost_model.tax, 0.0005)
        self.assertEqual(engine.cost_model.impact_model, "sqrt")
        self.assertEqual(engine.cost_model.k, 0.15)


if __name__ == "__main__":
    unittest.main()
