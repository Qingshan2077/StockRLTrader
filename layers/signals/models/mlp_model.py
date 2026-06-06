"""
PyTorch MLP 信号模型 — v3 接口 (继承 BaseAlphaModel)

架构: input_dim → hidden[0] → ... → hidden[-1] → 1
每层: Linear → BatchNorm1d → ReLU → Dropout
训练: MSE loss + Adam + ReduceLROnPlateau + Gradient Clipping + Early Stopping
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from sklearn.metrics import r2_score, mean_squared_error
from layers.signals.base_model import BaseAlphaModel


class _MLPNet(nn.Module):
    """内部 MLP 网络"""

    def __init__(self, input_dim: int, hidden_dims: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPSignalModel(BaseAlphaModel):
    """MLP Alpha 模型 — 实现 BaseAlphaModel 统一接口"""

    def __init__(self, name: str = "mlp", config: dict = None):
        super().__init__(name=name, config=config)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        input_dim = X_train.shape[1]
        hidden_dims = self.config.get("hidden_dims", [128, 64])
        dropout = self.config.get("dropout", 0.3)
        lr = self.config.get("learning_rate", 0.001)
        batch_size = self.config.get("batch_size", 64)
        epochs = self.config.get("epochs", 100)
        patience = self.config.get("patience", 15)
        seed = self.config.get("seed", 42)

        torch.manual_seed(seed)

        self.model = _MLPNet(input_dim, hidden_dims, dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(np.ravel(y_train), dtype=torch.float32).reshape(-1, 1)
        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            Xv_t = torch.tensor(X_val, dtype=torch.float32)
            yv_t = torch.tensor(np.ravel(y_val), dtype=torch.float32).reshape(-1, 1)
            val_ds = TensorDataset(Xv_t, yv_t)
            val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)
            history["train_loss"].append(train_loss)

            val_loss = float("inf")
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        val_loss += criterion(self.model(xb), yb).item() * len(xb)
                val_loss /= len(X_val)
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._fitted = True

        metrics = {"train_loss": train_loss}
        if val_loader is not None and len(X_val) > 0:
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(
                    torch.tensor(X_val, dtype=torch.float32)).numpy().ravel()
            metrics["val_r2"] = float(r2_score(np.ravel(y_val), val_preds))
            metrics["val_rmse"] = float(
                np.sqrt(mean_squared_error(np.ravel(y_val), val_preds)))
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(
                torch.tensor(X, dtype=torch.float32)).numpy().ravel()

    def save(self, path: str) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "name": self.name,
        }, path)

    def load(self, path: str) -> None:
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        data = torch.load(path, map_location="cpu",
                          weights_only=False)
        self.config = data.get("config", {})
        self.name = data.get("name", "mlp")
        # 从 config 重建模型架构
        if self.model is None:
            from .mlp_model import _MLPNet
            self.model = _MLPNet(
                input_dim=data.get("input_dim", 52),
                hidden_dims=self.config.get("hidden_dims", [128, 64]),
                dropout=self.config.get("dropout", 0.3),
            )
        self.model.load_state_dict(data["model_state"])
        self._fitted = True
