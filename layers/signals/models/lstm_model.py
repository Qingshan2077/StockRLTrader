"""
LSTM 序列模型
"""
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from layers.signals.base_model import BaseAlphaModel


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMSignalModel(BaseAlphaModel):
    """LSTM Alpha 模型"""

    def __init__(self, name: str = "lstm", config: dict = None):
        super().__init__(name=name, config=config)

    def _to_sequences(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        """将 2D 数据转为 LSTM 需要的 3D 序列"""
        n_samples = X.shape[0] - seq_len + 1
        n_features = X.shape[1]
        seqs = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        for i in range(n_samples):
            seqs[i] = X[i:i + seq_len]
        return seqs

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        seq_len = self.config.get("sequence_length", 20)
        hidden_size = self.config.get("hidden_size", 64)
        num_layers = self.config.get("num_layers", 2)
        dropout = self.config.get("dropout", 0.3)
        lr = self.config.get("learning_rate", 0.001)
        batch_size = self.config.get("batch_size", 64)
        epochs = self.config.get("epochs", 100)
        patience = self.config.get("patience", 15)

        X_seq = self._to_sequences(X_train, seq_len)
        y_seq = np.ravel(y_train)[seq_len - 1:]

        input_dim = X_train.shape[1]
        self.model = LSTMModel(input_dim, hidden_size, num_layers, dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32).reshape(-1, 1)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(ds)
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = float("inf")
            if X_val is not None and y_val is not None and len(X_val) > seq_len:
                X_v_seq = self._to_sequences(X_val, seq_len)
                y_v_seq = np.ravel(y_val)[seq_len - 1:]
                with torch.no_grad():
                    self.model.eval()
                    preds = self.model(torch.tensor(X_v_seq, dtype=torch.float32))
                    val_loss = float(criterion(
                        preds, torch.tensor(y_v_seq, dtype=torch.float32).reshape(-1, 1)))
                history["val_loss"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        self._fitted = True

        metrics = {"train_loss": train_loss}
        if X_val is not None and len(X_val) > seq_len:
            X_v_seq = self._to_sequences(X_val, seq_len)
            y_v_seq = np.ravel(y_val)[seq_len - 1:]
            with torch.no_grad():
                self.model.eval()
                val_preds = self.model(torch.tensor(X_v_seq, dtype=torch.float32)).numpy().ravel()
            metrics["val_loss"] = float(best_loss)
            metrics["val_r2"] = float(r2_score(y_v_seq, val_preds))
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_v_seq, val_preds)))
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        seq_len = self.config.get("sequence_length", 20)
        if len(X) < seq_len:
            return np.zeros(len(X))
        X_seq = self._to_sequences(X, seq_len)
        with torch.no_grad():
            self.model.eval()
            preds = self.model(torch.tensor(X_seq, dtype=torch.float32)).numpy().ravel()
        # 前面 seq_len-1 个点无法预测，用第一个预测值填充
        return np.concatenate([np.full(seq_len - 1, preds[0]), preds])
