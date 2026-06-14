"""
Transformer 序列模型 (positional encoding + multi-head attention)
"""
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from layers.signals.base_model import BaseAlphaModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])


class TransformerSignalModel(BaseAlphaModel):
    """Transformer Alpha 模型"""

    def __init__(self, name: str = "transformer", config: dict = None):
        super().__init__(name=name, config=config)

    def _to_sequences(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        n_samples = X.shape[0] - seq_len + 1
        n_features = X.shape[1]
        seqs = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        for i in range(n_samples):
            seqs[i] = X[i:i + seq_len]
        return seqs

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        seq_len = self.config.get("sequence_length", 20)
        d_model = self.config.get("d_model", 64)
        nhead = self.config.get("nhead", 4)
        num_layers = self.config.get("num_layers", 2)
        dim_feedforward = self.config.get("dim_feedforward", 128)
        dropout = self.config.get("dropout", 0.3)
        lr = self.config.get("learning_rate", 0.001)
        batch_size = self.config.get("batch_size", 64)
        epochs = self.config.get("epochs", 100)
        patience = self.config.get("patience", 15)

        X_seq = self._to_sequences(X_train, seq_len)
        y_seq = np.ravel(y_train)[seq_len - 1:]

        input_dim = X_train.shape[1]
        self.model = TransformerModel(input_dim, d_model, nhead, num_layers,
                                       dim_feedforward, dropout)
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

            val_loss = float("inf")
            if X_val is not None and len(X_val) > seq_len:
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
        return np.concatenate([np.full(seq_len - 1, preds[0]), preds])
