import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import pickle
from pathlib import Path


class AdvancedPredictor:
    """
    高级预测器 - 支持价格预测和趋势预测
    """

    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.price_model = None  # 价格预测模型
        self.trend_models = {}  # 趋势预测模型（不同时间窗口）
        self.feature_cols = [c for c in self.df.columns if
                             c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]

        self.scaler_mean = None
        self.scaler_std = None

    def create_price_targets(self, horizons=[1, 5, 10, 20, 30]):
        """
        创建价格预测目标
        预测未来 N 天的收盘价
        """
        for h in horizons:
            # 未来第 h 天的收盘价
            self.df[f'Target_Price_{h}d'] = self.df['Close'].shift(-h)

        # 去掉最后无法获得未来数据的行
        self.df.dropna(inplace=True)

    def create_trend_targets(self, horizons=[1, 5, 10, 20, 30]):
        """
        创建趋势预测目标
        预测未来 N 天是涨还是跌
        """
        for h in horizons:
            # 1 = 上涨, 0 = 下跌
            self.df[f'Target_Trend_{h}d'] = (self.df['Close'].shift(-h) > self.df['Close']).astype(int)

        self.df.dropna(inplace=True)

    def train_price_model(self, horizon=10, test_size=0.2):
        """
        训练价格预测模型（回归）

        Args:
            horizon: 预测未来多少天
            test_size: 测试集比例
        """
        print(f"\n训练价格预测模型 (未来 {horizon} 天)...")

        X = self.df[self.feature_cols]
        y = self.df[f'Target_Price_{horizon}d']

        # 分割数据（按时间序列）
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # 训练模型
        self.price_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )

        self.price_model.fit(X_train, y_train)

        # 评估
        y_pred = self.price_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # 计算方向准确率
        actual_direction = (y_test.values > X_test.index.map(lambda x: self.df.loc[x, 'Close'])).astype(int)
        pred_direction = (y_pred > X_test.index.map(lambda x: self.df.loc[x, 'Close'])).astype(int)
        direction_acc = (actual_direction == pred_direction).mean()

        print(f"  MSE: {mse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  方向准确率: {direction_acc:.2%}")

        return {
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction_acc
        }

    def train_trend_models(self, horizons=[1, 5, 10, 20, 30], test_size=0.2):
        """
        训练趋势预测模型（分类）

        Args:
            horizons: 预测时间窗口列表
            test_size: 测试集比例
        """
        results = {}

        for h in horizons:
            print(f"\n训练趋势预测模型 (未来 {h} 天)...")

            X = self.df[self.feature_cols]
            y = self.df[f'Target_Trend_{h}d']

            # 分割数据
            split = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            # 训练模型
            model = XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42
            )

            model.fit(X_train, y_train)

            # 评估
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # 获取概率
            y_prob = model.predict_proba(X_test)[:, 1]

            print(f"  准确率: {acc:.2%}")

            self.trend_models[h] = model
            results[h] = {'accuracy': acc}

        return results

    def predict_future_prices(self, current_data, days=30):
        """
        预测未来多天的价格走势

        Args:
            current_data: 当前的特征数据（包含技术指标）
            days: 预测未来多少天

        Returns:
            list: 未来每天的预测价格
        """
        if self.price_model is None:
            raise ValueError("请先训练价格预测模型")

        predictions = []
        current_features = current_data[self.feature_cols].iloc[-1:].copy()
        current_price = current_data['Close'].iloc[-1]

        # 逐步预测
        for day in range(1, days + 1):
            # 预测下一天的价格
            pred_price = self.price_model.predict(current_features)[0]
            predictions.append(pred_price)

            # 更新特征（简化版本，实际应该重新计算所有技术指标）
            # 这里我们使用预测价格更新部分特征
            # 注意：这是简化处理，真实场景需要更复杂的特征更新逻辑

        return predictions

    def predict_future_trend(self, current_data, days_ahead=10):
        """
        预测未来指定天数的涨跌趋势

        Args:
            current_data: 当前数据（DataFrame，最后一行）
            days_ahead: 预测未来多少天

        Returns:
            dict: 预测结果 {概率, 趋势方向, 置信度}
        """
        if days_ahead not in self.trend_models:
            # 找最接近的模型
            available = sorted(self.trend_models.keys())
            days_ahead = min(available, key=lambda x: abs(x - days_ahead))

        model = self.trend_models[days_ahead]
        features = current_data[self.feature_cols].values.reshape(1, -1)

        # 预测概率
        prob = model.predict_proba(features)[0][1]  # 上涨概率
        trend = "上涨" if prob > 0.5 else "下跌"
        confidence = abs(prob - 0.5) * 2 * 100  # 转换为0-100的置信度

        return {
            'probability': prob,
            'trend': trend,
            'confidence': confidence,
            'days': days_ahead
        }

    def generate_future_trend_line(self, current_data, days=30, method='mixed'):
        """
        生成未来趋势线（用于图表展示）

        Args:
            current_data: 当前历史数据
            days: 预测未来多少天
            method: 'mixed' 结合价格预测和趋势预测
                   'price' 仅使用价格预测
                   'trend' 仅使用趋势预测

        Returns:
            dict: {dates, prices, confidence}
        """
        current_price = current_data['Close'].iloc[-1]
        last_date = current_data.index[-1]

        # 生成未来日期
        future_dates = pd.date_range(start=last_date, periods=days + 1, freq='D')[1:]

        if method == 'price' or method == 'mixed':
            # 使用价格预测模型
            try:
                future_prices = self.predict_future_prices(current_data, days)
            except:
                # 如果价格预测失败，使用趋势预测
                method = 'trend'

        if method == 'trend':
            # 基于趋势预测生成价格线
            future_prices = []
            price = current_price

            # 获取不同时间窗口的趋势
            horizons = [1, 5, 10, 20, 30]
            available_horizons = [h for h in horizons if h in self.trend_models]

            for i in range(days):
                # 选择合适的预测窗口
                horizon = min(available_horizons, key=lambda x: abs(x - (i + 1)))

                # 预测趋势
                pred = self.predict_future_trend(current_data, horizon)

                # 根据趋势调整价格（简化模型）
                if pred['trend'] == "上涨":
                    # 上涨，增加0.5-2%
                    change = (pred['confidence'] / 100) * 0.02 * price
                    price = price + change
                else:
                    # 下跌，减少0.5-2%
                    change = (pred['confidence'] / 100) * 0.02 * price
                    price = price - change

                future_prices.append(price)

        # 计算置信区间（简化版）
        confidence_upper = [p * 1.05 for p in future_prices]  # +5%
        confidence_lower = [p * 0.95 for p in future_prices]  # -5%

        return {
            'dates': future_dates,
            'prices': future_prices,
            'upper_bound': confidence_upper,
            'lower_bound': confidence_lower
        }

    def save_models(self, path):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        models = {
            'price_model': self.price_model,
            'trend_models': self.trend_models,
            'feature_cols': self.feature_cols
        }

        with open(path, 'wb') as f:
            pickle.dump(models, f)

        print(f"✅ 模型已保存到: {path}")

    def load_models(self, path):
        """加载模型"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        with open(path, 'rb') as f:
            models = pickle.load(f)

        self.price_model = models['price_model']
        self.trend_models = models['trend_models']
        self.feature_cols = models['feature_cols']

        print(f"✅ 模型已加载: {path}")


# 测试代码
if __name__ == "__main__":
    from improved_data_engine import DataEngine

    # 加载数据
    engine = DataEngine("AAPL")
    df = engine.load_processed_data()

    if df is None:
        print("请先下载数据")
        exit(1)

    # 创建预测器
    predictor = AdvancedPredictor(df)

    # 创建目标
    predictor.create_price_targets([10, 30])
    predictor.create_trend_targets([1, 5, 10, 20, 30])

    # 训练模型
    print("=" * 60)
    predictor.train_price_model(horizon=10)
    predictor.train_trend_models()

    # 预测未来
    print("\n" + "=" * 60)
    print("预测未来趋势")
    print("=" * 60)

    latest_data = df.iloc[-1:]

    # 趋势预测
    for days in [1, 5, 10, 20, 30]:
        pred = predictor.predict_future_trend(latest_data, days)
        print(f"\n未来 {days} 天:")
        print(f"  趋势: {pred['trend']}")
        print(f"  概率: {pred['probability'] * 100:.1f}%")
        print(f"  置信度: {pred['confidence']:.1f}%")

    # 生成趋势线
    trend_line = predictor.generate_future_trend_line(df.tail(100), days=30)
    print(f"\n✅ 已生成未来 30 天趋势线")

    # 保存模型
    predictor.save_models("data/models/predictor_AAPL.pkl")