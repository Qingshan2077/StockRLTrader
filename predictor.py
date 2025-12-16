import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_engine import DataEngine


class ProbabilityPredictor:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.models = {}  # 存储不同时间窗口的模型
        self.feature_cols = [c for c in self.df.columns if
                             c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]

    def create_targets(self, horizons=[1, 5, 10]):
        """
        创建预测目标：未来N天是涨还是跌
        """
        for h in horizons:
            # 逻辑：如果 (未来第h天的收盘价) > (今天的收盘价)，标记为 1，否则为 0
            # shift(-h) 把未来的数据向上平移，让我们在“今天”这一行能看到“未来”
            self.df[f'Target_{h}d'] = (self.df['Close'].shift(-h) > self.df['Close']).astype(int)

        # 去掉最后无法获得未来数据的几行
        self.df.dropna(inplace=True)

    def train(self, horizons=[1, 5, 10]):
        """
        训练三个不同的模型，分别预测1天、5天、10天
        """
        for h in horizons:
            print(f"\n正在训练 [未来 {h} 天] 的预测模型...")
            X = self.df[self.feature_cols]
            y = self.df[f'Target_{h}d']

            # 分割训练集和测试集 (最后100天作为测试)
            split = len(X) - 200
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            # 初始化 XGBoost 分类器
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)

            # 评估
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"模型准确率 (测试集): {acc:.2%}")

            self.models[h] = model

    def predict_future(self, current_data_row):
        """
        输入今天的指标，返回未来涨跌概率
        """
        results = {}
        # 确保输入数据的格式正确
        features = current_data_row[self.feature_cols].values.reshape(1, -1)

        for h, model in self.models.items():
            # predict_proba 返回 [[跌的概率, 涨的概率]]
            prob = model.predict_proba(features)[0][1]
            results[h] = prob
        return results


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 获取数据
    symbol = "NVDA"  # 试试英伟达
    engine = DataEngine(symbol)
    engine.fetch_data()
    full_df = engine.add_technical_indicators()

    # 2. 准备训练
    # 我们不仅要训练，还要留出最后一行数据，模拟“今天”
    # 注意：在create_targets前备份最后一行，因为create_targets会删除最后无法计算未来的行
    latest_data = full_df.iloc[-1:]

    predictor = ProbabilityPredictor(full_df)
    predictor.create_targets()  # 这步之后，df里最后10行会被删掉，因为没有未来数据
    predictor.train()

    # 3. 模拟“今天”的预测
    print(f"\n--- {symbol} 预测报告 (基于最新收盘价: {latest_data['Close'].values[0]:.2f}) ---")
    probs = predictor.predict_future(latest_data)

    for days, prob in probs.items():
        direction = "看涨" if prob > 0.5 else "看跌"
        strength = abs(prob - 0.5) * 2  # 将 0.5-1.0 映射到 0-100% 强度
        print(f"未来 {days} 天: {direction} (概率: {prob:.1%})")