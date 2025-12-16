import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
proxy = 'http://127.0.0.1:7897' # 代理设置，此处修改

os.environ['HTTP_PROXY'] = proxy

os.environ['HTTPS_PROXY'] = proxy

class DataEngine:
    def __init__(self, ticker, start_date="2015-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.data = None

    def fetch_data(self):
        """下载基础数据"""
        print(f"正在下载 {self.ticker} 的数据...")
        self.data = yf.download(
            self.ticker,
            start=self.start_date,
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=False,
        )

        # 确保列名是扁平的（处理yfinance新版本的格式问题）
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        self.data.dropna(inplace=True)
        return self.data

    def add_technical_indicators(self):
        """添加技术指标作为特征"""
        if self.data is None:
            return None

        df = self.data.copy()

        # 1. 趋势指标
        # 简单移动平均线 (SMA)
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        # 距离均线的乖离率 (反映当前价格偏离趋势的程度)
        df['Dist_SMA_10'] = df['Close'] / df['SMA_10'] - 1

        # 2. 动量指标
        # 相对强弱指数 (RSI)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        # MACD
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)  # MACD会返回三列: MACD, Histogram, Signal

        # 3. 波动率指标
        # 布林带宽度 (Bandwidth)
        bb = ta.bbands(df['Close'], length=20)
        print("BBANDS columns:", bb.columns)
        df['BB_Width'] = (bb['BBU_20_2.0_2.0'] - bb['BBL_20_2.0_2.0']) / bb['BBM_20_2.0_2.0']

        # 4. 成交量变化
        df['Vol_Change'] = df['Volume'].pct_change()

        # 清除因为计算指标产生的空值（前几十行）
        df.dropna(inplace=True)
        self.data = df
        return df


# 测试一下
if __name__ == "__main__":
    engine = DataEngine("AAPL")  # 以苹果公司为例
    engine.fetch_data()
    df = engine.add_technical_indicators()
    print(df.tail())