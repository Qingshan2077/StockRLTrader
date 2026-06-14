import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

proxy = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy


class DataEngine:
    def __init__(self, ticker, data_dir="stock_data", start_date="2015-01-01"):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # 数据文件路径
        self.raw_data_path = self.data_dir / f"{self.ticker}_raw.csv"
        self.processed_data_path = self.data_dir / f"{self.ticker}_processed.csv"
        self.meta_path = self.data_dir / f"{self.ticker}_meta.json"

        self.data = None
        self.processed_data = None

    def _load_metadata(self):
        """加载元数据（记录上次更新时间等信息）"""
        if self.meta_path.exists():
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata):
        """保存元数据"""
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def fetch_data(self, force_update=False):
        """
        下载或更新股票数据
        force_update: 强制重新下载全部数据
        """
        metadata = self._load_metadata()

        # 判断是否需要增量更新
        if self.raw_data_path.exists() and not force_update:
            print(f"发现 {self.ticker} 的本地数据，准备增量更新...")
            existing_data = pd.read_csv(self.raw_data_path, index_col=0, parse_dates=True)

            # 从最后一天的下一天开始更新
            last_date = existing_data.index[-1]
            update_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            today = datetime.now().strftime('%Y-%m-%d')

            if update_start >= today:
                print(f"{self.ticker} 数据已是最新，无需更新。")
                self.data = existing_data
                return self.data

            print(f"从 {update_start} 更新到 {today}...")
            new_data = yf.download(
                self.ticker,
                start=update_start,
                interval="1d",
                progress=False,
                threads=False,
                auto_adjust=False,
            )

            if not new_data.empty:
                # 处理列名
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.get_level_values(0)

                # 合并数据
                self.data = pd.concat([existing_data, new_data])
                self.data = self.data[~self.data.index.duplicated(keep='last')]
                print(f"成功更新 {len(new_data)} 条新数据")
            else:
                print("没有新数据")
                self.data = existing_data
        else:
            # 全量下载
            print(f"正在下载 {self.ticker} 的完整数据（从 {self.start_date}）...")
            self.data = yf.download(
                self.ticker,
                start=self.start_date,
                interval="1d",
                progress=False,
                threads=False,
                auto_adjust=False,
            )

            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.get_level_values(0)

            print(f"成功下载 {len(self.data)} 条数据")

        self.data.dropna(inplace=True)

        # 保存原始数据
        self.data.to_csv(self.raw_data_path)

        # 更新元数据
        metadata['ticker'] = self.ticker
        metadata['custom_name'] = metadata.get('custom_name', '')  # 保留自定义名称
        metadata['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata['data_points'] = len(self.data)
        metadata['date_range'] = {
            'start': str(self.data.index[0]),
            'end': str(self.data.index[-1])
        }
        self._save_metadata(metadata)

        return self.data

    def add_technical_indicators(self):
        """添加技术指标"""
        if self.data is None:
            print("请先调用 fetch_data() 获取数据")
            return None

        df = self.data.copy()

        # 1. 趋势指标
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['Dist_SMA_10'] = df['Close'] / df['SMA_10'] - 1
        df['Dist_SMA_50'] = df['Close'] / df['SMA_50'] - 1

        # EMA 指数移动平均
        df['EMA_12'] = ta.ema(df['Close'], length=12)
        df['EMA_26'] = ta.ema(df['Close'], length=26)

        # 2. 动量指标
        df['RSI'] = ta.rsi(df['Close'], length=14)

        # MACD
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)

        # 3. 波动率指标
        bb = ta.bbands(df['Close'], length=20)
        df['BB_Width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
        df['BB_Position'] = (df['Close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])

        # ATR (平均真实波幅)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        # 4. 成交量指标
        df['Vol_Change'] = df['Volume'].pct_change()
        df['Vol_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA_20']

        # 5. 价格变化
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_20d'] = df['Close'].pct_change(20)

        # 6. 趋势强度
        df['Trend_Strength'] = (df['SMA_10'] - df['SMA_50']) / df['SMA_50']

        df.dropna(inplace=True)
        self.processed_data = df

        # 保存处理后的数据
        self.processed_data.to_csv(self.processed_data_path)
        print(f"技术指标已计算并保存到 {self.processed_data_path}")

        return df

    def load_processed_data(self):
        """从本地加载已处理的数据"""
        if self.processed_data_path.exists():
            self.processed_data = pd.read_csv(self.processed_data_path, index_col=0, parse_dates=True)
            print(f"已加载 {self.ticker} 的处理数据（{len(self.processed_data)} 条）")
            return self.processed_data
        else:
            print(f"未找到 {self.ticker} 的处理数据，请先运行完整流程")
            return None

    def get_latest_data(self):
        """获取最新一天的数据"""
        if self.processed_data is not None:
            return self.processed_data.iloc[-1:]
        return None

    def get_info(self):
        """获取股票信息摘要"""
        metadata = self._load_metadata()
        if metadata:
            print(f"\n{'=' * 50}")
            print(f"股票代码: {metadata.get('ticker', 'N/A')}")
            custom_name = metadata.get('custom_name', '')
            if custom_name:
                print(f"股票名称: {custom_name}")
            print(f"最后更新: {metadata.get('last_update', 'N/A')}")
            print(f"数据点数: {metadata.get('data_points', 'N/A')}")
            print(
                f"日期范围: {metadata.get('date_range', {}).get('start', 'N/A')} 至 {metadata.get('date_range', {}).get('end', 'N/A')}")
            print(f"{'=' * 50}\n")
        return metadata

    def set_custom_name(self, custom_name):
        """设置自定义股票名称"""
        metadata = self._load_metadata()
        metadata['custom_name'] = custom_name
        metadata['ticker'] = self.ticker
        self._save_metadata(metadata)
        print(f"✅ 已将 {self.ticker} 的名称设置为: {custom_name}")

    def get_custom_name(self):
        """获取自定义股票名称"""
        metadata = self._load_metadata()
        return metadata.get('custom_name', '')


class BatchDataEngine:
    """批量处理多个股票的数据引擎"""

    def __init__(self, data_dir="stock_data", start_date="2015-01-01"):
        self.data_dir = data_dir
        self.start_date = start_date
        self.engines = {}

    def process_ticker(self, ticker, force_update=False):
        """处理单个股票"""
        try:
            print(f"\n{'#' * 60}")
            print(f"处理股票: {ticker}")
            print(f"{'#' * 60}")

            engine = DataEngine(ticker, self.data_dir, self.start_date)
            engine.fetch_data(force_update=force_update)
            engine.add_technical_indicators()
            self.engines[ticker] = engine

            print(f"✓ {ticker} 处理完成")
            return True
        except Exception as e:
            print(f"✗ {ticker} 处理失败: {str(e)}")
            return False

    def process_batch(self, tickers, force_update=False):
        """批量处理多个股票"""
        success_count = 0
        fail_count = 0

        print(f"\n开始批量处理 {len(tickers)} 个股票...")
        print(f"数据保存目录: {self.data_dir}\n")

        for ticker in tickers:
            if self.process_ticker(ticker, force_update):
                success_count += 1
            else:
                fail_count += 1

        print(f"\n{'=' * 60}")
        print(f"批量处理完成！")
        print(f"成功: {success_count} | 失败: {fail_count}")
        print(f"{'=' * 60}\n")

        return self.engines

    def list_available_data(self):
        """列出所有可用的本地数据"""
        data_path = Path(self.data_dir)
        if not data_path.exists():
            print("数据目录不存在")
            return []

        csv_files = list(data_path.glob("*_processed.csv"))
        tickers = [f.stem.replace("_processed", "") for f in csv_files]

        print(f"\n本地可用数据 ({len(tickers)} 个):")
        for ticker in sorted(tickers):
            print(f"  - {ticker}")

        return tickers


# 测试代码
if __name__ == "__main__":
    # 单个股票测试
    print("=== 单个股票测试 ===")
    engine = DataEngine("AAPL")
    engine.fetch_data()
    df = engine.add_technical_indicators()
    engine.get_info()
    print(df.tail())

    # 批量处理测试
    print("\n\n=== 批量处理测试 ===")
    batch = BatchDataEngine()
    tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]
    batch.process_batch(tickers)

    # 列出可用数据
    batch.list_available_data()