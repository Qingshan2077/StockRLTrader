"""
数据验证和清理工具
用于在训练前检查和修复数据问题
"""

import numpy as np
import pandas as pd
from pathlib import Path


def validate_data(df, ticker=""):
    """
    验证数据质量

    Returns:
        bool: 数据是否有效
        list: 问题列表
    """
    issues = []

    # 1. 检查必需的列
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"缺少必需列: {missing_cols}")

    # 2. 检查 NaN 值
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        issues.append(f"发现 NaN 值: {dict(nan_cols)}")

    # 3. 检查 inf 值
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            issues.append(f"列 '{col}' 包含 {inf_count} 个无穷值")

    # 4. 检查数据长度
    if len(df) < 100:
        issues.append(f"数据长度不足: {len(df)} 条（建议至少 500 条）")

    # 5. 检查数值范围
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                if col_data.std() == 0:
                    issues.append(f"列 '{col}' 标准差为 0（无变化）")

                # 检查极端值
                if col_data.abs().max() > 1e6:
                    issues.append(f"列 '{col}' 包含极端值: max={col_data.max():.2e}")

    # 6. 检查价格逻辑
    if 'High' in df.columns and 'Low' in df.columns:
        invalid_hl = (df['High'] < df['Low']).sum()
        if invalid_hl > 0:
            issues.append(f"发现 {invalid_hl} 条记录的最高价低于最低价")

    if 'Close' in df.columns and 'High' in df.columns and 'Low' in df.columns:
        invalid_close = ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()
        if invalid_close > 0:
            issues.append(f"发现 {invalid_close} 条记录的收盘价超出高低价范围")

    is_valid = len(issues) == 0

    return is_valid, issues


def clean_data(df, verbose=True):
    """
    清理数据

    Args:
        df: 输入 DataFrame
        verbose: 是否打印清理信息

    Returns:
        DataFrame: 清理后的数据
    """
    if verbose:
        print("\n开始清理数据...")
        print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")

    df_clean = df.copy()

    # 1. 处理 inf 值
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df_clean[col]).sum()
        if inf_count > 0:
            if verbose:
                print(f"  - 列 '{col}': 替换 {inf_count} 个无穷值")
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

    # 2. 处理 NaN 值（向前填充）
    nan_before = df_clean.isnull().sum().sum()
    if nan_before > 0:
        if verbose:
            print(f"  - 发现 {nan_before} 个 NaN 值，使用前向填充")
        df_clean = df_clean.fillna(method='ffill')

        # 如果第一行还有 NaN，用后向填充
        df_clean = df_clean.fillna(method='bfill')

        # 如果还有 NaN，用 0 填充
        df_clean = df_clean.fillna(0)

    # 3. 限制数值范围（避免极端值）
    for col in numeric_cols:
        if col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            # 价格列不做限制
            continue

        # 对其他列使用百分位数限制
        if len(df_clean[col]) > 0:
            q1 = df_clean[col].quantile(0.01)
            q99 = df_clean[col].quantile(0.99)

            outliers = ((df_clean[col] < q1) | (df_clean[col] > q99)).sum()
            if outliers > 0:
                if verbose:
                    print(f"  - 列 '{col}': 限制 {outliers} 个异常值到 [{q1:.2f}, {q99:.2f}]")
                df_clean[col] = df_clean[col].clip(q1, q99)

    # 4. 删除重复的索引
    if df_clean.index.duplicated().any():
        dup_count = df_clean.index.duplicated().sum()
        if verbose:
            print(f"  - 删除 {dup_count} 个重复索引")
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]

    # 5. 按日期排序
    if not df_clean.index.is_monotonic_increasing:
        if verbose:
            print(f"  - 按日期排序")
        df_clean = df_clean.sort_index()

    if verbose:
        print(f"清理完成: {len(df_clean)} 行, {len(df_clean.columns)} 列\n")

    return df_clean


def prepare_for_training(df, min_length=100, verbose=True):
    """
    准备用于训练的数据

    Args:
        df: 输入 DataFrame
        min_length: 最小数据长度
        verbose: 是否打印信息

    Returns:
        DataFrame: 准备好的数据
        bool: 是否可用于训练
    """
    if verbose:
        print("\n" + "=" * 60)
        print("准备训练数据")
        print("=" * 60)

    # 1. 清理数据
    df_clean = clean_data(df, verbose=verbose)

    # 2. 验证数据
    is_valid, issues = validate_data(df_clean)

    if verbose:
        if is_valid:
            print("✅ 数据验证通过")
        else:
            print("⚠️ 数据存在以下问题:")
            for issue in issues:
                print(f"  - {issue}")

    # 3. 检查长度
    if len(df_clean) < min_length:
        if verbose:
            print(f"❌ 数据长度不足: {len(df_clean)} < {min_length}")
        return df_clean, False

    # 4. 最终检查
    feature_cols = [col for col in df_clean.columns if col not in
                    ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    if len(feature_cols) == 0:
        if verbose:
            print("❌ 未找到技术指标，请先运行 add_technical_indicators()")
        return df_clean, False

    if verbose:
        print(f"\n✅ 数据准备完成")
        print(f"  - 数据长度: {len(df_clean)}")
        print(f"  - 特征数量: {len(feature_cols)}")
        print(f"  - 日期范围: {df_clean.index[0]} ~ {df_clean.index[-1]}")
        print("=" * 60 + "\n")

    return df_clean, True


def split_train_test(df, train_ratio=0.8, verbose=True):
    """
    分割训练集和测试集

    Args:
        df: 输入 DataFrame
        train_ratio: 训练集比例
        verbose: 是否打印信息

    Returns:
        tuple: (train_data, test_data)
    """
    split_idx = int(len(df) * train_ratio)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()

    if verbose:
        print(f"\n数据分割:")
        print(f"  训练集: {len(train_data)} 条 ({train_data.index[0]} ~ {train_data.index[-1]})")
        print(f"  测试集: {len(test_data)} 条 ({test_data.index[0]} ~ {test_data.index[-1]})")

    return train_data, test_data


# 测试代码
if __name__ == "__main__":
    from improved_data_engine import DataEngine

    print("测试数据验证工具\n")

    # 加载数据
    engine = DataEngine("AAPL")
    df = engine.load_processed_data()

    if df is None:
        print("请先下载数据")
        exit(1)

    # 准备数据
    df_clean, is_ready = prepare_for_training(df, min_length=100)

    if is_ready:
        print("\n✅ 数据可用于训练！")

        # 分割数据
        train_data, test_data = split_train_test(df_clean, train_ratio=0.8)
    else:
        print("\n❌ 数据不适合训练，请检查数据质量")