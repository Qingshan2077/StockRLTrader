"""
截面因子处理管线 — 将单股票特征转为截面可用因子

每日期横截面处理流程:
  1. Winsorization (MAD 或 百分位截断)
  2. 行业中性化 (减去行业均值)
  3. 市值中性化 (回归取残差)
  4. Z-Score 标准化 (跨截面)
  5. (可选) 正交化去冗余

输出: 与输入相同形状的 DataFrame, index=(date, code)
"""

import numpy as np
import pandas as pd
from typing import Optional


def winsorize_mad(
    series: pd.Series,
    n_mad: float = 5.0,
) -> pd.Series:
    """MAD winsorization: 将超过 median ± n*MAD 的值截断

    Args:
        series: 因子值序列
        n_mad: MAD 倍数

    Returns:
        截断后的序列
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad < 1e-12:
        return series
    upper = median + n_mad * mad
    lower = median - n_mad * mad
    return series.clip(lower, upper)


def winsorize_percentile(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """百分位 winsorization

    Args:
        series: 因子值序列
        lower: 下尾百分位 (0.01 = 1%)
        upper: 上尾百分位 (0.99 = 99%)

    Returns:
        截断后的序列
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def zscore_normalize(series: pd.Series, cap: float = 3.0) -> pd.Series:
    """截面 Z-Score 标准化

    (x - mean) / std, 再截断到 ±cap

    Args:
        series: 因子值序列
        cap: 截断阈值

    Returns:
        标准化后的序列
    """
    mean = series.mean()
    std = series.std(ddof=1)
    if std < 1e-12:
        return pd.Series(0.0, index=series.index)
    z = (series - mean) / std
    return z.clip(-cap, cap)


def sector_neutralize(
    factor: pd.Series,
    sectors: pd.Series,
) -> pd.Series:
    """行业中性化: 减去行业均值

    Args:
        factor: 因子值, index=股票代码
        sectors: 行业标签, index=股票代码

    Returns:
        中性化后的因子值
    """
    result = factor.copy().astype(np.float64)
    for sec in sectors.dropna().unique():
        mask = sectors == sec
        sub = result[mask]
        result[mask] = sub - sub.mean()
    return result


def market_cap_neutralize(
    factor: pd.Series,
    market_cap: pd.Series,
) -> pd.Series:
    """市值中性化: 对 log(市值) 回归, 取残差

    Args:
        factor: 因子值, index=股票代码
        market_cap: 市值, index=股票代码

    Returns:
        中性化后的因子值
    """
    from sklearn.linear_model import LinearRegression

    valid = factor.notna() & market_cap.notna() & (market_cap > 0)
    if valid.sum() < 20:
        return factor

    x = np.log(market_cap[valid].values.astype(np.float64)).reshape(-1, 1)
    y = factor[valid].values.astype(np.float64)
    model = LinearRegression().fit(x, y)
    resid = y - model.predict(x)
    result = factor.copy()
    result[valid] = resid
    return result


def process_panel_factors(
    panel: pd.DataFrame,
    factor_cols: list[str],
    sector_col: str = "sector",
    market_cap_col: Optional[str] = None,
    method: str = "mad",
    n_mad: float = 5.0,
    zscore_cap: float = 3.0,
    do_sector_neutral: bool = True,
    do_market_neutral: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """对面板数据进行每日横截面因子处理

    Args:
        panel: MultiIndex DataFrame (date, code) × [features, sector, market_cap]
        factor_cols: 需要处理的因子列名列表
        sector_col: 行业列名
        market_cap_col: 市值列名 (为 None 则跳过市值中性化)
        method: 'mad' 或 'percentile'
        n_mad: MAD winsorization 倍数
        zscore_cap: Z-Score 截断
        do_sector_neutral: 是否做行业中性化
        do_market_neutral: 是否做市值中性化
        verbose: 是否打印进度

    Returns:
        处理后的 panel, 因子列被替换为标准化的值
        新增列: *_raw (原始因子值, 处理后加回用于分析)
    """
    result = panel.copy()
    dates = panel.index.get_level_values("date").unique()
    n_dates = len(dates)

    # 保存原始值
    for col in factor_cols:
        if col in result.columns:
            result[f"{col}_raw"] = result[col]

    for i, dt in enumerate(dates):
        mask = result.index.get_level_values("date") == dt
        chunk = result.loc[dt].copy()

        for col in factor_cols:
            if col not in chunk.columns:
                continue

            series = chunk[col]

            # 1. Winsorization
            if method == "mad":
                series = winsorize_mad(series, n_mad)
            else:
                series = winsorize_percentile(series)

            # 2. 行业中性化
            if do_sector_neutral and sector_col in chunk.columns:
                series = sector_neutralize(series, chunk[sector_col])

            # 3. 市值中性化
            if do_market_neutral and market_cap_col and market_cap_col in chunk.columns:
                series = market_cap_neutralize(series, chunk[market_cap_col])

            # 4. Z-Score
            series = zscore_normalize(series, zscore_cap)

            result.loc[dt, col] = series.values

        if verbose and (i + 1) % 100 == 0:
            print(f"  [截面处理] {i+1}/{n_dates} 日期完成")

    if verbose:
        print(f"  [截面处理] 完成: {n_dates} 个日期, {len(factor_cols)} 个因子")

    return result


def compute_cross_sectional_rank(
    panel: pd.DataFrame,
    score_col: str = "prediction",
    ascending: bool = False,
) -> pd.DataFrame:
    """计算每日横截面排名 (百分位)

    Args:
        panel: MultiIndex DataFrame (date, code) × cols
        score_col: 排名依据的列
        ascending: True=越小越好, False=越大越好

    Returns:
        panel 新增列 rank_pct ∈ [0, 1]
    """
    result = panel.copy()
    dates = result.index.get_level_values("date").unique()

    for dt in dates:
        mask = result.index.get_level_values("date") == dt
        sub = result.loc[dt].copy()
        scores = sub[score_col]
        rnk = scores.rank(method="average", ascending=ascending)
        result.loc[dt, "rank_pct"] = (rnk - 1) / max(len(rnk) - 1, 1)
        # 确保 0~1 范围
        result.loc[dt, "rank_pct"] = result.loc[dt, "rank_pct"].clip(0, 1)

    return result
