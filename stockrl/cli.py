"""Command-line workflows for local StockRL experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


def _add_experiment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--algorithm", choices=("PPO", "SAC"), default="PPO",
                        help="强化学习算法（默认：PPO）")
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="每个随机种子的训练步数（默认：10000）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="一个或多个随机种子（默认：42）")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments"),
                        help="实验产物根目录（默认：outputs/experiments）")
    parser.add_argument("--train-ratio", type=float, default=0.6,
                        help="训练收益区间占比（默认：0.6）")
    parser.add_argument("--validation-ratio", type=float, default=0.2,
                        help="验证收益区间占比（默认：0.2）")
    parser.add_argument("--episode-length", type=int, default=126,
                        help="训练回合最大交易日数（默认：126）")
    parser.add_argument("--initial-cash", type=float, default=10_000.0,
                        help="初始现金（默认：10000）")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="单边手续费率（默认：0.001）")
    parser.add_argument("--slippage", type=float, default=0.0005,
                        help="方向性滑点率（默认：0.0005）")
    parser.add_argument("--sell-tax", type=float, default=0.0,
                        help="卖出税率（默认：0）")
    parser.add_argument("--min-commission", type=float, default=0.0,
                        help="每笔最低手续费（默认：0）")
    parser.add_argument("--lot-size", type=int, default=1,
                        help="最小成交股数单位（默认：1）")
    parser.add_argument("--max-participation", type=float, default=0.01,
                        help="最大历史成交量参与率（默认：0.01）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockrl",
        description="单资产、只做多的强化学习交易研究工具（本地实验，不连接券商）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="用明确标注的合成行情运行技术演示")
    demo.add_argument("--rows", type=int, default=756, help="演示数据行数（默认：756）")
    demo.add_argument("--data-seed", type=int, default=42,
                      help="合成行情随机种子（默认：42）")
    _add_experiment_options(demo)

    train = subparsers.add_parser("train", help="用 CSV 或 Yahoo 日线行情训练")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="含 Date/Open/High/Low/Close/Volume 的 CSV")
    source.add_argument("--ticker", help="从 Yahoo 下载的证券代码，例如 AAPL")
    train.add_argument("--start", help="可选起始日期 YYYY-MM-DD")
    train.add_argument("--end", help="可选结束日期 YYYY-MM-DD")
    _add_experiment_options(train)

    evaluate = subparsers.add_parser("evaluate", help="原样重放已保存实验的测试区间")
    evaluate.add_argument("run_dir", type=Path, help="含 summary.json 的实验目录")
    evaluate.add_argument("--output", type=Path,
                          help="重放产物目录（默认写入实验目录下的新目录）")
    return parser


def _trading_config(args: argparse.Namespace):
    from stockrl.env import TradingConfig

    return TradingConfig(
        initial_cash=args.initial_cash,
        commission=args.commission,
        slippage=args.slippage,
        sell_tax=args.sell_tax,
        min_commission=args.min_commission,
        lot_size=args.lot_size,
        max_participation=args.max_participation,
    )


def _download_ticker(ticker: str, start: str | None, end: str | None):
    """Download one Yahoo symbol without turning the package into an online service."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - installation-specific branch
        raise RuntimeError("使用 --ticker 需要安装 yfinance") from exc

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("ticker 不能为空")
    kwargs = {"progress": False, "auto_adjust": False, "actions": False}
    if start:
        kwargs["start"] = start
    if end:
        kwargs["end"] = end
    if not start and not end:
        kwargs["period"] = "5y"
    frame = yf.download(symbol, **kwargs)
    if frame.empty:
        raise ValueError(f"Yahoo 未返回 {symbol} 的行情")
    if getattr(frame.columns, "nlevels", 1) > 1:
        frame.columns = frame.columns.get_level_values(0)
    from stockrl.data import validate_bars

    return validate_bars(frame), f"yahoo:{symbol}"


def _load_training_bars(args: argparse.Namespace):
    from stockrl.data import load_csv

    if args.csv:
        path = args.csv.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CSV 不存在：{path}")
        bars = load_csv(path)
        label = f"csv:{path.name}"
    else:
        bars, label = _download_ticker(args.ticker, args.start, args.end)
    if args.start:
        bars = bars.loc[args.start:]
    if args.end:
        bars = bars.loc[:args.end]
    if bars.empty:
        raise ValueError("日期筛选后没有行情数据")
    return bars, label


def _print_result(summary: dict, *, evaluation: bool = False) -> None:
    action = "评估已保存" if evaluation else "实验已保存"
    print(f"{action}：{summary['output_dir']}")
    print(f"算法：{summary['algorithm']}；数据：{summary['data_label']}；种子：{summary['seeds']}")
    aggregate = summary.get("aggregate", {})
    net_return = aggregate.get("total_return", {}).get("mean")
    sharpe = aggregate.get("sharpe", {}).get("mean")
    return_text = "未定义" if net_return is None else f"{net_return:.2%}"
    sharpe_text = "未定义" if sharpe is None else f"{sharpe:.3f}"
    print(f"测试区间平均净收益：{return_text}；Sharpe（无风险利率 0）：{sharpe_text}")
    print("结果仅用于研究与软件验证，不构成投资建议。")


def _run(args: argparse.Namespace) -> dict:
    from stockrl.experiments import evaluate_saved_run, run_experiment

    if args.command == "evaluate":
        return evaluate_saved_run(args.run_dir, args.output)
    if args.command == "demo":
        from stockrl.data import make_demo_data

        bars = make_demo_data(args.rows, args.data_seed)
        data_label = "synthetic_demo"
    else:
        bars, data_label = _load_training_bars(args)
    return run_experiment(
        bars,
        args.output,
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        seeds=tuple(args.seeds),
        config=_trading_config(args),
        train_ratio=args.train_ratio,
        val_ratio=args.validation_ratio,
        data_label=data_label,
        episode_length=args.episode_length,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if argv is None:
            raise
        return int(exc.code or 0)
    try:
        summary = _run(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    _print_result(summary, evaluation=args.command == "evaluate")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
