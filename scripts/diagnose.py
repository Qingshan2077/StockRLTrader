#!/usr/bin/env python3
"""
诊断工具 - 检查系统环境和数据质量
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np


def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 60)
    print("1. 检查依赖包")
    print("=" * 60)

    required = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yfinance': 'yfinance',
        'pandas_ta': 'pandas-ta',
        'xgboost': 'xgboost',
        'sklearn': 'scikit-learn',
        'plotly': 'plotly',
        'torch': 'torch'
    }

    all_ok = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            all_ok = False

    return all_ok


def check_data():
    """检查数据文件"""
    print("\n" + "=" * 60)
    print("2. 检查数据文件")
    print("=" * 60)

    data_dir = Path("stock_data")

    if not data_dir.exists():
        print("❌ 数据目录不存在: stock_data/")
        return False

    # 查找所有 processed 文件
    processed_files = list(data_dir.glob("*_processed.csv"))

    if not processed_files:
        print("❌ 未找到任何数据文件")
        print("   请先在'数据管理'页面下载股票数据")
        return False

    print(f"✅ 找到 {len(processed_files)} 个股票数据\n")

    # 检查每个文件
    issues_found = False
    for file in processed_files:
        ticker = file.stem.replace("_processed", "")
        print(f"\n检查 {ticker}:")

        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)

            # 基本检查
            print(f"  数据长度: {len(df)} 条")
            print(f"  日期范围: {df.index[0]} ~ {df.index[-1]}")
            print(f"  特征数量: {len(df.columns)}")

            # 检查 NaN
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                print(f"  ⚠️  包含 {nan_count} 个 NaN 值")
                issues_found = True
            else:
                print(f"  ✅ 无 NaN 值")

            # 检查 inf
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_count = 0
            for col in numeric_cols:
                inf_count += np.isinf(df[col]).sum()

            if inf_count > 0:
                print(f"  ⚠️  包含 {inf_count} 个无穷值")
                issues_found = True
            else:
                print(f"  ✅ 无无穷值")

            # 检查技术指标
            required_indicators = ['RSI', 'MACD_12_26_9', 'SMA_10', 'SMA_50']
            missing = [ind for ind in required_indicators if ind not in df.columns]

            if missing:
                print(f"  ⚠️  缺少技术指标: {missing}")
                issues_found = True
            else:
                print(f"  ✅ 技术指标完整")

            # 检查数据长度
            if len(df) < 500:
                print(f"  ⚠️  数据量较少（建议至少 500 条，当前 {len(df)} 条）")
                issues_found = True
            else:
                print(f"  ✅ 数据量充足")

        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            issues_found = True

    return not issues_found


def check_environment():
    """检查训练环境"""
    print("\n" + "=" * 60)
    print("3. 检查训练环境")
    print("=" * 60)

    all_ok = True

    # 检查 GPU（可选）
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  GPU 不可用（使用 CPU 训练，速度会较慢）")
    except:
        print("ℹ️  无法检测 GPU")

    # 检查模型目录
    model_dir = Path("data/models")
    if not model_dir.exists():
        print("ℹ️  创建模型目录: data/models/")
        model_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("✅ 模型目录存在")

        # 列出已有模型
        models = list(model_dir.glob("*.zip"))
        if models:
            print(f"   找到 {len(models)} 个已训练模型:")
            for model in models:
                print(f"   - {model.name}")

    return all_ok


def test_training():
    """测试训练流程"""
    print("\n" + "=" * 60)
    print("4. 测试训练流程（可选）")
    print("=" * 60)

    response = input("是否要运行快速训练测试？(y/n): ").strip().lower()

    if response != 'y':
        print("跳过训练测试")
        return True

    try:
        print("\n准备测试数据...")
        from improved_data_engine import DataEngine

        # 查找第一个可用的数据文件
        data_dir = Path("stock_data")
        raw_files = list(data_dir.glob("*_raw.csv"))

        if not raw_files:
            print("❌ 没有可用的数据文件")
            return False

        ticker = raw_files[0].stem.replace("_raw", "")
        print(f"使用 {ticker} 进行测试...")

        # 加载数据
        engine = DataEngine(ticker)
        df = engine.load_processed_data()

        if df is None or df.empty:
            print("❌ 无法加载数据")
            return False

        print(f"✅ 数据加载测试通过 (rows={len(df)}, cols={len(df.columns)})")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def provide_recommendations(deps_ok, data_ok, env_ok):
    """提供修复建议"""
    print("\n" + "=" * 60)
    print("5. 修复建议")
    print("=" * 60)

    if deps_ok and data_ok and env_ok:
        print("\n🎉 系统状态良好！\n")
        print("快速开始:")
        print("1. 运行: python run.py")
        print("2. 进入'▸ 模型训练'页面训练 Alpha 模型")
        print("3. 进入'▸ 交易执行'页面进行 RL 训练")
        return

    print("\n发现以下问题，请按顺序修复:\n")

    if not deps_ok:
        print("❌ 依赖包问题")
        print("   修复方法: pip install -r requirements.txt\n")

    if not data_ok:
        print("❌ 数据问题")
        print("   修复方法:")
        print("   1. 运行 python run.py 启动 Web 界面")
        print("   2. 进入'📊 数据管理'页面")
        print("   3. 下载股票数据（建议至少 2-3 年）")
        print("   4. 如果数据有问题，可以删除后重新下载\n")

    if not env_ok:
        print("❌ 环境问题")
        print("   修复方法: 确保有足够的磁盘空间用于存储模型\n")


def main():
    print("\n" + "=" * 60)
    print("AI 量化交易系统 v3 - 系统诊断工具")
    print("=" * 60)
    print("\n此工具将检查系统环境和数据质量")
    print("确保一切准备就绪可以开始 v3 量化管线\n")

    # 运行所有检查
    deps_ok = check_dependencies()
    data_ok = check_data()
    env_ok = check_environment()

    # 可选：测试训练
    if deps_ok and data_ok and env_ok:
        test_ok = test_training()
    else:
        test_ok = True  # 如果基础检查失败，跳过训练测试

    # 提供建议
    provide_recommendations(deps_ok and test_ok, data_ok, env_ok)

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n诊断被中断")
    except Exception as e:
        print(f"\n诊断过程出错: {e}")
        import traceback

        traceback.print_exc()