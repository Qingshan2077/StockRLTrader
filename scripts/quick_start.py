# !/usr/bin/env python3
"""
快速启动脚本 - 一键设置和运行股票数据系统
"""

import os
import sys
import json
from pathlib import Path


def check_dependencies():
    """检查依赖是否安装"""
    required = ['yfinance', 'pandas', 'pandas_ta']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return missing


def install_dependencies(packages):
    """安装缺失的依赖"""
    print(f"\n检测到缺失的依赖包: {', '.join(packages)}")
    install = input("是否自动安装？(y/n): ").strip().lower()

    if install == 'y':
        import subprocess
        for package in packages:
            print(f"\n正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("\n✓ 所有依赖已安装完成！")
        return True
    return False


def create_config():
    """创建配置文件"""
    if Path("../config.json").exists():
        print("✓ 配置文件已存在")
        return

    print("\n开始创建配置文件...")

    # 代理设置
    use_proxy = input("是否需要使用代理？(y/n): ").strip().lower() == 'y'
    proxy_url = ""
    if use_proxy:
        proxy_url = input("请输入代理地址 (如 http://127.0.0.1:7897): ").strip()

    # 数据目录
    data_dir = input("数据存储目录 (默认: stock_data): ").strip() or "stock_data"

    # 起始日期
    start_date = input("历史数据起始日期 (默认: 2015-01-01): ").strip() or "2015-01-01"

    config = {
        "proxy": {
            "enabled": use_proxy,
            "url": proxy_url
        },
        "data": {
            "directory": data_dir,
            "start_date": start_date
        },
        "watchlist": {
            "tech_giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "ai_chips": ["NVDA", "AMD", "INTC"],
            "my_stocks": []
        },
        "update_schedule": {
            "auto_update": False,
            "update_time": "09:30"
        }
    }

    with open("../config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("✓ 配置文件创建完成！")


def show_menu():
    """显示主菜单"""
    print("\n" + "=" * 60)
    print("股票数据管理系统 - 快速启动")
    print("=" * 60)
    print("\n请选择启动模式:")
    print("1. 交互式管理界面（推荐）")
    print("2. 快速下载示例数据（AAPL, MSFT, NVDA, TSLA）")
    print("3. 自定义下载")
    print("4. 仅查看配置")
    print("0. 退出")

    return input("\n请选择 (0-4): ").strip()


def quick_download():
    """快速下载示例数据"""
    from improved_data_engine import BatchDataEngine

    print("\n准备下载示例股票数据...")
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]

    confirm = input(f"将下载 {', '.join(tickers)} 的历史数据，继续吗？(y/n): ").strip().lower()
    if confirm != 'y':
        return

    batch = BatchDataEngine()
    batch.process_batch(tickers)

    print("\n✓ 示例数据下载完成！")
    print("\n你可以运行以下命令查看数据:")
    print("  python stock_data_manager.py --list")
    print("  python stock_data_manager.py --info AAPL")


def custom_download():
    """自定义下载"""
    from improved_data_engine import BatchDataEngine

    tickers_input = input("\n请输入股票代码（空格分隔，如 AAPL MSFT）: ").strip()
    if not tickers_input:
        print("未输入股票代码")
        return

    tickers = tickers_input.upper().split()
    print(f"\n准备下载: {', '.join(tickers)}")

    batch = BatchDataEngine()
    batch.process_batch(tickers)

    print("\n✓ 数据下载完成！")


def main():
    print("=" * 60)
    print("欢迎使用股票数据管理系统")
    print("=" * 60)

    # 检查依赖
    print("\n[1/3] 检查依赖...")
    missing = check_dependencies()
    if missing:
        if not install_dependencies(missing):
            print("\n请手动安装依赖:")
            print(f"pip install {' '.join(missing)}")
            return
    else:
        print("✓ 所有依赖已安装")

    # 创建配置
    print("\n[2/3] 检查配置...")
    if not Path("../config.json").exists():
        create_config()
    else:
        print("✓ 配置文件已存在")

    # 主菜单
    print("\n[3/3] 启动系统...")

    while True:
        choice = show_menu()

        if choice == "1":
            print("\n启动交互式管理界面...\n")
            try:
                from stock_manager_advanced import AdvancedStockManager
                manager = AdvancedStockManager()
                manager.interactive_menu()
            except KeyboardInterrupt:
                print("\n\n程序已退出")
                break

        elif choice == "2":
            quick_download()

        elif choice == "3":
            custom_download()

        elif choice == "4":
            with open("../config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            print("\n当前配置:")
            print(json.dumps(config, indent=2, ensure_ascii=False))

        elif choice == "0":
            print("\n再见！")
            break

        else:
            print("\n无效选项")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("请检查配置或联系开发者")