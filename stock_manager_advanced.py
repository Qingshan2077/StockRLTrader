#!/usr/bin/env python3
"""
增强版股票数据管理工具 - 支持配置文件和监视列表
"""

import json
import os
from pathlib import Path
from improved_data_engine import DataEngine, BatchDataEngine


class AdvancedStockManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()

        # 应用代理设置
        if self.config['proxy']['enabled']:
            os.environ['HTTP_PROXY'] = self.config['proxy']['url']
            os.environ['HTTPS_PROXY'] = self.config['proxy']['url']

        self.batch_engine = BatchDataEngine(
            data_dir=self.config['data']['directory'],
            start_date=self.config['data']['start_date']
        )

    def load_config(self):
        """加载配置文件"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return self.get_default_config()

    def get_default_config(self):
        """获取默认配置"""
        return {
            "proxy": {"enabled": False, "url": ""},
            "data": {"directory": "stock_data", "start_date": "2015-01-01"},
            "watchlist": {"my_stocks": []},
            "update_schedule": {"auto_update": False, "update_time": "09:30"}
        }

    def save_config(self):
        """保存配置"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"配置已保存到 {self.config_path}")

    def show_watchlists(self):
        """显示所有监视列表"""
        watchlists = self.config.get('watchlist', {})
        if not watchlists:
            print("没有配置监视列表")
            return

        print("\n" + "=" * 60)
        print("监视列表")
        print("=" * 60)
        for name, tickers in watchlists.items():
            print(f"\n【{name}】({len(tickers)} 个)")
            if tickers:
                for i, ticker in enumerate(tickers, 1):
                    print(f"  {i}. {ticker}")
            else:
                print("  (空)")

    def add_to_watchlist(self, watchlist_name, tickers):
        """添加股票到监视列表"""
        if 'watchlist' not in self.config:
            self.config['watchlist'] = {}

        if watchlist_name not in self.config['watchlist']:
            self.config['watchlist'][watchlist_name] = []

        for ticker in tickers:
            ticker = ticker.upper()
            if ticker not in self.config['watchlist'][watchlist_name]:
                self.config['watchlist'][watchlist_name].append(ticker)
                print(f"✓ 已添加 {ticker} 到 [{watchlist_name}]")
            else:
                print(f"✗ {ticker} 已存在于 [{watchlist_name}]")

        self.save_config()

    def update_watchlist(self, watchlist_name):
        """更新指定监视列表的所有股票"""
        if watchlist_name not in self.config.get('watchlist', {}):
            print(f"监视列表 [{watchlist_name}] 不存在")
            return

        tickers = self.config['watchlist'][watchlist_name]
        if not tickers:
            print(f"监视列表 [{watchlist_name}] 为空")
            return

        print(f"\n准备更新监视列表 [{watchlist_name}] 的 {len(tickers)} 个股票...")
        self.batch_engine.process_batch(tickers)

    def update_all_watchlists(self):
        """更新所有监视列表"""
        watchlists = self.config.get('watchlist', {})
        all_tickers = set()

        for name, tickers in watchlists.items():
            all_tickers.update(tickers)

        if not all_tickers:
            print("所有监视列表都是空的")
            return

        print(f"\n准备更新所有监视列表，共 {len(all_tickers)} 个不重复股票...")
        self.batch_engine.process_batch(list(all_tickers))

    def create_watchlist(self, name):
        """创建新的监视列表"""
        if 'watchlist' not in self.config:
            self.config['watchlist'] = {}

        if name in self.config['watchlist']:
            print(f"监视列表 [{name}] 已存在")
            return

        self.config['watchlist'][name] = []
        self.save_config()
        print(f"✓ 已创建监视列表 [{name}]")

    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n" + "=" * 60)
            print("股票数据管理系统 - 主菜单")
            print("=" * 60)
            print("1. 查看监视列表")
            print("2. 创建监视列表")
            print("3. 添加股票到监视列表")
            print("4. 更新指定监视列表")
            print("5. 更新所有监视列表")
            print("6. 快速添加单个股票")
            print("7. 查看本地所有数据")
            print("8. 查看股票详情")
            print("9. 配置管理")
            print("0. 退出")

            choice = input("\n请选择 (0-9): ").strip()

            if choice == "1":
                self.show_watchlists()

            elif choice == "2":
                name = input("输入新监视列表名称: ").strip()
                if name:
                    self.create_watchlist(name)

            elif choice == "3":
                self.show_watchlists()
                name = input("\n输入监视列表名称: ").strip()
                tickers_input = input("输入股票代码（空格分隔）: ").strip()
                if name and tickers_input:
                    tickers = tickers_input.upper().split()
                    self.add_to_watchlist(name, tickers)

            elif choice == "4":
                self.show_watchlists()
                name = input("\n输入要更新的监视列表名称: ").strip()
                if name:
                    self.update_watchlist(name)

            elif choice == "5":
                confirm = input("确认更新所有监视列表？(y/n): ").strip().lower()
                if confirm == 'y':
                    self.update_all_watchlists()

            elif choice == "6":
                tickers_input = input("输入股票代码（空格分隔）: ").strip()
                if tickers_input:
                    tickers = tickers_input.upper().split()
                    self.batch_engine.process_batch(tickers)

            elif choice == "7":
                self.batch_engine.list_available_data()

            elif choice == "8":
                ticker = input("输入股票代码: ").strip().upper()
                if ticker:
                    engine = DataEngine(ticker, data_dir=self.config['data']['directory'])
                    engine.get_info()
                    df = engine.load_processed_data()
                    if df is not None:
                        print("\n最新5天数据:")
                        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
                        available_cols = [c for c in cols if c in df.columns]
                        print(df[available_cols].tail())

            elif choice == "9":
                self.config_menu()

            elif choice == "0":
                print("\n再见！")
                break

            else:
                print("无效选项")

    def config_menu(self):
        """配置管理菜单"""
        while True:
            print("\n" + "=" * 60)
            print("配置管理")
            print("=" * 60)
            print("1. 查看当前配置")
            print("2. 修改代理设置")
            print("3. 修改数据目录")
            print("4. 修改起始日期")
            print("0. 返回主菜单")

            choice = input("\n请选择 (0-4): ").strip()

            if choice == "1":
                print("\n当前配置:")
                print(json.dumps(self.config, indent=2, ensure_ascii=False))

            elif choice == "2":
                enabled = input("启用代理？(y/n): ").strip().lower() == 'y'
                self.config['proxy']['enabled'] = enabled
                if enabled:
                    url = input("代理地址 (如 http://127.0.0.1:7897): ").strip()
                    if url:
                        self.config['proxy']['url'] = url
                self.save_config()
                print("✓ 代理设置已更新（重启程序生效）")

            elif choice == "3":
                directory = input("数据目录路径: ").strip()
                if directory:
                    self.config['data']['directory'] = directory
                    self.save_config()
                    print("✓ 数据目录已更新")

            elif choice == "4":
                date = input("起始日期 (YYYY-MM-DD): ").strip()
                if date:
                    self.config['data']['start_date'] = date
                    self.save_config()
                    print("✓ 起始日期已更新")

            elif choice == "0":
                break

            else:
                print("无效选项")


def main():
    import sys

    manager = AdvancedStockManager()

    # 如果有命令行参数
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "--list":
            manager.show_watchlists()

        elif command == "--update-all":
            manager.update_all_watchlists()

        elif command == "--update" and len(sys.argv) > 2:
            watchlist_name = sys.argv[2]
            manager.update_watchlist(watchlist_name)

        elif command == "--add" and len(sys.argv) > 3:
            watchlist_name = sys.argv[2]
            tickers = sys.argv[3:]
            manager.add_to_watchlist(watchlist_name, tickers)

        else:
            print("使用方法:")
            print("  python stock_manager_advanced.py                    # 交互式模式")
            print("  python stock_manager_advanced.py --list             # 显示监视列表")
            print("  python stock_manager_advanced.py --update-all       # 更新所有")
            print("  python stock_manager_advanced.py --update <列表名>   # 更新指定列表")
            print("  python stock_manager_advanced.py --add <列表名> <股票...>  # 添加股票")
    else:
        # 默认启动交互式界面
        manager.interactive_menu()


if __name__ == "__main__":
    main()