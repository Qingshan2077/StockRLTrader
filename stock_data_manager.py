#!/usr/bin/env python3
"""
股票数据管理工具 - 命令行界面
使用方法:
    python stock_data_manager.py --add AAPL MSFT      # 添加股票
    python stock_data_manager.py --update             # 更新所有股票
    python stock_data_manager.py --update AAPL        # 更新指定股票
    python stock_data_manager.py --list               # 列出所有股票
    python stock_data_manager.py --info AAPL          # 查看股票详情
    python stock_data_manager.py --interactive        # 交互式模式
"""

import argparse
import sys
from improved_data_engine import DataEngine, BatchDataEngine
from pathlib import Path


class StockDataManager:
    def __init__(self, data_dir="stock_data"):
        self.batch_engine = BatchDataEngine(data_dir=data_dir)
        self.data_dir = data_dir

    def add_stocks(self, tickers):
        """添加新股票"""
        print(f"\n准备添加 {len(tickers)} 个股票...")
        self.batch_engine.process_batch(tickers)

    def update_all(self):
        """更新所有现有股票"""
        tickers = self.batch_engine.list_available_data()
        if not tickers:
            print("没有找到本地数据，请先添加股票")
            return
        print(f"\n准备更新 {len(tickers)} 个股票...")
        self.batch_engine.process_batch(tickers)

    def update_specific(self, tickers):
        """更新指定股票"""
        print(f"\n准备更新指定的 {len(tickers)} 个股票...")
        self.batch_engine.process_batch(tickers)

    def list_stocks(self):
        """列出所有股票"""
        tickers = self.batch_engine.list_available_data()
        if tickers:
            print(f"\n找到 {len(tickers)} 个股票的本地数据")
        else:
            print("\n暂无本地数据")

    def show_info(self, ticker):
        """显示股票详细信息"""
        engine = DataEngine(ticker, data_dir=self.data_dir)
        metadata = engine.get_info()

        if metadata and engine.processed_data_path.exists():
            df = engine.load_processed_data()
            if df is not None:
                print("最新5天数据预览:")
                print(df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD_12_26_9']].tail())

    def interactive_mode(self):
        """交互式模式"""
        print("\n" + "=" * 60)
        print("股票数据管理工具 - 交互式模式")
        print("=" * 60)

        while True:
            print("\n请选择操作:")
            print("1. 添加股票")
            print("2. 更新所有股票")
            print("3. 更新指定股票")
            print("4. 查看股票列表")
            print("5. 查看股票详情")
            print("6. 删除股票数据")
            print("0. 退出")

            choice = input("\n请输入选项 (0-6): ").strip()

            if choice == "1":
                tickers_input = input("请输入股票代码（多个用空格分隔，如 AAPL MSFT）: ").strip()
                if tickers_input:
                    tickers = tickers_input.upper().split()
                    self.add_stocks(tickers)

            elif choice == "2":
                confirm = input("确认要更新所有股票吗？(y/n): ").strip().lower()
                if confirm == 'y':
                    self.update_all()

            elif choice == "3":
                tickers_input = input("请输入要更新的股票代码（多个用空格分隔）: ").strip()
                if tickers_input:
                    tickers = tickers_input.upper().split()
                    self.update_specific(tickers)

            elif choice == "4":
                self.list_stocks()

            elif choice == "5":
                ticker = input("请输入股票代码: ").strip().upper()
                if ticker:
                    self.show_info(ticker)

            elif choice == "6":
                ticker = input("请输入要删除的股票代码: ").strip().upper()
                if ticker:
                    self.delete_stock(ticker)

            elif choice == "0":
                print("\n感谢使用！再见！")
                break

            else:
                print("无效的选项，请重新选择")

    def delete_stock(self, ticker):
        """删除股票数据"""
        confirm = input(f"确认要删除 {ticker} 的所有数据吗？(y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return

        data_path = Path(self.data_dir)
        files_to_delete = [
            data_path / f"{ticker}_raw.csv",
            data_path / f"{ticker}_processed.csv",
            data_path / f"{ticker}_meta.json"
        ]

        deleted = 0
        for file in files_to_delete:
            if file.exists():
                file.unlink()
                deleted += 1

        if deleted > 0:
            print(f"✓ 已删除 {ticker} 的 {deleted} 个文件")
        else:
            print(f"✗ 未找到 {ticker} 的数据文件")


def main():
    parser = argparse.ArgumentParser(
        description="股票数据管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --add AAPL MSFT GOOGL        添加多个股票
  %(prog)s --update                     更新所有股票数据
  %(prog)s --update AAPL                更新指定股票
  %(prog)s --list                       列出所有本地股票
  %(prog)s --info AAPL                  查看股票详情
  %(prog)s --interactive                启动交互式模式
        """
    )

    parser.add_argument('--add', nargs='+', metavar='TICKER',
                        help='添加新股票（可以多个）')
    parser.add_argument('--update', nargs='*', metavar='TICKER',
                        help='更新股票数据（不指定则更新全部）')
    parser.add_argument('--list', action='store_true',
                        help='列出所有本地股票')
    parser.add_argument('--info', metavar='TICKER',
                        help='查看指定股票详情')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='启动交互式模式')
    parser.add_argument('--data-dir', default='stock_data',
                        help='数据存储目录（默认: stock_data）')

    args = parser.parse_args()

    # 如果没有任何参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    manager = StockDataManager(data_dir=args.data_dir)

    # 处理各种命令
    if args.interactive:
        manager.interactive_mode()
    elif args.add:
        manager.add_stocks([t.upper() for t in args.add])
    elif args.update is not None:
        if len(args.update) == 0:
            manager.update_all()
        else:
            manager.update_specific([t.upper() for t in args.update])
    elif args.list:
        manager.list_stocks()
    elif args.info:
        manager.show_info(args.info.upper())


if __name__ == "__main__":
    main()