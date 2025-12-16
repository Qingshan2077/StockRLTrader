#!/usr/bin/env python3
"""
启动脚本 - 自动检查环境并启动 Streamlit 应用
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 需要 Python 3.8 或更高版本")
        print(f"当前版本: Python {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python 版本: {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """检查依赖是否安装"""
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'yfinance': 'yfinance',
        'pandas_ta': 'pandas-ta',
        'xgboost': 'xgboost',
        'sklearn': 'scikit-learn',
        'plotly': 'plotly'
    }

    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (未安装)")
            missing.append(package)

    return missing


def install_dependencies(packages):
    """安装缺失的依赖"""
    print(f"\n正在安装 {len(packages)} 个依赖包...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", *packages
        ])
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依赖安装失败，请手动安装:")
        print(f"   pip install {' '.join(packages)}")
        return False


def check_project_structure():
    """检查项目结构"""
    required_files = [
        'improved_data_engine.py',
        'predictor.py',
        'config.json'
    ]

    required_dirs = [
        'stock_data',
        'frontend/pages'
    ]

    print("\n检查项目结构...")

    # 检查文件
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"⚠️  {file} (缺失)")

    # 检查目录
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"📁 创建目录: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ {dir_path}")


def create_default_config():
    """创建默认配置文件"""
    config_path = Path("config.json")
    if not config_path.exists():
        print("\n创建默认配置文件...")
        import json
        config = {
            "proxy": {
                "enabled": False,
                "url": "http://127.0.0.1:7897"
            },
            "data": {
                "directory": "stock_data",
                "start_date": "2015-01-01"
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
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("✅ 配置文件创建完成")


def start_streamlit():
    """启动 Streamlit 应用"""
    app_path = Path("frontend/app.py")

    if not app_path.exists():
        print(f"\n❌ 找不到应用文件: {app_path}")
        print("请确保 frontend/app.py 存在")
        return False

    print("\n" + "=" * 60)
    print("🚀 启动 AI股票交易助手")
    print("=" * 60)
    print("\n浏览器将自动打开应用界面...")
    print("如果没有自动打开，请访问: http://localhost:8501")
    print("\n按 Ctrl+C 停止应用\n")

    try:
        # 启动 Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 应用已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        return False

    return True


def main():
    """主函数"""
    print("=" * 60)
    print("AI股票交易助手 - 启动检查")
    print("=" * 60)

    # 1. 检查 Python 版本
    print("\n[1/5] 检查 Python 版本...")
    if not check_python_version():
        return

    # 2. 检查依赖
    print("\n[2/5] 检查依赖...")
    missing = check_dependencies()

    if missing:
        print(f"\n发现 {len(missing)} 个缺失的依赖包")
        install = input("是否自动安装? (y/n): ").strip().lower()

        if install == 'y':
            if not install_dependencies(missing):
                return
        else:
            print("\n请手动安装依赖:")
            print(f"pip install {' '.join(missing)}")
            return

    # 3. 检查项目结构
    print("\n[3/5] 检查项目结构...")
    check_project_structure()

    # 4. 创建配置
    print("\n[4/5] 检查配置文件...")
    create_default_config()

    # 5. 启动应用
    print("\n[5/5] 启动应用...")
    start_streamlit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()