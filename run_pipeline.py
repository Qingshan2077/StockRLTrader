"""Compatibility launcher for the StockRL command-line workflows."""

import sys

from stockrl.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
