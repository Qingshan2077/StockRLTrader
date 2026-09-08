"""Explicit local maintenance, never invoked by imports or normal API requests."""
import argparse
from pathlib import Path

from .settings import AppSettings
from .storage.database import Database
from .storage.migrations import backup_database, initialize_database, migrate_database


def initialize(settings: AppSettings) -> Database:
    settings.app_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.staging_dir.mkdir(parents=True, exist_ok=True)
    return initialize_database(settings.database_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='StockRL 本地存储维护')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init')
    commands.add_parser('check')
    commands.add_parser('import-legacy')
    backup = commands.add_parser('backup')
    backup.add_argument('destination', type=Path)
    migrate = commands.add_parser('migrate')
    migrate.add_argument('backup', type=Path)
    options = parser.parse_args(argv)
    settings = AppSettings.from_env()
    if options.command == 'init':
        initialize(settings)
    elif options.command == 'backup':
        backup_database(settings.database_path, options.destination)
    elif options.command == 'migrate':
        migrate_database(settings.database_path, options.backup)
    elif options.command == 'import-legacy':
        from .legacy import LegacyImporter
        records = LegacyImporter(settings, Database(settings.database_path)).scan()
        for record in records:
            print(f'{record.experiment_id} {record.integrity} replayable={record.replayable}')
    else:
        with Database(settings.database_path).connection():
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
