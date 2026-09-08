"""Syntax and import-graph checks only. Does not import project modules or run tests."""
import ast
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for name in ('stockrl', 'stockrl_app', 'api', 'scripts', 'tests'):
        files.extend((ROOT / name).rglob('*.py'))
    files.extend([ROOT / 'run.py', ROOT / 'run_pipeline.py'])
    for path in files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    # Follow only imports that execute at module level. Function-local learner imports
    # belong to the worker, and TYPE_CHECKING annotations do not execute.
    seen = set()

    def inspect(name):
        if name in seen:
            return
        seen.add(name)
        if name == 'torch' or name.startswith('stable_baselines3') or name in ('stockrl.training', 'stockrl.experiments'):
            raise ValueError(f'API imports learner at module load: {name}')
        path = ROOT.joinpath(*name.split('.')).with_suffix('.py')
        if not path.is_file():
            path = ROOT.joinpath(*name.split('.'), '__init__.py')
        if not path.is_file():
            return
        package = name if path.name == '__init__.py' else name.rpartition('.')[0]

        def visit(nodes):
            for node in nodes:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        inspect(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    prefix = package.split('.')[:len(package.split('.')) - node.level + 1] if node.level else []
                    module = '.'.join([*prefix, node.module] if node.module else prefix)
                    inspect(module)
                    for alias in node.names:
                        inspect(module + '.' + alias.name)
                elif isinstance(node, ast.If):
                    if not isinstance(node.test, ast.Name) or node.test.id != 'TYPE_CHECKING':
                        visit(node.body)
                        visit(node.orelse)
                elif isinstance(node, ast.Try):
                    visit(node.body + node.orelse + node.finalbody)
                    for handler in node.handlers:
                        visit(handler.body)
        visit(ast.parse(path.read_text(encoding='utf-8-sig')).body)

    inspect('api.main')
    print(f'AST syntax checked: {len(files)} Python files; TOML parsed; API source import graph contains no learner.')
    print('No application imports, tests, builds, services or training were executed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
