"""Generate transport types from model annotations without importing application code.

This bootstrap/static check intentionally does not claim OpenAPI runtime validation.
The separate export_openapi.py command is part of the deferred integration checks.
"""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'frontend-react/src/api/contracts.ts'
SKIP = {'StrictModel', 'DatasetRecord', 'JobRecord', 'ExperimentRecord', 'ArtifactStorageRecord', 'WorkerInstance'}


def type_name(node):
    if isinstance(node, ast.Name):
        return {'str': 'string', 'int': 'number', 'StrictInt': 'number', 'PositiveInt': 'number', 'float': 'number', 'bool': 'boolean', 'Any': 'unknown'}.get(node.id, node.id)
    if isinstance(node, ast.Constant):
        if node.value is None:
            return 'null'
        import json
        return json.dumps(node.value, ensure_ascii=False)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return f'{type_name(node.left)} | {type_name(node.right)}'
    if isinstance(node, ast.Subscript):
        name = type_name(node.value)
        args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if name == 'Annotated':
            return type_name(args[0])
        if name == 'Literal':
            return ' | '.join(type_name(item) for item in args)
        if name in ('list', 'tuple'):
            return f'Array<{type_name(args[0])}>'
        if name == 'dict':
            return f'Record<{type_name(args[0])}, {type_name(args[1])}>'
        return name + '<' + ', '.join(type_name(item) for item in args) + '>'
    raise ValueError(f'Unsupported annotation: {ast.dump(node)}')


def generate():
    parts = ['// Generated from Python annotations by scripts/generate_contracts.py. Do not edit.',
             '// All response fields are present; request defaults come from /capabilities.',
             '// Runtime OpenAPI parity remains a separate integration acceptance check.', '']
    aliases = {'JobStatus', 'JobKind', 'Phase', 'Integrity', 'Seed', 'Policy'}
    for name in ('models.py', 'result_models.py'):
        tree = ast.parse((ROOT / 'stockrl_app' / name).read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in aliases:
                parts.append(f'export type {node.targets[0].id} = {type_name(node.value)};')
            if not isinstance(node, ast.ClassDef) or node.name in SKIP:
                continue
            base = [type_name(item) for item in node.bases if isinstance(item, ast.Name) and item.id != 'StrictModel']
            suffix = '<T>' if node.name == 'Page' else ''
            inherit = ' extends ' + ', '.join(base) if base else ''
            parts.append(f'export interface {node.name}{suffix}{inherit} {{')
            for field in node.body:
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
                    parts.append(f'  {field.target.id}: {type_name(field.annotation)};')
            parts.extend(['}', ''])
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    options = parser.parse_args()
    content = generate()
    research = ['// Generated from stockrl/research/contracts.py. Do not edit.', '']
    tree = ast.parse((ROOT / 'stockrl/research/contracts.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name in ('Contract', 'ResearchError'):
            continue
        bases = [base.id for base in node.bases if isinstance(base, ast.Name) and base.id != 'Contract']
        research.append(f'export interface {node.name}' + (' extends ' + ', '.join(bases) if bases else '') + ' {')
        for field in node.body:
            if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
                research.append(f'  {field.target.id}: {type_name(field.annotation)};')
        research.extend(['}', ''])
    outputs = {TARGET: content, TARGET.with_name('research.contracts.ts'): '\n'.join(research)}
    if options.check:
        for target, value in outputs.items():
            if not target.is_file() or target.read_text(encoding='utf-8') != value:
                raise SystemExit('Transport types differ; run scripts/generate_contracts.py')
        print('Static transport annotations match.')
    else:
        for target, value in outputs.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
