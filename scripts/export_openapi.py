"""Future integration command; imports API models, never starts a server or a worker."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    from api.main import create_app
    document = create_app().openapi()
    if any(name == 'torch' or name.startswith('stable_baselines3') for name in sys.modules):
        raise RuntimeError('API contract export unexpectedly loaded a learner')
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
