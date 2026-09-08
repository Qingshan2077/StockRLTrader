"""Shared artifact format inspection, without importing a learner or an API."""

import hashlib
import csv
from datetime import datetime
from collections.abc import Callable
import statistics
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import AppError
from .settings import confined_path

ARTIFACT_SCHEMA_VERSION = 1
CORE_SEMANTICS_VERSION = 1
METRICS_VERSION = 1
BASELINES = ("cash", "buy_hold", "half", "trend")
METRICS = ("total_return", "annualized_return", "volatility", "sharpe", "max_drawdown",
           "total_turnover", "total_cost", "average_weight", "trade_count")
HISTORY_FIELDS = ("date", "nav", "cash", "shares", "weight", "requested_weight", "executed_weight",
                  "turnover", "cost")


def finite_number(value: Any) -> bool:
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def hash_file(path: Path, *, check: Callable[[], None] = lambda: None) -> str:
    digest = hashlib.sha256()
    check()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            check()
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    def reject_constant(value: str):
        raise ValueError(f"Invalid JSON number {value}")

    try:
        with path.open("rb") as source:
            body = source.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("JSON too large")
        result = json.loads(body.decode("utf-8"), parse_constant=reject_constant)
        if not isinstance(result, dict):
            raise ValueError("JSON object required")
        return result
    except (OSError, UnicodeError, ValueError) as exc:
        raise AppError("ARTIFACT_CORRUPT", "实验文件缺失或内容无效。", 409) from exc


def manifest_entries(root: Path, *, verify_files: bool = True,
                     expected_hash: str | None = None,
                     check: Callable[[], None] = lambda: None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    check()
    manifest_path = confined_path(root, "manifest.json", must_exist=True)
    if expected_hash is not None and hash_file(manifest_path, check=check) != expected_hash:
        raise AppError("ARTIFACT_CORRUPT", "实验产物清单已改变。", 409)
    manifest = read_json(manifest_path)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise AppError("UNSUPPORTED_ARTIFACT_VERSION", "实验产物版本暂不支持。", 409)
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries or len(entries) > 4096:
        raise AppError("ARTIFACT_CORRUPT", "实验产物清单不完整。", 409)
    seen: set[str] = set()
    for entry in entries:
        check()
        if not isinstance(entry, dict):
            raise AppError("ARTIFACT_CORRUPT", "产物记录格式无效。", 409)
        name, size, fingerprint = entry.get("path"), entry.get("size"), entry.get("sha256")
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or name == "manifest.json" or PurePosixPath(name).is_absolute()
                or name != PurePosixPath(name).as_posix()
                or any(part in {".", ".."} for part in PurePosixPath(name).parts)
                or name.casefold() in seen or type(size) is not int or size < 0
                or not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise AppError("ARTIFACT_CORRUPT", "产物文件名、大小或指纹无效。", 409)
        seen.add(name.casefold())
        path = confined_path(root, name, must_exist=True)
        if not path.is_file() or path.stat().st_size != size:
            raise AppError("ARTIFACT_CORRUPT", "实验文件缺失或大小不符。", 409)
        if verify_files and hash_file(path, check=check) != fingerprint:
            raise AppError("ARTIFACT_CORRUPT", "实验文件指纹不符。", 409)
    required = {"summary.json", "request.json", "bars.csv"}
    if not required.issubset(seen):
        raise AppError("ARTIFACT_CORRUPT", "缺少必要的实验产物。", 409)
    return manifest, entries


def metrics_only(raw: Any) -> dict[str, float | int | None]:
    if not isinstance(raw, dict) or not set(METRICS).issubset(raw):
        raise AppError("ARTIFACT_CORRUPT", "实验指标格式无效。", 409)
    result: dict[str, float | int | None] = {}
    for key in METRICS:
        value = raw.get(key)
        if value is not None and not finite_number(value):
            raise AppError("ARTIFACT_CORRUPT", "实验指标包含无效数值。", 409)
        result[key] = value
    return result


def public_runs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Explicit projection: stored absolute paths never become HTTP responses."""
    raw_runs = summary.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs or len(raw_runs) > 100:
        raise AppError("ARTIFACT_CORRUPT", "实验缺少可识别的 seed 结果。", 409)
    output = []
    seen = set()
    for raw in raw_runs:
        if not isinstance(raw, dict) or type(raw.get("seed")) is not int:
            raise AppError("ARTIFACT_CORRUPT", "seed 记录无效。", 409)
        seed = raw["seed"]
        if not 0 <= seed < 2**32 or seed in seen:
            raise AppError("ARTIFACT_CORRUPT", "seed 记录重复或超出范围。", 409)
        seen.add(seed)
        baselines = raw.get("baselines")
        if not isinstance(baselines, dict) or not set(BASELINES).issubset(baselines):
            raise AppError("ARTIFACT_CORRUPT", "缺少必要的基准结果。", 409)
        item: dict[str, Any] = {"seed": seed, "metrics": metrics_only(raw.get("metrics")), "baselines": {}}
        for name in BASELINES:
            baseline = baselines[name]
            if not isinstance(baseline, dict):
                raise AppError("ARTIFACT_CORRUPT", "基准结果格式无效。", 409)
            item["baselines"][name] = {"metrics": metrics_only(baseline.get("metrics"))}
        selection = raw.get("checkpoint_selection")
        item["checkpoint_selection"] = selection if selection == "validation_mean_reward" else "unknown"
        for key in ("actual_timesteps", "gradient_updates", "best_validation_mean_reward"):
            value = raw.get(key)
            if finite_number(value):
                item[key] = value
        output.append(item)
    return output


def public_aggregate(raw: Any, *, runs: list[dict] | None = None) -> dict[str, dict[str, float | int | None]]:
    if not isinstance(raw, dict) or not set(METRICS).issubset(raw):
        raise AppError("ARTIFACT_CORRUPT", "汇总指标格式无效。", 409)
    result = {}
    for name in METRICS:
        value = raw.get(name)
        if not isinstance(value, dict) or not {'mean', 'std', 'count'}.issubset(value):
            raise AppError("ARTIFACT_CORRUPT", "汇总指标格式无效。", 409)
        stats = {key: value.get(key) for key in ("mean", "std", "count")}
        if any(number is not None and not finite_number(number)
               for number in stats.values()):
            raise AppError("ARTIFACT_CORRUPT", "汇总指标包含无效数值。", 409)
        if (type(stats['count']) is not int or stats['count'] < 0
                or (stats['count'] == 0 and (stats['mean'] is not None or stats['std'] is not None))
                or (stats['std'] is not None and stats['std'] < 0)):
            raise AppError('ARTIFACT_CORRUPT', '汇总样本数或标准差无效。', 409)
        if runs is not None and stats['count'] != sum(run['metrics'][name] is not None for run in runs):
            raise AppError('ARTIFACT_CORRUPT', '汇总有效样本数与各 seed 不一致。', 409)
        result[name] = stats
    return result


def baseline_aggregate(runs: list[dict]) -> dict[str, dict]:
    """Summarize saved metrics across all seeds; never recalculate trading results."""
    result = {}
    for policy in BASELINES:
        result[policy] = {}
        for metric in METRICS:
            values = [run['baselines'][policy]['metrics'][metric] for run in runs
                      if run['baselines'][policy]['metrics'][metric] is not None]
            mean, deviation = None, None
            if values:
                try:
                    mean = statistics.mean(values)
                    deviation = statistics.pstdev(values)
                except (OverflowError, ValueError):
                    pass
            result[policy][metric] = {'mean': mean if finite_number(mean) else None,
                                     'std': deviation if finite_number(deviation) else None, 'count': len(values)}
    return result


def report_rows(path: Path, *, history: bool, max_rows: int = 100000,
                check: Callable[[], None] = lambda: None, interval: dict | None = None):
    """Bounded structural validation shared by import and HTTP report reads."""
    fields = HISTORY_FIELDS if history else ('date', 'side', 'shares', 'price', 'cost')
    previous = None
    first_date = None
    count = 0
    try:
        check()
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('oversize report')
        with path.open('r', encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or not set(fields).issubset(reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('missing or duplicate report fields')
            for count, row in enumerate(reader, start=1):
                if count % 1000 == 0:
                    check()
                if count > max_rows + 1 or None in row:
                    raise ValueError('invalid row count/shape')
                date = datetime.fromisoformat(row['date'])
                if first_date is None:
                    first_date = date
                if previous is not None and date <= previous:
                    raise ValueError('non-increasing dates')
                previous = date
                if interval is not None and not history:
                    if not (datetime.fromisoformat(interval['first_reward_date']).date() <= date.date()
                            <= datetime.fromisoformat(interval['last_reward_date']).date()):
                        raise ValueError('trade outside evaluation interval')
                output: dict[str, Any] = {'date': row['date']}
                for name in fields:
                    if name == 'date':
                        continue
                    if name == 'side':
                        if row[name] not in ('buy', 'sell'):
                            raise ValueError('invalid side')
                        output[name] = row[name]
                        continue
                    value = float(row[name])
                    if not math.isfinite(value):
                        raise ValueError('invalid number')
                    output[name] = value
                yield output
        if history and not count:
            raise ValueError('empty history')
        if history and interval is not None:
            if (count != interval['reward_count'] + 1
                    or first_date.date() != datetime.fromisoformat(interval['observation_start_date']).date()
                    or previous.date() != datetime.fromisoformat(interval['last_reward_date']).date()):
                raise ValueError('history does not match evaluation interval')
    except (OSError, UnicodeError, csv.Error, ValueError, TypeError, OverflowError) as exc:
        raise AppError('ARTIFACT_CORRUPT', '结果 CSV 格式、日期或数值无效。', 409) from exc
