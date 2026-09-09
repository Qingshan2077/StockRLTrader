"""Read-only, fail-closed validation of auditable six-file market bundles."""
import csv
import hashlib
import io
import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .contracts import (Bar, CorporateAction, MarketDatasetPreview, MarketDatasetV2,
                        MarketProfile, MarketSession, Tradability)

BUNDLE_FILES = ('metadata.json', 'bars.csv', 'sessions.csv', 'actions.csv',
                'tradability.csv', 'market-profile.json')
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024


def canonical_instrument_id(metadata):
    exchange, symbol = metadata.get('exchange'), metadata.get('symbol')
    if exchange not in ('SSE', 'SZSE') or not isinstance(symbol,str) or not re.fullmatch(r'[0-9]{6}',symbol):
        raise ValueError('supported instrument identity requires exchange and six digit symbol')
    return f'{exchange}:{symbol}'


class MarketDataError(ValueError):
    def __init__(self, code, preview):
        super().__init__(code)
        self.code = code
        self.preview = preview


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('non-finite number')
    return result


def _optional_number(value):
    return None if value in ('', None, 'null') else _number(value)


def _boolean(value, optional=False):
    if optional and value in ('', None, 'null'):
        return None
    if value is True or value == 'true':
        return True
    if value is False or value == 'false':
        return False
    raise ValueError('boolean must be true or false')


def _day(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError('session must be ISO date')
    return parsed


def _time(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('timestamp requires timezone')
    return parsed


def validate_market_bundle(bundle, *, expected_hashes=None, max_file_bytes=MAX_FILE_BYTES,
                           max_total_bytes=MAX_TOTAL_BYTES):
    """Preview bytes by exact filename or a local directory; never write input files.

    Provenance declarations are recorded, never treated as a price transformation.
    A ready preview establishes contract consistency, not supplier truthfulness.
    """
    reasons, issues, raw, hashes = [], [], {}, {}
    unsupported = False

    def issue(code, detail='', unsupported_issue=False):
        nonlocal unsupported
        if code not in reasons:
            reasons.append(code)
        issues.append(f'{code}: {detail}' if detail else code)
        unsupported |= unsupported_issue

    total = 0
    for name in BUNDLE_FILES:
        try:
            if isinstance(bundle, (str, Path)):
                path = Path(bundle) / name
                if path.is_symlink():
                    raise ValueError('symlink inputs are not accepted')
                if path.stat().st_size > max_file_bytes:
                    issue('BUNDLE_TOO_LARGE', name)
                    continue
                content = path.read_bytes()
            else:
                content = bundle[name]
                if isinstance(content, Path):
                    if content.is_symlink() or content.stat().st_size > max_file_bytes:
                        raise ValueError('linked or oversized input file')
                    content = content.read_bytes()
            if not isinstance(content, bytes):
                raise ValueError('bundle values must be bytes')
            total += len(content)
            if len(content) > max_file_bytes or total > max_total_bytes:
                issue('BUNDLE_TOO_LARGE', name)
                continue
            raw[name] = content
            hashes[name] = hashlib.sha256(content).hexdigest()
        except (KeyError, FileNotFoundError):
            issue('FILE_MISSING:' + name)
        except (OSError, ValueError, TypeError) as exc:
            issue('DATA_INVALID', f'{name}: {exc}')
    if expected_hashes is not None:
        for name, digest in expected_hashes.items():
            if name not in hashes or hashes[name] != digest:
                issue('DATA_INVALID', f'fingerprint mismatch: {name}')
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def read_json(name):
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f'duplicate JSON field: {key}')
                result[key] = value
            return result
        try:
            value = json.loads(raw.get(name, b'{}'), object_pairs_hook=unique_object)
            if not isinstance(value, dict):
                raise ValueError('JSON object required')
            return value
        except (ValueError, UnicodeError) as exc:
            issue('DATA_INVALID', f'{name}: {exc}')
            return {}

    metadata = read_json('metadata.json')
    required = ('dataset_schema_version instrument_id symbol exchange security_type currency timezone source '
                'source_version retrieved_at coverage_start coverage_end adjustment corporate_actions_complete '
                'completeness_source').split()
    for key in required:
        if key not in metadata or metadata[key] in ('', None):
            issue('METADATA_MISSING:' + key)
    if metadata.get('dataset_schema_version') != 2:
        issue('DATASET_SCHEMA_UNSUPPORTED', unsupported_issue=True)
    if metadata.get('adjustment') != 'raw':
        issue('RAW_PRICES_REQUIRED')
    if not metadata.get('price_basis_source'):
        issue('RAW_PRICE_PROVENANCE_MISSING')
    if metadata.get('volume_unit') != 'shares':
        issue('VOLUME_UNIT_MISSING')
    if metadata.get('corporate_actions_complete') is not True or not metadata.get('completeness_source'):
        issue('CORPORATE_ACTION_COVERAGE_MISSING')
    if (metadata.get('exchange') not in ('SSE', 'SZSE') or
            metadata.get('security_type') != 'common_stock' or metadata.get('currency') != 'CNY' or
            metadata.get('timezone') != 'Asia/Shanghai'):
        issue('MARKET_PROFILE_UNSUPPORTED', 'only explicit SSE/SZSE CNY common stock coverage', True)
    start = end = None
    try:
        start, end = _day(metadata['coverage_start']), _day(metadata['coverage_end'])
        _time(metadata['retrieved_at'])
        if start > end:
            raise ValueError('coverage reversed')
    except (KeyError, ValueError, TypeError) as exc:
        issue('DATA_INVALID', f'metadata dates: {exc}')

    def rows(name, columns):
        try:
            reader = csv.DictReader(io.StringIO(raw.get(name, b'').decode('utf-8-sig')))
            if reader.fieldnames is None or not set(columns.split()).issubset(reader.fieldnames):
                issue('COLUMNS_MISSING:' + name)
                return []
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError('duplicate CSV column')
            maximum = 10000 if name == 'actions.csv' else 100000
            result = []
            for row in reader:
                if len(result) >= maximum:
                    raise ValueError(f'CSV exceeds {maximum} rows')
                result.append(row)
            if any(None in row or any(value is None for value in row.values()) for row in result):
                raise ValueError('malformed CSV row')
            return result
        except (UnicodeError, csv.Error, ValueError) as exc:
            issue('DATA_INVALID', f'{name}: {exc}')
            return []

    bars, sessions, actions, tradability = [], [], [], []
    def parse_records(name, columns, constructor, target):
        for index, row in enumerate(rows(name, columns), 2):
            try:
                target.append(constructor(row))
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                issue('DATA_INVALID', f'{name}:{index}: {exc}')

    parse_records('bars.csv', 'session Open High Low Close Volume',
                  lambda r: Bar(r['session'], *(_number(r[k]) for k in ('Open','High','Low','Close','Volume'))), bars)
    parse_records('sessions.csv', 'session open_at close_at is_open',
                  lambda r: MarketSession(r['session'], r['open_at'], r['close_at'], _boolean(r['is_open'])), sessions)
    parse_records('actions.csv', 'action_id kind effective_session available_at pay_session cash_per_old_share split_ratio',
                  lambda r: CorporateAction(r['action_id'],r['kind'],r['effective_session'],r['available_at'],
                                            r['pay_session'] if r['pay_session'] not in ('','null') else None,
                                            _optional_number(r['cash_per_old_share']), _optional_number(r['split_ratio'])), actions)
    parse_records('tradability.csv', 'session available_at can_buy_open can_sell_open limit_up limit_down reference_close reason',
                  lambda r: Tradability(r['session'],r['available_at'],_boolean(r['can_buy_open'],True),
                                        _boolean(r['can_sell_open'],True),_optional_number(r['limit_up']),
                                        _optional_number(r['limit_down']),_optional_number(r['reference_close']),r['reason']), tradability)
    for records, label in ((bars,'bars'), (sessions,'sessions'), (tradability,'tradability')):
        keys = [r.session for r in records]
        if keys != sorted(set(keys)):
            issue('DATA_INVALID', f'{label} sessions must be unique and increasing')
        for key in keys:
            try:
                day = _day(key)
                if start and end and not start <= day <= end:
                    raise ValueError('outside coverage')
            except (ValueError, TypeError) as exc:
                issue('DATA_INVALID', f'{label} session: {exc}')
    calendar = {r.session:r for r in sessions}
    prices = {r.session:r for r in bars}
    statuses = {r.session:r for r in tradability}
    if start and end and (end-start).days > 100000:
        issue('DATA_INVALID', 'coverage exceeds supported calendar span')
    if start and end and (end-start).days <= 100000:
        current = start
        while current <= end:
            if current.isoformat() not in calendar:
                issue('SESSION_COVERAGE_MISSING', current.isoformat())
            current += timedelta(days=1)
    if not bars:
        issue('RAW_PRICES_MISSING')
    for bar in bars:
        if min(bar.open,bar.high,bar.low,bar.close) <= 0 or bar.volume < 0 or not (
                bar.low <= min(bar.open,bar.close) <= max(bar.open,bar.close) <= bar.high):
            issue('DATA_INVALID', f'invalid OHLCV: {bar.session}')
        if bar.session not in calendar or not calendar[bar.session].is_open:
            issue('DATA_INVALID', 'bar has no open session')
    try:
        market_timezone = ZoneInfo(metadata.get('timezone', ''))
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        market_timezone = None
        issue('DATA_INVALID', 'unknown market timezone')
    previous_close = None
    for session in sessions:
        try:
            if _time(session.open_at) >= _time(session.close_at):
                raise ValueError('session times reversed')
            if market_timezone is not None and any(_time(value).astimezone(market_timezone).date().isoformat() != session.session
                                                  for value in (session.open_at, session.close_at)):
                raise ValueError('opening/closing local date differs from session')
            if previous_close is not None and _time(session.open_at) <= previous_close:
                raise ValueError('session timestamps overlap')
            previous_close = _time(session.close_at)
        except (ValueError, TypeError) as exc:
            issue('DATA_INVALID', str(exc))
        if session.is_open:
            status = statuses.get(session.session)
            if status is None:
                issue('TRADABILITY_MISSING', session.session)
            if session.session not in prices:
                if not status or status.reason != 'suspended' or status.can_buy_open is not False or status.can_sell_open is not False:
                    issue('SESSION_PRICE_GAP_UNEXPLAINED', session.session)
                if not any(b.session < session.session for b in bars):
                    issue('VALUATION_PRICE_MISSING', session.session)
    for status in tradability:
        try:
            if status.session not in calendar:
                raise ValueError('status outside calendar')
            if _time(status.available_at) > _time(calendar[status.session].open_at):
                issue('TRADABILITY_NOT_KNOWN_AT_OPEN', status.session)
        except (ValueError, TypeError) as exc:
            issue('DATA_INVALID', str(exc))
        if status.can_buy_open is None or status.can_sell_open is None:
            issue('TRADABILITY_DIRECTION_UNKNOWN', status.session)
        if any(x is None or x <= 0 for x in (status.limit_up,status.limit_down,status.reference_close)):
            issue('PRICE_LIMIT_DATA_MISSING', status.session)
        elif not status.limit_down <= status.reference_close <= status.limit_up:
            issue('DATA_INVALID', 'price limits inconsistent')
    ids = set()
    for action in actions:
        if not action.action_id or action.action_id in ids:
            issue('DUPLICATE_ACTION', action.action_id)
        ids.add(action.action_id)
        try:
            _day(action.effective_session)
            if action.effective_session not in calendar or not calendar[action.effective_session].is_open:
                issue('ACTION_SESSION_MISSING', action.action_id)
            elif _time(action.available_at) > _time(calendar[action.effective_session].open_at):
                issue('ACTION_NOT_KNOWN_AT_EFFECTIVE_OPEN', action.action_id)
            if action.pay_session and _day(action.pay_session) < _day(action.effective_session):
                raise ValueError('payment before entitlement')
        except (ValueError, TypeError) as exc:
            issue('DATA_INVALID', f'action {action.action_id}: {exc}')
        if metadata.get('action_share_basis') != 'old_shares':
            issue('ACTION_SHARE_BASIS_MISSING')
        if action.kind == 'split':
            if action.split_ratio is None or action.split_ratio < 1 or not action.split_ratio.is_integer():
                issue('CORPORATE_ACTION_UNSUPPORTED', action.action_id, True)
            if action.pay_session is not None or action.cash_per_old_share is not None:
                issue('DATA_INVALID', 'non-applicable split fields must be null')
        elif action.kind == 'cash_dividend':
            if action.pay_session is None:
                issue('DIVIDEND_PAYMENT_MISSING', action.action_id)
            if action.cash_per_old_share is None or action.cash_per_old_share < 0 or action.split_ratio is not None:
                issue('DATA_INVALID', 'invalid dividend fields')
        else:
            issue('CORPORATE_ACTION_UNSUPPORTED', action.kind, True)
        if action.effective_session not in prices:
            status = statuses.get(action.effective_session)
            if not status or status.reference_close is None or not metadata.get('suspension_valuation_source'):
                issue('SUSPENSION_ACTION_VALUATION_MISSING', action.action_id)
    profile_data = read_json('market-profile.json')
    profile = None
    try:
        profile = MarketProfile(**profile_data)
        if not profile.rule_sources or not all(isinstance(s,str) and s.strip() for s in profile.rule_sources):
            raise ValueError('rule sources required')
        if any(getattr(profile,k) != metadata.get(k) for k in ('exchange','security_type','currency')):
            raise ValueError('profile instrument scope mismatch')
        if not start or not end or _day(profile.effective_start) > start or _day(profile.effective_end) < end:
            raise ValueError('profile does not cover dates')
        if type(profile.buy_lot) is not int or profile.buy_lot <= 0:
            raise ValueError('positive integer lot required')
        _boolean(profile.sell_odd_lot)
        _boolean(profile.t_plus_one)
        for key in ('commission_rate','commission_min','sell_tax_rate','other_fee_rate','dividend_tax_rate','slippage','participation_cap'):
            value = getattr(profile,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or _number(value) < 0:
                raise ValueError(f'invalid {key}')
        if not 0 < profile.participation_cap <= 1 or profile.slippage >= 1 or profile.dividend_tax_rate > 1:
            raise ValueError('invalid execution fraction')
        if any(getattr(profile, key) >= 1 for key in ('commission_rate', 'sell_tax_rate', 'other_fee_rate')):
            raise ValueError('proportional trading rates must be below one')
        if profile.fee_model != 'proportional_minimum' or not profile.profile_id or not profile.version:
            raise ValueError('unsupported fee model or missing profile identity')
        next_start = _day(profile.effective_start)
        for interval in profile.fee_intervals:
            if _day(interval.effective_start) != next_start or _day(interval.effective_end) < next_start:
                raise ValueError('fee interval gap, overlap or unordered range')
            next_start = _day(interval.effective_end) + timedelta(days=1)
            if not interval.rule_sources or not all(isinstance(s,str) and s.strip() for s in interval.rule_sources):
                raise ValueError('interval rule sources required')
            for key in ('commission_rate','commission_min','sell_tax_rate','other_fee_rate','dividend_tax_rate','slippage','participation_cap'):
                value = getattr(interval,key)
                if isinstance(value,bool) or not isinstance(value,(int,float)) or _number(value) < 0:
                    raise ValueError(f'invalid interval {key}')
            if not 0 < interval.participation_cap <= 1 or interval.slippage >= 1 or interval.dividend_tax_rate > 1:
                raise ValueError('invalid interval execution fraction')
            if any(getattr(interval, key) >= 1 for key in ('commission_rate', 'sell_tax_rate', 'other_fee_rate')):
                raise ValueError('interval proportional trading rates must be below one')
        if profile.fee_intervals and next_start != _day(profile.effective_end) + timedelta(days=1):
            raise ValueError('fee intervals do not cover full profile')
    except (TypeError, ValueError) as exc:
        issue('MARKET_PROFILE_UNSUPPORTED', str(exc), True)
    qualification = 'unsupported' if unsupported else 'incomplete' if reasons else 'ready'
    dataset = None
    if qualification == 'ready':
        dataset = MarketDatasetV2(metadata,tuple(bars),tuple(sessions),tuple(actions),tuple(tradability),profile,hashes,fingerprint)
    return MarketDatasetPreview(qualification,tuple(reasons),tuple(issues),hashes,fingerprint,dataset)


def load_market_bundle(bundle, **kwargs):
    preview = validate_market_bundle(bundle, **kwargs)
    if preview.dataset is None:
        code = ('DATA_INVALID' if 'DATA_INVALID' in preview.blocking_reasons else
                'MARKET_PROFILE_UNSUPPORTED' if preview.qualification == 'unsupported' else 'DATA_CONTRACT_INCOMPLETE')
        raise MarketDataError(code, preview)
    return preview.dataset
