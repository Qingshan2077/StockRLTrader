"""Version two market contracts. All sessions use ISO calendar dates."""
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Mapping, Any


class FrozenDict(dict):
    """JSON/pickle friendly immutable copy of an input mapping."""
    def _immutable(self, *args, **kwargs):
        raise TypeError('immutable mapping')
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __reduce__(self):
        return FrozenDict, (dict(self),)


def freeze(value):
    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def to_dict(value):
    if is_dataclass(value):
        return {field.name: to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_dict(item) for item in value]
    return value


@dataclass(frozen=True)
class Bar:
    session: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MarketSession:
    session: str
    open_at: str
    close_at: str
    is_open: bool


@dataclass(frozen=True)
class CorporateAction:
    action_id: str
    kind: str
    effective_session: str
    available_at: str
    pay_session: str | None = None
    cash_per_old_share: float | None = None
    split_ratio: float | None = None


@dataclass(frozen=True)
class Tradability:
    session: str
    available_at: str
    can_buy_open: bool | None
    can_sell_open: bool | None
    limit_up: float | None
    limit_down: float | None
    reference_close: float | None
    reason: str


@dataclass(frozen=True)
class MarketFeeInterval:
    effective_start: str
    effective_end: str
    commission_rate: float
    commission_min: float
    sell_tax_rate: float
    other_fee_rate: float
    dividend_tax_rate: float
    slippage: float
    participation_cap: float
    rule_sources: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, 'rule_sources', tuple(self.rule_sources))


@dataclass(frozen=True)
class MarketProfile:
    profile_id: str
    version: str
    exchange: str
    security_type: str
    currency: str
    effective_start: str
    effective_end: str
    buy_lot: int
    sell_odd_lot: bool
    t_plus_one: bool
    commission_rate: float
    commission_min: float
    sell_tax_rate: float
    other_fee_rate: float
    dividend_tax_rate: float
    slippage: float
    participation_cap: float
    rule_sources: tuple[str, ...]
    fee_model: str = 'proportional_minimum'
    fee_intervals: tuple[MarketFeeInterval, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'rule_sources', tuple(self.rule_sources))
        object.__setattr__(self, 'fee_intervals', tuple(
            MarketFeeInterval(**item) if isinstance(item, dict) else item for item in self.fee_intervals))

    def for_session(self, session: str):
        if not self.effective_start <= session <= self.effective_end:
            raise ValueError('market profile does not cover session')
        if not self.fee_intervals:
            return self
        matches = [item for item in self.fee_intervals if item.effective_start <= session <= item.effective_end]
        if len(matches) != 1:
            raise ValueError('market fee interval coverage is ambiguous or missing')
        return replace(self, **to_dict(matches[0]), fee_intervals=())


@dataclass(frozen=True)
class MarketDatasetV2:
    metadata: Mapping[str, Any]
    bars: tuple[Bar, ...]
    sessions: tuple[MarketSession, ...]
    actions: tuple[CorporateAction, ...]
    tradability: tuple[Tradability, ...]
    market_profile: MarketProfile
    file_hashes: Mapping[str, str]
    fingerprint: str

    def __post_init__(self):
        for name in ('metadata', 'file_hashes'):
            object.__setattr__(self, name, freeze(dict(getattr(self, name))))
        for name in ('bars', 'sessions', 'actions', 'tradability'):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class MarketDatasetPreview:
    qualification: str
    blocking_reasons: tuple[str, ...]
    issues: tuple[str, ...]
    file_hashes: Mapping[str, str]
    fingerprint: str
    dataset: MarketDatasetV2 | None = None

    def __post_init__(self):
        object.__setattr__(self, 'file_hashes', freeze(dict(self.file_hashes)))


@dataclass(frozen=True)
class AccountState:
    cash: float
    shares: int
    sellable_shares: int
    receivables: Mapping[str, float]
    nav: float
    peak_nav: float

    def __post_init__(self):
        object.__setattr__(self, 'receivables', freeze(dict(self.receivables)))


@dataclass(frozen=True)
class DecisionIntent:
    target_weight: float
    decision_session: str
    decision_at: str
    input_fingerprint: str


@dataclass(frozen=True)
class Fill:
    session: str
    side: str
    requested_qty: int
    filled_qty: int
    price: float
    commission: float
    sell_tax: float
    other_fees: float


@dataclass(frozen=True)
class DecisionRecord:
    intent: DecisionIntent | None
    requested_qty: int
    filled_qty: int
    no_fill_reason: str | None


@dataclass(frozen=True)
class SessionOutcome:
    state: AccountState
    fills: tuple[Fill, ...]
    reward: float
    decision_record: DecisionRecord
