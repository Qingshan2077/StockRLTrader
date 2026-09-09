import pytest
from dataclasses import replace
from stockrl.market.contracts import AccountState
from tests.v2.test_corporate_actions import bundle,intent


def test_commission_once_and_bidirectional_slippage():
    from stockrl.market.accounting import advance_session
    b=bundle((50,50,50),commission_rate=.001)
    out=advance_session(AccountState(1000,0,0,{},1000,1000),intent(.5),'2024-01-03',b)
    assert out.state.cash==499.5 and out.state.nav==999.5
    assert out.fills[0].commission==.5
    for weight,shares,cash,price in [(1,0,1000,100.1),(0,10,0,99.9)]:
        out=advance_session(AccountState(cash,shares,shares,{},1000,1000),intent(weight),'2024-01-03',bundle((100,100,100),slippage=.001))
        assert out.fills[0].price==pytest.approx(price)


def test_direction_limits_and_missing_quote_do_not_fill():
    from stockrl.market.accounting import advance_session
    b=bundle((100,110,110))
    state=AccountState(1000,0,0,{},1000,1000)
    out=advance_session(state,intent(1),'2024-01-03',b)
    assert out.fills==() and out.decision_record.no_fill_reason=='BUY_AT_LIMIT_UP'
    b=replace(b,bars=(b.bars[0],b.bars[2]))
    out=advance_session(state,intent(1),'2024-01-03',b)
    assert out.fills==() and out.decision_record.no_fill_reason=='NO_OPEN_QUOTE'


def test_minimum_fee_affordability_and_lots():
    from stockrl.market.accounting import advance_session
    b=bundle((100,100,100),commission_min=5)
    b=replace(b,market_profile=replace(b.market_profile,buy_lot=10))
    out=advance_session(AccountState(1000,0,0,{},1000,1000),intent(1),'2024-01-03',b)
    assert out.fills==() and out.state.cash==1000

def test_t_plus_one_direct_execution_and_next_session_settlement():
    from stockrl.market.execution import execute_open
    from stockrl.market.accounting import advance_session
    b=bundle((100,100,100))
    state=AccountState(0,10,0,{},1000,1000)
    out,fills,record=execute_open(state,intent(0),'2024-01-03',b)
    assert not fills and record.no_fill_reason=='T_PLUS_ONE_OR_LOT'
    settled=advance_session(state,intent(0),'2024-01-03',b)
    assert settled.state.shares==0 and settled.state.cash==1000


def test_sell_odd_lot_and_dated_fee_profile():
    from stockrl.market.accounting import advance_session
    from stockrl.market.contracts import MarketFeeInterval
    b=bundle((100,100,100))
    interval=MarketFeeInterval('2024-01-03','2024-01-03',.01,0,.02,.005,0,0,1,('synthetic',))
    b=replace(b,market_profile=replace(b.market_profile,buy_lot=100,fee_intervals=(interval,)))
    out=advance_session(AccountState(0,7,7,{},700,700),intent(0),'2024-01-03',b)
    assert out.state.cash==675.5 and out.state.shares==0
    assert out.fills[0].commission==7 and out.fills[0].sell_tax==14


def test_suspension_action_requires_reference_and_marks_stale():
    from stockrl.market.accounting import advance_session
    from stockrl.market.features import session_view
    from tests.v2.test_corporate_actions import dividend
    b=bundle(actions=[dividend()])
    b=replace(b,bars=(b.bars[0],b.bars[2]))
    with pytest.raises(ValueError,match='SUSPENSION_ACTION_VALUATION_MISSING'):
        advance_session(AccountState(0,10,10,{},1000,1000),None,'2024-01-03',b)
    statuses=tuple(replace(s,reference_close=98) if s.session=='2024-01-03' else s for s in b.tradability)
    b=replace(b,metadata=dict(b.metadata,suspension_valuation_source='audited fixture'),tradability=statuses)
    out=advance_session(AccountState(0,10,10,{},1000,1000),None,'2024-01-03',b)
    assert out.state.nav==1000 and session_view(b).stale.iloc[1]
