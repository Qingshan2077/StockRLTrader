from dataclasses import replace
import pytest
from stockrl.market.contracts import AccountState, CorporateAction, DecisionIntent
from stockrl.market.validation import load_market_bundle
from tests.v2.market_fixtures import synthetic_bundle


def bundle(prices=(100,98,98), actions=(), **fees):
    result = load_market_bundle(synthetic_bundle())
    bars = tuple(replace(b,open=p,close=p,high=p,low=p) for b,p in zip(result.bars,prices))
    return replace(result,bars=bars,actions=tuple(actions),market_profile=replace(result.market_profile,buy_lot=1,**fees))


def dividend(pay='2024-01-04'):
    return CorporateAction('d','cash_dividend','2024-01-03','2024-01-02T12:00:00+08:00',pay,2,None)


def intent(weight):
    return DecisionIntent(weight,'2024-01-02','2024-01-02T15:00:00+08:00','test')


def test_dividend_receivable_then_payment_conserve_nav():
    from stockrl.market.accounting import advance_session
    b = bundle(actions=[dividend()])
    state = AccountState(0,10,10,{},1000,1000)
    ex = advance_session(state,None,'2024-01-03',b)
    assert (ex.state.cash,ex.state.shares,dict(ex.state.receivables),ex.state.nav)==(0,10,{'d':20},1000)
    paid = advance_session(ex.state,None,'2024-01-04',b)
    assert (paid.state.cash,dict(paid.state.receivables),paid.state.nav)==(20,{},1000)
    assert paid.reward == 0


def test_no_spending_receivables_and_old_share_basis():
    from stockrl.market.accounting import advance_session
    split = CorporateAction('s','split','2024-01-03','2024-01-02T12:00:00+08:00',split_ratio=2)
    b = bundle((100,49,49),[dividend(),split])
    out = advance_session(AccountState(0,10,10,{},1000,1000),intent(1),'2024-01-03',b)
    assert out.state.shares == 20 and out.state.nav == 1000
    assert out.state.receivables == {'d':20} and out.fills == ()


def test_net_tax_and_gross_feature_are_distinct():
    from stockrl.market.accounting import advance_session
    from stockrl.market.features import build_feature_prices
    b=bundle(actions=[dividend()],dividend_tax_rate=.1)
    out=advance_session(AccountState(0,10,10,{},1000,1000),None,'2024-01-03',b)
    assert out.state.nav == 998 and out.state.receivables['d']==18
    assert build_feature_prices(b).Close.iloc[1]==100


def test_unknown_payment_and_fractional_split_rejected():
    from stockrl.market.accounting import advance_session
    for action in [dividend(None),CorporateAction('s','split','2024-01-03','2024-01-02T12:00:00+08:00',split_ratio=1.5)]:
        with pytest.raises(ValueError):
            advance_session(AccountState(0,10,10,{},1000,1000),None,'2024-01-03',bundle(actions=[action]))

def test_split_only_conservation_and_payment_on_closed_date_next_open():
    from stockrl.market.accounting import advance_session
    split=CorporateAction('s','split','2024-01-03','2024-01-02T12:00:00+08:00',split_ratio=2)
    out=advance_session(AccountState(0,10,10,{},1000,1000),None,'2024-01-03',bundle((100,50,50),[split]))
    assert out.state.shares==20 and out.state.nav==1000 and not out.fills
    b=bundle(actions=[dividend('2024-01-03')])
    b=replace(b,sessions=(b.sessions[0],b.sessions[1],replace(b.sessions[2],session='2024-01-08',open_at='2024-01-08T09:30:00+08:00',close_at='2024-01-08T15:00:00+08:00')),
              bars=(b.bars[0],b.bars[1],replace(b.bars[2],session='2024-01-08')),actions=(replace(dividend(),pay_session='2024-01-06'),))
    ex=advance_session(AccountState(0,10,10,{},1000,1000),None,'2024-01-03',b)
    paid=advance_session(ex.state,None,'2024-01-08',b)
    assert paid.state.cash==20 and paid.state.nav==1000 and not paid.state.receivables
