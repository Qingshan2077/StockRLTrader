"""Single open auction approximation under a supplied dated market profile."""
from dataclasses import replace
import math
from .contracts import Fill, DecisionRecord
from .features import historical_capacity, known


def execute_open(state,intent,session,bundle,profile=None):
    profile=(profile or bundle.market_profile).for_session(session)
    def empty(reason,requested=0):
        return state,(),DecisionRecord(intent,requested,0,reason)
    if intent is None:
        return empty('NO_ORDER')
    if not math.isfinite(intent.target_weight) or not 0<=intent.target_weight<=1:
        raise ValueError('target weight must be in [0,1]')
    calendar=next(s for s in bundle.sessions if s.session==session)
    if intent.decision_session>=session or not known(intent.decision_at,calendar.open_at):
        raise ValueError('decision must precede execution')
    bar=next((b for b in bundle.bars if b.session==session),None)
    if bar is None:
        return empty('NO_OPEN_QUOTE')
    nav=state.cash+state.shares*bar.open+sum(state.receivables.values())
    desired=(state.shares+math.floor(state.cash/bar.open/profile.buy_lot)*profile.buy_lot if intent.target_weight==1 else math.floor(intent.target_weight*nav/bar.open))
    delta=desired-state.shares
    requested=abs(delta)
    if not delta:
        return empty('AT_TARGET')
    side='buy' if delta>0 else 'sell'
    status=next((s for s in bundle.tradability if s.session==session),None)
    if not status or not known(status.available_at,calendar.open_at):
        return empty('TRADABILITY_UNKNOWN',requested)
    allowed=status.can_buy_open if side=='buy' else status.can_sell_open
    if allowed is not True:
        return empty('BUY_NOT_TRADABLE' if side=='buy' else 'SELL_NOT_TRADABLE',requested)
    if side=='buy' and status.limit_up is not None and bar.open>=status.limit_up:
        return empty('BUY_AT_LIMIT_UP',requested)
    if side=='sell' and status.limit_down is not None and bar.open<=status.limit_down:
        return empty('SELL_AT_LIMIT_DOWN',requested)
    capacity=historical_capacity(bundle,intent.decision_session,session,profile)
    quantity=min(requested,capacity)
    price=bar.open*(1+profile.slippage if side=='buy' else 1-profile.slippage)
    if side=='buy':
        affordable=min(state.cash/(price*(1+profile.commission_rate+profile.other_fee_rate)),max(0,state.cash-profile.commission_min)/(price*(1+profile.other_fee_rate)))
        quantity=min(quantity,math.floor(affordable+1e-12))
        quantity=quantity//profile.buy_lot*profile.buy_lot
    else:
        quantity=min(quantity,state.sellable_shares if profile.t_plus_one else state.shares)
        if not profile.sell_odd_lot:
            quantity=quantity//profile.buy_lot*profile.buy_lot
    if quantity<=0:
        reason='CAPACITY' if capacity==0 else ('INSUFFICIENT_CASH_OR_LOT' if side=='buy' else 'T_PLUS_ONE_OR_LOT')
        return empty(reason,requested)
    notional=quantity*price
    commission=max(profile.commission_min,notional*profile.commission_rate)
    tax=notional*profile.sell_tax_rate if side=='sell' else 0.
    other=notional*profile.other_fee_rate
    cash=state.cash+(-notional if side=='buy' else notional)-commission-tax-other
    if cash < -1e-9:
        return empty('INSUFFICIENT_CASH',requested)
    shares=state.shares+(quantity if side=='buy' else -quantity)
    sellable=state.sellable_shares-(quantity if side=='sell' else 0)
    if not profile.t_plus_one:
        sellable=shares
    updated=replace(state,cash=max(0.,cash),shares=shares,sellable_shares=sellable)
    fill=Fill(session,side,requested,quantity,price,commission,tax,other)
    return updated,(fill,),DecisionRecord(intent,requested,quantity,'PARTIAL_FILL' if quantity<requested else None)
