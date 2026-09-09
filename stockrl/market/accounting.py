"""One session transition: entitlement, split, payment, open fill, close NAV."""
from dataclasses import replace
import math
from .contracts import SessionOutcome
from .execution import execute_open
from .features import actions_for,split_factor,valuation_close


def advance_session(state,intent,session,bundle,profile=None):
    calendar=next(s for s in bundle.sessions if s.session==session)
    if not calendar.is_open:
        raise ValueError('execution requires open market session')
    rules=(profile or bundle.market_profile).for_session(session)
    actions=actions_for(bundle,session,calendar.open_at)
    receivables=dict(state.receivables)
    for action in actions:
        if action.kind=='cash_dividend':
            if action.pay_session is None or action.cash_per_old_share is None:
                raise ValueError('DIVIDEND_PAYMENT_MISSING')
            receivables[action.action_id]=state.shares*action.cash_per_old_share*(1-rules.dividend_tax_rate)
        elif action.kind!='split':
            raise ValueError('CORPORATE_ACTION_UNSUPPORTED')
    split=split_factor(actions)
    cash=state.cash
    ledger={a.action_id:a for a in bundle.actions}
    for action_id,amount in tuple(receivables.items()):
        if action_id not in ledger or ledger[action_id].pay_session is None:
            raise ValueError('DIVIDEND_PAYMENT_MISSING')
        if ledger[action_id].pay_session<=session:
            cash+=amount
            del receivables[action_id]
    # One call is one t -> t+1 transition, so every prior-close share settles now.
    state_at_open=replace(state,cash=cash,shares=state.shares*split,sellable_shares=state.shares*split,receivables=receivables)
    updated,fills,record=execute_open(state_at_open,intent,session,bundle,rules)
    close=valuation_close(bundle,session)
    nav=updated.cash+updated.shares*close+sum(updated.receivables.values())
    updated=replace(updated,nav=nav,peak_nav=max(state.peak_nav,nav))
    reward=math.log(max(nav,1e-300)/state.nav)
    return SessionOutcome(updated,fills,reward,record)
