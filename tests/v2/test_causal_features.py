from dataclasses import replace
import numpy as np
from tests.v2.test_corporate_actions import bundle, dividend


def test_future_mutations_do_not_change_features_or_training_normalizer(tmp_path):
    from stockrl.market.features import build_features, NormalizerV2
    b=bundle(actions=[dividend()])
    future=replace(b,bars=(*b.bars[:2],replace(b.bars[2],open=200,high=200,low=200,close=200)))
    a,c=build_features(b),build_features(future)
    assert a.iloc[:2].equals(c.iloc[:2])
    fit=NormalizerV2.fit(a.iloc[:2],input_fingerprint='train')
    other=NormalizerV2.fit(c.iloc[:2],input_fingerprint='train')
    assert np.array_equal(fit.mean,other.mean)
    fit.save(tmp_path/'n.json')
    assert NormalizerV2.load(tmp_path/'n.json').input_fingerprint=='train'


def test_split_adjusted_volume_capacity():
    from stockrl.market.contracts import CorporateAction
    from stockrl.market.features import historical_capacity
    split=CorporateAction('s','split','2024-01-03','2024-01-02T12:00:00+08:00',split_ratio=2)
    b=bundle((100,50,50),[split])
    assert historical_capacity(b,'2024-01-02','2024-01-03')==2000
    b=replace(b,bars=(b.bars[0],replace(b.bars[1],volume=999999),b.bars[2]))
    assert historical_capacity(b,'2024-01-02','2024-01-03')==2000

def test_future_missing_suspension_reference_cannot_break_prior_execution():
    from dataclasses import replace
    from stockrl.market.contracts import CorporateAction,AccountState
    from stockrl.market.accounting import advance_session
    from tests.v2.test_corporate_actions import intent
    b=bundle((100,100,50))
    split=CorporateAction('future','split','2024-01-04','2024-01-03T12:00:00+08:00',split_ratio=2)
    b=replace(b,bars=b.bars[:2],actions=(split,))
    out=advance_session(AccountState(1000,0,0,{},1000,1000),intent(1),'2024-01-03',b)
    assert out.state.nav==1000 and out.state.shares==10
