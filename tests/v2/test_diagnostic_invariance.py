import copy
import random
import numpy as np
from stockrl.research.diagnostics import summarize_diagnostics


def test_diagnostics_does_not_mutate_decisions_or_rng():
    rows=[{'raw_action':1.2,'clipped_action':1,'requested_weight':1,'fees':.4,'nav':99.6,'filled_qty':1}]
    before=copy.deepcopy(rows)
    random.seed(41); np.random.seed(42)
    py=random.getstate(); numeric=np.random.get_state()
    summarize_diagnostics(rows)
    assert rows==before
    assert random.getstate()==py
    after=np.random.get_state()
    assert after[0]==numeric[0]
    assert np.array_equal(after[1],numeric[1])
    assert after[2:]==numeric[2:]
