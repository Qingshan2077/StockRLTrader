from stockrl.research.baselines import match_fixed_weight


def test_matching_uses_validation_grid_and_lower_tie():
    calls = []
    def evaluate(w):
        calls.append(w)
        return w*.2
    result = match_fixed_weight(.105, evaluate)
    assert calls == [i/20 for i in range(21)]
    assert result['weight'] == .5
    assert abs(result['calibration_error']-.005) < 1e-12
