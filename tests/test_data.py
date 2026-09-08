import numpy as np
import pandas as pd
import pytest
from stockrl.data import load_csv, make_demo_data
from stockrl.features import build_features, ObservationNormalizer


def test_csv_selects_ohlcv_and_preserves_prices(tmp_path):
    path = tmp_path / 'bars.csv'
    path.write_text('Date,Open,High,Low,Close,Volume,Adj Close\n2024-01-01,100,110,90,105,1000,50\n2024-01-02,105,115,95,110,2000,51\n')
    frame = load_csv(path)
    assert list(frame.columns) == ['Open','High','Low','Close','Volume']
    assert frame.iloc[0]['Close'] == 105
    assert frame.index.name == 'Date'


@pytest.mark.parametrize('data', [
    'bad,100,110,90,105,1000',
    '2024-01-01,100,99,90,105,1000',
    '2024-01-01,100,110,90,105,-1',
    '2024-01-01,0,110,0,105,1000',
    '2024-01-01,100,110,90,nan,1000',
    '2024-01-01,100,110,90,105,1000\n2024-01-01,100,110,90,105,1000',
    '2024-01-02,100,110,90,105,1000\n2024-01-01,100,110,90,105,1000',
])
def test_csv_rejects_invalid_input_explanatorily(tmp_path, data):
    path = tmp_path/'bad.csv'
    path.write_text('Date,Open,High,Low,Close,Volume\n'+data+'\n')
    with pytest.raises(ValueError, match='Date|date|OHLC|Volume|finite|positive'): load_csv(path)


def test_causal_features_cannot_see_changed_future():
    frame = make_demo_data(n=100)
    altered = frame.copy()
    altered.iloc[50:, :4] *= 10
    altered.iloc[50:, 4] *= 20
    original = build_features(frame)
    pd.testing.assert_frame_equal(original.iloc[:50], build_features(altered).iloc[:50])
    assert original.index.equals(frame.index)
    assert np.isfinite(original.to_numpy()).all()


def test_normalizer_fits_only_supplied_training_rows_and_roundtrips(tmp_path):
    train = pd.DataFrame({'x':[1.,3.], 'constant':[5.,5.]})
    normalizer = ObservationNormalizer.fit(train)
    result = normalizer.transform(pd.DataFrame({'x':[5.], 'constant':[5.]}))
    assert result.iloc[0,0] == 3
    assert result.iloc[0,1] == 0
    path = tmp_path/'normalizer.json'
    normalizer.save(path)
    pd.testing.assert_frame_equal(result, ObservationNormalizer.load(path).transform(pd.DataFrame({'x':[5.], 'constant':[5.]})))
    assert abs(normalizer.transform(pd.DataFrame({'x':[1e100], 'constant':[5.]})).iloc[0,0]) <= 10
    with pytest.raises(ValueError): normalizer.transform(pd.DataFrame({'wrong':[1.]}))


def test_demo_repeatability_and_validity(tmp_path):
    frame = make_demo_data(n=30, seed=5)
    pd.testing.assert_frame_equal(frame, make_demo_data(n=30, seed=5))
    assert not frame.equals(make_demo_data(n=30, seed=6))
    path = tmp_path/'demo.csv'
    frame.to_csv(path)
    assert len(load_csv(path)) == 30


def test_saved_market_prices_reload_bit_for_bit(tmp_path):
    frame = make_demo_data(n=20,seed=97)
    path = tmp_path/'market.csv'
    frame.to_csv(path,float_format='%.17g')
    np.testing.assert_array_equal(load_csv(path).to_numpy(),frame.to_numpy())
