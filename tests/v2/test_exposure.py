from stockrl.research.contracts import ExposureRecord
from stockrl.research.report import exposure_qualification
from test_research_verdict import fixture


def test_exposure_uses_instrument_and_actual_overlap_not_dataset_or_protocol_id():
    p,_=fixture()
    record=ExposureRecord(instrument_id='A',start_session='2020-01-01',end_session='2020-01-02',scope='exposed_test',reason='seen',recorded_at='2026-01-01',source='previous copy',protocol_id='different')
    assert exposure_qualification(p,[record])=='exploratory'
    assert exposure_qualification(p,[record.model_copy(update={'instrument_id':'B'})])=='declared_holdout'
    assert exposure_qualification(p,[record.model_copy(update={'end_session':'2019-12-31','start_session':'2019-01-01'})])=='declared_holdout'
    assert exposure_qualification(p,[record.model_copy(update={'scope':'whole_period_knowledge'})])=='exploratory'
