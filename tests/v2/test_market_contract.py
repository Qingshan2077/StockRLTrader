import importlib
import json

import pytest

from market_fixtures import synthetic_bundle, change_json


def validator():
    try:
        return importlib.import_module('stockrl.market.validation')
    except ModuleNotFoundError:
        pytest.fail('market bundle validation has not been implemented')


def test_ready_bundle_is_immutable_and_content_addressed():
    api = validator()
    bundle = synthetic_bundle()
    preview = api.validate_market_bundle(bundle)
    assert preview.qualification == 'ready', preview.issues
    dataset = api.load_market_bundle(bundle)
    assert dataset.bars[0].close == 100
    with pytest.raises(TypeError):
        dataset.metadata['source'] = 'changed'
    assert len(dataset.file_hashes) == 6
    bundle['metadata.json'] += b' '
    assert api.validate_market_bundle(bundle).fingerprint != preview.fingerprint


@pytest.mark.parametrize('mutation,reason,qualification', [
    (lambda b: b.pop('bars.csv'), 'FILE_MISSING:bars.csv', 'incomplete'),
    (lambda b: change_json(b,'metadata.json',currency='UNKNOWN'), 'MARKET_PROFILE_UNSUPPORTED', 'unsupported'),
    (lambda b: change_json(b,'metadata.json',corporate_actions_complete=False), 'CORPORATE_ACTION_COVERAGE_MISSING','incomplete'),
    (lambda b: change_json(b,'metadata.json',price_basis_source=''), 'RAW_PRICE_PROVENANCE_MISSING','incomplete'),
    (lambda b: change_json(b,'metadata.json',adjustment='forward'), 'RAW_PRICES_REQUIRED','incomplete'),
    (lambda b: b.update({'bars.csv':b['bars.csv'].replace(b',100,101,99,100,',b',0,101,99,100,')}), 'DATA_INVALID','incomplete'),
    (lambda b: b.update({'tradability.csv':b['tradability.csv'].replace(b'T09:00:',b'T10:00:')}), 'TRADABILITY_NOT_KNOWN_AT_OPEN','incomplete'),
    (lambda b: change_json(b,'market-profile.json',effective_end='2024-01-03'), 'MARKET_PROFILE_UNSUPPORTED','unsupported'),
])
def test_rejects_unsafe_research_inputs(mutation, reason, qualification):
    api = validator()
    bundle = synthetic_bundle()
    mutation(bundle)
    preview = api.validate_market_bundle(bundle)
    assert preview.qualification == qualification
    assert reason in preview.blocking_reasons
    with pytest.raises(api.MarketDataError):
        api.load_market_bundle(bundle)


def test_reports_all_missing_fields_and_missing_action_coverage():
    bundle = synthetic_bundle()
    bundle['metadata.json'] = json.dumps({'dataset_schema_version':1}).encode()
    preview = validator().validate_market_bundle(bundle)
    assert 'METADATA_MISSING:source' in preview.blocking_reasons
    assert 'METADATA_MISSING:currency' in preview.blocking_reasons
    assert 'CORPORATE_ACTION_COVERAGE_MISSING' in preview.blocking_reasons


def test_hash_size_and_duplicate_events_fail_closed():
    api = validator()
    bundle = synthetic_bundle()
    assert 'DATA_INVALID' in api.validate_market_bundle(bundle, expected_hashes={'bars.csv':'bad'}).blocking_reasons
    assert 'BUNDLE_TOO_LARGE' in api.validate_market_bundle(bundle, max_file_bytes=10).blocking_reasons
    row = b'a,split,2024-01-03,2024-01-02T09:00:00+08:00,,,2\r\n'
    bundle['actions.csv'] += row + row
    assert 'DUPLICATE_ACTION' in api.validate_market_bundle(bundle).blocking_reasons


def test_late_action_and_non_integer_split_are_not_ready():
    bundle = synthetic_bundle()
    bundle['actions.csv'] += b'a,split,2024-01-03,2024-01-03T10:00:00+08:00,,,1.5\r\n'
    preview = validator().validate_market_bundle(bundle)
    assert preview.qualification == 'unsupported'
    assert 'ACTION_NOT_KNOWN_AT_EFFECTIVE_OPEN' in preview.blocking_reasons
    assert 'CORPORATE_ACTION_UNSUPPORTED' in preview.blocking_reasons


def test_profile_resolves_historical_rates_and_rejects_gaps():
    bundle = synthetic_bundle()
    profile = json.loads(bundle['market-profile.json'])
    fee_fields = ('commission_rate','commission_min','sell_tax_rate','other_fee_rate',
                  'dividend_tax_rate','slippage','participation_cap','rule_sources')
    old = {k:profile[k] for k in fee_fields}
    old.update(effective_start='2024-01-01',effective_end='2024-01-02',sell_tax_rate=0.002)
    new = dict(old,effective_start='2024-01-03',effective_end='2024-12-31',sell_tax_rate=0.001)
    change_json(bundle,'market-profile.json',fee_intervals=[old,new])
    dataset = validator().load_market_bundle(bundle)
    assert dataset.market_profile.for_session('2024-01-02').sell_tax_rate == 0.002
    assert dataset.market_profile.for_session('2024-01-03').sell_tax_rate == 0.001
    with pytest.raises(ValueError):
        dataset.market_profile.for_session('2025-01-01')
    new['effective_start'] = '2024-01-04'
    change_json(bundle,'market-profile.json',fee_intervals=[old,new])
    assert validator().validate_market_bundle(bundle).qualification == 'unsupported'


def test_canonical_identity_does_not_trust_importer_alias():
    api = validator()
    metadata = {'exchange':'SSE','symbol':'600000','instrument_id':'alias'}
    assert api.canonical_instrument_id(metadata) == 'SSE:600000'
    with pytest.raises(ValueError):
        api.canonical_instrument_id(dict(metadata,symbol='../alias'))


def test_directory_load_hashes_bytes_and_serializes_without_mutable_aliases(tmp_path):
    from stockrl.market.contracts import to_dict
    bundle = synthetic_bundle()
    for name, content in bundle.items():
        (tmp_path/name).write_bytes(content)
    dataset = validator().load_market_bundle(tmp_path)
    exported = to_dict(dataset)
    exported['metadata']['source'] = 'modified'
    assert dataset.metadata['source'] == 'synthetic fixture'
    assert validator().load_market_bundle(bundle).fingerprint == dataset.fingerprint
    (tmp_path/'bars.csv').write_bytes(bundle['bars.csv'] + b'\n')
    with pytest.raises(validator().MarketDataError, match='DATA_INVALID'):
        validator().load_market_bundle(tmp_path, expected_hashes=dataset.file_hashes)


@pytest.mark.parametrize('content,reason', [
    (b'a,cash_dividend,2024-01-03,2024-01-02T09:00:00+08:00,,2,\r\n','DIVIDEND_PAYMENT_MISSING'),
    (b'a,merger,2024-01-03,2024-01-02T09:00:00+08:00,,,\r\n','CORPORATE_ACTION_UNSUPPORTED'),
])
def test_action_gaps_are_exposed(content, reason):
    bundle = synthetic_bundle()
    bundle['actions.csv'] += content
    assert reason in validator().validate_market_bundle(bundle).blocking_reasons


def test_missing_bar_requires_explicit_suspension_and_known_valuation():
    bundle = synthetic_bundle()
    lines = bundle['bars.csv'].splitlines(keepends=True)
    bundle['bars.csv'] = b''.join(lines[:2]+lines[3:])
    assert 'SESSION_PRICE_GAP_UNEXPLAINED' in validator().validate_market_bundle(bundle).blocking_reasons
    bundle['tradability.csv'] = bundle['tradability.csv'].replace(
        b'2024-01-03,2024-01-03T09:00:00+08:00,true,true,110,90,100,normal',
        b'2024-01-03,2024-01-03T09:00:00+08:00,false,false,110,90,100,suspended')
    assert validator().validate_market_bundle(bundle).qualification == 'ready'


def test_duplicate_json_fields_cannot_hide_adjustment_or_source():
    bundle = synthetic_bundle()
    bundle['metadata.json'] = bundle['metadata.json'][:-1] + b',"adjustment":"forward","adjustment":"raw"}'
    assert 'DATA_INVALID' in validator().validate_market_bundle(bundle).blocking_reasons
def test_session_close_cannot_overlap_next_execution():
    from market_fixtures import synthetic_bundle
    from stockrl.market.validation import validate_market_bundle
    files = synthetic_bundle()
    files['sessions.csv'] = files['sessions.csv'].replace(b'2024-01-02T15:00:00', b'2024-01-03T15:00:00')
    assert 'DATA_INVALID' in validate_market_bundle(files).blocking_reasons
