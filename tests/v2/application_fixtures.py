"""Small offline research fixture with real calendar boundaries and synthetic quotes."""
import csv
import io
import json

import pandas as pd

from stockrl.research.contracts import ResearchProtocolDraft, TrainingBudget
from market_fixtures import synthetic_bundle


def research_files():
    files = synthetic_bundle()
    days = pd.date_range('2014-12-01', '2022-01-03', freq='D')
    meta = json.loads(files['metadata.json'])
    meta.update(coverage_start='2014-12-01', coverage_end='2022-01-03', retrieved_at='2022-01-04T18:00:00+08:00')
    profile = json.loads(files['market-profile.json'])
    profile.update(effective_start='2014-01-01', effective_end='2023-01-01', buy_lot=1, participation_cap=1)
    files['metadata.json'], files['market-profile.json'] = json.dumps(meta).encode(), json.dumps(profile).encode()
    def table(name, fields, rows):
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(fields.split(','))
        writer.writerows(rows)
        files[name] = stream.getvalue().encode()
    sessions = [day.date().isoformat() for day in days]
    opens = [day.date().isoformat() for day in days if day.weekday() < 5]
    table('sessions.csv', 'session,open_at,close_at,is_open',
          [[day, day+'T09:30:00+08:00', day+'T15:00:00+08:00', str(day in opens).lower()] for day in sessions])
    table('bars.csv', 'session,Open,High,Low,Close,Volume', [[day, 100, 101, 99, 100, 100000] for day in opens])
    table('tradability.csv', 'session,available_at,can_buy_open,can_sell_open,limit_up,limit_down,reference_close,reason',
          [[day, day+'T09:00:00+08:00', 'true', 'true', 110, 90, 100, 'synthetic'] for day in opens])
    return files


def draft_for(dataset):
    return ResearchProtocolDraft(protocol_id='test-protocol', instrument_ids=('SSE:600000',),
        dataset_ids=(dataset['dataset_id'],), dataset_fingerprints={dataset['dataset_id']: dataset['fingerprint']},
        market_profile_ids=('synthetic',), market_profile_fingerprints={'synthetic': dataset['file_hashes']['market-profile.json']},
        asset_selection_note='Synthetic integration fixture, never financial evidence', first_test_session='2021-01-01',
        fold_count=1, seeds=(42,), training_budget=TrainingBudget(requested_timesteps=8,episode_length=20),checkpoint_budget=2)
