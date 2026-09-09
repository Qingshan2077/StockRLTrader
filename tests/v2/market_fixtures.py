"""Explicit synthetic inputs; these are not verified exchange rules."""
import csv
import io
import json


def synthetic_bundle():
    metadata = dict(dataset_schema_version=2, instrument_id='SSE:600000', symbol='600000', exchange='SSE',
                    security_type='common_stock', currency='CNY', timezone='Asia/Shanghai',
                    source='synthetic fixture', source_version='1', retrieved_at='2024-01-05T18:00:00+08:00',
                    coverage_start='2024-01-02', coverage_end='2024-01-04', adjustment='raw',
                    corporate_actions_complete=True, completeness_source='synthetic complete ledger',
                    price_basis_source='synthetic raw generator', volume_unit='shares',
                    action_share_basis='old_shares')
    profile = dict(profile_id='synthetic', version='1', exchange='SSE', security_type='common_stock',
                   currency='CNY', effective_start='2024-01-01', effective_end='2024-12-31', buy_lot=100,
                   sell_odd_lot=True, t_plus_one=True, commission_rate=0, commission_min=0,
                   sell_tax_rate=0, other_fee_rate=0, dividend_tax_rate=0, slippage=0,
                   participation_cap=0.1, rule_sources=['synthetic fixture'], fee_model='proportional_minimum')
    result = {'metadata.json': json.dumps(metadata).encode(), 'market-profile.json': json.dumps(profile).encode()}
    def table(name, fields, rows):
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(fields.split(','))
        writer.writerows(rows)
        result[name] = stream.getvalue().encode()
    dates = ['2024-01-02', '2024-01-03', '2024-01-04']
    table('bars.csv', 'session,Open,High,Low,Close,Volume', [[d,100,101,99,100,10000] for d in dates])
    table('sessions.csv', 'session,open_at,close_at,is_open',
          [[d,d+'T09:30:00+08:00',d+'T15:00:00+08:00','true'] for d in dates])
    table('actions.csv', 'action_id,kind,effective_session,available_at,pay_session,cash_per_old_share,split_ratio', [])
    table('tradability.csv', 'session,available_at,can_buy_open,can_sell_open,limit_up,limit_down,reference_close,reason',
          [[d,d+'T09:00:00+08:00','true','true',110,90,100,'normal'] for d in dates])
    return result


def change_json(bundle, name, **changes):
    value = json.loads(bundle[name])
    value.update(changes)
    bundle[name] = json.dumps(value).encode()
