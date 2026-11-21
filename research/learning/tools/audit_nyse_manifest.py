"""Offline audit of the fixed November 18 data acquisition; never runs a strategy.

Usage: python research/learning/tools/audit_nyse_manifest.py RAW_DIR OUTPUT_JSON
Raw inputs are the byte-exact Dolt responses and SEC cover extraction evidence
listed in the output inventory. No downloads, orders or model imports occur.
"""
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from sec_cover_rows import read_cover_rows

CALENDAR_URL = 'https://ir.theice.com/press/news-details/2019/NYSE-Group-Announces-2020-2021-and-2022-Holiday-and-Early-Closings-Calendar/default.aspx'
HOLIDAYS = set('2020-12-25 2021-01-01 2021-01-18 2021-02-15 2021-04-02 2021-05-31 2021-07-05 2021-09-06 2021-11-25 2021-12-24 2022-01-17'.split())
EARLY = {'2020-12-24', '2021-11-26'}


def audit(root):
    def read(path):
        return json.loads((root / path).read_text())

    def verify(record):
        raw = (root / record['path']).read_bytes()
        assert len(raw) == record['bytes'], record['path']
        assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['path']

    freeze = read('freeze.json')
    symbols = freeze['universe']
    assert len(symbols) == len(set(symbols)) == 12
    first = date.fromisoformat(freeze['history_start'])
    last = date.fromisoformat(freeze['settlement_calendar_through'])
    weekdays, sessions, calendar = [], [], []
    d = first
    zone = ZoneInfo('America/New_York')
    while d <= last:
        day = d.isoformat()
        if d.weekday() < 5:
            weekdays.append(day)
            if day not in HOLIDAYS:
                sessions.append(day)
                calendar.append({'session': day,
                    'open_at': datetime(d.year, d.month, d.day, 9, 30, tzinfo=zone).isoformat(),
                    'close_at': datetime(d.year, d.month, d.day, 13 if day in EARLY else 16, tzinfo=zone).isoformat()})
        d += timedelta(days=1)
    inventory = read('daily-source-audit.json')
    assert sorted(r['date'] for r in inventory) == weekdays, 'incomplete query batch'
    assert not any('error' in r for r in inventory), 'query failures'
    counts, found, bad, duplicate, unexpected = Counter(), set(), [], [], []
    invalid_rows = []
    for record in inventory:
        verify(record)
        response = read(record['path'])
        assert response['query_execution_status'] == 'Success'
        assert len(response['rows']) == record['rows']
        for row in response['rows']:
            key = (row['act_symbol'], row['date'])
            if key in found:
                duplicate.append(key)
            found.add(key)
            if key[0] not in symbols or key[1] != record['date'] or key[1] not in sessions:
                unexpected.append(key)
            try:
                o, h, l, c = (Decimal(row[k]) for k in ('open', 'high', 'low', 'close'))
                v = Decimal(row['volume'])
                valid = all(x.is_finite() and x > 0 for x in (o, h, l, c, v))
                valid = valid and l <= min(o, c) <= max(o, c) <= h and v == v.to_integral_value()
            except (InvalidOperation, TypeError, KeyError):
                valid = False
            if not valid:
                bad.append(key)
                invalid_rows.append(row)
            counts[key[0]] += 1
    missing = [(s, d) for s in symbols for d in sessions if (s, d) not in found]
    securities = read('security-audit.json')
    assert {s['symbol'] for s in securities} == set(symbols)
    security_results = []
    for security in securities:
        assert 'error' not in security
        for record in security['source_records']:
            verify(record)
        filing_path = next(r['path'] for r in security['source_records'] if r['path'].endswith('-filing.htm'))
        matches = [r for r in read_cover_rows(root / filing_path) if r['symbol'] == security['symbol']]
        assert matches == security['cover_matches']
        assert len(matches) == 1
        cover = matches[0]
        assert cover['exchange'] in ('New York Stock Exchange', 'The New York Stock Exchange')
        assert 'common stock' in cover['title'].lower()
        filing = security['selected_filing']
        assert filing['filingDate'] <= '2021-12-31'
        security_results.append({'ticker': security['symbol'], 'cik': security['cik'],
            'title': cover['title'], 'exchange': cover['exchange'],
            'filing_accession': filing['accessionNumber'], 'filing_date': filing['filingDate'],
            'catalog_reported_acceptance': filing['acceptanceDateTime'],
            'source_records': security['source_records'],
            'permanent_security_id': None, 'complete_membership_interval': None})
    metadata = read('sources.json')
    for record in metadata:
        verify(record)
    controls = read('missing-session-controls.json')
    assert sorted(r['date'] for r in controls) == sorted({d for _, d in missing})
    for record in controls:
        verify(record)
        assert int(read(record['path'])['rows'][0]['n']) == record['rows_all_symbols']
    action_counts = {k: len(read(k + '.json')['rows']) for k in ('dividends', 'splits')}
    version = read('version.json')['rows']
    assert len(version) == 1 and version[0]['commit_hash'] == freeze['raw_data_revision']
    start, end = freeze['evaluation_start'], freeze['evaluation_end']
    evaluation = [d for d in sessions if start <= d <= end]
    warmup = [d for d in sessions if d < start]
    following = [d for d in sessions if d > end]
    blockers = [
        {'id': 'PIT_AVAILABILITY', 'scope': 'development',
         'finding': 'The pinned February 3, 2022 snapshot postdates every January decision. Per-row original publication/first-seen timestamps are absent. The Dolt log time has no UTC offset. Current acquisition times cannot be backdated to bar closes.',
         'resolution': 'Obtain verified decision-admissible source vintages and timestamp semantics for the warmup and evaluation rows, or start a separately frozen prospective diagnostic with observed receipt timestamps.'},
        {'id': 'ACTION_COMPLETENESS', 'scope': 'development',
         'finding': 'Zero dividend and split rows are a provider observation, not a complete action ledger. No independent interval-wide reconciliation of cash/stock actions, renames, class continuity, exits and terminal proceeds was completed. Prior audit found a real omitted DDS dividend outside this twelve-name scope.',
         'resolution': 'Reconcile all twelve classes across the full warmup/evaluation interval against primary action notices; retain confirmed no-action findings and receipt/pay dates.'},
        {'id': 'PRICE_BASIS', 'scope': 'development',
         'finding': 'Vendor raw-versus-adjusted OHLC convention and an as-of total-return momentum index are unverified. Coherent OHLC does not establish either.',
         'resolution': 'Document upstream price adjustment conventions and construct a causal total-return series from the reconciled action ledger without future adjustments.'},
        {'id': 'SECURITY_CONTINUITY', 'scope': 'development',
         'finding': 'SEC covers establish common-stock classification at twelve filing dates, not permanent class IDs or complete exchange eligibility intervals. CIK identifies a filer.',
         'resolution': 'Assign documented class-level IDs and verify listing/class continuity, including any exits; do not treat ticker strings or CIKs as permanent security IDs.'},
        {'id': 'UNIVERSE_SELECTION', 'scope': 'external_validity',
         'finding': 'This is a deliberately narrow, retrospectively named twelve-stock diagnostic, not a point-in-time census or a survivorship-neutral NYSE sample.',
         'resolution': 'Keep the fixed scope and claims narrow. A later broad-universe confirmation needs independently frozen point-in-time membership and delisted securities.'},
        {'id': 'EXECUTABLE_QUOTES', 'scope': 'executable_evidence',
         'finding': 'Daily OHLCV provides no order-time bid/ask, quote size, auction eligibility or halt state. ATR and raw opens are not executable quotes.',
         'resolution': 'After admissible signals freeze intended whole-share orders, verify the relevant following-session quotes/auction data, sizes and halts; retain unfilled orders.'},
        {'id': 'RIGHTS', 'scope': 'distribution',
         'finding': 'Public access does not establish upstream market-data redistribution rights; no new license approval is claimed.',
         'resolution': 'Verify upstream provider terms before redistribution or deployment. Raw market files remain outside Git.'},
    ]
    if missing or bad or duplicate or unexpected:
        blockers.insert(0, {'id': 'PRICE_INTEGRITY', 'scope': 'development', 'finding': 'Coverage/integrity exceptions retained below; no symbol removed.', 'resolution': 'Resolve source exceptions without changing the frozen universe.'})
    return {'schema': 'nyse_market_data_audit_v1', 'audit_date': '2026-11-18',
        'status': 'audit_complete_dataset_blocked', 'runner_ready': False,
        'runner_manifest': None, 'reason_no_runner_manifest': 'Unverified required fields are not fabricated to satisfy the runner schema.',
        'freeze': freeze, 'calendar': {'source': CALENDAR_URL, 'published': '2019-12-09',
            'timezone': 'America/New_York', 'sessions': calendar,
            'decision_at': None, 'decision_policy': 'Not assigned until causal feed availability is established.'},
        'coverage': {'queried_weekdays': len(weekdays), 'expected_sessions': len(sessions),
            'holiday_control_queries': len(weekdays)-len(sessions), 'expected_cells': len(sessions)*len(symbols),
            'observed_rows': sum(counts.values()), 'unique_cells': len(found),
            'per_symbol_rows': dict(sorted(counts.items())), 'missing_cells': missing,
            'invalid_ohlcv_cells': bad, 'invalid_ohlcv_source_rows': invalid_rows,
            'valid_ohlcv_rows': sum(counts.values()) - len(bad),
            'missing_session_controls': controls, 'duplicate_cells': duplicate, 'unexpected_cells': unexpected,
            'evaluation_sessions': len(evaluation), 'warmup_sessions_before_start': len(warmup),
            'settlement_sessions_after_end': following,
            'numeric_history_length_passes_runner_minimum': len(warmup) >= 254,
            'note': 'Numeric length does not establish admissible availability or complete total returns.'},
        'security_class_evidence': security_results, 'provider_action_rows': action_counts,
        'source_revision': version[0], 'source_revision_timezone': None,
        'retrieval_dates_are_not_historical_availability': True,
        'daily_source_url_template': inventory[0]['url'].replace(inventory[0]['date'], '{session}'),
        'daily_source_url_rule': 'Replace {session} with the inventory date to recover each exact query URL.',
        'daily_source_inventory': [{k: v for k, v in r.items() if k != 'url'} for r in inventory], 'metadata_source_inventory': metadata,
        'blockers': blockers,
        'evidence': {'audit_cycles': 1, 'strategy_investment_trials': 0,
            'strategy_evaluation': 'skipped_blocked_market_data', 'neural_training_runs': 0,
            'unseen_confirmations': 0, 'net_return': None, 'drawdown': None, 'sharpe': None,
            'profit_factor': None, 'trade_count': None, 'turnover': None, 'execution_costs': None,
            'expected_profit': None, 'operating_expenses': None,
            'expenses_note': 'No purchases made. Unknown data/hosting/broker expenses are not zero.'},
        'research_contract': {'cash_usd': 5000, 'whole_shares': True, 'position_entry_cap_usd': 1000,
            'stress_bps_round_trip': 86, 'stress_bps_each_side': 43, 'orders_allowed': False,
            'rule': 'Frozen monthly 12-minus-1 top-five versus retain-through-rank-ten; no strategy results evaluated.'}}


if __name__ == '__main__':
    result = audit(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'coverage': result['coverage'],
                      'blocker_ids': [x['id'] for x in result['blockers']]}))
