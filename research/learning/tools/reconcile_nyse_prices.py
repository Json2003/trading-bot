"""Offline data reconciliation; no rankings, strategy or return calculations."""
from pathlib import Path
from datetime import datetime
from decimal import Decimal,ROUND_HALF_UP
from zoneinfo import ZoneInfo
from collections import Counter
import argparse
import json,hashlib,html,re
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--secondary',type=Path,required=True)
parser.add_argument('--original',type=Path,required=True)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--decisions',type=Path,required=True)
args=parser.parse_args()
root=args.secondary
original=args.original
freeze=json.loads((original/'freeze.json').read_text())
expected=json.loads(args.manifest.read_text())
assert expected['freeze']==freeze
for record in expected['daily_source_inventory']:
 b=(original/record['path']).read_bytes()
 assert len(b)==record['bytes'] and hashlib.sha256(b).hexdigest()==record['sha256']
secondary={}
for s in freeze['universe']:
 alias='XYZ' if s=='SQ' else s
 p=root/('yahoo-'+alias+'.json');b=p.read_bytes();x=json.loads(b)['chart']['result'][0]
 assert x['meta']['symbol']==alias
 q=x['indicators']['quote'][0];records={}
 for i,t in enumerate(x['timestamp']):
  d=datetime.fromtimestamp(t,ZoneInfo('America/New_York')).date().isoformat()
  assert d not in records
  records[d]={k:v[i] for k,v in q.items()}
 secondary[s]=records
prices={}
for record in expected['daily_source_inventory']:
 for row in json.loads((original/record['path']).read_text())['rows']:
  key=(row['act_symbol'],row['date'])
  assert key not in prices
  prices[key]=row
fields=('open','high','low','close','volume');matches=Counter();mismatch=Counter();secondary_invalid=[];missing_secondary=[];bad_reconciled=[];all_days=[x['session'] for x in expected['calendar']['sessions']]
by_stock={}
volume_bands=Counter();large_volume=[];price_disagreements=[]
def value(raw,field,symbol):
 v=Decimal(str(raw));factor=Decimal(5) if symbol=='NOW' else Decimal(1)
 return v/factor if field=='volume' else (v*factor).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
for s in freeze['universe']:
 same=Counter();different=Counter()
 for d in all_days:
  r=secondary[s].get(d)
  if r is None:missing_secondary.append([s,d]);continue
  v={k:value(r[k],k,s) for k in fields}
  valid=all(x.is_finite() and x>0 for x in v.values()) and v['low']<=min(v['open'],v['close'])<=max(v['open'],v['close'])<=v['high'] and v['volume']==v['volume'].to_integral_value()
  if not valid:secondary_invalid.append([s,d])
  primary=prices.get((s,d))
  if primary:
   agree={k:abs(v[k]-Decimal(primary[k]))<=(Decimal(0) if k=='volume' else Decimal('.01')) for k in fields}
   delta=abs(v['volume']-Decimal(primary['volume']))
   volume_bands['equal' if delta==0 else 'nonzero_within_100_shares' if delta<=100 else 'over_100_shares']+=1
   relative=delta/max(v['volume'],Decimal(primary['volume']))
   if relative>Decimal('.01'):large_volume.append({'symbol':s,'session':d,'absolute_difference_divided_by_larger_volume':str(relative)})
   if any(not agree[k] for k in fields if k!='volume'):price_disagreements.append({'symbol':s,'session':d,'fields':[k for k in fields if k!='volume' and not agree[k]]})
   for k,ok in agree.items():
    (matches if ok else mismatch)[k]+=1;(same if ok else different)[k]+=1
   if [s,d] in expected['coverage']['invalid_ohlcv_cells']:
    bad_reconciled.append({'symbol':s,'session':d,'secondary_bar_passes_ohlcv':bool(valid),'fields_agree_with_original_within_one_cent':agree,'original_unchanged':True})
 by_stock[s]={'secondary_sessions':len(secondary[s]),'overlap_sessions':sum(s2==s for s2,d in prices),'fields_agree_within_one_cent':dict(same),'field_disagreements':dict(different)}
inventory=[]
for s in freeze['universe']:
 alias='XYZ' if s=='SQ' else s;p=root/('yahoo-'+alias+'.json');b=p.read_bytes()
 inventory.append({'historical_ticker':s,'provider_request_symbol':alias,'path':p.name,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'url':'https://query1.finance.yahoo.com/v8/finance/chart/'+alias+'?period1=1606780800&period2=1643846400&interval=1d&events=div%2Csplits&includeAdjustedClose=true','publication_cutoff_proved':False,'events_in_requested_interval':json.loads(b)['chart']['result'][0].get('events',{})})
cash_evidence=json.loads((root/'cash-dividend-evidence.json').read_text())
for item in cash_evidence:
 for src in item['source_records']:
  p=Path(src['path']) if src['path'].startswith('market-reconciliation/../') else root/src['path']
  b=p.read_bytes()
  assert len(b)==src['bytes'] and hashlib.sha256(b).hexdigest()==src['sha256']
  if src['path'].endswith('-annual.htm'):
   text=re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',b.decode())))
   assert item['cash_dividend_evidence']['quote'] in text
  if src['path'].startswith('market-reconciliation/../'):
   src['path']='prior-acquisition/sec/'+Path(src['path']).name
for src in json.loads((root/'action-primary-index.json').read_text()):
 b=(root/src['path']).read_bytes()
 assert len(b)==src['bytes'] and hashlib.sha256(b).hexdigest()==src['sha256']
assert len(cash_evidence)==12 and {x['symbol'] for x in cash_evidence}==set(freeze['universe'])
result={'schema':'nyse_price_reconciliation_v1','audit_date':'2026-11-18','scope':freeze,'basis':{'price_tolerance_usd':'.01','volume_tolerance':0,'now_normalization':'Multiply historical OHLC by five and divide historical volume by five to undo the documented December 2025 split for comparison only. Not a 2021 action.','sq_alias':'Request current XYZ, preserve historical SQ and its class; alias evidence is a January 2025 issuer/SEC record. Not a 2021 ticker change.','source_independence':'Separately retrieved Yahoo/CSI historical endpoint versus Dolt archive. Dolt upstream provenance is unknown; full upstream independence is not established.','secondary_retrieval_cutoff':'2026-11-18; not historical 2022 availability','rights':'Yahoo source data not republished here. Only audit counts, field-agreement flags, URLs and hashes retained; raw prices remain outside Git.'},'coverage':{'secondary_expected_cells':len(all_days)*12,'secondary_observed_cells':sum(len(x) for x in secondary.values()),'secondary_missing_cells':missing_secondary,'secondary_invalid_cells':secondary_invalid,'original_missing_cells_present_in_secondary':sum(d in secondary[s] for s,d in expected['coverage']['missing_cells']),'original_invalid_rows_reconciled':bad_reconciled,'volume_absolute_difference_bands':dict(volume_bands),'volume_difference_over_one_percent':large_volume,'price_disagreements':price_disagreements,'overlap_field_agreements':dict(matches),'overlap_field_disagreements':dict(mismatch),'per_stock':by_stock},'cash_dividend_evidence':cash_evidence,'primary_action_sources':json.loads((root/'action-primary-index.json').read_text()),'source_inventory':inventory,'status':'crosscheck_complete_not_authorized_runner_input','runner_ready':False,'original_data_modified':False,'strategy_trials':0,'strategy_metrics':None,'timing_gap_resolved':False,'corporate_action_completeness_resolved':False}
decisions=json.loads(args.decisions.read_text())
assert set(decisions)=={'timing_resolution','corporate_action_resolution','failed_acquisition_probes','primary_method_sources','scope_limit'}
result.update(decisions)
args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'coverage':{k:v for k,v in result['coverage'].items() if k not in ('per_stock','volume_difference_over_one_percent','original_invalid_rows_reconciled','price_disagreements')},'strategy_trials':0,'runner_ready':False}))
