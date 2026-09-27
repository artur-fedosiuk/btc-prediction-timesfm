# Reproduce audit diagnostics

These are audit characterization probes, not regression acceptance tests. Assertions confirm defects present at commit `3f6782ce8aef432df0e79852dc289c506bf465e5`. They do not establish correctness. All CSV mutations target temporary copies; entry-point I/O is mocked. No checkpoint is loaded by this script.

Save the following Python block to a temporary file, then run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 /path/to/temporary/probes.py
```

Requires the existing local Python environment. The future-date diagnostic uses the execution clock, so that timestamp changes on replay.

```python
"""Read-only audit probes; all mutations use temporary files or patched functions."""
import csv
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from evaluation import binance, coingecko, price_verifier, run_predict, run_verify
from evaluation.csv_manager import load_predictions, save_prediction, update_verification
from evaluation.predictor import PredictionResult, predict_with_timesfm
from evaluation.path_metrics import ForecastPathMetrics
from evaluation.trading.types import PredictionRecord, StrategyConfig, TradingCosts
from evaluation.trading.signal import generate_signal, generate_signals
from evaluation.trading.backtest_engine import run_backtest
from evaluation.trading.metrics import compute_sharpe_ratio, compute_sortino_ratio

logging.disable(logging.CRITICAL)
out = {}
utc = timezone.utc
t0 = datetime(2026, 9, 16, 12, tzinfo=utc)

with tempfile.TemporaryDirectory() as directory:
    p = Path(directory) / 'copy.csv'
    shutil.copyfile('data/bitcoin_predictions.csv', p)
    save_prediction({'timestamp_utc': '2026-09-26T12:00:00Z', 'experiment_version': 'timesfm_h24_v1', 'forecast_horizon': '24'}, p)
    try:
        load_predictions(p)
    except Exception as exc:
        out['legacy_append'] = {'error': type(exc).__name__, 'message': str(exc).strip()}
    assert out['legacy_append']['error'] == 'ParserError'

    p = Path(directory) / 'update.csv'
    shutil.copyfile('data/bitcoin_predictions.csv', p)
    before = list(csv.DictReader(p.open()))
    update_verification(before[0]['timestamp_utc'], 1.0, 2.0, 3.0, 'up', True, True, path=p)
    after = list(csv.DictReader(p.open()))
    out['verification_overwrite'] = {'old_actual': before[0]['actual_price_24h'], 'new_actual': after[0]['actual_price_24h'], 'columns_before': len(before[0]), 'columns_after': len(after[0]), 'legacy_fields_added': after[0]['experiment_version']}
    assert after[0]['actual_price_24h'] == '1.0'

prices = [(t0 + timedelta(hours=i+1), float(100+i)) for i in range(24)]
with patch.object(coingecko, 'get_btc_hourly_prices', return_value=prices), patch.object(binance, 'get_btc_hourly_prices', return_value=prices):
    v = price_verifier.get_verified_price(t0 + timedelta(hours=24))
    out['target_mismatch'] = {'stored_actual': v.price, 'true_endpoint': prices[-1][1], 'confidence': v.confidence}
    assert v.price == 111.5 and prices[-1][1] == 123

with patch.object(coingecko, 'get_btc_hourly_prices', return_value=prices[:1]), patch.object(binance, 'get_btc_hourly_prices', return_value=prices[:1]):
    v = price_verifier.get_verified_price(t0 + timedelta(hours=24))
    out['incomplete_verification'] = {'points_per_source': 1, 'confidence': v.confidence, 'price': v.price}
    assert v.confidence == 'high'

invalids = []
for value in [0, -1, 'NaN', 'Infinity']:
    with patch.object(coingecko, '_request_with_retry', return_value={'bitcoin': {'usd': value}}):
        invalids.append(str(coingecko.get_current_btc_price()))
out['invalid_spot_accepted'] = invalids
raw = {'prices': [[int(t0.timestamp()*1000), 100], [int(t0.timestamp()*1000), -1], [int((t0+timedelta(days=999)).timestamp()*1000), 200]]}
with patch.object(coingecko, '_request_with_retry', return_value=raw):
    out['invalid_history_accepted'] = [(t.isoformat(), p) for t,p in coingecko.get_btc_history()]

klines = [[int((t0+timedelta(hours=i)).timestamp()*1000), '100', '120', '90', str(100+i), '20', int((t0+timedelta(hours=i+1)).timestamp()*1000)-1] for i in range(25)]
with patch.object(binance, '_request_with_retry', return_value=klines):
    rows = binance.get_btc_hourly_prices(t0,24)
    out['binance_window'] = {'points': len(rows), 'last_label': rows[-1][0].isoformat(), 'last_close_time': datetime.fromtimestamp(klines[-1][6]/1000,utc).isoformat(), 'target': (t0+timedelta(hours=24)).isoformat()}
    assert len(rows) == 25

path = list(np.arange(101.,125.))
q = np.stack([np.arange(91.,115.)+i for i in range(9)],axis=-1)
fake = SimpleNamespace(predict_batch=lambda *a,**kw: iter([SimpleNamespace(forecast=np.array(path),quantiles=q)]))
with patch('timesfm3.TimesFM3Evaluator', return_value=fake) as ctor:
    result = predict_with_timesfm(np.arange(600.)+100)
    out['adapter_index_mapping'] = {'path_length': len(result.forecast_path), 't1': result.forecast_path[0], 't24': result.forecast, 'lower_t24':result.min_price,'upper_t24':result.max_price,'checkpoint': ctor.call_args.args[0].checkpoint_path}
    assert result.forecast == 124 and result.forecast_path[0] == 101

synthetic = PredictionResult(100,99,101,'timesfm', forecast_path=[100.]*24,min_path=[99.]*24,max_path=[101.]*24)
acceptance = {}
for label, end in [('future',datetime.now(utc)+timedelta(days=5)),('stale',t0)]:
    history = [(end-timedelta(hours=39-i),float(100+i)) for i in range(40)]
    with patch.object(sys,'argv',['audit','--use-timesfm']), patch.object(run_predict,'is_experiment_active',return_value=True), patch.object(run_predict,'get_current_btc_price',return_value=100), patch.object(run_predict,'get_btc_history',return_value=history), patch.object(run_predict,'predict_with_timesfm',return_value=synthetic) as predict, patch.object(run_predict,'save_prediction') as save:
        rc=run_predict.main()
        acceptance[label]={'exit_code':rc,'model_called':predict.called,'save_called':save.called,'history_end':end.isoformat()}
        assert rc==0 and save.called
out['cutoff_and_freshness'] = acceptance

out['nonfinite_path_accepted'] = str(ForecastPathMetrics.calculate(100,[float('nan')]*24).t24)

def pred(pid,ts,entry=100,forecast=110,actual=120):
    return PredictionRecord(pid,ts,entry,forecast,90,130,actual)
out['flat_zero_threshold'] = generate_signal(pred('flat',t0.isoformat(),forecast=100), StrategyConfig(threshold_pct=0)).direction
assert out['flat_zero_threshold']=='LONG'
ps=[pred('one','2026-09-16T12:00:00Z'),pred('two','2026-09-16T13:00:00Z')]
bt=run_backtest(ps,StrategyConfig(threshold_pct=0),TradingCosts(0,0,0,0))
out['future_equity_position_size'] = [{'entry':t.entry_timestamp,'exit':t.exit_timestamp,'position':t.position_size,'net_pnl':t.net_pnl} for t in bt.trades]
assert bt.trades[1].position_size==1020
dup=run_backtest([pred('dup',t0.isoformat(),actual=110),pred('dup',(t0+timedelta(days=1)).isoformat(),actual=90)],StrategyConfig(threshold_pct=0))
out['duplicate_id_exit'] = [t.exit_price for t in dup.trades]
assert out['duplicate_id_exit']==[90,90]
signals=generate_signals([pred('one','2026-09-16T12:00:00Z'),pred('two','2026-09-17T13:00:00+02:00')],StrategyConfig(threshold_pct=0,position_mode='single'))
out['timezone_overlap']=[s.direction for s in signals]
assert out['timezone_overlap']==['LONG','LONG']
out['small_sample_ratios']={'n':2,'sharpe':compute_sharpe_ratio([1,2]),'sortino':str(compute_sortino_ratio([1,2]))}

df=load_predictions().iloc[[0]].copy()
df.loc[:,'actual_price_24h']=''
with patch.object(run_verify,'load_predictions',return_value=df), patch.object(run_verify,'get_verified_price',side_effect=RuntimeError('sources unavailable')), patch.object(run_verify,'generate_report'):
    out['verification_failure_exit_code']=run_verify.main()
    assert out['verification_failure_exit_code']==0

from chat_app.app import generate_future_dates,app
out['crypto_calendar']=generate_future_dates('2026-09-25',2)
out['chat_root_status']=app.test_client().get('/').status_code

rows=list(csv.DictReader(open('data/bitcoin_predictions.csv')))
r=next(r for r in rows if r['timestamp_utc']=='2026-09-19T15:43:25Z')
entry=float(r['initial_price']); forecast=float(r['predicted_price_24h']); actual=float(r['actual_price_24h'])
rec=pred('real',r['timestamp_utc'],entry,forecast,actual)
t=run_backtest([rec],StrategyConfig(threshold_pct=0)).trades[0]
out['real_row_reconciliation']={'timestamp':r['timestamp_utc'],'entry':entry,'forecast':forecast,'actual_stored_twap':actual,'forecast_return_pct':(forecast-entry)/entry*100,'actual_return_pct':(actual-entry)/entry*100,'python_position':t.position_size,'python_gross':t.gross_pnl,'python_fees':t.fees,'python_spread':t.spread_cost,'python_slippage':t.slippage_cost,'python_net':t.net_pnl,'dashboard_signal_log_net_using_rounded_demo_price':(-((80824.83-entry)/entry*100)-.35)*10000/100}
verified=[r for r in rows if r['actual_price_24h']]
model_mae=np.mean([abs(float(r['predicted_price_24h'])-float(r['actual_price_24h'])) for r in verified])
naive_mae=np.mean([abs(float(r['initial_price'])-float(r['actual_price_24h'])) for r in verified])
out['legacy_only_diagnostic_not_edge']={'n':len(verified),'timesfm_vs_wrong_twap_target_mae':float(model_mae),'naive_vs_wrong_twap_target_mae':float(naive_mae),'mean_abs_predicted_return_pct':float(np.mean([abs(float(r['predicted_price_24h'])/float(r['initial_price'])-1)*100 for r in rows]))}
print(json.dumps(out,indent=2,allow_nan=False))
```
