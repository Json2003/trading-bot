# Friday research brief to model feature

The weekly brief nominates a measurable hypothesis; its prose, conclusions,
and Yes/Maybe/No recommendation are **not** model inputs or trade signals.
Each nomination records the paper/report URL, publication date, exact proposed
measurement, venue, instrument, data source and availability lag, direction of
the hypothesis, and a frozen test plan. Keep the brief and the candidate data
version together in the research artifact. No unverified report may edit an
active model tag or inference threshold.

The current repository has an older LSTM inference API in `scripts/infer_api.py`.
Its `scripts/train_lstm.py` import of `data.feature_store` is absent from this
checkout. This pipeline therefore evaluates a separate research model; it does
not claim to retrain or deploy the LSTM. A later integration must repair and
validate that training path, register the feature order and source version in
the model artifact, and require the same feature schema at inference time.

## Measured input contract

Supply one candidate at a time. Store its brief nomination as a JSON file with
`candidate_id`, `source_url`, `published_at`, `measurement`, `formula`,
`data_source`, `symbol`, `venue`, and `max_age_hours`. The evaluation artifact
retains that record and input SHA-256 hashes. A measurement CSV has
`timestamp,available_at,value`. `timestamp` is when the underlying market
observation occurred; `available_at` is the first instant the specific numeric
value was available to the bot. Both require explicit timezone offsets.
Duplicate observation timestamps are rejected, as revisions need an explicit
point-in-time history. The feature is joined to a decision only if already
available and within a frozen maximum age. Missing or stale values have a
separate flag; coverage below 80% fails. Preserve the source raw data, release
schedule, transformations, and checksums with the result.

Example candidates from future briefs include funding-rate surprise,
liquidation imbalance, bid/ask depth imbalance, and cross-venue price or flow
divergence. Each needs its own measured feed and reproducible formula. A paper's
claimed effect or a text sentiment score cannot stand in for the feed.

For a diagnostic comparison, supply bars with `decision_ts`, numeric baseline
feature columns, and `next_bar_executable_return`. The latter must be measured
from the next candle's executable entry after the decision, with realistic
fills; close-to-close returns are invalid. Run:

```bash
python scripts/evaluate_microstructure_feature.py \
  --candidate path/to/friday_candidate.json \
  --bars path/to/decision_bars.csv \
  --measurements path/to/point_in_time_measurements.csv \
  --baseline-columns momentum volatility \
  --output artifacts/research/microstructure_candidate.json
```

The comparison fits separate baseline and augmented logistic models on the
first 60% of chronological rows, uses the next 20% for validation, and reports
the final 20% as a diagnostic holdout. Scaling fits on training data only. The
fixed threshold is 0.55 and each selected entry pays 86 bps. The metric is an
uncompounded sum over hypothetical single-bar entries, not portfolio PnL.

This is a screening tool. Promotion requires an experiment frozen before new
data arrives, non-overlapping forward confirmation with enough independent
events per block, realistic fill and funding accounting, positive net outcomes
under the stress model, and a separate reviewed inference integration. Neither
the script nor the Friday brief changes orders, risk controls, leverage, the
active model, or promotion state.
