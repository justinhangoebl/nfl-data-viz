"""Local inference server script with a minimal `predict` implementation.

This `predict` returns the last-observed (x,y) for each (game_id, play_id, nfl_id)
found in `test_input`. It satisfies the evaluation API contract and is useful for
local end-to-end testing with `run_local_gateway`.

Run locally from the repo root:

    python inference.py

Or to simulate Kaggle rerun behavior (not needed locally):

    set KAGGLE_IS_COMPETITION_RERUN=1; python inference.py

"""
from __future__ import annotations

import os
from typing import Union

import pandas as pd
import polars as pl

import kaggle_evaluation.nfl_inference_server


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> Union[pl.DataFrame, pd.DataFrame]:
    """Minimal predict implementation.

    For each row in `test` (one row per prediction request) we look up the
    last-observed x,y for that player in `test_input` and return that as the
    prediction. This returns a Polars DataFrame with columns ['x','y'] and the
    same number of rows as `test`.

    This is intentionally simple so it can be used as a baseline and to verify
    the end-to-end evaluation wiring locally.
    """

    # Normalize to pandas for simple index-based lookups
    if isinstance(test_input, pl.DataFrame):
        ti = test_input.to_pandas()
    else:
        ti = test_input

    if isinstance(test, pl.DataFrame):
        test_pd = test.to_pandas()
    else:
        test_pd = test

    # Build a mapping from (game_id, play_id, nfl_id) -> last observed (x,y)
    # Use the row with the max frame_id per player as the latest observation.
    key_cols = ['game_id', 'play_id', 'nfl_id']
    if not set(key_cols).issubset(set(ti.columns)):
        # Fallback: return zeros if input doesn't have expected structure
        preds = pd.DataFrame({'x': [0.0] * len(test_pd), 'y': [0.0] * len(test_pd)})
        return pl.from_pandas(preds)

    # Ensure frame_id exists; otherwise just take the last occurrence order-wise
    if 'frame_id' in ti.columns:
        sort_cols = key_cols + ['frame_id']
        ti_sorted = ti.sort_values(sort_cols)
        last_obs = ti_sorted.groupby(key_cols).last().reset_index()
    else:
        last_obs = ti.groupby(key_cols).last().reset_index()

    # Create lookup dict
    lookup = {}
    for _, row in last_obs.iterrows():
        k = (row['game_id'], row['play_id'], row['nfl_id'])
        lookup[k] = (float(row.get('x', 0.0)), float(row.get('y', 0.0)))

    preds_x = []
    preds_y = []
    for _, r in test_pd.iterrows():
        k = (r['game_id'], r['play_id'], r['nfl_id'])
        x, y = lookup.get(k, (0.0, 0.0))
        preds_x.append(x)
        preds_y.append(y)

    preds = pd.DataFrame({'x': preds_x, 'y': preds_y})

    # Return Polars DataFrame for performance / consistency
    return pl.from_pandas(preds)


if __name__ == '__main__':
    # Wire up the inference server and run locally using the gateway's paths
    inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

    # If running on Kaggle rerun, the environment variable will be set and serve() will block
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        # Local path to the published competition data (matches README/data layout)
        data_path = os.path.join('data', 'nfl-big-data-bowl-2026-prediction')
        inference_server.run_local_gateway((data_path,))
