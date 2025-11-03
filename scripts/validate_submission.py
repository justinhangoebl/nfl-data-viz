"""Local submission validator for the NFL gateway

Usage:
    python scripts/validate_submission.py --submission submission.csv --data-dir data/nfl-big-data-bowl-2026-prediction

This script will:
 - Load your `submission.csv` (pandas CSV)
 - Iterate the gateway's `generate_data_batches()` and for each batch
   - extract the expected row ids
   - select matching rows from your submission (by `id`) in the same order
   - call the gateway's `competition_specific_validation()` to run the same checks used on Kaggle

Exit code: 0 on success, non-zero on validation failure.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import polars as pl

from kaggle_evaluation.nfl_gateway import NFLGateway
from kaggle_evaluation.core.base_gateway import GatewayRuntimeError


def to_pandas_series_first_col(obj):
    """Convert a polars or pandas single-column frame/series to a pandas Series preserving order."""
    if isinstance(obj, pl.DataFrame):
        df = obj.to_pandas()
        # take first column
        return df.iloc[:, 0]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    if isinstance(obj, pd.Series):
        return obj
    raise ValueError(f"Unsupported row_ids type: {type(obj)}")


def validate(submission_csv: Path, data_dir: Path) -> None:
    if not submission_csv.exists():
        print(f"Submission file not found: {submission_csv}")
        sys.exit(2)

    # Basic load
    sub = pd.read_csv(submission_csv)
    expected_cols = {"x", "y"}
    if "id" not in sub.columns:
        print("submission.csv must contain the `id` column matching test.csv's `id` values.")
        sys.exit(2)

    if not expected_cols.issubset(set(sub.columns)):
        print(f"submission.csv must include columns {expected_cols}. Found: {list(sub.columns)}")
        sys.exit(2)

    # init gateway
    gw = NFLGateway(data_paths=(str(data_dir),))
    gw.unpack_data_paths()

    total_rows = 0
    batch_idx = 0
    try:
        for data_batch, row_ids in gw.generate_data_batches():
            batch_idx += 1
            # row_ids is produced by the gateway as a DataFrame/Series holding the id(s)
            row_ids_series = to_pandas_series_first_col(row_ids)
            row_id_list = row_ids_series.tolist()

            # select rows from submission by id preserving order
            try:
                sub_indexed = sub.set_index("id")
                sel = sub_indexed.loc[row_id_list]
            except KeyError as e:
                print(f"Missing id(s) from submission for batch {batch_idx}: {e}")
                sys.exit(2)

            # sel now contains rows in the requested order (may be DataFrame or Series)
            # Build a DataFrame with exactly the expected columns (no id column)
            sel_xy = sel[["x", "y"]].reset_index(drop=True)

            # Run competition-specific validation which mirrors Kaggle checks
            gw.competition_specific_validation(sel_xy, row_ids, data_batch)

            batch_rows = len(sel_xy)
            total_rows += batch_rows
            print(f"Batch {batch_idx}: validated {batch_rows} rows")

    except GatewayRuntimeError as gre:
        print("Validation failed:")
        print(f"  ErrorType: {gre.error_type}")
        print(f"  Details: {gre.error_details}")
        sys.exit(3)

    print(f"All done — validated {total_rows} rows across {batch_idx} batch(es). Submission looks valid.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"), help="Path to your submission CSV")
    parser.add_argument("--data-dir", type=Path, default=Path("data/nfl-big-data-bowl-2026-prediction"), help="Path to the competition data directory containing test.csv and test_input.csv")
    args = parser.parse_args()

    validate(args.submission, args.data_dir)


if __name__ == "__main__":
    main()
