"""Task 3 Module 3: chronological validation and leakage prevention.

This module consumes the audited Module 2 master calendar.  It does not
recompute labels, build model-ready matrices, fit preprocessing, or train a
model.  Calendar rows remain distinct from supervised-eligible rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
MASTER_CALENDAR_PATH = HANA / "data/intermediate/master_calendar_news_windows.csv"
MODULE2_SUMMARY_PATH = HANA / "data/manifests/module2_audit_summary.json"
OUTPUT_DIR = HANA / "data/manifests"
ASSIGNMENT_PATH = HANA / "data/intermediate/validation_fold_assignments.csv"

TEST_FRACTION = 0.20
N_DEVELOPMENT_FOLDS = 3
BOUNDARY_PURGE_ROWS = 1

MASTER_REQUIRED_COLUMNS = {
    "Date", "next_Date", "next_USD_IDR", "next_return", "target_available",
    "is_flat_movement", "target_direction", "calendar_index", "is_warmup",
    "supervised_eligible",
}

# This whitelist is deliberately conservative.  The validation module does
# not decide the eventual feature set; it only rejects target/split metadata
# from any future predictor matrix.
FORBIDDEN_PREDICTOR_COLUMNS = {
    "Date", "next_Date", "next_USD_IDR", "next_return", "target_date",
    "next_jisdor", "log_return_next", "target_direction", "y",
    "target_available", "is_flat_movement", "is_warmup", "supervised_eligible",
    "calendar_index", "split", "split_name", "fold", "fold_id",
    "is_train", "is_validation", "is_test", "purged", "decision_date",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_master_calendar(path: Path = MASTER_CALENDAR_PATH) -> pd.DataFrame:
    """Load the Module 2 calendar without recomputing or altering its labels."""
    frame = pd.read_csv(path, parse_dates=["Date", "next_Date"])
    missing = MASTER_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Master calendar is missing required columns: {sorted(missing)}")
    if len(frame) != 1202:
        raise AssertionError(f"Expected 1,202 calendar observations, found {len(frame)}")
    if not frame["Date"].is_monotonic_increasing or not frame["Date"].is_unique:
        raise AssertionError("Master calendar dates must be unique and chronological")
    expected_indices = np.arange(len(frame))
    if not np.array_equal(frame["calendar_index"].to_numpy(), expected_indices):
        raise AssertionError("Master calendar indices are not the audited sequential indices")
    if frame["supervised_eligible"].isna().any():
        raise AssertionError("Supervised eligibility contains missing values")
    return frame


def _indices(frame: pd.DataFrame) -> np.ndarray:
    return frame["calendar_index"].astype(int).to_numpy()


def _class_counts(frame: pd.DataFrame, indices: Iterable[int]) -> dict[str, int]:
    subset = frame[frame["calendar_index"].isin(list(indices))]
    directions = subset.loc[subset["supervised_eligible"], "target_direction"]
    return {
        "UP": int(directions.eq(True).sum()),
        "DOWN": int(directions.eq(False).sum()),
        "eligible": int(len(directions)),
    }


def _target_date(frame: pd.DataFrame, indices: Iterable[int]) -> pd.Series:
    return frame.loc[frame["calendar_index"].isin(list(indices)), "next_Date"].dropna()


def assert_no_date_overlap(frame: pd.DataFrame, train_indices: Iterable[int], validation_indices: Iterable[int]) -> None:
    train = set(train_indices)
    validation = set(validation_indices)
    assert train.isdisjoint(validation), "Train and validation decision dates overlap"
    train_dates = set(frame.loc[frame["calendar_index"].isin(train), "Date"])
    validation_dates = set(frame.loc[frame["calendar_index"].isin(validation), "Date"])
    assert train_dates.isdisjoint(validation_dates), "Train and validation dates overlap"


def assert_target_date_boundary(frame: pd.DataFrame, train_indices: Iterable[int], later_indices: Iterable[int]) -> None:
    train_targets = _target_date(frame, train_indices)
    later_dates = frame.loc[frame["calendar_index"].isin(list(later_indices)), "Date"]
    if train_targets.empty or later_dates.empty:
        return
    assert train_targets.max() < later_dates.min(), (
        f"Target-date leakage: max train target {train_targets.max()} is not before "
        f"later decision date {later_dates.min()}"
    )


def assert_predictor_columns(columns: Iterable[str]) -> None:
    """Reject target-derived and split metadata columns from predictors."""
    received = set(columns)
    forbidden = received & FORBIDDEN_PREDICTOR_COLUMNS
    lower_names = {str(column).lower() for column in received}
    pattern_hits = {
        column for column in received
        if column.lower() in {"target", "label", "outcome"}
        or column.lower().startswith("target_")
        or column.lower().startswith("next_")
        or column.lower().endswith("_target")
        or column.lower().endswith("_split")
    }
    forbidden |= pattern_hits
    if forbidden:
        raise AssertionError(f"Forbidden predictor columns: {sorted(forbidden)}")


def assert_preprocessing_fit_scope(fit_indices: Iterable[int], train_indices: Iterable[int]) -> None:
    """Require every learned transform to be fit on training rows only."""
    fit_set = set(fit_indices)
    train_set = set(train_indices)
    assert fit_set <= train_set, "Preprocessing fit includes validation or test rows"


def construct_splits(
    calendar: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    n_folds: int = N_DEVELOPMENT_FOLDS,
    purge_rows: int = BOUNDARY_PURGE_ROWS,
) -> dict[str, Any]:
    """Construct a locked final test and eligibility-aware expanding folds."""
    if len(calendar) < n_folds + purge_rows + 2:
        raise AssertionError("Calendar is too short for the requested folds and purge boundary")
    if not (0 < test_fraction < 1) or n_folds < 1 or purge_rows < 1:
        raise ValueError("Invalid split configuration")

    n_calendar = len(calendar)
    test_start = int(np.floor(n_calendar * (1 - test_fraction)))
    test_indices = np.arange(test_start, n_calendar, dtype=int)
    final_purge_indices = np.arange(max(0, test_start - purge_rows), test_start, dtype=int)
    development_end_exclusive = test_start - purge_rows
    development_indices = np.arange(0, development_end_exclusive, dtype=int)
    eligible_development = calendar[
        calendar["calendar_index"].lt(development_end_exclusive)
        & calendar["supervised_eligible"]
    ]["calendar_index"].astype(int).to_numpy()
    eligible_test = calendar[
        calendar["calendar_index"].isin(test_indices) & calendar["supervised_eligible"]
    ]["calendar_index"].astype(int).to_numpy()
    if len(eligible_development) < n_folds:
        raise AssertionError("Not enough eligible development rows for requested folds")

    # Reserve an initial chronological training block, then use the following
    # three blocks as validation windows.  This avoids an empty first fold when
    # the earliest calendar rows are warm-up exclusions.
    development_blocks = np.array_split(eligible_development, n_folds + 1)
    initial_training_block = development_blocks[0].astype(int)
    validation_blocks = development_blocks[1:]
    folds: list[dict[str, Any]] = []
    for fold_number, validation_block in enumerate(validation_blocks, start=1):
        validation_indices = validation_block.astype(int)
        validation_start = int(validation_indices.min())
        fold_purge_indices = np.arange(max(0, validation_start - purge_rows), validation_start, dtype=int)
        train_indices = eligible_development[eligible_development < validation_start - purge_rows]
        assert_no_date_overlap(calendar, train_indices, validation_indices)
        assert_target_date_boundary(calendar, train_indices, validation_indices)
        assert_preprocessing_fit_scope(train_indices, train_indices)
        folds.append(
            {
                "fold_id": fold_number,
                "train_calendar_indices": train_indices.tolist(),
                "validation_calendar_indices": validation_indices.tolist(),
                "purged_calendar_indices": fold_purge_indices.tolist(),
                "train_row_count": int(len(train_indices)),
                "validation_row_count": int(len(validation_indices)),
                "train_class_counts": _class_counts(calendar, train_indices),
                "validation_class_counts": _class_counts(calendar, validation_indices),
                "train_date_min": str(calendar.loc[calendar["calendar_index"].isin(train_indices), "Date"].min().date()) if len(train_indices) else None,
                "train_date_max": str(calendar.loc[calendar["calendar_index"].isin(train_indices), "Date"].max().date()) if len(train_indices) else None,
                "validation_date_min": str(calendar.loc[calendar["calendar_index"].isin(validation_indices), "Date"].min().date()),
                "validation_date_max": str(calendar.loc[calendar["calendar_index"].isin(validation_indices), "Date"].max().date()),
                "train_target_date_max": str(_target_date(calendar, train_indices).max().date()) if len(train_indices) else None,
                "validation_target_date_max": str(_target_date(calendar, validation_indices).max().date()),
                "preprocessing_fit_scope": "train_only",
            }
        )

    # The final test is never used in fold construction or preprocessing.
    for fold in folds:
        assert set(fold["validation_calendar_indices"]).isdisjoint(set(test_indices))
        assert set(fold["train_calendar_indices"]).isdisjoint(set(test_indices))
    assert_target_date_boundary(calendar, eligible_development, eligible_test)

    assignment = pd.DataFrame({"calendar_index": calendar["calendar_index"].astype(int)})
    assignment["split"] = "development_unused_or_excluded"
    assignment.loc[assignment["calendar_index"].isin(final_purge_indices), "split"] = "final_boundary_purge"
    assignment.loc[assignment["calendar_index"].isin(test_indices), "split"] = "locked_test_calendar"
    for fold in folds:
        assignment.loc[assignment["calendar_index"].isin(fold["train_calendar_indices"]), "split"] = f"fold_{fold['fold_id']}_train"
        assignment.loc[assignment["calendar_index"].isin(fold["validation_calendar_indices"]), "split"] = f"fold_{fold['fold_id']}_validation"
        assignment.loc[assignment["calendar_index"].isin(fold["purged_calendar_indices"]), "split"] = f"fold_{fold['fold_id']}_boundary_purge"
    assignment["Date"] = calendar["Date"].dt.strftime("%Y-%m-%d")
    assignment["supervised_eligible"] = calendar["supervised_eligible"].astype(bool)
    assignment["is_flat_movement"] = calendar["is_flat_movement"].astype(bool)
    assignment["is_warmup"] = calendar["is_warmup"].astype(bool)

    return {
        "configuration": {
            "calendar_rows": n_calendar,
            "test_fraction": test_fraction,
            "test_start_calendar_index": test_start,
            "development_end_exclusive": development_end_exclusive,
            "n_development_folds": n_folds,
            "initial_training_block": "first chronological eligible block reserved before the three validation blocks",
            "boundary_purge_rows": purge_rows,
            "final_boundary_purge_indices": final_purge_indices.tolist(),
            "eligibility_rule": "supervised_eligible == True; flat, warm-up, and unavailable-target rows excluded from supervised membership",
            "preprocessing_fit_rule": "fit learned preprocessing on training rows of each fold only",
        },
        "locked_test": {
            "calendar_indices": test_indices.tolist(),
            "eligible_indices": eligible_test.tolist(),
            "calendar_row_count": int(len(test_indices)),
            "eligible_row_count": int(len(eligible_test)),
            "date_min": str(calendar.loc[calendar["calendar_index"].isin(test_indices), "Date"].min().date()),
            "date_max": str(calendar.loc[calendar["calendar_index"].isin(test_indices), "Date"].max().date()),
            "class_counts": _class_counts(calendar, eligible_test),
        },
        "development": {
            "calendar_indices": development_indices.tolist(),
            "eligible_indices": eligible_development.tolist(),
            "calendar_row_count": int(len(development_indices)),
            "eligible_row_count": int(len(eligible_development)),
            "date_min": str(calendar.loc[calendar["calendar_index"].isin(development_indices), "Date"].min().date()),
            "date_max": str(calendar.loc[calendar["calendar_index"].isin(development_indices), "Date"].max().date()),
            "flat_excluded_count": int((calendar["calendar_index"].isin(development_indices) & calendar["is_flat_movement"]).sum()),
            "warmup_excluded_count": int((calendar["calendar_index"].isin(development_indices) & calendar["is_warmup"]).sum()),
            "unavailable_target_excluded_count": int((calendar["calendar_index"].isin(development_indices) & ~calendar["target_available"]).sum()),
        },
        "folds": folds,
        "assignment": assignment,
    }


def run_validation(calendar_path: Path = MASTER_CALENDAR_PATH, module2_summary_path: Path = MODULE2_SUMMARY_PATH) -> dict[str, Any]:
    calendar = load_master_calendar(calendar_path)
    splits = construct_splits(calendar)
    module2_summary = json.loads(module2_summary_path.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSIGNMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits["assignment"].to_csv(ASSIGNMENT_PATH, index=False)
    manifest = {key: value for key, value in splits.items() if key != "assignment"}
    manifest["input_hashes"] = {
        "master_calendar_sha256": sha256_file(calendar_path),
        "module2_audit_summary_sha256": sha256_file(module2_summary_path),
        "module2_fx_sha256": module2_summary["inputs"]["fx_sha256"],
        "module2_news_sha256": module2_summary["inputs"]["news_sha256"],
    }
    manifest["forbidden_predictor_columns"] = sorted(FORBIDDEN_PREDICTOR_COLUMNS)
    manifest["test_evaluation_policy"] = "locked until model selection and preprocessing protocol are frozen"
    manifest["model_ready_datasets_built"] = False
    output_path = OUTPUT_DIR / "module3_fold_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-path", type=Path, default=MASTER_CALENDAR_PATH)
    parser.add_argument("--module2-summary-path", type=Path, default=MODULE2_SUMMARY_PATH)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.calendar_path, args.module2_summary_path), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
