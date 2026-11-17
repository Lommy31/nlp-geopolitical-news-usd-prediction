"""Phase 5 Module 5A: fold-local TF-IDF/SVD for controlled B2.

The approved Module 4 B2 input is a daily-title document table.  Each daily
title string is therefore one document.  Vectorizer and SVD fitting are done
only on the fold's training calendar indices; validation rows are transformed
with those fitted objects and are never refit.  No locked-test rows are
transformed or used here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
PROCESSED = HANA / "data/processed"
MANIFESTS = HANA / "data/manifests"
OUTPUT_DIR = PROCESSED / "tfidf_svd"

FOLD_MANIFEST_PATH = MANIFESTS / "module3_fold_manifest.json"
MODULE4_MANIFEST_PATH = MANIFESTS / "module4_feature_manifest.json"
CALENDAR_ROWS = 1202
MAX_FEATURES = 500
N_COMPONENTS = 5
RANDOM_STATE = 42
TFIDF_FEATURE_PREFIX = "tfidf_svd_"
INPUT_COLUMNS = {"Date", "calendar_index", "daily_titles"}
FORBIDDEN_TEXT_COLUMNS = {
    "target_direction", "target_date", "next_Date", "next_return", "next_USD_IDR",
    "next_jisdor", "log_return_next", "y", "split", "fold", "fold_id",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fold_manifest(path: Path = FOLD_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_daily_inputs(timing_mode: str) -> pd.DataFrame:
    if timing_mode not in {"strict", "same_date"}:
        raise ValueError("timing_mode must be 'strict' or 'same_date'")
    path = PROCESSED / f"b2_{'strict' if timing_mode == 'strict' else 'same_date'}_daily_titles_input.csv"
    frame = pd.read_csv(path, parse_dates=["Date"], keep_default_na=False)
    if set(frame.columns) != INPUT_COLUMNS:
        raise AssertionError(f"Unexpected B2 input columns: {list(frame.columns)}")
    if len(frame) != CALENDAR_ROWS:
        raise AssertionError(f"Expected {CALENDAR_ROWS} B2 daily rows, found {len(frame)}")
    if not frame["calendar_index"].equals(pd.Series(range(CALENDAR_ROWS), name="calendar_index")):
        raise AssertionError("B2 daily-title input indices do not match the audited calendar")
    if frame["Date"].duplicated().any() or not frame["Date"].is_monotonic_increasing:
        raise AssertionError("B2 daily-title input dates must be unique and chronological")
    if FORBIDDEN_TEXT_COLUMNS & set(frame.columns):
        raise AssertionError("B2 text input contains forbidden target or split metadata")
    frame["daily_titles"] = frame["daily_titles"].fillna("").astype(str)
    return frame


def load_availability(timing_mode: str) -> pd.DataFrame:
    path = PROCESSED / f"b0_{'strict' if timing_mode == 'strict' else 'same_date'}_predictors.csv"
    frame = pd.read_csv(path)
    required = {"n_articles", "has_news", "log1p_n_articles"}
    if not required <= set(frame.columns) or len(frame) != CALENDAR_ROWS:
        raise AssertionError("Timing-matched B0 availability artifact is invalid")
    return frame[list(required)].copy()


def fold_indices(manifest: dict[str, Any], fold_id: int) -> tuple[np.ndarray, np.ndarray]:
    matches = [fold for fold in manifest["folds"] if fold["fold_id"] == fold_id]
    if len(matches) != 1:
        raise ValueError(f"Fold {fold_id} is not uniquely defined in Module 3 manifest")
    fold = matches[0]
    train = np.asarray(fold["train_calendar_indices"], dtype=int)
    validation = np.asarray(fold["validation_calendar_indices"], dtype=int)
    if np.intersect1d(train, validation).size:
        raise AssertionError(f"Fold {fold_id} train and validation membership overlaps")
    if np.intersect1d(train, np.asarray(manifest["locked_test"]["calendar_indices"], dtype=int)).size:
        raise AssertionError(f"Fold {fold_id} train membership overlaps locked test")
    if np.intersect1d(validation, np.asarray(manifest["locked_test"]["calendar_indices"], dtype=int)).size:
        raise AssertionError(f"Fold {fold_id} validation membership overlaps locked test")
    return train, validation


def _rank_of_sparse_matrix(matrix) -> int:
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 0
    return int(np.linalg.matrix_rank(matrix.toarray()))


def fit_transform_fold(
    daily: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit TF-IDF/SVD on train documents and transform validation only."""
    train_mask = daily["calendar_index"].isin(train_indices)
    validation_mask = daily["calendar_index"].isin(validation_indices)
    train_docs = daily.loc[train_mask, "daily_titles"].tolist()
    validation_docs = daily.loc[validation_mask, "daily_titles"].tolist()
    if not train_docs:
        raise AssertionError("Cannot fit TF-IDF without training documents")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )
    train_tfidf = vectorizer.fit_transform(train_docs)
    fitted_vocabulary = len(vectorizer.vocabulary_)
    if fitted_vocabulary == 0:
        raise ValueError("Training fold produced an empty TF-IDF vocabulary")
    validation_tfidf = vectorizer.transform(validation_docs)
    rank = _rank_of_sparse_matrix(train_tfidf)
    effective_components = min(N_COMPONENTS, fitted_vocabulary, rank, max(1, len(train_docs) - 1))
    if effective_components < 1:
        raise ValueError("Training TF-IDF matrix has no usable SVD component")

    svd = TruncatedSVD(n_components=effective_components, random_state=random_state)
    train_svd = svd.fit_transform(train_tfidf)
    validation_svd = svd.transform(validation_tfidf)
    train_output = np.zeros((len(train_docs), N_COMPONENTS), dtype=float)
    validation_output = np.zeros((len(validation_docs), N_COMPONENTS), dtype=float)
    train_output[:, :effective_components] = train_svd
    validation_output[:, :effective_components] = validation_svd
    details = {
        "train_daily_document_count": len(train_docs),
        "validation_daily_document_count": len(validation_docs),
        "vectorizer_fit_calendar_indices": train_indices.tolist(),
        "validation_transform_calendar_indices": validation_indices.tolist(),
        "validation_refit": False,
        "vocabulary_size": fitted_vocabulary,
        "tfidf_matrix_shape_train": list(train_tfidf.shape),
        "tfidf_matrix_shape_validation": list(validation_tfidf.shape),
        "training_matrix_rank": rank,
        "requested_svd_components": N_COMPONENTS,
        "effective_svd_components": effective_components,
        "output_svd_dimensions": N_COMPONENTS,
        "explained_variance_ratio": [float(value) for value in svd.explained_variance_ratio_] + [0.0] * (N_COMPONENTS - effective_components),
        "explained_variance_sum": float(svd.explained_variance_ratio_.sum()),
        "random_state": random_state,
        "tfidf_config": {
            "ngram_range": [1, 2],
            "stop_words": "english",
            "max_features": MAX_FEATURES,
            "sublinear_tf": True,
        },
    }
    return train_output, validation_output, details


def build_fold_daily_output(
    daily: pd.DataFrame,
    availability: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    train_output: np.ndarray,
    validation_output: np.ndarray,
) -> pd.DataFrame:
    output = daily[["Date", "calendar_index"]].copy()
    output = pd.concat([output, availability.reset_index(drop=True)], axis=1)
    output["empty_news_day"] = output["n_articles"].eq(0)
    output["fold_role"] = "not_fit_or_transformed"
    output["feature_available"] = False
    output.loc[output["calendar_index"].isin(train_indices), "fold_role"] = "train"
    output.loc[output["calendar_index"].isin(validation_indices), "fold_role"] = "validation"
    output.loc[output["calendar_index"].isin(train_indices), "feature_available"] = True
    output.loc[output["calendar_index"].isin(validation_indices), "feature_available"] = True
    feature_columns = [f"{TFIDF_FEATURE_PREFIX}{index}" for index in range(1, N_COMPONENTS + 1)]
    for column in feature_columns:
        output[column] = np.nan
    output.loc[output["calendar_index"].isin(train_indices), feature_columns] = train_output
    output.loc[output["calendar_index"].isin(validation_indices), feature_columns] = validation_output
    return output


def run_module5a() -> dict[str, Any]:
    manifest = load_fold_manifest()
    all_fold_summaries: dict[str, Any] = {}
    output_records: list[dict[str, Any]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for timing_mode in ("strict", "same_date"):
        daily = load_daily_inputs(timing_mode)
        availability = load_availability(timing_mode)
        all_fold_summaries[timing_mode] = {}
        for fold_id in (1, 2, 3):
            train_indices, validation_indices = fold_indices(manifest, fold_id)
            train_output, validation_output, details = fit_transform_fold(daily, train_indices, validation_indices)
            output = build_fold_daily_output(daily, availability, train_indices, validation_indices, train_output, validation_output)
            output_path = OUTPUT_DIR / f"b2_tfidf_svd_{timing_mode}_fold{fold_id}_daily.csv"
            output.to_csv(output_path, index=False)
            train_articles = int(availability.loc[availability.index.isin(train_indices), "n_articles"].sum())
            validation_articles = int(availability.loc[availability.index.isin(validation_indices), "n_articles"].sum())
            details.update(
                {
                    "fold_id": fold_id,
                    "timing_mode": timing_mode,
                    "train_article_count_from_availability": train_articles,
                    "validation_article_count_from_availability": validation_articles,
                    "train_empty_news_days": int(availability.loc[availability.index.isin(train_indices), "n_articles"].eq(0).sum()),
                    "validation_empty_news_days": int(availability.loc[availability.index.isin(validation_indices), "n_articles"].eq(0).sum()),
                    "output_path": str(output_path.relative_to(HANA)),
                    "output_rows": len(output),
                    "locked_test_transformed": False,
                }
            )
            all_fold_summaries[timing_mode][str(fold_id)] = details
            output_records.append({"timing_mode": timing_mode, "fold_id": fold_id, "path": str(output_path.relative_to(HANA)), "rows": len(output)})

    summary = {
        "module": "phase5_module5a",
        "primary_input": "data/processed/b2_strict_daily_titles_input.csv",
        "sensitivity_input": "data/processed/b2_same_date_daily_titles_input.csv",
        "fold_manifest_sha256": sha256_file(FOLD_MANIFEST_PATH),
        "module4_manifest_sha256": sha256_file(MODULE4_MANIFEST_PATH),
        "calendar_rows_preserved": CALENDAR_ROWS,
        "fit_policy": "fit vectorizer and SVD on fold train daily-title documents only; transform validation only",
        "locked_test_used": False,
        "global_transform_fitted": False,
        "all_fold_summaries": all_fold_summaries,
        "outputs": output_records,
    }
    manifest_path = MANIFESTS / "module5a_tfidf_svd_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_module5a(), indent=2, sort_keys=True, default=str))
