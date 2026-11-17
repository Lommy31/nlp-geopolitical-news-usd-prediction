"""Module 5E: fold-safe exponential-decay aggregation of accepted embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
INTERMEDIATE = HANA / "data/intermediate"
MANIFESTS = HANA / "data/manifests"
OUTPUT_DIR = HANA / "embeddings/decay"
CALENDAR_PATH = INTERMEDIATE / "master_calendar_news_windows.csv"
FOLD_MANIFEST_PATH = MANIFESTS / "module3_fold_manifest.json"
MODULE4_MANIFEST_PATH = MANIFESTS / "module4_feature_manifest.json"
CALENDAR_ROWS = 1202
HALF_LIVES = (1, 3, 5)
MODELS = ("finbert", "roberta")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calendar() -> pd.DataFrame:
    frame = pd.read_csv(CALENDAR_PATH, parse_dates=["Date"])
    required = {"Date", "calendar_index"}
    if len(frame) != CALENDAR_ROWS or not required <= set(frame):
        raise AssertionError("Invalid master calendar")
    if not frame["calendar_index"].equals(pd.Series(range(CALENDAR_ROWS), name="calendar_index")):
        raise AssertionError("Calendar indices are not 0..1201")
    if frame["Date"].duplicated().any() or not frame["Date"].is_monotonic_increasing:
        raise AssertionError("Calendar is not unique and chronological")
    return frame[["Date", "calendar_index"]].copy()


def load_fold_manifest() -> dict[str, Any]:
    return json.loads(FOLD_MANIFEST_PATH.read_text(encoding="utf-8"))


def fold_indices(manifest: dict[str, Any], fold_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matches = [x for x in manifest["folds"] if x["fold_id"] == fold_id]
    if len(matches) != 1:
        raise ValueError(f"Fold {fold_id} is not uniquely defined")
    train = np.asarray(matches[0]["train_calendar_indices"], dtype=int)
    validation = np.asarray(matches[0]["validation_calendar_indices"], dtype=int)
    locked = np.asarray(manifest["locked_test"]["calendar_indices"], dtype=int)
    if np.intersect1d(train, validation).size or np.intersect1d(np.r_[train, validation], locked).size:
        raise AssertionError("Fold overlap or locked-test overlap")
    return train, validation, locked


def load_embedding_inputs(model: str, timing_mode: str, fold_id: int) -> tuple[pd.DataFrame, np.ndarray, Path, Path]:
    if model not in MODELS or timing_mode not in {"strict", "same_date"}:
        raise ValueError("Unknown model or timing mode")
    base = HANA / ("data/processed/finbert" if model == "finbert" else "embeddings/roberta")
    stem = f"{model}_{timing_mode}_fold{fold_id}"
    metadata_path = base / f"{stem}_article_metadata.csv"
    vector_path = base / f"{stem}_article_vectors.npz"
    metadata = pd.read_csv(metadata_path, parse_dates=["Date", "Aligned_Date"])
    vectors = np.load(vector_path)["vectors"]
    if len(metadata) != len(vectors) or metadata["article_id"].duplicated().any() or vectors.ndim != 2 or vectors.shape[1] != 768:
        raise AssertionError(f"Invalid {model} input artifacts")
    required = {"article_id", "Date", "Aligned_Date", "calendar_index", "fold_role", "valid_alignment", "successfully_embedded"}
    if not required <= set(metadata):
        raise AssertionError(f"Missing metadata fields: {required - set(metadata)}")
    forbidden = {"target_direction", "target_date", "next_Date", "next_return", "log_return_next", "y", "split", "module3_split"} & set(metadata)
    if forbidden:
        raise AssertionError(f"Forbidden predictor fields in embedding metadata: {forbidden}")
    return metadata, vectors.astype(np.float32, copy=False), metadata_path, vector_path


def decay_lambda(half_life: float) -> float:
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return float(np.log(2.0) / half_life)


def trading_age(source_date: pd.Timestamp, prediction_date: pd.Timestamp, trading_dates: pd.DatetimeIndex) -> int:
    if pd.isna(source_date) or pd.isna(prediction_date) or source_date > prediction_date:
        raise AssertionError("Future article relative to prediction date")
    left = int(trading_dates.searchsorted(source_date, side="right"))
    right = int(trading_dates.searchsorted(prediction_date, side="right"))
    age = right - left
    if age < 0:
        raise AssertionError("Negative trading age")
    return age


def exponential_weights(ages: np.ndarray, half_life: float) -> np.ndarray:
    weights = np.exp(-decay_lambda(half_life) * np.asarray(ages, dtype=float))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise AssertionError("Invalid decay weight total")
    return weights / total


def aggregate_articles(article_dates: pd.Series, prediction_date: pd.Timestamp, vectors: np.ndarray, trading_dates: pd.DatetimeIndex, half_life: float) -> tuple[np.ndarray, np.ndarray]:
    if len(article_dates) != len(vectors) or not len(vectors):
        raise ValueError("At least one article vector is required")
    ages = np.asarray([trading_age(date, prediction_date, trading_dates) for date in pd.to_datetime(article_dates)], dtype=int)
    weights = exponential_weights(ages, half_life)
    result = np.average(vectors.astype(np.float64), axis=0, weights=weights).astype(np.float32)
    return result, weights


def aggregate_daily(calendar: pd.DataFrame, availability: pd.DataFrame, metadata: pd.DataFrame, vectors: np.ndarray, train: np.ndarray, validation: np.ndarray, half_life: float, model: str) -> pd.DataFrame:
    if len(metadata) != len(vectors):
        raise AssertionError("Metadata/vector length mismatch")
    allowed = set(np.r_[train, validation].tolist())
    train_set = set(train.tolist())
    output = calendar.merge(availability, on=["Date", "calendar_index"], validate="one_to_one")
    output["scope_status"] = output["calendar_index"].map(lambda i: "train" if i in train_set else "validation" if i in allowed else "not_fit_or_transformed")
    output["within_fold_scope"] = output["calendar_index"].isin(allowed)
    output["empty_news_day"] = output["n_articles"].eq(0)
    output["n_embedded_articles"] = 0
    output["feature_available"] = False
    output["unavailable_due_to_scope"] = (~output["within_fold_scope"]) & output["n_articles"].gt(0)
    output["embedding_unavailable_nonempty"] = False
    columns = [f"{model}_decay_hl{half_life:g}_{i}" for i in range(1, 769)]
    output = pd.concat([output, pd.DataFrame(np.nan, index=output.index, columns=columns)], axis=1)
    trading_dates = pd.DatetimeIndex(calendar["Date"])
    for index, group in metadata.groupby("calendar_index", sort=False):
        if int(index) not in allowed:
            raise AssertionError("Out-of-scope article entered aggregation")
        eligible = group["successfully_embedded"].to_numpy(dtype=bool)
        if not eligible.any():
            continue
        positions = group.index.to_numpy()[eligible]
        prediction_date = calendar.loc[calendar["calendar_index"].eq(int(index)), "Date"]
        if len(prediction_date) != 1:
            raise AssertionError("Article calendar index has no unique prediction date")
        mean_vector, weights = aggregate_articles(group.loc[eligible, "Date"], prediction_date.iloc[0], vectors[positions], trading_dates, half_life)
        row = output["calendar_index"].eq(int(index))
        output.loc[row, columns] = mean_vector
        output.loc[row, "n_embedded_articles"] = len(positions)
        output.loc[row, "feature_available"] = True
        if not np.isclose(weights.sum(), 1.0):
            raise AssertionError("Weights are not normalized")
    output.loc[output["empty_news_day"], columns] = 0.0
    output.loc[output["empty_news_day"] & output["within_fold_scope"], "feature_available"] = True
    output["embedding_unavailable_nonempty"] = output["n_articles"].gt(0) & ~output["feature_available"]
    return output


def load_availability(timing_mode: str, calendar: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(HANA / f"data/processed/b0_{timing_mode}_predictors.csv")
    required = {"n_articles", "has_news", "log1p_n_articles"}
    if len(frame) != CALENDAR_ROWS or not required <= set(frame):
        raise AssertionError("Invalid availability input")
    result = calendar.copy()
    result["n_articles"] = frame["n_articles"].astype(int).to_numpy()
    result["has_news"] = frame["has_news"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
    result["log1p_n_articles"] = frame["log1p_n_articles"].astype(float).to_numpy()
    if not np.array_equal(result["has_news"], result["n_articles"].gt(0)):
        raise AssertionError("Availability mismatch")
    return result


def run_module5e() -> dict[str, Any]:
    calendar = load_calendar()
    manifest = load_fold_manifest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    outputs: list[dict[str, Any]] = []
    source_hashes = {"calendar": sha256_file(CALENDAR_PATH), "fold_manifest": sha256_file(FOLD_MANIFEST_PATH), "module4_manifest": sha256_file(MODULE4_MANIFEST_PATH)}
    for timing_mode in ("strict", "same_date"):
        availability = load_availability(timing_mode, calendar)
        summaries[timing_mode] = {}
        for fold_id in (1, 2, 3):
            train, validation, locked = fold_indices(manifest, fold_id)
            summaries[timing_mode][str(fold_id)] = {}
            for model in MODELS:
                metadata, vectors, metadata_path, vector_path = load_embedding_inputs(model, timing_mode, fold_id)
                if metadata["calendar_index"].isin(locked).any() or metadata["fold_role"].isin({"locked_test", "test"}).any():
                    raise AssertionError("Locked-test article entered decay input")
                if not metadata["calendar_index"].isin(np.r_[train, validation]).all():
                    raise AssertionError("Embedding input contains out-of-scope calendar index")
                source_hashes[f"{model}_{timing_mode}_fold{fold_id}_metadata"] = sha256_file(metadata_path)
                source_hashes[f"{model}_{timing_mode}_fold{fold_id}_vectors"] = sha256_file(vector_path)
                for half_life in HALF_LIVES:
                    daily = aggregate_daily(calendar, availability, metadata, vectors, train, validation, half_life, model)
                    if len(daily) != CALENDAR_ROWS:
                        raise AssertionError("Decay output did not preserve calendar")
                    stem = f"{model}_decay_{timing_mode}_fold{fold_id}_hl{half_life}d"
                    path = OUTPUT_DIR / f"{stem}_daily.csv"
                    daily.to_csv(path, index=False)
                    key = f"{model}_hl{half_life}"
                    summaries[timing_mode][str(fold_id)][key] = {"model": model, "half_life_trading_days": half_life, "lambda": decay_lambda(half_life), "training_article_count": int(metadata["fold_role"].eq("train").sum()), "validation_article_count": int(metadata["fold_role"].eq("validation").sum()), "embedded_article_count": int(metadata["successfully_embedded"].sum()), "daily_output_shape": list(daily.shape), "empty_news_days": int(daily["empty_news_day"].sum()), "unavailable_nonempty_days": int(daily["unavailable_due_to_scope"].sum()), "embedding_unavailable_nonempty_days": int(daily["embedding_unavailable_nonempty"].sum()), "locked_test_articles_used": False, "locked_test_calendar_used": False, "output_path": str(path.relative_to(HANA)), "output_sha256": sha256_file(path)}
                    outputs.append({"model": model, "timing_mode": timing_mode, "fold_id": fold_id, "half_life_trading_days": half_life, "path": str(path.relative_to(HANA)), "rows": len(daily)})
    result = {"module": "phase5_module5e", "models": list(MODELS), "timing_modes": ["strict", "same_date"], "half_lives_trading_days": list(HALF_LIVES), "aggregation": "normalized exp(-lambda * trading_age) weighted mean", "age_definition": "number of master-calendar observations strictly after article Date and through prediction Aligned_Date", "calendar_rows_preserved": CALENDAR_ROWS, "locked_test_articles_used": False, "locked_test_calendar_used": False, "source_hashes": source_hashes, "summaries": summaries, "outputs": outputs}
    (MANIFESTS / "module5e_decay_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_module5e(), indent=2, sort_keys=True))
