"""Module 5C.3: real-data, headline-only FinBERT CLS extraction.

The frozen ProsusAI/finbert encoder is used only to transform Title_clean.
Fold membership comes from Module 3; the locked test is never encoded.  The
module writes one article-level artifact and one 1,202-row daily artifact for
each timing track and development fold.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
INTERMEDIATE = HANA / "data/intermediate"
PROCESSED = HANA / "data/processed"
MANIFESTS = HANA / "data/manifests"
OUTPUT_DIR = PROCESSED / "finbert"
CACHE_DIR = HANA / "cache/huggingface/hub"

FOLD_MANIFEST_PATH = MANIFESTS / "module3_fold_manifest.json"
MODULE4_MANIFEST_PATH = MANIFESTS / "module4_feature_manifest.json"
CALENDAR_PATH = INTERMEDIATE / "master_calendar_news_windows.csv"
MODEL_REPOSITORY = "ProsusAI/finbert"
MODEL_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
MODEL_SNAPSHOT = CACHE_DIR / f"models--ProsusAI--finbert/snapshots/{MODEL_REVISION}"
EMBEDDING_DIMENSION = 768
CALENDAR_ROWS = 1202
BATCH_SIZE = 32
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ARTICLE_REQUIRED_COLUMNS = {
    "article_id",
    "Title_clean",
    "Date",
    "Aligned_Date",
    "alignment_mode",
    "alignment_failed",
    "out_of_calendar_or_invalid_date_flag",
    "source_row_number",
    "provenance_status",
    "publication_timestamp_present",
    "publication_timestamp_verified",
    "timing_uncertainty_flag",
    "confirmed_contamination_flag",
    "heuristic_contamination_flag",
}
FORBIDDEN_COLUMNS = {
    "target_direction",
    "target_date",
    "next_Date",
    "next_return",
    "next_USD_IDR",
    "next_jisdor",
    "log_return_next",
    "y",
    "split",
    "fold",
    "fold_id",
    "module3_split",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "false", "1", "0"})
    if unknown.any():
        raise AssertionError(f"Unrecognized boolean values in {series.name}: {normalized[unknown].unique()}")
    return normalized.isin({"true", "1"})


def load_fold_manifest(path: Path = FOLD_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_calendar(path: Path = CALENDAR_PATH) -> pd.DataFrame:
    calendar = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "calendar_index"}
    if not required <= set(calendar.columns):
        raise AssertionError(f"Calendar missing columns: {required - set(calendar.columns)}")
    if len(calendar) != CALENDAR_ROWS:
        raise AssertionError(f"Expected {CALENDAR_ROWS} calendar rows, found {len(calendar)}")
    if not calendar["calendar_index"].equals(pd.Series(range(CALENDAR_ROWS), name="calendar_index")):
        raise AssertionError("Calendar indices are not the audited 0..1201 sequence")
    if calendar["Date"].duplicated().any() or not calendar["Date"].is_monotonic_increasing:
        raise AssertionError("Calendar dates must be unique and chronological")
    return calendar[["Date", "calendar_index"]].copy()


def load_availability(timing_mode: str, calendar: pd.DataFrame) -> pd.DataFrame:
    if timing_mode not in {"strict", "same_date"}:
        raise ValueError("timing_mode must be strict or same_date")
    path = PROCESSED / f"b0_{timing_mode}_predictors.csv"
    availability = pd.read_csv(path)
    required = {"n_articles", "has_news", "log1p_n_articles"}
    if len(availability) != CALENDAR_ROWS or not required <= set(availability.columns):
        raise AssertionError(f"Invalid availability artifact: {path}")
    output = calendar.copy()
    output["n_articles"] = availability["n_articles"].astype(int).to_numpy()
    output["has_news"] = parse_bool(availability["has_news"]).to_numpy()
    output["log1p_n_articles"] = availability["log1p_n_articles"].astype(float).to_numpy()
    if not np.array_equal(output["has_news"].to_numpy(), output["n_articles"].gt(0).to_numpy()):
        raise AssertionError("Availability has_news is inconsistent with n_articles")
    return output


def load_aligned_articles(timing_mode: str, calendar: pd.DataFrame) -> pd.DataFrame:
    path = INTERMEDIATE / f"articles_aligned_{timing_mode}.csv"
    frame = pd.read_csv(path, keep_default_na=False, na_filter=False)
    missing = ARTICLE_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise AssertionError(f"Article corpus missing columns: {missing}")
    forbidden = FORBIDDEN_COLUMNS & set(frame.columns)
    if forbidden:
        raise AssertionError(f"Article corpus exposes forbidden target/split columns: {forbidden}")
    if frame["article_id"].duplicated().any():
        raise AssertionError("Module 2 article IDs must remain unique")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Aligned_Date"] = pd.to_datetime(frame["Aligned_Date"], errors="coerce")
    for column in [
        "alignment_failed",
        "out_of_calendar_or_invalid_date_flag",
        "publication_timestamp_present",
        "publication_timestamp_verified",
        "timing_uncertainty_flag",
        "confirmed_contamination_flag",
        "heuristic_contamination_flag",
    ]:
        frame[column] = parse_bool(frame[column])
    frame["Title_clean"] = frame["Title_clean"].astype(str)
    date_map = calendar.set_index("Date")["calendar_index"]
    valid = (~frame["alignment_failed"] & ~frame["out_of_calendar_or_invalid_date_flag"] & frame["Aligned_Date"].notna())
    frame["calendar_index"] = frame["Aligned_Date"].map(date_map).astype("Int64")
    frame["valid_alignment"] = valid & frame["calendar_index"].notna()
    frame["headline_empty"] = frame["Title_clean"].eq("")
    columns = [
        "article_id", "source_row_number", "Date", "Aligned_Date", "calendar_index", "Title_clean",
        "alignment_mode", "provenance_status", "publication_timestamp_present",
        "publication_timestamp_verified", "timing_uncertainty_flag", "confirmed_contamination_flag",
        "heuristic_contamination_flag", "alignment_failed", "out_of_calendar_or_invalid_date_flag",
        "valid_alignment", "headline_empty",
    ]
    return frame[columns].copy()


def fold_indices(manifest: dict[str, Any], fold_id: int) -> tuple[np.ndarray, np.ndarray]:
    folds = [fold for fold in manifest["folds"] if fold["fold_id"] == fold_id]
    if len(folds) != 1:
        raise ValueError(f"Fold {fold_id} is not uniquely defined")
    train = np.asarray(folds[0]["train_calendar_indices"], dtype=int)
    validation = np.asarray(folds[0]["validation_calendar_indices"], dtype=int)
    locked = np.asarray(manifest["locked_test"]["calendar_indices"], dtype=int)
    if np.intersect1d(train, validation).size or np.intersect1d(np.concatenate([train, validation]), locked).size:
        raise AssertionError(f"Fold {fold_id} overlaps validation/train or locked test")
    return train, validation


def fold_scope(articles: pd.DataFrame, train: np.ndarray, validation: np.ndarray) -> pd.DataFrame:
    allowed = np.concatenate([train, validation])
    scope = articles.loc[articles["valid_alignment"] & articles["calendar_index"].isin(allowed)].copy()
    train_set = set(train.tolist())
    scope["fold_role"] = scope["calendar_index"].astype(int).map(lambda value: "train" if value in train_set else "validation")
    if not bool(scope["calendar_index"].isin(set(allowed.tolist())).all()):
        raise AssertionError("Article scope contains an unauthorized calendar index")
    return scope.reset_index(drop=True)


def inspect_headline_lengths(frame: pd.DataFrame, tokenizer, maximum_length: int) -> pd.DataFrame:
    output = frame.copy()
    output["headline_status"] = np.where(output["headline_empty"], "empty", "candidate")
    output["token_count"] = 0
    output["overlength_headline"] = False
    candidates = output.index[~output["headline_empty"]]
    if len(candidates):
        texts = output.loc[candidates, "Title_clean"].tolist()
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        lengths = np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=int)
        output.loc[candidates, "token_count"] = lengths
        output.loc[candidates, "overlength_headline"] = lengths > maximum_length
        output.loc[candidates, "headline_status"] = np.where(lengths > maximum_length, "overlength", "candidate")
    return output


def encode_article_embeddings(frame: pd.DataFrame, tokenizer, model, device: str, batch_size: int = BATCH_SIZE) -> tuple[pd.DataFrame, np.ndarray]:
    output = frame.copy()
    vectors = np.zeros((len(output), EMBEDDING_DIMENSION), dtype=np.float32)
    output["successfully_embedded"] = False
    candidate_mask = output["headline_status"].eq("candidate")
    candidate_positions = np.flatnonzero(candidate_mask.to_numpy())
    for start in range(0, len(candidate_positions), batch_size):
        positions = candidate_positions[start:start + batch_size]
        texts = output.iloc[positions]["Title_clean"].tolist()
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        input_length = int(inputs["input_ids"].shape[1])
        if input_length > int(model.config.max_position_embeddings):
            raise AssertionError("A candidate batch exceeds the model limit after preflight")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            cls = model(**inputs).last_hidden_state[:, 0, :].detach().to("cpu").numpy()
        if cls.shape != (len(positions), EMBEDDING_DIMENSION):
            raise AssertionError(f"Unexpected CLS shape: {cls.shape}")
        if not np.isfinite(cls).all() or not np.any(cls):
            raise AssertionError("Non-finite or all-zero CLS output")
        vectors[positions] = cls.astype(np.float32, copy=False)
        output.loc[output.index[positions], "successfully_embedded"] = True
    return output, vectors


def aggregate_daily(calendar: pd.DataFrame, availability: pd.DataFrame, articles: pd.DataFrame, vectors: np.ndarray, train: np.ndarray, validation: np.ndarray) -> pd.DataFrame:
    output = calendar.merge(availability, on=["Date", "calendar_index"], validate="one_to_one")
    allowed = set(np.concatenate([train, validation]).tolist())
    train_set = set(train.tolist())
    output["scope_status"] = output["calendar_index"].map(lambda value: "train" if value in train_set else "validation" if value in allowed else "not_fit_or_transformed")
    output["within_fold_scope"] = output["calendar_index"].isin(allowed)
    output["empty_news_day"] = output["n_articles"].eq(0)
    output["n_embedded_articles"] = 0
    output["n_invalid_headlines"] = 0
    output["n_overlength_headlines"] = 0
    output["feature_available"] = False
    output["unavailable_due_to_scope"] = (~output["within_fold_scope"]) & output["n_articles"].gt(0)
    output["embedding_unavailable_nonempty"] = False
    feature_columns = [f"finbert_{index}" for index in range(1, EMBEDDING_DIMENSION + 1)]
    output = pd.concat([output, pd.DataFrame(np.nan, index=output.index, columns=feature_columns)], axis=1)
    for calendar_index, group in articles.groupby("calendar_index", sort=False):
        row = output["calendar_index"].eq(int(calendar_index))
        output.loc[row, "n_embedded_articles"] = int(group["successfully_embedded"].sum())
        output.loc[row, "n_invalid_headlines"] = int(group["headline_status"].eq("empty").sum())
        output.loc[row, "n_overlength_headlines"] = int(group["headline_status"].eq("overlength").sum())
        successful = group["successfully_embedded"].to_numpy(dtype=bool)
        if successful.any():
            mean_vector = vectors[group.index.to_numpy()[successful]].mean(axis=0)
            output.loc[row, feature_columns] = mean_vector
            output.loc[row, "feature_available"] = True
    output.loc[output["empty_news_day"], feature_columns] = 0.0
    output.loc[output["empty_news_day"] & output["within_fold_scope"], "feature_available"] = True
    output["embedding_unavailable_nonempty"] = output["n_articles"].gt(0) & ~output["feature_available"]
    return output


def save_article_output(frame: pd.DataFrame, vectors: np.ndarray, npz_path: Path, metadata_path: Path) -> None:
    metadata = frame.drop(columns=["Title_clean"], errors="ignore").copy()
    metadata["Aligned_Date"] = metadata["Aligned_Date"].dt.strftime("%Y-%m-%d")
    np.savez_compressed(npz_path, article_id=metadata["article_id"].astype(str).to_numpy(), vectors=vectors.astype(np.float32), calendar_index=metadata["calendar_index"].astype(int).to_numpy())
    metadata.to_csv(metadata_path, index=False)


def load_frozen_model() -> tuple[Any, Any, str]:
    model_file = MODEL_SNAPSHOT / "model.safetensors"
    if not model_file.exists() or not (MODEL_SNAPSHOT / "config.json").exists():
        raise FileNotFoundError(f"Pinned local FinBERT snapshot is incomplete: {MODEL_SNAPSHOT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPOSITORY, revision=MODEL_REVISION, cache_dir=str(CACHE_DIR), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_REPOSITORY, revision=MODEL_REVISION, cache_dir=str(CACHE_DIR), local_files_only=True, use_safetensors=True)
    if int(model.config.hidden_size) != EMBEDDING_DIMENSION:
        raise AssertionError(f"Expected hidden size {EMBEDDING_DIMENSION}, got {model.config.hidden_size}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("FinBERT parameters are not frozen")
    model.eval().to(DEVICE)
    return tokenizer, model, DEVICE


def run_module5c() -> dict[str, Any]:
    manifest = load_fold_manifest()
    calendar = load_calendar()
    tokenizer, model, device = load_frozen_model()
    source_paths = {
        "calendar": CALENDAR_PATH,
        "fold_manifest": FOLD_MANIFEST_PATH,
        "module4_manifest": MODULE4_MANIFEST_PATH,
        "articles_strict": INTERMEDIATE / "articles_aligned_strict.csv",
        "articles_same_date": INTERMEDIATE / "articles_aligned_same_date.csv",
        "model_config": MODEL_SNAPSHOT / "config.json",
        "model_safetensors": MODEL_SNAPSHOT / "model.safetensors",
        "tokenizer_config": MODEL_SNAPSHOT / "tokenizer_config.json",
        "special_tokens_map": MODEL_SNAPSHOT / "special_tokens_map.json",
        "vocab": MODEL_SNAPSHOT / "vocab.txt",
    }
    summaries: dict[str, dict[str, Any]] = {}
    outputs: list[dict[str, Any]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for timing_mode in ("strict", "same_date"):
        articles = load_aligned_articles(timing_mode, calendar)
        availability = load_availability(timing_mode, calendar)
        valid_counts = articles.loc[articles["valid_alignment"]].groupby("calendar_index").size()
        expected = availability.set_index("calendar_index")["n_articles"]
        if not valid_counts.reindex(expected.index, fill_value=0).astype(int).equals(expected.astype(int)):
            raise AssertionError(f"Aligned article counts do not match availability for {timing_mode}")
        summaries[timing_mode] = {}
        for fold_id in (1, 2, 3):
            train, validation = fold_indices(manifest, fold_id)
            locked = set(manifest["locked_test"]["calendar_indices"])
            scope = fold_scope(articles, train, validation)
            if scope["calendar_index"].isin(locked).any():
                raise AssertionError("Locked-test article entered FinBERT scope")
            scope = inspect_headline_lengths(scope, tokenizer, int(model.config.max_position_embeddings))
            encoded, article_vectors = encode_article_embeddings(scope, tokenizer, model, device)
            daily = aggregate_daily(calendar, availability, encoded, article_vectors, train, validation)
            if len(daily) != CALENDAR_ROWS:
                raise AssertionError("Daily output did not preserve the complete calendar")
            article_path = OUTPUT_DIR / f"finbert_{timing_mode}_fold{fold_id}_article_vectors.npz"
            metadata_path = OUTPUT_DIR / f"finbert_{timing_mode}_fold{fold_id}_article_metadata.csv"
            daily_path = OUTPUT_DIR / f"finbert_{timing_mode}_fold{fold_id}_daily.csv"
            save_article_output(encoded, article_vectors, article_path, metadata_path)
            daily.to_csv(daily_path, index=False)
            train_rows = encoded["fold_role"].eq("train")
            validation_rows = encoded["fold_role"].eq("validation")
            train_dates = calendar.loc[calendar["calendar_index"].isin(train), "Date"]
            validation_dates = calendar.loc[calendar["calendar_index"].isin(validation), "Date"]
            summary = {
                "fold_id": fold_id,
                "timing_mode": timing_mode,
                "train_calendar_indices": train.tolist(),
                "validation_calendar_indices": validation.tolist(),
                "train_date_range": [train_dates.min().strftime("%Y-%m-%d"), train_dates.max().strftime("%Y-%m-%d")],
                "validation_date_range": [validation_dates.min().strftime("%Y-%m-%d"), validation_dates.max().strftime("%Y-%m-%d")],
                "training_article_count": int(train_rows.sum()),
                "validation_article_count": int(validation_rows.sum()),
                "training_successfully_embedded_articles": int(encoded.loc[train_rows, "successfully_embedded"].sum()),
                "validation_successfully_embedded_articles": int(encoded.loc[validation_rows, "successfully_embedded"].sum()),
                "empty_headline_count": int(encoded["headline_status"].eq("empty").sum()),
                "overlength_headline_count": int(encoded["headline_status"].eq("overlength").sum()),
                "invalid_alignment_count_source": int((~articles["valid_alignment"]).sum()),
                "embedding_dimensions": EMBEDDING_DIMENSION,
                "daily_output_shape": list(daily.shape),
                "article_output_shape": list(article_vectors.shape),
                "device": device,
                "locked_test_articles_used": False,
                "model_safetensors_sha256": sha256_file(source_paths["model_safetensors"]),
                "article_vector_path": str(article_path.relative_to(HANA)),
                "article_vector_sha256": sha256_file(article_path),
                "article_metadata_path": str(metadata_path.relative_to(HANA)),
                "article_metadata_sha256": sha256_file(metadata_path),
                "daily_output_path": str(daily_path.relative_to(HANA)),
                "daily_output_sha256": sha256_file(daily_path),
            }
            summaries[timing_mode][str(fold_id)] = summary
            outputs.append({"timing_mode": timing_mode, "fold_id": fold_id, "path": str(daily_path.relative_to(HANA)), "rows": len(daily)})
    result = {
        "module": "phase5_module5c3",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_class": "BertModel via AutoModel",
        "text_source": "Title_clean only",
        "text_exclusions": ["Text_clean", "lead extraction", "full-text processing", "headline truncation"],
        "embedding_method": "last_hidden_state[:, 0, :]",
        "embedding_dimensions": EMBEDDING_DIMENSION,
        "weights_frozen": True,
        "aggregation": "uniform mean of successfully embedded article-level CLS vectors",
        "calendar_rows_preserved": CALENDAR_ROWS,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "locked_test_articles_used": False,
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "summaries": summaries,
        "outputs": outputs,
    }
    manifest_path = MANIFESTS / "module5c_finbert_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_module5c(), indent=2, sort_keys=True))
