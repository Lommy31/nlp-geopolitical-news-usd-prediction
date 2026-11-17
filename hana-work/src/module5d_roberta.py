"""Module 5D.2: frozen RoBERTa headline embeddings for authorized development folds."""

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
MANIFESTS = HANA / "data/manifests"
OUTPUT_DIR = HANA / "embeddings/roberta"
CACHE_DIR = HANA / "cache/huggingface/hub"
FOLD_MANIFEST_PATH = MANIFESTS / "module3_fold_manifest.json"
MODULE4_MANIFEST_PATH = MANIFESTS / "module4_feature_manifest.json"
CALENDAR_PATH = INTERMEDIATE / "master_calendar_news_windows.csv"
MODEL_REPOSITORY = "FacebookAI/roberta-base"
MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
MODEL_SNAPSHOT = CACHE_DIR / f"models--FacebookAI--roberta-base/snapshots/{MODEL_REVISION}"
EMBEDDING_DIMENSION = 768
CALENDAR_ROWS = 1202
BATCH_SIZE = 32
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ARTICLE_REQUIRED_COLUMNS = {
    "article_id", "Title_clean", "Date", "Aligned_Date", "alignment_mode",
    "alignment_failed", "out_of_calendar_or_invalid_date_flag", "source_row_number",
    "provenance_status", "publication_timestamp_present", "publication_timestamp_verified",
    "timing_uncertainty_flag", "confirmed_contamination_flag", "heuristic_contamination_flag",
}
FORBIDDEN_COLUMNS = {
    "target_direction", "target_date", "next_Date", "next_return", "next_USD_IDR",
    "next_jisdor", "log_return_next", "y", "split", "fold", "fold_id", "module3_split",
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
    if (~normalized.isin({"true", "false", "1", "0"})).any():
        raise AssertionError(f"Unrecognized boolean values in {series.name}")
    return normalized.isin({"true", "1"})


def load_fold_manifest(path: Path = FOLD_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_calendar(path: Path = CALENDAR_PATH) -> pd.DataFrame:
    calendar = pd.read_csv(path, parse_dates=["Date"])
    if len(calendar) != CALENDAR_ROWS or not {"Date", "calendar_index"} <= set(calendar):
        raise AssertionError("Invalid audited master calendar")
    if not calendar["calendar_index"].equals(pd.Series(range(CALENDAR_ROWS), name="calendar_index")):
        raise AssertionError("Calendar indices are not 0..1201")
    if calendar["Date"].duplicated().any() or not calendar["Date"].is_monotonic_increasing:
        raise AssertionError("Calendar must be unique and chronological")
    return calendar[["Date", "calendar_index"]].copy()


def load_availability(timing_mode: str, calendar: pd.DataFrame) -> pd.DataFrame:
    if timing_mode not in {"strict", "same_date"}:
        raise ValueError("timing_mode must be strict or same_date")
    frame = pd.read_csv(HANA / f"data/processed/b0_{timing_mode}_predictors.csv")
    required = {"n_articles", "has_news", "log1p_n_articles"}
    if len(frame) != CALENDAR_ROWS or not required <= set(frame):
        raise AssertionError("Invalid availability artifact")
    result = calendar.copy()
    result["n_articles"] = frame["n_articles"].astype(int).to_numpy()
    result["has_news"] = parse_bool(frame["has_news"]).to_numpy()
    result["log1p_n_articles"] = frame["log1p_n_articles"].astype(float).to_numpy()
    if not np.array_equal(result["has_news"], result["n_articles"].gt(0)):
        raise AssertionError("Availability indicators disagree")
    return result


def load_aligned_articles(timing_mode: str, calendar: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(INTERMEDIATE / f"articles_aligned_{timing_mode}.csv", keep_default_na=False, na_filter=False)
    missing = ARTICLE_REQUIRED_COLUMNS - set(frame)
    if missing:
        raise AssertionError(f"Missing article columns: {missing}")
    forbidden = FORBIDDEN_COLUMNS & set(frame)
    if forbidden or frame["article_id"].duplicated().any():
        raise AssertionError(f"Forbidden article fields or duplicate IDs: {forbidden}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Aligned_Date"] = pd.to_datetime(frame["Aligned_Date"], errors="coerce")
    for column in ("alignment_failed", "out_of_calendar_or_invalid_date_flag", "publication_timestamp_present", "publication_timestamp_verified", "timing_uncertainty_flag", "confirmed_contamination_flag", "heuristic_contamination_flag"):
        frame[column] = parse_bool(frame[column])
    frame["Title_clean"] = frame["Title_clean"].astype(str)
    frame["calendar_index"] = frame["Aligned_Date"].map(calendar.set_index("Date")["calendar_index"]).astype("Int64")
    frame["valid_alignment"] = (~frame["alignment_failed"] & ~frame["out_of_calendar_or_invalid_date_flag"] & frame["Aligned_Date"].notna() & frame["calendar_index"].notna())
    frame["headline_empty"] = frame["Title_clean"].eq("")
    columns = ["article_id", "source_row_number", "Date", "Aligned_Date", "calendar_index", "Title_clean", "alignment_mode", "provenance_status", "publication_timestamp_present", "publication_timestamp_verified", "timing_uncertainty_flag", "confirmed_contamination_flag", "heuristic_contamination_flag", "alignment_failed", "out_of_calendar_or_invalid_date_flag", "valid_alignment", "headline_empty"]
    return frame[columns].copy()


def fold_indices(manifest: dict[str, Any], fold_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold = [item for item in manifest["folds"] if item["fold_id"] == fold_id]
    if len(fold) != 1:
        raise ValueError(f"Fold {fold_id} is not uniquely defined")
    train = np.asarray(fold[0]["train_calendar_indices"], dtype=int)
    validation = np.asarray(fold[0]["validation_calendar_indices"], dtype=int)
    locked = np.asarray(manifest["locked_test"]["calendar_indices"], dtype=int)
    if np.intersect1d(train, validation).size or np.intersect1d(np.r_[train, validation], locked).size:
        raise AssertionError("Fold overlaps or locked-test membership detected")
    return train, validation, locked


def fold_scope(articles: pd.DataFrame, train: np.ndarray, validation: np.ndarray, locked: np.ndarray) -> pd.DataFrame:
    allowed = np.r_[train, validation]
    scope = articles.loc[articles["valid_alignment"] & articles["calendar_index"].isin(allowed)].copy()
    if scope["calendar_index"].isin(locked).any():
        raise AssertionError("Locked-test article entered scope")
    train_set = set(train.tolist())
    scope["fold_role"] = scope["calendar_index"].astype(int).map(lambda i: "train" if i in train_set else "validation")
    return scope.reset_index(drop=True)


def inspect_headline_lengths(frame: pd.DataFrame, tokenizer: Any, maximum_length: int) -> pd.DataFrame:
    output = frame.copy()
    output["headline_status"] = np.where(output["headline_empty"], "empty", "candidate")
    output["token_count"] = 0
    output["overlength_headline"] = False
    positions = output.index[~output["headline_empty"]]
    if len(positions):
        encoded = tokenizer(output.loc[positions, "Title_clean"].tolist(), add_special_tokens=True, truncation=False, padding=False)
        lengths = np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=int)
        output.loc[positions, "token_count"] = lengths
        output.loc[positions, "overlength_headline"] = lengths > maximum_length
        output.loc[positions, "headline_status"] = np.where(lengths > maximum_length, "overlength", "candidate")
    return output


def encode_article_embeddings(frame: pd.DataFrame, tokenizer: Any, model: Any, device: str, batch_size: int = BATCH_SIZE) -> tuple[pd.DataFrame, np.ndarray]:
    output = frame.copy()
    vectors = np.zeros((len(output), EMBEDDING_DIMENSION), dtype=np.float32)
    output["successfully_embedded"] = False
    positions = np.flatnonzero(output["headline_status"].eq("candidate").to_numpy())
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start:start + batch_size]
        inputs = tokenizer(output.iloc[batch_positions]["Title_clean"].tolist(), return_tensors="pt", padding=True, truncation=False)
        if int(inputs["input_ids"].shape[1]) > int(model.config.max_position_embeddings):
            raise AssertionError("Candidate headline batch exceeds model limit")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            cls = model(**inputs).last_hidden_state[:, 0, :].detach().to("cpu").numpy()
        if cls.shape != (len(batch_positions), EMBEDDING_DIMENSION) or not np.isfinite(cls).all() or not np.any(cls):
            raise AssertionError("Invalid RoBERTa CLS output")
        vectors[batch_positions] = cls.astype(np.float32, copy=False)
        output.loc[output.index[batch_positions], "successfully_embedded"] = True
    return output, vectors


def aggregate_daily(calendar: pd.DataFrame, availability: pd.DataFrame, articles: pd.DataFrame, vectors: np.ndarray, train: np.ndarray, validation: np.ndarray) -> pd.DataFrame:
    output = calendar.merge(availability, on=["Date", "calendar_index"], validate="one_to_one")
    allowed = set(np.r_[train, validation].tolist())
    train_set = set(train.tolist())
    output["scope_status"] = output["calendar_index"].map(lambda i: "train" if i in train_set else "validation" if i in allowed else "not_fit_or_transformed")
    output["within_fold_scope"] = output["calendar_index"].isin(allowed)
    output["empty_news_day"] = output["n_articles"].eq(0)
    output["n_embedded_articles"] = 0
    output["n_invalid_headlines"] = 0
    output["n_overlength_headlines"] = 0
    output["feature_available"] = False
    output["unavailable_due_to_scope"] = (~output["within_fold_scope"]) & output["n_articles"].gt(0)
    output["embedding_unavailable_nonempty"] = False
    names = [f"roberta_{i}" for i in range(1, EMBEDDING_DIMENSION + 1)]
    output = pd.concat([output, pd.DataFrame(np.nan, index=output.index, columns=names)], axis=1)
    for index, group in articles.groupby("calendar_index", sort=False):
        row = output["calendar_index"].eq(int(index))
        output.loc[row, "n_embedded_articles"] = int(group["successfully_embedded"].sum())
        output.loc[row, "n_invalid_headlines"] = int(group["headline_status"].eq("empty").sum())
        output.loc[row, "n_overlength_headlines"] = int(group["headline_status"].eq("overlength").sum())
        success = group["successfully_embedded"].to_numpy(dtype=bool)
        if success.any():
            output.loc[row, names] = vectors[group.index.to_numpy()[success]].mean(axis=0)
            output.loc[row, "feature_available"] = True
    output.loc[output["empty_news_day"], names] = 0.0
    output.loc[output["empty_news_day"] & output["within_fold_scope"], "feature_available"] = True
    output["embedding_unavailable_nonempty"] = output["n_articles"].gt(0) & ~output["feature_available"]
    return output


def save_article_output(frame: pd.DataFrame, vectors: np.ndarray, vector_path: Path, metadata_path: Path) -> None:
    metadata = frame.drop(columns=["Title_clean"], errors="ignore").copy()
    metadata["Aligned_Date"] = metadata["Aligned_Date"].dt.strftime("%Y-%m-%d")
    np.savez_compressed(vector_path, article_id=metadata["article_id"].astype(str).to_numpy(), vectors=vectors.astype(np.float32), calendar_index=metadata["calendar_index"].astype(int).to_numpy())
    metadata.to_csv(metadata_path, index=False)


def load_frozen_model() -> tuple[Any, Any, str]:
    if not (MODEL_SNAPSHOT / "model.safetensors").exists() or not (MODEL_SNAPSHOT / "config.json").exists():
        raise FileNotFoundError(f"Pinned local RoBERTa snapshot is incomplete: {MODEL_SNAPSHOT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPOSITORY, revision=MODEL_REVISION, cache_dir=str(CACHE_DIR), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_REPOSITORY, revision=MODEL_REVISION, cache_dir=str(CACHE_DIR), local_files_only=True, use_safetensors=True, add_pooling_layer=False)
    config = model.config
    if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.vocab_size, config.max_position_embeddings) != (768, 12, 12, 50265, 514):
        raise AssertionError("Unexpected pinned RoBERTa configuration")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval().to(DEVICE)
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise AssertionError("RoBERTa model is not frozen/eval")
    return tokenizer, model, DEVICE


def run_module5d() -> dict[str, Any]:
    manifest = load_fold_manifest()
    calendar = load_calendar()
    tokenizer, model, device = load_frozen_model()
    model_files = {name: MODEL_SNAPSHOT / name for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json")}
    source_paths = {"calendar": CALENDAR_PATH, "fold_manifest": FOLD_MANIFEST_PATH, "module4_manifest": MODULE4_MANIFEST_PATH, "articles_strict": INTERMEDIATE / "articles_aligned_strict.csv", "articles_same_date": INTERMEDIATE / "articles_aligned_same_date.csv", **{f"model_{key}": value for key, value in model_files.items()}}
    summaries: dict[str, dict[str, Any]] = {}
    outputs: list[dict[str, Any]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for timing_mode in ("strict", "same_date"):
        articles = load_aligned_articles(timing_mode, calendar)
        availability = load_availability(timing_mode, calendar)
        valid_counts = articles.loc[articles["valid_alignment"]].groupby("calendar_index").size()
        expected = availability.set_index("calendar_index")["n_articles"]
        if not valid_counts.reindex(expected.index, fill_value=0).astype(int).equals(expected.astype(int)):
            raise AssertionError(f"Article counts do not match availability: {timing_mode}")
        summaries[timing_mode] = {}
        for fold_id in (1, 2, 3):
            train, validation, locked = fold_indices(manifest, fold_id)
            scope = inspect_headline_lengths(fold_scope(articles, train, validation, locked), tokenizer, int(model.config.max_position_embeddings))
            if scope["calendar_index"].isin(locked).any():
                raise AssertionError("Locked-test article entered RoBERTa scope")
            encoded, vectors = encode_article_embeddings(scope, tokenizer, model, device)
            daily = aggregate_daily(calendar, availability, encoded, vectors, train, validation)
            if len(daily) != CALENDAR_ROWS:
                raise AssertionError("Daily output did not preserve 1,202 calendar rows")
            stem = f"roberta_{timing_mode}_fold{fold_id}"
            vector_path = OUTPUT_DIR / f"{stem}_article_vectors.npz"
            metadata_path = OUTPUT_DIR / f"{stem}_article_metadata.csv"
            daily_path = OUTPUT_DIR / f"{stem}_daily.csv"
            save_article_output(encoded, vectors, vector_path, metadata_path)
            daily.to_csv(daily_path, index=False)
            train_mask = encoded["fold_role"].eq("train")
            validation_mask = encoded["fold_role"].eq("validation")
            train_dates = calendar.loc[calendar["calendar_index"].isin(train), "Date"]
            validation_dates = calendar.loc[calendar["calendar_index"].isin(validation), "Date"]
            summaries[timing_mode][str(fold_id)] = {
                "fold_id": fold_id, "timing_mode": timing_mode,
                "train_calendar_indices": train.tolist(), "validation_calendar_indices": validation.tolist(),
                "train_date_range": [train_dates.min().strftime("%Y-%m-%d"), train_dates.max().strftime("%Y-%m-%d")],
                "validation_date_range": [validation_dates.min().strftime("%Y-%m-%d"), validation_dates.max().strftime("%Y-%m-%d")],
                "training_article_count": int(train_mask.sum()), "validation_article_count": int(validation_mask.sum()),
                "training_successfully_embedded_articles": int(encoded.loc[train_mask, "successfully_embedded"].sum()), "validation_successfully_embedded_articles": int(encoded.loc[validation_mask, "successfully_embedded"].sum()),
                "empty_headline_count": int(encoded["headline_status"].eq("empty").sum()), "overlength_headline_count": int(encoded["headline_status"].eq("overlength").sum()), "failed_article_count": int((~encoded["successfully_embedded"]).sum()),
                "embedding_dimensions": EMBEDDING_DIMENSION, "article_output_shape": list(vectors.shape), "daily_output_shape": list(daily.shape), "device": device, "batch_size": BATCH_SIZE, "locked_test_articles_used": False, "locked_test_calendar_used": False,
                "model_safetensors_sha256": sha256_file(model_files["model.safetensors"]),
                "article_vector_path": str(vector_path.relative_to(HANA)), "article_vector_sha256": sha256_file(vector_path), "article_metadata_path": str(metadata_path.relative_to(HANA)), "article_metadata_sha256": sha256_file(metadata_path), "daily_output_path": str(daily_path.relative_to(HANA)), "daily_output_sha256": sha256_file(daily_path),
            }
            outputs.append({"fold_id": fold_id, "timing_mode": timing_mode, "path": str(daily_path.relative_to(HANA)), "rows": len(daily)})
    result = {
        "module": "phase5_module5d2", "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION, "model_class": "RobertaModel via AutoModel", "text_source": "Title_clean only", "embedding_method": "last_hidden_state[:, 0, :]", "embedding_dimensions": EMBEDDING_DIMENSION, "model_config": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "vocab_size": 50265, "max_position_embeddings": 514}, "weights_frozen": True, "pooler_created": False, "aggregation": "uniform mean of successfully embedded article-level CLS vectors", "calendar_rows_preserved": CALENDAR_ROWS, "batch_size": BATCH_SIZE, "device": device, "locked_test_articles_used": False, "locked_test_calendar_used": False, "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()}, "summaries": summaries, "outputs": outputs,
    }
    manifest_path = MANIFESTS / "module5d_roberta_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_module5d(), indent=2, sort_keys=True))
