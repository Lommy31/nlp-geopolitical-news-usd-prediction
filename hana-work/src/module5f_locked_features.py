"""Module 5F: leakage-safe locked-test feature transformation.

This module never fits a transformation on locked-test data.  It reuses the
fold-3 Word2Vec model, the frozen FinBERT/RoBERTa encoders, and the accepted
article alignment.  B2 is reported as unavailable because Module 5A did not
persist its fitted vectorizer/SVD objects.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
INTERMEDIATE = HANA / "data/intermediate"
PROCESSED = HANA / "data/processed"
MANIFESTS = HANA / "data/manifests"
OUT = PROCESSED / "module5f_locked"
ARTICLE_OUT = OUT / "articles"
CALENDAR_ROWS = 1202
LOCKED_START = 961
LOCKED_END = 1200  # index 1201 has no target and is retained in metadata only
FOLD_ID = 3
SEED = 42
EMBEDDING_DIM = 768
BASE_FINANCIAL = ["return_t", "return_lag1", "return_lag2", "return_lag3", "return_lag5", "volatility_5d", "volatility_20d", "ma_ratio_5d", "ma_ratio_20d", "momentum_5d", "momentum_10d", "rsi_14"]
AVAILABILITY = ["n_articles", "has_news", "log1p_n_articles"]
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_calendar() -> pd.DataFrame:
    frame = pd.read_csv(INTERMEDIATE / "master_calendar_news_windows.csv", parse_dates=["Date"])
    if len(frame) != CALENDAR_ROWS or frame["calendar_index"].tolist() != list(range(CALENDAR_ROWS)):
        raise AssertionError("Invalid canonical calendar")
    return frame[["Date", "calendar_index"]]


def locked_indices() -> np.ndarray:
    manifest = read_json(MANIFESTS / "module3_fold_manifest.json")
    indices = np.asarray(manifest["locked_test"]["calendar_indices"], dtype=int)
    if indices.tolist() != list(range(LOCKED_START, CALENDAR_ROWS)):
        raise AssertionError("Locked-test membership differs from Module 3 manifest")
    return indices


def load_availability(timing_mode: str) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / f"b0_{timing_mode}_predictors.csv")
    if len(frame) != CALENDAR_ROWS or not set(AVAILABILITY) <= set(frame):
        raise AssertionError("Invalid availability source")
    result = frame[AVAILABILITY].copy()
    result["has_news"] = result["has_news"].astype(str).str.lower().isin({"true", "1"})
    if not result["has_news"].eq(result["n_articles"].gt(0)).all():
        raise AssertionError("Availability indicators disagree")
    return result


def load_articles(timing_mode: str) -> pd.DataFrame:
    frame = pd.read_csv(INTERMEDIATE / f"articles_aligned_{timing_mode}.csv", keep_default_na=False, na_filter=False)
    required = {"article_id", "Title_clean", "Date", "Aligned_Date", "alignment_failed", "out_of_calendar_or_invalid_date_flag"}
    if not required <= set(frame) or frame["article_id"].duplicated().any():
        raise AssertionError("Invalid aligned article corpus")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Aligned_Date"] = pd.to_datetime(frame["Aligned_Date"], errors="coerce")
    bad = frame["alignment_failed"].astype(str).str.lower().isin({"true", "1"}) | frame["out_of_calendar_or_invalid_date_flag"].astype(str).str.lower().isin({"true", "1"})
    calendar = load_calendar()
    date_map = calendar.set_index("Date")["calendar_index"]
    frame["calendar_index"] = frame["Aligned_Date"].map(date_map)
    valid = ~bad & frame["calendar_index"].notna() & frame["Title_clean"].astype(str).ne("")
    frame = frame.loc[valid].copy()
    frame["calendar_index"] = frame["calendar_index"].astype(int)
    return frame.reset_index(drop=True)


def locked_articles(timing_mode: str) -> pd.DataFrame:
    frame = load_articles(timing_mode)
    return frame[frame["calendar_index"].between(LOCKED_START, LOCKED_END)].copy().reset_index(drop=True)


def _save_article_vectors(model_name: str, timing_mode: str, articles: pd.DataFrame, vectors: np.ndarray, success: np.ndarray) -> dict[str, Any]:
    ARTICLE_OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_{timing_mode}_locked_fold3"
    vector_path = ARTICLE_OUT / f"{stem}_article_vectors.npz"
    metadata_path = ARTICLE_OUT / f"{stem}_article_metadata.csv"
    meta = articles[["article_id", "Date", "Aligned_Date", "calendar_index"]].copy()
    meta["successfully_embedded"] = success
    meta["Date"] = meta["Date"].dt.strftime("%Y-%m-%d")
    meta["Aligned_Date"] = meta["Aligned_Date"].dt.strftime("%Y-%m-%d")
    np.savez_compressed(vector_path, article_id=meta["article_id"].astype(str).to_numpy(), calendar_index=meta["calendar_index"].to_numpy(dtype=np.int64), vectors=vectors.astype(np.float32))
    meta.to_csv(metadata_path, index=False)
    return {"article_vectors": str(vector_path.relative_to(HANA)), "article_metadata": str(metadata_path.relative_to(HANA)), "article_vectors_sha256": sha256_file(vector_path), "article_metadata_sha256": sha256_file(metadata_path), "article_count": len(meta), "embedded_count": int(success.sum())}


def _uniform_daily(model_name: str, timing_mode: str, articles: pd.DataFrame, vectors: np.ndarray, success: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    calendar = load_calendar(); availability = load_availability(timing_mode); locked = locked_indices()
    output = pd.concat([calendar.iloc[locked].reset_index(drop=True), availability.iloc[locked].reset_index(drop=True)], axis=1)
    output["scope_status"] = "locked_test_transformed"
    output["feature_available"] = False
    output["empty_news_day"] = output["n_articles"].eq(0)
    output["n_embedded_articles"] = 0
    output["embedding_unavailable_nonempty"] = False
    names = [f"{model_name}_{i}" for i in range(1, vectors.shape[1] + 1)]
    output[names] = np.nan
    good = articles.loc[success].copy(); good["_position"] = np.flatnonzero(success)[good.index]
    for idx, group in good.groupby("calendar_index", sort=False):
        row = output["calendar_index"].eq(int(idx))
        mean = vectors[group["_position"].to_numpy()].mean(axis=0)
        output.loc[row, names] = mean
        output.loc[row, "n_embedded_articles"] = len(group)
        output.loc[row, "feature_available"] = True
    output.loc[output["empty_news_day"], names] = 0.0
    output.loc[output["empty_news_day"], "feature_available"] = True
    output["embedding_unavailable_nonempty"] = output["n_articles"].gt(0) & ~output["feature_available"]
    return output, {"aggregation": "uniform_mean", "empty_news_days": int(output["empty_news_day"].sum()), "unavailable_nonempty_days": int(output["embedding_unavailable_nonempty"].sum())}


def _load_fold3_vectors(model_name: str, timing_mode: str) -> tuple[pd.DataFrame, np.ndarray]:
    base = PROCESSED / model_name if model_name == "finbert" else HANA / "embeddings/roberta"
    stem = f"{model_name}_{timing_mode}_fold3"
    metadata = pd.read_csv(base / f"{stem}_article_metadata.csv", parse_dates=["Date", "Aligned_Date"])
    vectors = np.load(base / f"{stem}_article_vectors.npz")["vectors"].astype(np.float32)
    return metadata, vectors


def _decay_daily(model_name: str, timing_mode: str, current_articles: pd.DataFrame, current_vectors: np.ndarray, current_success: np.ndarray, half_life: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    calendar = load_calendar(); availability = load_availability(timing_mode); locked = locked_indices()
    old_meta, old_vec = _load_fold3_vectors(model_name, timing_mode)
    old_success = old_meta["successfully_embedded"].astype(bool).to_numpy()
    old_meta = old_meta.loc[old_success].copy().reset_index(drop=True); old_vec = old_vec[old_success]
    new_meta = current_articles.loc[current_success].copy().reset_index(drop=True); new_vec = current_vectors[current_success]
    old_meta["Date"] = pd.to_datetime(old_meta["Date"]); new_meta["Date"] = pd.to_datetime(new_meta["Date"])
    meta = pd.concat([old_meta[["article_id", "Date", "Aligned_Date", "calendar_index"]], new_meta[["article_id", "Date", "Aligned_Date", "calendar_index"]]], ignore_index=True)
    vec = np.vstack([old_vec, new_vec])
    output = pd.concat([calendar.iloc[locked].reset_index(drop=True), availability.iloc[locked].reset_index(drop=True)], axis=1)
    output["scope_status"] = "locked_test_transformed"; output["feature_available"] = False; output["empty_news_day"] = output["n_articles"].eq(0); output["n_embedded_articles"] = 0; output["embedding_unavailable_nonempty"] = False
    names = [f"{model_name}_decay_hl{half_life}_{i}" for i in range(1, vec.shape[1] + 1)]; output[names] = np.nan
    dates = pd.DatetimeIndex(calendar["Date"])
    lam = np.log(2.0) / half_life
    for idx in locked:
        row_date = calendar.loc[calendar["calendar_index"].eq(int(idx)), "Date"].iloc[0]
        row = output["calendar_index"].eq(int(idx))
        if bool(output.loc[row, "empty_news_day"].iloc[0]):
            output.loc[row, names] = 0.0; output.loc[row, "feature_available"] = True; continue
        eligible = (meta["Aligned_Date"] <= row_date) & (meta["Date"] <= row_date)
        if not eligible.any():
            output.loc[row, "embedding_unavailable_nonempty"] = True; continue
        selected = meta.loc[eligible]
        positions = np.flatnonzero(eligible.to_numpy())
        left = dates.searchsorted(pd.to_datetime(selected["Date"]).to_numpy(), side="right")
        right = int(dates.searchsorted(row_date, side="right")); ages = right - left
        weights = np.exp(-lam * ages.astype(float)); weights = weights / weights.sum()
        output.loc[row, names] = np.average(vec[positions].astype(np.float64), axis=0, weights=weights).astype(np.float32)
        output.loc[row, "n_embedded_articles"] = len(positions); output.loc[row, "feature_available"] = True
    output.loc[output["empty_news_day"], names] = 0.0; output.loc[output["empty_news_day"], "feature_available"] = True
    output["embedding_unavailable_nonempty"] = output["n_articles"].gt(0) & ~output["feature_available"]
    return output, {"aggregation": "normalized_exponential_decay", "half_life_trading_days": half_life, "lambda": float(lam), "empty_news_days": int(output["empty_news_day"].sum()), "unavailable_nonempty_days": int(output["embedding_unavailable_nonempty"].sum())}


def _encode_transformer(model_name: str, timing_mode: str, articles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if model_name == "finbert":
        repo, revision = "ProsusAI/finbert", "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
        kwargs = {}
    else:
        repo, revision = "FacebookAI/roberta-base", "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
        kwargs = {"add_pooling_layer": False}
    cache = HANA / "cache/huggingface/hub"
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, cache_dir=str(cache), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(repo, revision=revision, cache_dir=str(cache), local_files_only=True, use_safetensors=True, **kwargs)
    for parameter in model.parameters(): parameter.requires_grad = False
    model.eval().to(DEVICE)
    texts = articles["Title_clean"].astype(str).tolist(); success = np.ones(len(texts), dtype=bool); vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    encoded = tok(texts, add_special_tokens=True, truncation=False, padding=False)
    lengths = np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=int)
    if np.any(lengths > int(model.config.max_position_embeddings)):
        success[lengths > int(model.config.max_position_embeddings)] = False
    positions = np.flatnonzero(success)
    for start in range(0, len(positions), 32):
        pos = positions[start:start + 32]; inputs = tok([texts[i] for i in pos], return_tensors="pt", padding=True, truncation=False)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.inference_mode(): out = model(**inputs).last_hidden_state[:, 0, :].detach().cpu().numpy()
        vectors[pos] = out.astype(np.float32); 
        if not np.isfinite(out).all() or not np.any(out): raise AssertionError("Invalid transformer embedding")
    return vectors, success, {"model_repository": repo, "model_revision": revision, "device": DEVICE, "overlength_count": int((~success).sum())}


def _word2vec_transform(timing_mode: str, articles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = HANA / "embeddings/word2vec" / f"word2vec_{timing_mode}_fold3.model"
    model = Word2Vec.load(str(path)); vectors = np.zeros((len(articles), 100), dtype=np.float32); success = np.zeros(len(articles), dtype=bool)
    for i, title in enumerate(articles["Title_clean"].astype(str)):
        tokens = TOKEN_PATTERN.findall(title.lower()); iv = [token for token in tokens if token in model.wv.key_to_index]
        if iv: vectors[i] = model.wv[iv].mean(axis=0); success[i] = True
    return vectors, success, {"model_path": str(path.relative_to(HANA)), "model_sha256": sha256_file(path), "dimension": 100}


def run_module5f() -> dict[str, Any]:
    locked = locked_indices(); summaries = []; outputs = []
    source_paths = [MANIFESTS / "module3_fold_manifest.json", INTERMEDIATE / "master_calendar_news_windows.csv"]
    for timing in ("strict", "same_date"):
        articles = locked_articles(timing)
        for model_name in ("word2vec", "finbert", "roberta"):
            vectors, success, model_info = _word2vec_transform(timing, articles) if model_name == "word2vec" else _encode_transformer(model_name, timing, articles)
            article_info = _save_article_vectors(model_name, timing, articles, vectors, success)
            daily, info = _uniform_daily(model_name, timing, articles, vectors, success)
            path = OUT / f"{model_name}_{timing}_locked_fold3_uniform_daily.csv"; OUT.mkdir(parents=True, exist_ok=True); daily.to_csv(path, index=False)
            record = {"model": model_name, "timing_mode": timing, "fold_id": 3, **model_info, **article_info, **info, "daily_path": str(path.relative_to(HANA)), "daily_sha256": sha256_file(path), "daily_shape": list(daily.shape), "locked_test_rows": len(daily)}
            summaries.append(record); outputs.append(record)
            if model_name in {"finbert", "roberta"}:
                for hl in (1, 3, 5):
                    decay, decay_info = _decay_daily(model_name, timing, articles, vectors, success, hl)
                    dpath = OUT / f"{model_name}_{timing}_locked_fold3_hl{hl}d_daily.csv"; decay.to_csv(dpath, index=False)
                    summaries.append({"model": model_name, "timing_mode": timing, "fold_id": 3, **model_info, **article_info, **decay_info, "daily_path": str(dpath.relative_to(HANA)), "daily_sha256": sha256_file(dpath), "daily_shape": list(decay.shape), "locked_test_rows": len(decay)})
    manifest = {"module": "phase5_module5f", "fold_id": 3, "locked_calendar_indices": locked.tolist(), "locked_test_rows": len(locked), "models": ["word2vec", "finbert", "roberta"], "timing_modes": ["strict", "same_date"], "uniform_only_for_word2vec": True, "b2_status": "unavailable", "b2_reason": "Module 5A accepted artifacts contain daily outputs but no persisted fitted TF-IDF vectorizer or SVD object; refitting is prohibited", "source_hashes": {str(p.relative_to(HANA)): sha256_file(p) for p in source_paths}, "summaries": summaries}
    MANIFESTS.mkdir(parents=True, exist_ok=True); path = MANIFESTS / "module5f_locked_features_manifest.json"; path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_sha256"] = sha256_file(path); path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__": print(json.dumps(run_module5f(), indent=2, sort_keys=True))
