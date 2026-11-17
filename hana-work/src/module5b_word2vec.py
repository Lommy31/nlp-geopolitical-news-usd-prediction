"""Phase 5B: fold-local headline-only Word2Vec features.

This module trains one independent Gensim Word2Vec model per timing track and
Module 3 fold.  Only ``Title_clean`` is tokenized.  Articles aligned to the
fold's training calendar indices determine the vocabulary and weights; the
validation articles are inferred with that already-fitted model.  Locked-test
articles are never used or transformed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
INTERMEDIATE = HANA / "data/intermediate"
PROCESSED = HANA / "data/processed"
MANIFESTS = HANA / "data/manifests"
MODEL_DIR = HANA / "embeddings/word2vec"
OUTPUT_DIR = PROCESSED / "word2vec"
FOLD_MANIFEST_PATH = MANIFESTS / "module3_fold_manifest.json"
CALENDAR_PATH = INTERMEDIATE / "master_calendar_news_windows.csv"
MODULE4_MANIFEST_PATH = MANIFESTS / "module4_feature_manifest.json"

CALENDAR_ROWS = 1202
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
WORKERS = 1
SG = 1
SEED = 42
EPOCHS = 8

ARTICLE_COLUMNS = {
    "article_id",
    "Title_clean",
    "Aligned_Date",
    "alignment_mode",
    "alignment_failed",
    "out_of_calendar_or_invalid_date_flag",
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
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize_title(value: str) -> list[str]:
    """Lowercase and tokenize the complete cleaned headline without truncation."""
    return TOKEN_PATTERN.findall(str(value).lower())


def load_fold_manifest(path: Path = FOLD_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_calendar() -> pd.DataFrame:
    calendar = pd.read_csv(CALENDAR_PATH, parse_dates=["Date"])
    required = {"Date", "calendar_index"}
    if not required <= set(calendar.columns):
        raise AssertionError(f"Calendar is missing required columns: {required - set(calendar.columns)}")
    if len(calendar) != CALENDAR_ROWS:
        raise AssertionError(f"Expected {CALENDAR_ROWS} calendar rows, found {len(calendar)}")
    if not calendar["calendar_index"].equals(pd.Series(range(CALENDAR_ROWS), name="calendar_index")):
        raise AssertionError("Calendar indices are not the audited sequential indices")
    if calendar["Date"].duplicated().any() or not calendar["Date"].is_monotonic_increasing:
        raise AssertionError("Calendar dates must be unique and chronological")
    return calendar[["Date", "calendar_index"]].copy()


def load_availability(timing_mode: str, calendar: pd.DataFrame) -> pd.DataFrame:
    path = PROCESSED / f"b0_{timing_mode}_predictors.csv"
    frame = pd.read_csv(path)
    required = {"n_articles", "has_news", "log1p_n_articles"}
    if not required <= set(frame.columns):
        raise AssertionError(f"Availability is missing columns: {required - set(frame.columns)}")
    if len(frame) != CALENDAR_ROWS:
        raise AssertionError(f"Expected {CALENDAR_ROWS} availability rows, found {len(frame)}")
    result = calendar.copy()
    result["n_articles"] = frame["n_articles"].astype(int).to_numpy()
    result["has_news"] = frame["has_news"].astype(bool).to_numpy()
    result["log1p_n_articles"] = frame["log1p_n_articles"].astype(float).to_numpy()
    if not result["has_news"].equals(result["n_articles"].gt(0)):
        raise AssertionError("Availability has_news does not match n_articles")
    return result


def load_aligned_articles(timing_mode: str, calendar: pd.DataFrame) -> pd.DataFrame:
    path = INTERMEDIATE / f"articles_aligned_{timing_mode}.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    if not ARTICLE_COLUMNS <= set(frame.columns):
        raise AssertionError(f"Article corpus is missing columns: {ARTICLE_COLUMNS - set(frame.columns)}")
    if FORBIDDEN_COLUMNS & set(frame.columns):
        raise AssertionError(f"Article corpus exposes forbidden target/split columns: {FORBIDDEN_COLUMNS & set(frame.columns)}")
    frame = frame[list(ARTICLE_COLUMNS)].copy()
    frame["Aligned_Date"] = pd.to_datetime(frame["Aligned_Date"], errors="coerce")
    valid = (~frame["alignment_failed"].astype(bool)) & (~frame["out_of_calendar_or_invalid_date_flag"].astype(bool))
    valid &= frame["Aligned_Date"].notna() & frame["Title_clean"].ne("")
    frame = frame.loc[valid].copy()
    if frame["article_id"].duplicated().any():
        raise AssertionError("Aligned article IDs must be unique after Module 2 deduplication")
    date_map = calendar.set_index("Date")["calendar_index"]
    frame["calendar_index"] = frame["Aligned_Date"].map(date_map)
    if frame["calendar_index"].isna().any():
        raise AssertionError("A valid aligned article did not map to the audited calendar")
    frame["calendar_index"] = frame["calendar_index"].astype(int)
    frame["tokens"] = frame["Title_clean"].map(tokenize_title)
    if frame["tokens"].map(len).eq(0).any():
        raise AssertionError("A nonempty Title_clean produced no tokens")
    return frame.reset_index(drop=True)


def fold_indices(manifest: dict[str, Any], fold_id: int) -> tuple[np.ndarray, np.ndarray]:
    matches = [fold for fold in manifest["folds"] if fold["fold_id"] == fold_id]
    if len(matches) != 1:
        raise ValueError(f"Fold {fold_id} is not uniquely defined")
    fold = matches[0]
    train = np.asarray(fold["train_calendar_indices"], dtype=int)
    validation = np.asarray(fold["validation_calendar_indices"], dtype=int)
    locked_test = np.asarray(manifest["locked_test"]["calendar_indices"], dtype=int)
    if np.intersect1d(train, validation).size:
        raise AssertionError("Train and validation calendar membership overlaps")
    if np.intersect1d(np.concatenate([train, validation]), locked_test).size:
        raise AssertionError("Fold training/validation membership overlaps locked test")
    return train, validation


def train_fold_model(articles: pd.DataFrame, train_indices: np.ndarray) -> Word2Vec:
    training = articles.loc[articles["calendar_index"].isin(train_indices)]
    if training.empty:
        raise AssertionError("Cannot train Word2Vec without training articles")
    sentences = training["tokens"].tolist()
    return Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=SG,
        seed=SEED,
        epochs=EPOCHS,
    )


def article_vectors(articles: pd.DataFrame, model: Word2Vec, train_indices: np.ndarray, validation_indices: np.ndarray) -> pd.DataFrame:
    scope = articles.loc[articles["calendar_index"].isin(np.concatenate([train_indices, validation_indices]))].copy()
    train_set = set(train_indices.tolist())
    vectors: list[np.ndarray] = []
    embedded: list[bool] = []
    token_counts: list[int] = []
    in_vocab_counts: list[int] = []
    for tokens in scope["tokens"]:
        in_vocab = [token for token in tokens if token in model.wv.key_to_index]
        vector = model.wv[in_vocab].mean(axis=0) if in_vocab else np.zeros(VECTOR_SIZE, dtype=np.float32)
        vectors.append(np.asarray(vector, dtype=np.float32))
        embedded.append(bool(in_vocab))
        token_counts.append(len(tokens))
        in_vocab_counts.append(len(in_vocab))
    result = scope[["article_id", "Aligned_Date", "calendar_index"]].copy()
    result["fold_role"] = result["calendar_index"].map(lambda value: "train" if value in train_set else "validation")
    result["token_count"] = token_counts
    result["in_vocab_token_count"] = in_vocab_counts
    result["oov_token_count"] = result["token_count"] - result["in_vocab_token_count"]
    result["successfully_embedded"] = embedded
    result["vector"] = vectors
    return result


def aggregate_daily(calendar: pd.DataFrame, availability: pd.DataFrame, article_vectors_frame: pd.DataFrame, train_indices: np.ndarray, validation_indices: np.ndarray) -> pd.DataFrame:
    output = calendar.merge(availability, on=["Date", "calendar_index"], how="left", validate="one_to_one")
    output["n_embedded_articles"] = 0
    output["n_oov_only_articles"] = 0
    output["empty_news_day"] = output["n_articles"].eq(0)
    output["oov_only_news_day"] = False
    output["fold_role"] = "not_fit_or_transformed"
    output["feature_available"] = False
    output.loc[output["calendar_index"].isin(train_indices), "fold_role"] = "train"
    output.loc[output["calendar_index"].isin(validation_indices), "fold_role"] = "validation"
    output.loc[output["calendar_index"].isin(np.concatenate([train_indices, validation_indices])), "feature_available"] = True
    feature_columns = [f"word2vec_{index}" for index in range(1, VECTOR_SIZE + 1)]
    output = pd.concat([output, pd.DataFrame(np.nan, index=output.index, columns=feature_columns)], axis=1)
    scoped = article_vectors_frame.loc[article_vectors_frame["successfully_embedded"]]
    for calendar_index, group in scoped.groupby("calendar_index", sort=False):
        if calendar_index not in set(np.concatenate([train_indices, validation_indices]).tolist()):
            raise AssertionError("An article outside train/validation was transformed")
        mean_vector = np.vstack(group["vector"].to_numpy()).mean(axis=0)
        row_mask = output["calendar_index"].eq(calendar_index)
        output.loc[row_mask, feature_columns] = mean_vector
        output.loc[row_mask, "n_embedded_articles"] = len(group)
    scoped_all = article_vectors_frame.groupby("calendar_index", sort=False)
    for calendar_index, group in scoped_all:
        row_mask = output["calendar_index"].eq(calendar_index)
        output.loc[row_mask, "n_oov_only_articles"] = int((~group["successfully_embedded"]).sum())
    transformed = output["feature_available"]
    output.loc[transformed, "oov_only_news_day"] = (
        output.loc[transformed, "n_articles"].gt(0) & output.loc[transformed, "n_embedded_articles"].eq(0)
    )
    output.loc[transformed & output["n_embedded_articles"].eq(0), feature_columns] = 0.0
    # Empty-news rows require no learned representation and are safe to expose
    # as deterministic zero vectors even outside a train/validation scope.
    output.loc[output["empty_news_day"], feature_columns] = 0.0
    return output


def save_article_vectors(frame: pd.DataFrame, npz_path: Path, metadata_path: Path) -> None:
    metadata = frame.drop(columns=["vector", "tokens"], errors="ignore").copy()
    vectors = np.vstack(frame["vector"].to_numpy()).astype(np.float32)
    np.savez_compressed(
        npz_path,
        article_id=metadata["article_id"].to_numpy(dtype=str),
        aligned_date=metadata["Aligned_Date"].dt.strftime("%Y-%m-%d").to_numpy(dtype=str),
        calendar_index=metadata["calendar_index"].to_numpy(dtype=np.int64),
        vectors=vectors,
    )
    metadata.to_csv(metadata_path, index=False)


def run_module5b() -> dict[str, Any]:
    manifest = load_fold_manifest()
    calendar = load_calendar()
    source_paths = {
        "calendar": CALENDAR_PATH,
        "fold_manifest": FOLD_MANIFEST_PATH,
        "module4_manifest": MODULE4_MANIFEST_PATH,
        "articles_strict": INTERMEDIATE / "articles_aligned_strict.csv",
        "articles_same_date": INTERMEDIATE / "articles_aligned_same_date.csv",
    }
    summaries: dict[str, Any] = {}
    outputs: list[dict[str, Any]] = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for timing_mode in ("strict", "same_date"):
        availability = load_availability(timing_mode, calendar)
        articles = load_aligned_articles(timing_mode, calendar)
        counts = articles.groupby("calendar_index").size()
        expected_counts = availability.set_index("calendar_index")["n_articles"]
        if not counts.reindex(expected_counts.index, fill_value=0).equals(expected_counts.astype(int)):
            raise AssertionError(f"Article counts do not match availability for {timing_mode}")
        summaries[timing_mode] = {}
        for fold_id in (1, 2, 3):
            train_indices, validation_indices = fold_indices(manifest, fold_id)
            model = train_fold_model(articles, train_indices)
            model_path = MODEL_DIR / f"word2vec_{timing_mode}_fold{fold_id}.model"
            model.save(str(model_path))
            model_artifacts = sorted(model_path.parent.glob(model_path.name + "*"))
            vectors_frame = article_vectors(articles, model, train_indices, validation_indices)
            article_npz = OUTPUT_DIR / f"word2vec_{timing_mode}_fold{fold_id}_article_vectors.npz"
            article_metadata = OUTPUT_DIR / f"word2vec_{timing_mode}_fold{fold_id}_article_metadata.csv"
            save_article_vectors(vectors_frame, article_npz, article_metadata)
            daily = aggregate_daily(calendar, availability, vectors_frame, train_indices, validation_indices)
            daily_path = OUTPUT_DIR / f"word2vec_{timing_mode}_fold{fold_id}_daily.csv"
            daily.to_csv(daily_path, index=False)
            train_articles = vectors_frame["fold_role"].eq("train")
            validation_articles = vectors_frame["fold_role"].eq("validation")
            train_dates = calendar.loc[calendar["calendar_index"].isin(train_indices), "Date"]
            validation_dates = calendar.loc[calendar["calendar_index"].isin(validation_indices), "Date"]
            summary = {
                "fold_id": fold_id,
                "timing_mode": timing_mode,
                "train_calendar_indices": train_indices.tolist(),
                "validation_calendar_indices": validation_indices.tolist(),
                "train_date_range": [train_dates.min().strftime("%Y-%m-%d"), train_dates.max().strftime("%Y-%m-%d")],
                "validation_date_range": [validation_dates.min().strftime("%Y-%m-%d"), validation_dates.max().strftime("%Y-%m-%d")],
                "training_article_count": int(train_articles.sum()),
                "validation_article_count": int(validation_articles.sum()),
                "training_successfully_embedded_articles": int(vectors_frame.loc[train_articles, "successfully_embedded"].sum()),
                "validation_successfully_embedded_articles": int(vectors_frame.loc[validation_articles, "successfully_embedded"].sum()),
                "training_oov_only_articles": int((~vectors_frame.loc[train_articles, "successfully_embedded"]).sum()),
                "validation_oov_only_articles": int((~vectors_frame.loc[validation_articles, "successfully_embedded"]).sum()),
                "training_empty_news_days": int(daily.loc[daily["calendar_index"].isin(train_indices), "empty_news_day"].sum()),
                "validation_empty_news_days": int(daily.loc[daily["calendar_index"].isin(validation_indices), "empty_news_day"].sum()),
                "training_oov_only_news_days": int(daily.loc[daily["calendar_index"].isin(train_indices), "oov_only_news_day"].sum()),
                "validation_oov_only_news_days": int(daily.loc[daily["calendar_index"].isin(validation_indices), "oov_only_news_day"].sum()),
                "vocabulary_size": len(model.wv),
                "embedding_dimensions": VECTOR_SIZE,
                "aggregation": "uniform_mean_of_successfully_embedded_article_vectors",
                "locked_test_articles_used": False,
                "model_artifacts": [{"path": str(p.relative_to(HANA)), "sha256": sha256_file(p)} for p in model_artifacts],
                "article_vector_path": str(article_npz.relative_to(HANA)),
                "article_vector_sha256": sha256_file(article_npz),
                "article_metadata_path": str(article_metadata.relative_to(HANA)),
                "article_metadata_sha256": sha256_file(article_metadata),
                "daily_output_path": str(daily_path.relative_to(HANA)),
                "daily_output_sha256": sha256_file(daily_path),
                "daily_output_rows": len(daily),
            }
            summaries[timing_mode][str(fold_id)] = summary
            outputs.append({"timing_mode": timing_mode, "fold_id": fold_id, "path": str(daily_path.relative_to(HANA)), "rows": len(daily)})
    result = {
        "module": "phase5_module5b",
        "text_source": "Title_clean only",
        "text_exclusions": ["Text_clean", "lead extraction", "headline truncation"],
        "tokenizer": "lowercase Unicode word-token extraction with no headline truncation or stopword removal",
        "word2vec_configuration": {
            "vector_size": VECTOR_SIZE, "window": WINDOW, "min_count": MIN_COUNT,
            "workers": WORKERS, "sg": SG, "seed": SEED, "epochs": EPOCHS,
        },
        "aggregation": "uniform mean only",
        "decay_parameters": None,
        "calendar_rows_preserved": CALENDAR_ROWS,
        "global_model_trained": False,
        "locked_test_articles_used": False,
        "source_hashes": {name: sha256_file(path) for name, path in source_paths.items()},
        "fold_manifest_sha256": sha256_file(FOLD_MANIFEST_PATH),
        "module4_manifest_sha256": sha256_file(MODULE4_MANIFEST_PATH),
        "summaries": summaries,
        "outputs": outputs,
    }
    manifest_path = MANIFESTS / "module5b_word2vec_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_module5b(), indent=2, sort_keys=True))
