"""Module 6A: assemble leakage-safe development datasets from accepted features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
PROCESSED = HANA / "data/processed"
INTERMEDIATE = HANA / "data/intermediate"
MANIFESTS = HANA / "data/manifests"
OUTPUT_DIR = PROCESSED / "module6"
CALENDAR_ROWS = 1202
BASE_FINANCIAL = ["return_t", "return_lag1", "return_lag2", "return_lag3", "return_lag5", "volatility_5d", "volatility_20d", "ma_ratio_5d", "ma_ratio_20d", "momentum_5d", "momentum_10d", "rsi_14"]
AVAILABILITY = ["n_articles", "has_news", "log1p_n_articles"]
BASIC_NLP = ["vader_title_compound", "vader_title_pos", "vader_title_neg", "lm_title_polarity", "lm_title_subjectivity", "title_sentiment_decay3", "title_sentiment_decay5", "title_shock_extremity", "lm_title_shock", "sent_vol_interact", "vader_body_compound", "vader_body_pos", "vader_body_neg", "lm_body_polarity", "lm_body_subjectivity"]
METADATA = ["Date", "calendar_index", "split_role", "target_date", "next_Date", "target_direction", "target_available", "is_flat_movement", "is_warmup", "supervised_eligible", "module3_split", "experiment_id", "timing_mode", "fold_id"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fold_manifest() -> dict[str, Any]:
    return read_json(MANIFESTS / "module3_fold_manifest.json")


def load_calendar() -> pd.DataFrame:
    frame = pd.read_csv(INTERMEDIATE / "master_calendar_news_windows.csv", parse_dates=["Date", "next_Date"])
    if len(frame) != CALENDAR_ROWS or frame["calendar_index"].tolist() != list(range(CALENDAR_ROWS)):
        raise AssertionError("Invalid master calendar")
    return frame


def load_metadata(timing_mode: str) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / f"baseline_{timing_mode}_metadata.csv", parse_dates=["Date", "next_Date"])
    required = {"Date", "calendar_index", "next_Date", "target_direction", "target_available", "is_flat_movement", "is_warmup", "supervised_eligible", "module3_split"}
    if len(frame) != CALENDAR_ROWS or not required <= set(frame):
        raise AssertionError("Invalid Module 4 target metadata")
    if frame["calendar_index"].tolist() != list(range(CALENDAR_ROWS)) or frame["calendar_index"].duplicated().any():
        raise AssertionError("Target metadata is not a unique master-calendar index")
    frame["target_date"] = frame["next_Date"]
    return frame


def fold_indices(manifest: dict[str, Any], fold_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [x for x in manifest["folds"] if x["fold_id"] == fold_id]
    if len(rows) != 1:
        raise ValueError(f"Fold {fold_id} is not unique")
    train = np.asarray(rows[0]["train_calendar_indices"], dtype=int)
    validation = np.asarray(rows[0]["validation_calendar_indices"], dtype=int)
    locked = np.asarray(manifest["locked_test"]["calendar_indices"], dtype=int)
    if np.intersect1d(train, validation).size or np.intersect1d(np.r_[train, validation], locked).size:
        raise AssertionError("Fold overlap or locked-test overlap")
    return train, validation, locked


def parse_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    return s.astype(str).str.lower().isin({"true", "1"})


def feature_source(experiment: str, timing_mode: str, fold_id: int, authorized_indices: np.ndarray, half_life: int | None = None) -> tuple[pd.DataFrame, list[str], str, Path]:
    if experiment == "A2":
        path = PROCESSED / f"a2_{timing_mode}_predictors.csv"; columns = BASE_FINANCIAL; source = "Module 4 A2"
    elif experiment == "B0":
        path = PROCESSED / f"b0_{timing_mode}_predictors.csv"; columns = BASE_FINANCIAL + AVAILABILITY; source = "Module 4 B0"
    elif experiment == "B1":
        path = PROCESSED / f"b1_{timing_mode}_predictors.csv"; columns = BASE_FINANCIAL + AVAILABILITY + BASIC_NLP; source = "Module 4 B1"
    elif experiment == "B2":
        path = PROCESSED / f"tfidf_svd/b2_tfidf_svd_{timing_mode}_fold{fold_id}_daily.csv"; base_path = PROCESSED / f"b2_{timing_mode}_predictors.csv"; columns = BASE_FINANCIAL + AVAILABILITY + [f"tfidf_svd_{i}" for i in range(1, 6)]; source = "Module 4 B2 + Module 5A"
    elif experiment == "C1":
        path = PROCESSED / f"word2vec/word2vec_{timing_mode}_fold{fold_id}_daily.csv"; base_path = PROCESSED / f"b0_{timing_mode}_predictors.csv"; columns = BASE_FINANCIAL + AVAILABILITY + [f"word2vec_{i}" for i in range(1, 101)]; source = "Module 4 B0 + Module 5B Word2Vec"
    elif experiment == "C2":
        path = PROCESSED / f"finbert/finbert_{timing_mode}_fold{fold_id}_daily.csv"; base_path = PROCESSED / f"b0_{timing_mode}_predictors.csv"; columns = BASE_FINANCIAL + AVAILABILITY + [f"finbert_{i}" for i in range(1, 769)]; source = "Module 4 B0 + Module 5C FinBERT uniform mean"
    elif experiment == "C3":
        path = HANA / f"embeddings/roberta/roberta_{timing_mode}_fold{fold_id}_daily.csv"; base_path = PROCESSED / f"b0_{timing_mode}_predictors.csv"; columns = BASE_FINANCIAL + AVAILABILITY + [f"roberta_{i}" for i in range(1, 769)]; source = "Module 4 B0 + Module 5D RoBERTa uniform mean"
    elif experiment in {"C4", "C5"}:
        model = "finbert" if experiment == "C4" else "roberta"
        path = HANA / f"embeddings/decay/{model}_decay_{timing_mode}_fold{fold_id}_hl{half_life}d_daily.csv"; base_path = PROCESSED / f"b0_{timing_mode}_predictors.csv"
        columns = BASE_FINANCIAL + AVAILABILITY + [f"{model}_decay_hl{half_life}_{i}" for i in range(1, 769)]
        source = f"Module 5E {model} decay {half_life}d"
    else:
        raise ValueError(experiment)
    header = pd.read_csv(path, nrows=0).columns
    read_rows = int(np.max(authorized_indices)) + 1
    frame = pd.read_csv(path, nrows=read_rows, parse_dates=["Date"] if "Date" in header else None)
    if len(frame) != read_rows:
        raise AssertionError(f"Feature artifact ends before authorized development rows: {path}")
    if "calendar_index" not in frame:
        frame.insert(0, "calendar_index", np.arange(read_rows, dtype=int))
    if frame["calendar_index"].tolist() != list(range(read_rows)):
        raise AssertionError(f"Feature calendar indices are not canonical: {path}")
    if "base_path" in locals():
        base_header = pd.read_csv(base_path, nrows=0).columns
        base = pd.read_csv(base_path, nrows=read_rows, parse_dates=["Date"] if "Date" in base_header else None)
        if len(base) != read_rows:
            raise AssertionError(f"Base predictor artifact ends before authorized development rows: {base_path}")
        if "calendar_index" not in base:
            base.insert(0, "calendar_index", np.arange(read_rows, dtype=int))
        if base["calendar_index"].tolist() != list(range(read_rows)):
            raise AssertionError(f"Base predictor indices are not canonical: {base_path}")
        merged = base[["calendar_index"] + [c for c in BASE_FINANCIAL + AVAILABILITY if c in base.columns]].copy()
        for column in columns:
            if column not in merged and column in frame:
                merged[column] = frame[column].to_numpy()
        frame = merged
    if not set(columns) <= set(frame.columns):
        raise AssertionError(f"Missing feature columns in {path}: {set(columns) - set(frame.columns)}")
    return frame, columns, source, path


def last_observed_direction(metadata: pd.DataFrame) -> pd.Series:
    eligible = metadata["supervised_eligible"].astype(bool)
    direction = metadata["target_direction"].map(lambda x: str(x).lower() in {"true", "1"})
    result = pd.Series(np.nan, index=metadata.index, dtype="object")
    previous = np.nan
    for i in metadata.index:
        result.loc[i] = previous
        if eligible.loc[i]:
            previous = bool(direction.loc[i])
    return result


def assemble_dataset(experiment: str, timing_mode: str, fold_id: int, manifest: dict[str, Any], half_life: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    calendar = load_calendar(); metadata = load_metadata(timing_mode)
    train, validation, locked = fold_indices(manifest, fold_id)
    role = pd.Series("excluded", index=metadata.index)
    role.loc[metadata["calendar_index"].isin(train)] = "train"
    role.loc[metadata["calendar_index"].isin(validation)] = "validation"
    role.loc[metadata["calendar_index"].isin(locked)] = "locked_test"
    metadata["split_role"] = role
    metadata["experiment_id"] = experiment + (f"_hl{half_life}d" if half_life else "")
    metadata["timing_mode"] = timing_mode; metadata["fold_id"] = fold_id
    if experiment in {"A0", "A1"}:
        columns = []
        features = pd.DataFrame(index=metadata.index)
        source = "Module 3 target metadata"
        path = PROCESSED / f"baseline_{timing_mode}_metadata.csv"
        if experiment == "A1":
            features["last_observed_direction"] = last_observed_direction(metadata); columns = ["last_observed_direction"]
    else:
        authorized = np.r_[train, validation]
        features_frame, columns, source, path = feature_source(experiment, timing_mode, fold_id, authorized, half_life)
        if not set(authorized.tolist()) <= set(features_frame["calendar_index"].tolist()):
            raise AssertionError(f"Feature artifact lacks authorized calendar indices: {path}")
        features = features_frame.set_index("calendar_index").reindex(metadata["calendar_index"])[columns].copy()
        for c in ("has_news",):
            if c in features: features[c] = parse_bool(features[c])
    development = role.isin({"train", "validation"}) & metadata["supervised_eligible"].astype(bool)
    if metadata.loc[development, "calendar_index"].isin(locked).any():
        raise AssertionError("Locked-test row entered dataset")
    result = pd.concat([metadata.loc[development, METADATA].reset_index(drop=True), features.loc[development, columns].reset_index(drop=True)], axis=1)
    if result["calendar_index"].duplicated().any() or len(result) != int(development.sum()):
        raise AssertionError("Duplicate or missing development rows")
    if result["Date"].gt(result["target_date"]).any():
        raise AssertionError("Decision date is after target date")
    if result[columns].columns.duplicated().any() or set(columns) & set(METADATA):
        raise AssertionError("Predictor/metadata column collision")
    info = {"experiment_id": result["experiment_id"].iloc[0] if len(result) else experiment, "experiment": experiment, "feature_source": source, "source_path": str(path.relative_to(HANA)), "feature_columns": columns, "feature_dimension": len(columns), "timing_mode": timing_mode, "fold_id": fold_id, "train_rows": int((result["split_role"] == "train").sum()), "validation_rows": int((result["split_role"] == "validation").sum()), "missing_value_count": int(result[columns].isna().sum().sum()), "unavailable_value_count": int(result[columns].isna().sum().sum()), "target_column": "target_direction", "target_definition": "next JISDOR direction; flat movements excluded by supervised_eligible", "output_columns": list(result.columns)}
    return result, info


def run_module6a() -> dict[str, Any]:
    manifest = read_json(MANIFESTS / "module3_fold_manifest.json"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiments = ["A0", "A1", "A2", "B0", "B1", "B2", "C1", "C2", "C3"] + [f"C4_hl{x}d" for x in (1,3,5)] + [f"C5_hl{x}d" for x in (1,3,5)]
    summaries=[]; outputs=[]
    for timing_mode in ("strict", "same_date"):
        for fold_id in (1,2,3):
            for exp in experiments:
                base = exp.split("_hl")[0]; hl = int(exp.split("_hl")[1][:-1]) if "_hl" in exp else None
                data, info = assemble_dataset(base, timing_mode, fold_id, manifest, hl)
                path = OUTPUT_DIR / f"{exp.lower()}_{timing_mode}_fold{fold_id}.csv"
                data.to_csv(path, index=False); info["output_path"] = str(path.relative_to(HANA)); info["output_sha256"] = sha256_file(path); summaries.append(info); outputs.append(info["output_path"])
    metadata = pd.DataFrame(summaries); metadata_path = OUTPUT_DIR / "module6a_dataset_metadata.csv"; metadata.to_csv(metadata_path, index=False)
    source_paths = [MANIFESTS / "module3_fold_manifest.json", MANIFESTS / "module4_feature_manifest.json", MANIFESTS / "module5a_tfidf_svd_manifest.json", MANIFESTS / "module5b_word2vec_manifest.json", MANIFESTS / "module5c_finbert_manifest.json", MANIFESTS / "module5d_roberta_manifest.json", MANIFESTS / "module5e_decay_manifest.json", INTERMEDIATE / "master_calendar_news_windows.csv"]
    result = {"module":"phase5_module6a", "calendar_rows":CALENDAR_ROWS, "experiments":experiments, "experiment_variant_count":len(experiments), "dataset_count":len(outputs), "metadata_path":str(metadata_path.relative_to(HANA)), "metadata_sha256":sha256_file(metadata_path), "outputs":outputs, "summaries":summaries, "source_manifest_hashes":{p.name:sha256_file(p) for p in source_paths}, "locked_test_rows_in_outputs":False, "preprocessing_fitted":False, "model_training_performed":False}
    (MANIFESTS / "module6a_dataset_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__": print(json.dumps(run_module6a(), indent=2, sort_keys=True))
