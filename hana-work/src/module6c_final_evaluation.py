"""Module 6C: final evaluation on the locked test, without refitting."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
PROCESSED = HANA / "data/processed"
MANIFESTS = HANA / "data/manifests"
MODEL_DIR = HANA / "models/module6"
FEATURE_DIR = PROCESSED / "module5f_locked"
RESULT_DIR = HANA / "results/module6c"
FOLD_ID = 3
LOCKED_START = 961
LOCKED_END = 1200
BASE_FINANCIAL = ["return_t", "return_lag1", "return_lag2", "return_lag3", "return_lag5", "volatility_5d", "volatility_20d", "ma_ratio_5d", "ma_ratio_20d", "momentum_5d", "momentum_10d", "rsi_14"]
AVAILABILITY = ["n_articles", "has_news", "log1p_n_articles"]
BASIC_NLP = ["vader_title_compound", "vader_title_pos", "vader_title_neg", "lm_title_polarity", "lm_title_subjectivity", "title_sentiment_decay3", "title_sentiment_decay5", "title_shock_extremity", "lm_title_shock", "sent_vol_interact", "vader_body_compound", "vader_body_pos", "vader_body_neg", "lm_body_polarity", "lm_body_subjectivity"]
EXPERIMENTS = ["A0", "A1", "A2", "B0", "B1", "B2", "C1", "C2", "C3", "C4_hl1d", "C4_hl3d", "C4_hl5d", "C5_hl1d", "C5_hl3d", "C5_hl5d"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def locked_metadata(timing: str) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / f"baseline_{timing}_metadata.csv", parse_dates=["Date", "next_Date"])
    frame["target_date"] = frame["next_Date"]
    locked = frame[frame["calendar_index"].between(LOCKED_START, LOCKED_END)].copy()
    if len(locked) != 240 or locked["calendar_index"].duplicated().any():
        raise AssertionError("Unexpected locked metadata membership")
    eligible = locked["supervised_eligible"].astype(bool)
    if not eligible.any():
        raise AssertionError("Locked test has no supervised rows")
    return locked


def last_observed_direction(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    previous = np.nan
    for idx, row in frame.sort_values("calendar_index").iterrows():
        result.loc[idx] = previous
        if bool(row["supervised_eligible"]):
            previous = float(bool(row["target_direction"]))
    return result


def load_base_predictors(experiment: str, timing: str, metadata: pd.DataFrame) -> pd.DataFrame:
    if experiment == "A2": columns = BASE_FINANCIAL; path = PROCESSED / f"a2_{timing}_predictors.csv"
    elif experiment == "B0": columns = BASE_FINANCIAL + AVAILABILITY; path = PROCESSED / f"b0_{timing}_predictors.csv"
    elif experiment == "B1": columns = BASE_FINANCIAL + AVAILABILITY + BASIC_NLP; path = PROCESSED / f"b1_{timing}_predictors.csv"
    else: raise ValueError(experiment)
    frame = pd.read_csv(path)
    if len(frame) != 1202 or not set(columns) <= set(frame): raise AssertionError(f"Invalid predictor input {path}")
    return frame.loc[metadata.index, columns].reset_index(drop=True)


def load_embedding_features(experiment: str, timing: str, metadata: pd.DataFrame) -> pd.DataFrame:
    if experiment == "C1":
        model, suffix = "word2vec", "uniform"
    elif experiment == "C2": model, suffix = "finbert", "uniform"
    elif experiment == "C3": model, suffix = "roberta", "uniform"
    else:
        model = "finbert" if experiment.startswith("C4") else "roberta"
        suffix = "hl" + experiment.split("hl")[1].replace("d", "") + "d"
    if suffix == "uniform": path = FEATURE_DIR / f"{model}_{timing}_locked_fold3_uniform_daily.csv"
    else: path = FEATURE_DIR / f"{model}_{timing}_locked_fold3_{suffix}_daily.csv"
    frame = pd.read_csv(path)
    if not set(metadata["calendar_index"]) <= set(frame["calendar_index"]): raise AssertionError(f"Feature calendar mismatch: {path}")
    frame = frame.set_index("calendar_index").loc[metadata["calendar_index"]].reset_index()
    if experiment == "C1": names = [f"word2vec_{i}" for i in range(1, 101)]
    elif suffix == "uniform": names = [f"{model}_{i}" for i in range(1, 769)]
    else: names = [f"{model}_decay_{suffix.replace('d', '')}_{i}" for i in range(1, 769)]
    if not set(names) <= set(frame): raise AssertionError(f"Missing embedding columns in {path}")
    return frame[names].copy()


def assemble_test(experiment: str, timing: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    metadata = locked_metadata(timing)
    if experiment == "A0": return metadata, pd.DataFrame(index=metadata.index), []
    full_meta = pd.read_csv(PROCESSED / f"baseline_{timing}_metadata.csv", parse_dates=["Date", "next_Date"])
    if experiment == "A1":
        full_meta["last_observed_direction"] = last_observed_direction(full_meta)
        features = full_meta.loc[metadata.index, ["last_observed_direction"]].reset_index(drop=True)
        return metadata, features, ["last_observed_direction"]
    if experiment in {"A2", "B0", "B1"}:
        features = load_base_predictors(experiment, timing, metadata)
        return metadata, features, list(features.columns)
    if experiment == "B2": raise FileNotFoundError("B2 locked-test TF-IDF/SVD objects were not persisted by Module 5A")
    embedding = load_embedding_features(experiment, timing, metadata)
    base = load_base_predictors("B0", timing, metadata)
    features = pd.concat([base.reset_index(drop=True), embedding.reset_index(drop=True)], axis=1)
    return metadata, features, list(features.columns)


def metric_dict(y_true: np.ndarray, prediction: np.ndarray, score: np.ndarray | None) -> dict[str, Any]:
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, prediction, labels=[False, True], zero_division=0)
    result: dict[str, Any] = {"mcc": float(matthews_corrcoef(y_true, prediction)), "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)), "accuracy": float(accuracy_score(y_true, prediction)), "roc_auc": None, "precision_down": float(precision[0]), "precision_up": float(precision[1]), "recall_down": float(recall[0]), "recall_up": float(recall[1]), "f1_down": float(fscore[0]), "f1_up": float(fscore[1]), "support_down": int(support[0]), "support_up": int(support[1])}
    if score is not None and len(np.unique(y_true)) == 2: result["roc_auc"] = float(roc_auc_score(y_true, score))
    cm = confusion_matrix(y_true, prediction, labels=[False, True])
    result.update({"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])})
    return result


def transform_with_artifact(features: pd.DataFrame, columns: list[str], artifact: dict[str, Any]) -> np.ndarray:
    prep = artifact["preprocessor"]
    expected = prep.get("feature_columns", columns) if prep else columns
    if list(expected) != list(columns): raise AssertionError("Feature columns differ from fitted Module 6B artifact")
    x = features[columns].apply(pd.to_numeric, errors="coerce")
    if prep is None: return x.to_numpy(dtype=float)
    x = prep["imputer"].transform(x); x = prep["scaler"].transform(x)
    if prep.get("pca") is not None: x = prep["pca"].transform(x)
    return x


def artifact_for(experiment: str, timing: str, model_name: str) -> Path:
    stem = experiment.lower()
    suffix = "majority_baseline" if model_name == "majority_baseline" else model_name
    return MODEL_DIR / f"{stem}_{timing}_{suffix}_fold3.pkl"


def run_module6c() -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    module6b = read_json(MANIFESTS / "module6b_training_manifest.json")
    module5f = read_json(MANIFESTS / "module5f_locked_features_manifest.json")
    records: list[dict[str, Any]] = []; prediction_paths: list[dict[str, Any]] = []; unavailable: list[dict[str, Any]] = []
    for timing in ("strict", "same_date"):
        for experiment in EXPERIMENTS:
            if experiment == "B2":
                unavailable.append({"experiment": experiment, "timing_mode": timing, "reason": "Module 5A did not persist fitted TF-IDF/SVD objects; locked-test transformation would require prohibited refitting"}); continue
            models = ["majority_baseline"] if experiment == "A0" else ["logistic_regression", "mlp"]
            try: metadata, features, columns = assemble_test(experiment, timing)
            except (FileNotFoundError, AssertionError) as exc:
                unavailable.append({"experiment": experiment, "timing_mode": timing, "reason": str(exc)}); continue
            eligible = metadata["supervised_eligible"].astype(bool).to_numpy(); metadata = metadata.loc[eligible].reset_index(drop=True); features = features.loc[eligible].reset_index(drop=True)
            y = metadata["target_direction"].astype(bool).to_numpy()
            for model_name in models:
                path = artifact_for(experiment, timing, model_name)
                if not path.exists(): unavailable.append({"experiment": experiment, "timing_mode": timing, "model": model_name, "reason": f"Missing artifact: {path.relative_to(HANA)}"}); continue
                expected_hash = next((x["sha256"] for x in module6b["model_artifacts"] if x["path"] == str(path.relative_to(HANA))), None)
                actual_hash = sha256_file(path)
                if expected_hash and expected_hash != actual_hash: raise AssertionError(f"Model artifact hash mismatch: {path}")
                with path.open("rb") as handle: artifact = pickle.load(handle)
                if model_name == "majority_baseline":
                    prediction = np.full(len(y), bool(artifact["majority_direction"]), dtype=bool); score = None
                else:
                    x = transform_with_artifact(features, columns, artifact); model = artifact["model"]; prediction = model.predict(x).astype(bool); score = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)
                metrics = metric_dict(y, prediction, score)
                stem = f"{experiment.lower()}_{timing}_{model_name}_fold3"
                pred_path = RESULT_DIR / f"{stem}_predictions.csv"; pred = pd.DataFrame({"calendar_index": metadata["calendar_index"], "Date": metadata["Date"].dt.strftime("%Y-%m-%d"), "target_date": metadata["target_date"].dt.strftime("%Y-%m-%d"), "target_direction": y, "prediction": prediction, "score": score if score is not None else np.nan}); pred.to_csv(pred_path, index=False)
                cm_path = RESULT_DIR / f"{stem}_confusion_matrix.json"; cm_path.write_text(json.dumps({"labels": ["DOWN", "UP"], "matrix": [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]}, indent=2) + "\n")
                records.append({"experiment": experiment, "timing_mode": timing, "model": model_name, "fold_id": FOLD_ID, "test_rows": len(pred), "artifact_path": str(path.relative_to(HANA)), "artifact_sha256": actual_hash, "prediction_path": str(pred_path.relative_to(HANA)), "prediction_sha256": sha256_file(pred_path), "confusion_matrix_path": str(cm_path.relative_to(HANA)), "confusion_matrix_sha256": sha256_file(cm_path), **metrics})
                prediction_paths.append(records[-1])
    result_df = pd.DataFrame(records); result_path = RESULT_DIR / "module6c_locked_test_metrics.csv"; result_df.to_csv(result_path, index=False)
    manifest = {"module": "phase5_module6c", "locked_test_indices": list(range(LOCKED_START, LOCKED_END + 1)), "locked_test_calendar_rows": 241, "evaluated_rows_exclude_unavailable_target": True, "primary_timing_mode": "strict", "sensitivity_timing_mode": "same_date", "fold_model_used": 3, "no_refit": True, "no_tuning_on_test": True, "module6b_manifest_sha256": sha256_file(MANIFESTS / "module6b_training_manifest.json"), "module5f_manifest_sha256": sha256_file(MANIFESTS / "module5f_locked_features_manifest.json"), "metrics_path": str(result_path.relative_to(HANA)), "metrics_sha256": sha256_file(result_path), "evaluated_count": len(records), "unavailable": unavailable, "artifacts": prediction_paths}
    manifest_path = MANIFESTS / "module6c_final_evaluation_manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__": print(json.dumps(run_module6c(), indent=2, sort_keys=True))
