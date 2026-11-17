"""Module 6B: fold-local training and validation tuning on Module 6A datasets."""

from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
HANA = ROOT / "hana-work"
DATA_DIR = HANA / "data/processed/module6"
MANIFEST_DIR = HANA / "data/manifests"
MODEL_DIR = HANA / "models/module6"
RESULT_DIR = HANA / "results/module6"
SEED = 42
META_COLUMNS = {"Date", "calendar_index", "split_role", "target_date", "next_Date", "target_direction", "target_available", "is_flat_movement", "is_warmup", "supervised_eligible", "module3_split", "experiment_id", "timing_mode", "fold_id"}
LR_GRID = list(itertools.product([0.01, 0.1, 1.0, 10.0, 100.0], [None, "balanced"]))
MLP_GRID = list(itertools.product([(64,), (128,), (128, 64)], [0.0001, 0.001, 0.01], [0.001, 0.0005]))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module6a_manifest() -> dict[str, Any]:
    return json.loads((MANIFEST_DIR / "module6a_dataset_manifest.json").read_text())


def load_fold_manifest() -> dict[str, Any]:
    return json.loads((MANIFEST_DIR / "module3_fold_manifest.json").read_text())


def fold_indices(manifest: dict[str, Any], fold_id: int) -> tuple[set[int], set[int], set[int]]:
    row = [x for x in manifest["folds"] if x["fold_id"] == fold_id]
    if len(row) != 1:
        raise ValueError(f"Fold {fold_id} is not unique")
    train = set(row[0]["train_calendar_indices"]); validation = set(row[0]["validation_calendar_indices"]); locked = set(manifest["locked_test"]["calendar_indices"])
    if train & validation or (train | validation) & locked:
        raise AssertionError("Invalid fold overlap")
    return train, validation, locked


def load_dataset(path: Path, fold_manifest: dict[str, Any], fold_id: int) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, parse_dates=["Date", "target_date", "next_Date"])
    train, validation, locked = fold_indices(fold_manifest, fold_id)
    if frame["calendar_index"].duplicated().any() or set(frame["calendar_index"]) & locked:
        raise AssertionError(f"Invalid or locked dataset: {path}")
    expected = (train | validation) & set(frame.loc[frame["supervised_eligible"].astype(bool), "calendar_index"])
    if set(frame["calendar_index"]) != expected:
        raise AssertionError(f"Module 6A membership mismatch: {path}")
    if set(frame["split_role"]) != {"train", "validation"} or frame["Date"].gt(frame["target_date"]).any():
        raise AssertionError(f"Invalid split or target dates: {path}")
    feature_columns = [c for c in frame.columns if c not in META_COLUMNS]
    if len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Duplicate feature columns")
    return frame, feature_columns


def make_preprocessor(feature_columns: list[str], train_rows: int) -> tuple[Any, int, bool]:
    high_dim = len(feature_columns) > 64 and any(name.startswith(("word2vec_", "finbert_", "roberta_")) for name in feature_columns)
    n_components = min(64, len(feature_columns), max(1, train_rows - 1)) if high_dim else None
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    return {"imputer": imputer, "scaler": scaler, "pca": PCA(n_components=n_components, random_state=SEED) if n_components else None, "feature_columns": feature_columns, "pca_components": n_components}, (n_components or len(feature_columns)), high_dim


def fit_transform(train: pd.DataFrame, validation: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prep, dimensions, high_dim = make_preprocessor(feature_columns, len(train))
    x_train = prep["imputer"].fit_transform(train[feature_columns].apply(pd.to_numeric, errors="coerce"))
    x_validation = prep["imputer"].transform(validation[feature_columns].apply(pd.to_numeric, errors="coerce"))
    x_train = prep["scaler"].fit_transform(x_train); x_validation = prep["scaler"].transform(x_validation)
    if prep["pca"] is not None:
        prep["pca"].fit(x_train); x_train = prep["pca"].transform(x_train); x_validation = prep["pca"].transform(x_validation)
    prep["high_dimensional_pca"] = high_dim; prep["output_dimensions"] = dimensions
    return x_train, x_validation, prep


def metrics(y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray | None) -> dict[str, Any]:
    precision, recall, _, _ = precision_recall_fscore_support(y_true, predictions, labels=[False, True], zero_division=0)
    result = {"mcc": float(matthews_corrcoef(y_true, predictions)), "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)), "accuracy": float(accuracy_score(y_true, predictions)), "precision_down": float(precision[0]), "precision_up": float(precision[1]), "recall_down": float(recall[0]), "recall_up": float(recall[1]), "roc_auc": None}
    if scores is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    return result


def fit_model(model_name: str, params: dict[str, Any], x: np.ndarray, y: np.ndarray) -> Any:
    if model_name == "logistic_regression":
        model = LogisticRegression(C=float(params["C"]), class_weight=params["class_weight"], solver="lbfgs", max_iter=2000, random_state=SEED)
    elif model_name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=tuple(params["hidden_layer_sizes"]), alpha=float(params["alpha"]), learning_rate_init=float(params["learning_rate_init"]), early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, max_iter=300, random_state=SEED)
    else:
        raise ValueError(model_name)
    model.fit(x, y); return model


def predict_scores(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    predictions = model.predict(x).astype(bool)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x)
    else:
        scores = None
    return predictions, scores


def candidate_params(model_name: str) -> list[dict[str, Any]]:
    if model_name == "logistic_regression":
        return [{"C": c, "class_weight": w} for c, w in LR_GRID]
    return [{"hidden_layer_sizes": list(h), "alpha": a, "learning_rate_init": lr} for h, a, lr in MLP_GRID]


def tune_group(experiment_id: str, timing_mode: str, model_name: str, module6a: dict[str, Any], fold_manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame, pd.DataFrame]]]:
    rows = [x for x in module6a["summaries"] if x["experiment_id"] == experiment_id and x["timing_mode"] == timing_mode]
    if len(rows) != 3:
        raise AssertionError(f"Expected three fold datasets for {experiment_id}/{timing_mode}")
    prepared = {}
    for row in rows:
        fold_id = int(row["fold_id"]); frame, feature_columns = load_dataset(HANA / row["output_path"], fold_manifest, fold_id)
        train = frame[frame["split_role"] == "train"]; validation = frame[frame["split_role"] == "validation"]
        y_train = train["target_direction"].astype(bool).to_numpy(); y_validation = validation["target_direction"].astype(bool).to_numpy()
        if model_name == "majority_baseline":
            prepared[fold_id] = (None, None, {"feature_columns": feature_columns, "output_dimensions": 0}, train, validation); continue
        x_train, x_validation, prep = fit_transform(train, validation, feature_columns)
        prepared[fold_id] = (x_train, x_validation, prep, train, validation)
    if model_name == "majority_baseline":
        return {"strategy": "training-fold majority"}, [], prepared
    tuning=[]
    for params in candidate_params(model_name):
        fold_scores=[]
        for fold_id in (1,2,3):
            x_train,x_val,_,train,val=prepared[fold_id]; model=fit_model(model_name,params,x_train,train["target_direction"].astype(bool).to_numpy()); pred,score=predict_scores(model,x_val); fold_scores.append(metrics(val["target_direction"].astype(bool).to_numpy(),pred,score))
        tuning.append({"params":params,"mean_mcc":float(np.mean([x["mcc"] for x in fold_scores])),"fold_metrics":fold_scores})
    best=max(tuning,key=lambda x:(x["mean_mcc"], -candidate_params(model_name).index(x["params"])))
    return best, tuning, prepared


def run_module6b() -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True); RESULT_DIR.mkdir(parents=True, exist_ok=True)
    module6a=load_module6a_manifest(); fold_manifest=load_fold_manifest(); experiment_ids=[]
    for row in module6a["summaries"]:
        if row["experiment_id"] not in experiment_ids: experiment_ids.append(row["experiment_id"])
    all_results=[]; tuning_records=[]; artifact_records=[]
    groups=[(x,mode,model) for x in experiment_ids for mode in ("strict","same_date") for model in (("majority_baseline",) if x=="A0" else ("logistic_regression","mlp"))]
    for experiment_id,timing_mode,model_name in groups:
        started=time.time(); best,tuning,prepared=tune_group(experiment_id,timing_mode,model_name,module6a,fold_manifest)
        tuning_records.append({"experiment_id":experiment_id,"timing_mode":timing_mode,"model":model_name,"best_params":best.get("params",best),"mean_validation_mcc":best.get("mean_mcc")})
        for fold_id in (1,2,3):
            x_train,x_val,prep,train,val=prepared[fold_id]; y_train=train["target_direction"].astype(bool).to_numpy(); y_val=val["target_direction"].astype(bool).to_numpy()
            if model_name=="majority_baseline":
                majority=bool(pd.Series(y_train).mode().iloc[0]); model=DummyClassifier(strategy="most_frequent",random_state=SEED).fit(np.zeros((len(y_train),1)),y_train); pred=np.full(len(y_val),majority,dtype=bool); score=None; artifact={"model":model,"preprocessor":prep,"majority_direction":majority}
            else:
                model=fit_model(model_name,best["params"],x_train,y_train); pred,score=predict_scores(model,x_val); artifact={"model":model,"preprocessor":prep,"params":best["params"],"seed":SEED}
            model_path=MODEL_DIR / f"{experiment_id.lower()}_{timing_mode}_{model_name}_fold{fold_id}.pkl"; model_path.write_bytes(pickle.dumps(artifact,protocol=pickle.HIGHEST_PROTOCOL)); artifact_records.append({"path":str(model_path.relative_to(HANA)),"sha256":sha256_file(model_path)})
            pred_path=RESULT_DIR / f"{experiment_id.lower()}_{timing_mode}_{model_name}_fold{fold_id}_predictions.csv"; output=pd.DataFrame({"calendar_index":val["calendar_index"].to_numpy(),"Date":val["Date"].dt.strftime("%Y-%m-%d"),"target_direction":y_val,"prediction":pred,"score":score if score is not None else np.nan}); output.to_csv(pred_path,index=False)
            fold_metrics=metrics(y_val,pred,score); all_results.append({"experiment_id":experiment_id,"timing_mode":timing_mode,"model":model_name,"fold_id":fold_id,"train_rows":len(train),"validation_rows":len(val),"preprocessing_dimensions":prep.get("output_dimensions",0),"pca_used":bool(prep.get("high_dimensional_pca",False)),"best_params":json.dumps(best.get("params",best),sort_keys=True),"training_seconds":time.time()-started,**fold_metrics,"prediction_path":str(pred_path.relative_to(HANA)),"prediction_sha256":sha256_file(pred_path)})
    results=pd.DataFrame(all_results); results.to_csv(RESULT_DIR/"module6b_fold_results.csv",index=False); tuning=pd.DataFrame(tuning_records); tuning.to_csv(RESULT_DIR/"module6b_tuning_summary.csv",index=False)
    source_paths=[MANIFEST_DIR/"module6a_dataset_manifest.json",MANIFEST_DIR/"module3_fold_manifest.json",HANA/"config/module6_experiment_matrix.json"]
    manifest={"module":"phase5_module6b","models":["logistic_regression","mlp"],"seed":SEED,"selection_metric":"mean validation MCC across three folds","locked_test_used":False,"preprocessing":"fold-local median imputation, standard scaling, PCA up to 64 components for embedding feature groups >64 dimensions","hyperparameter_grids":{"logistic_regression":{"C":[0.01,0.1,1,10,100],"class_weight":[None,"balanced"],"solver":"lbfgs"},"mlp":{"hidden_layer_sizes":[[64],[128],[128,64]],"alpha":[0.0001,0.001,0.01],"learning_rate_init":[0.001,0.0005],"early_stopping":True}},"baseline":"A0 majority classifier; no learned feature model","experiment_count":len(experiment_ids),"training_group_count":len(groups),"fold_result_count":len(results),"source_hashes":{p.name:sha256_file(p) for p in source_paths},"model_artifacts":artifact_records,"results_sha256":sha256_file(RESULT_DIR/"module6b_fold_results.csv"),"tuning_sha256":sha256_file(RESULT_DIR/"module6b_tuning_summary.csv")}
    (MANIFEST_DIR/"module6b_training_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


if __name__=="__main__": print(json.dumps(run_module6b(),indent=2,sort_keys=True))
