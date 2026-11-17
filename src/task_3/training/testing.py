from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata

METRICS = ["mcc", "macro_f1", "balanced_accuracy", "accuracy", "roc_auc", "brier", "log_loss", "ece_10"]

def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def test_context(root, model_id):
    root = Path(root)
    directory = root / "src/task_3/output/model/level_1/refit_test_v1"
    path = directory / "frozen_config.json"
    configuration = json.loads(path.read_text())
    for relative, expected in configuration["input_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    condition = configuration["conditions"][model_id]
    development = root / condition["development_directory"]
    current = json.loads((development / "run_metadata.json").read_text())
    if current["predictors"] != condition["predictors"] or current["training_seeds"] != condition["seeds"]:
        raise ValueError("Predictors or seeds differ from the frozen configuration.")
    if current.get("model_config") != condition["model_config"]:
        raise ValueError("Model configuration changed after freezing.")
    if condition["seeds"]:
        fits = pd.read_csv(development / "fits.csv")
        epochs = {str(seed): int(value) for seed, value in fits.groupby("seed").best_epoch.median().items()}
        if epochs != condition["epochs"]:
            raise ValueError("Development epoch choices changed after freezing.")
    model_directory = directory / model_id
    model_directory.mkdir(parents=True, exist_ok=True)
    return configuration, condition, model_directory

def mark_test_access(root, model_id):
    root = Path(root)
    directory = root / "src/task_3/output/model/level_1/refit_test_v1"
    path = directory / "test_access.json"
    if path.exists():
        record = json.loads(path.read_text())
    else:
        record = {"evaluation_started_at_wib": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
                  "first_notebook_model": model_id,
                  "frozen_config_sha256": sha256(directory / "frozen_config.json"),
                  "test_sha256": sha256(root / "data/final/test.csv"),
                  "prior_access": "Original-corpus evaluation used this test period on 2026-11-11.",
                  "interpretation": "Subsequent combined-corpus evaluation; not a new untouched holdout."}
        path.write_text(json.dumps(record, indent=2) + "\n")
    if record["frozen_config_sha256"] != sha256(directory / "frozen_config.json"):
        raise ValueError("The frozen test configuration changed.")
    return record

def circular_indices(n, repeats=1000, block_length=10, seed=20261110):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repeats, int(np.ceil(n / block_length))))
    return ((starts[:, :, None] + np.arange(block_length)) % n).reshape(repeats, -1)[:, :n]

def bootstrap_metrics(y, probability, indices):
    actual = np.asarray(y, dtype=int)[indices]
    p = np.asarray(probability, dtype=float)[indices]
    predicted = p >= 0.5
    n = actual.shape[1]
    up = actual.sum(axis=1).astype(float)
    down = n - up
    tp = (predicted & (actual == 1)).sum(axis=1).astype(float)
    fp = (predicted & (actual == 0)).sum(axis=1).astype(float)
    fn, tn = up - tp, down - fp
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    denominator = np.sqrt((tp + fp) * up * down * (tn + fn))
    mcc = divide(tp * tn - fp * fn, denominator)
    macro_f1 = (divide(2 * tp, 2 * tp + fp + fn) + divide(2 * tn, 2 * tn + fp + fn)) / 2
    balanced = (divide(tp, up) + divide(tn, down)) / 2
    ranks = rankdata(p, method="average", axis=1)
    auc = divide((ranks * actual).sum(axis=1) - up * (up + 1) / 2, up * down)
    both = (up > 0) & (down > 0)
    mcc[~both], balanced[~both], auc[~both] = np.nan, np.nan, np.nan
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.abs(np.where(bins == b, p - actual, 0).sum(axis=1)) for b in range(10)) / n
    return np.column_stack([
        mcc, macro_f1, balanced, (tp + tn) / n, auc, ((p - actual) ** 2).mean(axis=1),
        -(actual * np.log(clipped) + (1 - actual) * np.log1p(-clipped)).mean(axis=1), ece,
    ])

def save_test_results(root, model_id, notebook_path, predictions, metrics, fits, history, per_class, extra_inputs=None):
    root = Path(root)
    configuration, condition, directory = test_context(root, model_id)
    summary = metrics[METRICS].agg(["mean", "std"]).rename_axis("statistic")
    groups = [g.sort_values("Date") for _, g in predictions.groupby("seed", dropna=False)]
    indices = circular_indices(len(groups[0]), **{k: configuration["bootstrap"][k] for k in ["repeats", "block_length", "seed"]})
    draws = np.mean([bootstrap_metrics(g.y, g.p_up, indices) for g in groups], axis=0)
    bounds = np.nanquantile(draws, [0.025, 0.975], axis=0)
    intervals = pd.DataFrame({"metric": METRICS, "mean": summary.loc["mean"].to_numpy(),
                              "low_95": bounds[0], "high_95": bounds[1],
                              "valid_resamples": np.isfinite(draws).sum(axis=0)})
    for filename, frame in {"predictions.csv": predictions, "seed_metrics.csv": metrics,
                            "summary.csv": summary.reset_index(), "fits.csv": fits,
                            "history.csv": history, "per_class.csv": per_class, "intervals.csv": intervals}.items():
        frame.to_csv(directory / filename, index=False)
    np.savez_compressed(directory / "bootstrap_draws.npz", metrics=np.asarray(METRICS), draws=draws)
    notebook = json.loads(Path(notebook_path).read_text())
    source = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    helper_paths = [root / "src/task_3/training" / name for name in ["testing.py", "baselines.py", "availability.py", "fusion.py", "finbert.py"]]
    inputs = {**configuration["input_sha256"],
              "src/task_3/output/model/level_1/refit_test_v1/frozen_config.json": sha256(directory.parent / "frozen_config.json"),
              **{str(p.relative_to(root)): sha256(p) for p in helper_paths}, **(extra_inputs or {})}
    for relative, expected in inputs.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Input changed: {relative}")
    figure_directory = root / "src/task_3/output/figure/level_1/refit_test_v1" / model_id
    artifacts = [p for p in directory.iterdir() if p.is_file() and p.name != "run_metadata.json"]
    if figure_directory.exists():
        artifacts += [p for p in figure_directory.iterdir() if p.is_file()]
    metadata = {"model_id": model_id, "phase": "refit_test", "fit_rows": 930, "test_rows": 235,
                "evaluated_at_wib": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
                "fit_from_scratch": True, "epochs": condition["epochs"], "training_seeds": condition["seeds"],
                "model_config": condition["model_config"], "predictors": condition["predictors"],
                "test_used_for_fitting_or_stopping": False, "threshold": configuration["threshold"],
                "test_evaluated": True, "prior_test_access": configuration["test_history"],
                "score_aggregation": "mean individual-seed test metrics; no ensemble",
                "standard_deviation": "across training seeds; undefined for deterministic baselines",
                "bootstrap": configuration["bootstrap"], "bootstrap_scope": "conditional on fixed fitted models; shared test-date blocks",
                "input_sha256": inputs, "notebook_code_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
                "saved_model_predictions_verified": True,
                "artifact_sha256": {str(p.relative_to(root)): sha256(p) for p in artifacts}}
    (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return summary, intervals
