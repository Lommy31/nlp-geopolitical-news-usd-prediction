from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

def test_context(root, model_id, development_directory):
    root, development = Path(root), Path(development_directory)
    directory = root / "src/task_3/output/model/level_3/refit_test_v1" / model_id
    directory.mkdir(parents=True, exist_ok=True)
    current = json.loads((development / "run_metadata.json").read_text())
    fits = pd.read_csv(development / "fits.csv")
    condition = {key: current[key] for key in ["predictors", "model_config", "text_pipeline", "flags", "level_3_contract_sha256", "representation_selection_sha256"]}
    condition["seeds"] = current["training_seeds"]
    condition["epochs"] = {str(seed): int(value) for seed, value in fits.groupby("seed").best_epoch.median().items()}
    path = directory / "frozen_config.json"
    if not path.exists():
        snapshot = directory / "development_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name in ["run_metadata.json", "fits.csv", "summary.csv"]:
            shutil.copy2(development / name, snapshot / name)
        configuration = {
            "model_id": model_id, "frozen_at_wib": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
            "condition": condition, "fit_rows": 930, "test_rows": 235, "threshold": 0.5,
            "epoch_rule": "Per-seed median selected epoch across three development folds",
            "fit_from_scratch": True, "test_early_stopping": False,
            "input_sha256": {str(p.relative_to(root)): sha256(p) for p in (root / "data/final").glob("*") if p.is_file()},
            "development_snapshot_sha256": {str(p.relative_to(root)): sha256(p) for p in snapshot.iterdir()},
            "bootstrap": {"repeats": 1000, "block_length": 10, "seed": 20261110},
            "test_history": "This period was evaluated on the original corpus on 2026-11-11 and combined-corpus Level 1 on 2026-11-15; subsequent exploratory evaluation.",
        }
        path.write_text(json.dumps(configuration, indent=2) + "\n")
    configuration = json.loads(path.read_text())
    if condition != configuration["condition"]:
        raise ValueError("Development settings or epoch choices differ from the frozen configuration.")
    for relative, expected in {**configuration["input_sha256"], **configuration["development_snapshot_sha256"]}.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    return configuration, condition, directory

def mark_test_access(directory):
    directory = Path(directory)
    path = directory / "test_access.json"
    expected = sha256(directory / "frozen_config.json")
    if not path.exists():
        path.write_text(json.dumps({"evaluated_at_wib": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
                                    "frozen_config_sha256": expected,
                                    "interpretation": "Subsequent evaluation of a previously accessed test period"}, indent=2) + "\n")
    if json.loads(path.read_text())["frozen_config_sha256"] != expected:
        raise ValueError("Frozen test configuration changed after test access.")

def save_test_results(root, model_id, notebook_path, directory, predictions, metrics, fits, history, per_class, extra_inputs):
    root, directory = Path(root), Path(directory)
    configuration = json.loads((directory / "frozen_config.json").read_text())
    summary = metrics[METRICS].agg(["mean", "std"]).rename_axis("statistic")
    groups = [group.sort_values("Date") for _, group in predictions.groupby("seed")]
    indices = circular_indices(configuration["test_rows"], **configuration["bootstrap"])
    draws = np.mean([bootstrap_metrics(group.y, group.p_up, indices) for group in groups], axis=0)
    bounds = np.nanquantile(draws, [.025, .975], axis=0)
    intervals = pd.DataFrame({"metric": METRICS, "mean": summary.loc["mean"].to_numpy(),
                              "low_95": bounds[0], "high_95": bounds[1]})
    frames = {"predictions.csv": predictions, "seed_metrics.csv": metrics,
              "summary.csv": summary.reset_index(), "fits.csv": fits, "history.csv": history,
              "per_class.csv": per_class, "intervals.csv": intervals}
    for name, frame in frames.items():
        frame.to_csv(directory / name, index=False)
    np.savez_compressed(directory / "bootstrap_draws.npz", metrics=np.asarray(METRICS), draws=draws)
    helpers = [root / "src/task_3/training" / name for name in ["testing.py", "component_testing.py", "controlled_fusion.py", "word2vec.py"]]
    inputs = {**configuration["input_sha256"], **extra_inputs,
              **{str(p.relative_to(root)): sha256(p) for p in helpers},
              str((directory / "frozen_config.json").relative_to(root)): sha256(directory / "frozen_config.json")}
    for relative, expected in inputs.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Input changed: {relative}")
    notebook = json.loads(Path(notebook_path).read_text())
    source = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    figures = root / "src/task_3/output/figure/level_3/refit_test_v1" / model_id
    artifacts = [p for p in directory.iterdir() if p.is_file() and p.name != "run_metadata.json"] + list(figures.glob("*"))
    metadata = {
        "model_id": model_id, "phase": "refit_test", "fit_rows": 930, "test_rows": 235,
        "fit_from_scratch": True, "test_used_for_fitting_or_stopping": False,
        "condition": configuration["condition"], "threshold": configuration["threshold"],
        "evaluated_at_wib": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "test_evaluated": True, "test_history": configuration["test_history"],
        "score_aggregation": "Mean individual-seed metrics; no ensemble; SD across training seeds",
        "bootstrap": configuration["bootstrap"], "bootstrap_scope": "Fixed fitted models; shared circular blocks of test decisions",
        "input_sha256": inputs, "notebook_code_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "artifact_sha256": {str(p.relative_to(root)): sha256(p) for p in artifacts},
        "saved_model_predictions_verified": True,
    }
    (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return summary, intervals
