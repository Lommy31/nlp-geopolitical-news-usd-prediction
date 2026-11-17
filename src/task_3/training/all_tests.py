from pathlib import Path
import argparse
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.task_3.training.experiment import Experiment, STUDIES, ROOT, OUTPUT, FIGURES, read_json, notebook_code_hash, check_inputs
from src.task_3.training.data import raw_data, final_data
from src.task_3.training.evaluation import now, write_json, scored, write_phase, plot_results, paired_comparison
from src.task_3.training.testing import METRICS, sha256

TEST_OUTPUT = OUTPUT / "all_conditions_test_v1"
TEST_FIGURES = FIGURES / "all_conditions_test_v1"
REFERENCES = {"L1-a": "L1-a", "L1-b": "L1-b", "L1-c": "price_matched", "L1-e": "availability_matched", "A000": "A000",
              "price_matched": "price_matched", "price_tuned": "price_tuned", "availability_matched": "availability_matched"}

def prepare():
    path = TEST_OUTPUT / "frozen_config.json"
    if path.exists():
        check_inputs()
        return read_json(path)
    previous = TEST_OUTPUT / "previous_scope"
    manifest = read_json(previous / "manifest.json")
    for relative, digest in manifest.items():
        assert sha256(previous / relative) == digest, relative
    base = read_json(OUTPUT / "protocol.json")
    conditions = []
    prior = read_json(OUTPUT / "final_test/frozen_config.json")
    previous_configs = {cfg["model_id"]: cfg for cfg in prior["conditions"]}
    studies = [(level, model_id) for level, models in STUDIES.items() for model_id in models]
    studies += [(1, model_id) for model_id in ["price_matched", "price_tuned", "availability_matched"]]
    inputs = {}
    for level, model_id in studies:
        experiment = Experiment(level, model_id)
        fits = pd.read_csv(experiment.directory / "fits.csv")
        epochs = {str(int(seed)): int(value) for seed, value in fits.groupby("seed").best_epoch.median().items()}
        files = [experiment.directory / name for name in ["run_metadata.json", "fits.csv", "summary.csv", "predictions.csv"]]
        inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in files})
        reuse = None
        if model_id in REFERENCES:
            source_id = REFERENCES[model_id]
            source_config = previous_configs[source_id]
            assert source_config["config"] == experiment.config and source_config["epochs"] == epochs
            assert source_config["representation"] == experiment.rep and source_config["flags"] == experiment.flags
            directory = OUTPUT / "final_test" / source_id
            source_meta = read_json(directory / "run_metadata.json")
            assert not source_meta["test_used_for_fitting_or_stopping"]
            assert source_meta["frozen_config_sha256"] == sha256(OUTPUT / "final_test/frozen_config.json")
            for name, digest in source_meta["artifact_sha256"].items():
                assert sha256(directory / name) == digest, directory / name
            inputs.update({str(p.relative_to(ROOT)): sha256(p) for p in directory.iterdir() if p.is_file()})
            reuse = str(directory.relative_to(ROOT))
        conditions.append({"level": level, "model_id": model_id, "representation": experiment.rep, "config": experiment.config,
                           "flags": experiment.flags, "epochs": epochs, "seeds": experiment.seeds, "reuse_source": reuse,
                           "alias_of": "L1-d" if model_id == "R3" else None})
    amendments, notebooks = {}, {}
    for relative, digest in base["input_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != digest:
            snapshot = previous / relative
            assert sha256(snapshot) == digest
            amendments[relative] = {"before_sha256": digest, "after_sha256": actual, "snapshot": str(snapshot.relative_to(ROOT))}
    for relative, digest in base["notebook_code_sha256"].items():
        actual = notebook_code_hash(ROOT / relative)
        if actual != digest:
            snapshot = previous / relative
            assert notebook_code_hash(snapshot) == digest
            notebooks[relative] = {"before_sha256": digest, "after_sha256": actual, "snapshot": str(snapshot.relative_to(ROOT))}
    implementation = [Path(__file__), ROOT / "src/task_3/training/experiment.py", ROOT / "src/task_3/training/run_all.py", ROOT / "src/task_3/training/PROTOCOL.md"]
    write_json(path, {"frozen_at_wib": now(), "authorization": "User explicitly requested final test for every ablation condition after the previous selected-only evaluation",
               "base_protocol_sha256": sha256(OUTPUT / "protocol.json"), "base_test_config_sha256": sha256(OUTPUT / "final_test/frozen_config.json"),
               "scope": "All fifteen primary notebook conditions plus the three existing extra controls; no VADER", "conditions": conditions,
               "selection": "Preserve every validation-selected configuration and each seed's median validation epoch; no test-driven changes or new grid",
               "threshold": .5, "refit_rows": len(raw_data()), "test_used_for_selection": False,
               "reference_policy": "Reuse already verified identical refits; R3 aliases the identical L1-d model and predictions",
               "test_history": "Repeated earlier test access; all comparisons exploratory; validation winner remains unchanged",
               "input_amendments": amendments, "notebook_amendments": notebooks, "implementation_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in implementation}, "input_sha256": inputs})
    check_inputs()
    print(f"Frozen {len(conditions)} conditions before new refits.", flush=True)
    return read_json(path)

def evaluate(level, model_id):
    frozen = prepare()
    cfg = next(row for row in frozen["conditions"] if row["level"] == level and row["model_id"] == model_id)
    directory = TEST_OUTPUT / model_id
    directory.mkdir(exist_ok=True)
    if (directory / "run_metadata.json").exists():
        meta = read_json(directory / "run_metadata.json")
        assert meta["frozen_config_sha256"] == sha256(TEST_OUTPUT / "frozen_config.json")
        for name, digest in meta["artifact_sha256"].items():
            assert sha256(directory / name) == digest, directory / name
        return pd.read_csv(directory / "summary.csv").set_index("statistic")
    experiment = Experiment(level, model_id)
    assert cfg["config"] == experiment.config and cfg["representation"] == experiment.rep and cfg["flags"] == experiment.flags
    for relative, digest in frozen["input_sha256"].items():
        assert sha256(ROOT / relative) == digest, relative
    alias = cfg["alias_of"]
    if alias:
        evaluate(1, alias)
    source = ROOT / cfg["reuse_source"] if cfg["reuse_source"] else TEST_OUTPUT / alias if alias else None
    text = not experiment.naive and experiment.config["view"] == "text"
    dev, vectors, dev_source = final_data(experiment.rep) if text else (raw_data(), None, {})
    fits, histories = [], []
    for seed in cfg["seeds"]:
        duration = cfg["epochs"][str(seed)]
        path = directory / f"seed_{seed}.joblib"
        if source:
            shutil.copy2(source / path.name, path)
            records = pd.read_csv(source / "fits.csv")
            fit = records.loc[records.seed.eq(seed)].iloc[0].to_dict()
            assert fit["train_n"] == len(dev) and fit["fixed_epochs"] == duration and fit["evaluation_is_none"]
            fits.append({**fit, "reused": True, "source_model": str((source / path.name).relative_to(ROOT))})
            trace = pd.read_csv(source / "history.csv")
            if "seed" in trace:
                histories.extend(trace.loc[trace.seed.eq(seed)].to_dict("records"))
            print(f"{model_id} seed{seed}: reused verified {duration}-epoch model", flush=True)
        else:
            evaluation = None
            model, best, ran, trace = experiment.fit(dev, evaluation, vectors, seed, epochs=duration)
            joblib.dump(model, path)
            fits.append({"seed": seed, "fold": 0, "train_n": len(dev), "epochs": duration, "epochs_run": ran, "best_epoch": best,
                         "evaluation_role": "none", "evaluation_is_none": evaluation is None, "fixed_epochs": duration, "reused": False, "source_model": None})
            histories.extend({"fold": 0, "seed": seed, **row} for row in trace)
            print(f"{model_id} seed{seed}: refit {duration} epochs", flush=True)
        write_json(TEST_OUTPUT / "status.json", {"phase": "refit", "updated_at_wib": now(), "level": level, "model_id": model_id, "seed": seed})
    pd.DataFrame(fits).to_csv(directory / "refit_records.csv", index=False)
    write_json(directory / "test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(TEST_OUTPUT / "frozen_config.json"),
               "refit_records_sha256": sha256(directory / "refit_records.csv"), "models_reused": source is not None, "source": str(source.relative_to(ROOT)) if source else None})
    test, test_vectors, test_source = final_data(experiment.rep, "test") if text else (raw_data("test"), None, {})
    assert dev.target_date.max() < test.Date.min()
    predictions, metrics, classes = [], [], []
    for seed in cfg["seeds"]:
        path = directory / f"seed_{seed}.joblib"
        p = experiment.predict(joblib.load(path), test, test_vectors)
        np.testing.assert_array_equal(p, experiment.predict(joblib.load(path), test, test_vectors))
        a, b, c = scored(test, p, model_id, seed, 0)
        predictions.append(a); metrics.append(b); classes.extend(c)
    predictions = pd.concat(predictions, ignore_index=True)
    if source:
        old = pd.read_csv(source / "predictions.csv", parse_dates=["Date", "target_date"]).sort_values(["seed", "Date"])
        current = predictions.sort_values(["seed", "Date"])
        pd.testing.assert_frame_equal(current[["Date", "target_date", "y"]].reset_index(drop=True), old[["Date", "target_date", "y"]].reset_index(drop=True))
        np.testing.assert_allclose(current.p_up, old.p_up, atol=1e-14, rtol=0)
    summary, _ = write_phase(directory, predictions, pd.DataFrame(metrics), pd.DataFrame(fits), pd.DataFrame(histories, columns=None if histories else ["epoch"]), pd.DataFrame(classes),
                             {"model_id": model_id, "level": level, "phase": "test", "config": cfg["config"], "representation": cfg["representation"], "flags": cfg["flags"],
                              "protocol_sha256": sha256(OUTPUT / "protocol.json"), "frozen_config_sha256": sha256(TEST_OUTPUT / "frozen_config.json"),
                              "encoder_sources": {"development": dev_source, "test": test_source}, "exploratory": True, "test_used_for_selection": False,
                              "new_fits": 0 if source else len(fits), "reuse_source": str(source.relative_to(ROOT)) if source else None, "alias": alias})
    figures = plot_results(predictions, pd.DataFrame(fits), pd.DataFrame(histories), TEST_FIGURES / model_id, f"{model_id}: test")
    import matplotlib.pyplot as plt
    for figure in figures:
        plt.close(figure)
    print(f"Completed test {model_id}: MCC{summary.loc['mean', 'mcc']:.4f}, AUC{summary.loc['mean', 'roc_auc']:.4f}", flush=True)
    report()
    return summary

def markdown_table(frame, level=False):
    headers = (["Level"] if level else []) + ["Model", *METRICS]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * (2 if level else 1) + ["---:"] * len(METRICS)) + " |"]
    for row in frame.itertuples():
        labels = ([str(row.level)] if level else []) + [row.model_id]
        lines.append("| " + " | ".join(labels + [f"{getattr(row, metric):.4f}" for metric in METRICS]) + " |")
    return "\n".join(lines) + "\n"

def report():
    frozen = read_json(TEST_OUTPUT / "frozen_config.json")
    rows, predictions = [], {}
    for cfg in frozen["conditions"]:
        directory = TEST_OUTPUT / cfg["model_id"]
        if not (directory / "summary.csv").exists():
            continue
        summary = pd.read_csv(directory / "summary.csv").set_index("statistic").loc["mean"]
        rows.append({"level": cfg["level"], "model_id": cfg["model_id"], **summary.to_dict()})
        predictions[cfg["model_id"]] = pd.read_csv(directory / "predictions.csv")
    table = pd.DataFrame(rows)
    if not len(table):
        return
    table.to_csv(TEST_OUTPUT / "test_summary.csv", index=False)
    main = table.loc[table.apply(lambda row: row.model_id in STUDIES[row.level], axis=1)]
    for level, models in STUDIES.items():
        part = main.loc[main.level.eq(level)]
        if len(part) == len(models):
            (ROOT / f"src/task_3/training/level_{level}/level_{level}_test_result.md").write_text(markdown_table(part))
    if len(rows) != len(frozen["conditions"]):
        return table
    (ROOT / "src/task_3/training/final_test_result.md").write_text(markdown_table(table, level=True))
    comparisons = [("L1-d", "L1-c"), ("L1-d", "L1-e"), ("L1-e", "L1-c"), ("T1", "R3")]
    comparisons += [(name, "A000") for name in STUDIES[3] if name != "A000"]
    contrasts = []
    for candidate, reference in comparisons:
        result, _ = paired_comparison(predictions[candidate], predictions[reference])
        result["comparison"] = f"{candidate}_minus_{reference}"
        contrasts.append(result)
    pd.concat(contrasts, ignore_index=True).to_csv(TEST_OUTPUT / "paired_contrasts.csv", index=False)
    counts = []
    for model_id, group in predictions.items():
        for seed, part in group.groupby("seed"):
            counts.append({"model_id": model_id, "seed": seed, "n": len(part), "predicted_up": int(part.y_pred.sum()), "p_min": part.p_up.min(), "p_max": part.p_up.max()})
    pd.DataFrame(counts).to_csv(TEST_OUTPUT / "prediction_distribution.csv", index=False)
    lines = ["# Test evaluation of every ablation condition", "", "All fifteen primary notebook conditions and the three existing additional controls are evaluated on identical final-test dates. Configurations and per-seed median epochs were frozen from completed validation before the new refits. No VADER, hyperparameter search, threshold tuning or test-based model reselection is included.", "",
             markdown_table(table, level=True).strip(), "", "R3 reuses the identical L1-d model; equivalent previously verified final refits are reused and recorded. These aliases are not independent evidence. Validation still selects FinBERT and A000; descriptive test rankings do not change those choices.", "",
             "Test has been accessed repeatedly. All results and paired block-bootstrap intervals are exploratory and conditional on fitted models; selection uncertainty and multiplicity are not included in nominal intervals. The failed validation success gate remains a limitation.", "",
             "Per-condition directories contain saved models, fit records, predictions, eight metrics, per-class scores and temporal intervals. Figures are under output/figure/improved_v1/all_conditions_test_v1/. Previous selected-only evaluation and its source snapshots remain preserved."]
    (TEST_OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    write_json(TEST_OUTPUT / "status.json", {"phase": "completed", "completed_at_wib": now(), "conditions": len(rows), "primary_notebooks": len(main), "test_used_for_selection": False})
    return table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "run"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        frozen = prepare()
        if args.phase == "run":
            for cfg in frozen["conditions"]:
                evaluate(cfg["level"], cfg["model_id"])
            report()

if __name__ == "__main__":
    main()
