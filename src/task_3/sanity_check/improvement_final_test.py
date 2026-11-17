from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import platform
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
sys.path.insert(0, str(ROOT))
from src.task_3.sanity_check.level1_improvement_models import SEEDS, fit_small
from src.task_3.training.baselines import LastDirectionClassifier
from src.task_3.training.controlled_fusion import score_binary
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

OUTPUT = ROOT / "src/task_3/output/model/sanity_check/level2_improvement_test_v1"
FIGURES = ROOT / "src/task_3/output/figure/sanity_check/level2_improvement_test_v1"
L1 = ROOT / "src/task_3/output/model/sanity_check/level1_improvement_v1"
L2 = ROOT / "src/task_3/output/model/sanity_check/level2_improvement_v1"
OLD = ROOT / "src/task_3/output/model/level_1/refit_test_v1"
FEATURE_CACHE = ROOT / "src/task_3/output/model/sanity_check/movement_magnitude_v1"

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def predict(model, method, frame, features):
    if method == "training_prior":
        return model.predict_proba(np.zeros((len(frame), 1)))[:, 1]
    if method == "last_direction":
        return model.predict_proba(frame[["log_return_t"]])[:, 1]
    return model.predict_proba(frame[features])[:, 1]

def evaluate(model, method, frame, seed, path, features):
    p = predict(model, method, frame, features)
    np.testing.assert_array_equal(predict(joblib.load(path), method, frame, features), p)
    result = frame[["Date", "target_date", "y", "n_berita", "has_news"]].copy()
    result["method"], result["seed"] = method, seed
    result["p_up"], result["y_pred"] = p, (p >= .5).astype(int)
    metrics = {"method": method, "seed": seed, "n": len(frame), **score_binary(frame.y, p)}
    precision, recall, f1, support = precision_recall_fscore_support(frame.y, result.y_pred, labels=[0, 1], zero_division=0)
    classes = [{"method": method, "seed": seed, "class": name, "support": int(support[i]), "precision": precision[i],
                "recall": recall[i], "f1": f1[i]} for i, name in enumerate(["DOWN", "UP"])]
    return result, metrics, classes

def main():
    if OUTPUT.exists():
        raise FileExistsError("Use a new version instead of overwriting a completed evaluation.")
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    assert json.loads((L1 / "verification.json").read_text())["verified"]
    assert json.loads((L2 / "verification.json").read_text())["verified"]
    selection = json.loads((L2 / "frozen_selection.json").read_text())
    rep = selection["global_winner_representation"]
    assert rep == "R3"
    selected = selection["winners"][rep]
    source = ROOT / selected["reuse_source"]
    source_config = json.loads((source / "complete.json").read_text())["candidate"]
    assert selected["local_id"] == source_config["id"]
    matched_id, tuned_id = selected["matched_price"], selection["best_price"]
    configurations = {"finbert_improved": {"candidate": source_config, "source": source},
                      "price_matched": {"candidate": json.loads((L1 / "candidates" / matched_id / "complete.json").read_text())["candidate"], "source": L1 / "candidates" / matched_id},
                      "price_tuned": {"candidate": json.loads((L1 / "candidates" / tuned_id / "complete.json").read_text())["candidate"], "source": L1 / "candidates" / tuned_id}}
    r3 = json.loads((ROOT / "src/task_3/output/model/level_1/04_market_news_concat/run_metadata.json").read_text())
    market, text, availability = [r3["feature_blocks"][key] for key in ["market", "text", "availability"]]
    features = market + text + availability
    dev = joblib.load(L1 / "development_features.joblib").sort_values("Date").reset_index(drop=True)
    assert dev.Date.is_unique
    epochs = {}
    for method, spec in configurations.items():
        fits = pd.read_csv(spec["source"] / "fits.csv")
        epochs[method] = {str(int(seed)): int(value) for seed, value in fits.groupby("seed").best_epoch.median().items()}
        assert fits.evaluation_role.eq("development_validation").all()
        assert not fits.has_test_evaluation.any()
    assert epochs["finbert_improved"] == selected["final_median_epochs"]
    assert configurations["finbert_improved"]["candidate"]["alpha"] == configurations["price_matched"]["candidate"]["alpha"]
    assert configurations["finbert_improved"]["candidate"]["dropout"] == configurations["price_matched"]["candidate"]["dropout"]
    sources = [Path(__file__), ROOT / "src/task_3/sanity_check/level1_improvement_models.py", ROOT / "src/task_3/training/baselines.py",
               ROOT / "src/task_3/training/controlled_fusion.py", ROOT / "src/task_3/training/fusion.py", ROOT / "src/task_3/training/testing.py"]
    sources += [L1 / name for name in ["verification.json", "run_metadata.json", "development_features.joblib", "frozen_selection.json"]]
    sources += [L2 / name for name in ["verification.json", "run_metadata.json", "frozen_selection.json", "protocol.json"]]
    sources += [spec["source"] / name for spec in configurations.values() for name in ["complete.json", "fits.csv", "seed_metrics.csv", "predictions.csv"]]
    sources += [p for p in (ROOT / "data/final").glob("*") if p.is_file()]
    sources += [FEATURE_CACHE / name for name in ["test_features.joblib", "run_metadata.json", "verification.json"]]
    sources += [OLD / directory / name for directory in ["L1-a", "L1-b", "L1-d"] for name in ["predictions.csv", "summary.csv", "run_metadata.json"]]
    sources += [p for level in ["level_1", "level_2", "level_3"] for p in (ROOT / "src/task_3/training" / level).glob("*") if p.is_file()]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    write_json("frozen_config.json", {"frozen_at_wib": now(), "selected_representation": rep, "selected_candidate": selected,
               "selection_criterion": selection["criterion"], "source_selection_sha256": sha256(L2 / "frozen_selection.json"),
               "validation_gate_passed": selection["test_eligible"], "reason_for_evaluation": "Explicit requested exploratory follow-up despite the failed validation gate",
               "conditions": {method: {"candidate": spec["candidate"], "source": str(spec["source"].relative_to(ROOT)), "epochs": epochs[method]} for method, spec in configurations.items()},
               "fit_rows": len(dev), "fit_first_date": dev.Date.min().strftime("%Y-%m-%d"), "fit_last_date": dev.Date.max().strftime("%Y-%m-%d"),
               "features": features, "feature_blocks": {"market": market, "text": text, "availability": availability},
               "training_seeds": SEEDS, "threshold": .5, "fit_from_scratch": True, "test_used_for_selection": False,
               "bootstrap": {"repeats": 1000, "block_length": 10, "seed": 20261110}, "input_sha256": hashes,
               "interpretation": "Previously accessed test period; exploratory only. No threshold, candidate or epoch selection uses this evaluation."})
    models, fits, history = {}, [], []
    for method, spec in configurations.items():
        for seed in SEEDS:
            fixed_epochs = epochs[method][str(seed)]
            evaluation = None
            model, best, ran, trace = fit_small(dev, evaluation, seed, spec["candidate"], market, text, availability, epochs=fixed_epochs)
            assert best == ran == fixed_epochs
            path = OUTPUT / f"{method}_seed_{seed}.joblib"
            joblib.dump(model, path)
            models[method, seed] = model, path
            history.extend({"method": method, "seed": seed, **row} for row in trace)
            fits.append({"method": method, "seed": seed, "train_n": len(dev), "epochs": fixed_epochs, "best_epoch": best, "epochs_run": ran,
                         "optimizer_steps": model.optimizer_steps, "evaluation_is_none": evaluation is None, "fixed_epochs": fixed_epochs,
                         "test_used_for_fitting_or_stopping": bool(evaluation is not None or fixed_epochs is None), "model_file": str(path.relative_to(ROOT))})
            print(f"Refit {method}, seed {seed}: {len(dev)} rows, {fixed_epochs} epochs", flush=True)
    prior = DummyClassifier(strategy="prior").fit(np.zeros((len(dev), 1)), dev.y)
    momentum = LastDirectionClassifier().fit(dev[["log_return_t"]], dev.y)
    for method, model in [("training_prior", prior), ("last_direction", momentum)]:
        path = OUTPUT / f"{method}.joblib"
        joblib.dump(model, path)
        models[method, -1] = model, path
    pd.DataFrame(fits).to_csv(OUTPUT / "fits.csv", index=False)
    pd.DataFrame(history).to_csv(OUTPUT / "history.csv", index=False)
    write_json("test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(OUTPUT / "frozen_config.json"),
               "final_fits_sha256": sha256(OUTPUT / "fits.csv"), "purpose": "One frozen candidate plus prespecified price and naive controls"})
    test = pd.read_csv(ROOT / "data/final/test.csv", parse_dates=["Date", "target_date"]).sort_values("Date").reset_index(drop=True)
    cached = joblib.load(FEATURE_CACHE / "test_features.joblib").set_index("Date").loc[test.Date].reset_index()
    old_cache_metadata = json.loads((FEATURE_CACHE / "run_metadata.json").read_text())
    assert sha256(FEATURE_CACHE / "test_features.joblib") == old_cache_metadata["artifact_sha256"][str((FEATURE_CACHE / "test_features.joblib").relative_to(ROOT))]
    pd.testing.assert_frame_equal(cached[["Date", "target_date", "n_berita", "has_news", *market]], test[["Date", "target_date", "n_berita", "has_news", *market]])
    test = pd.concat([test.drop(columns="articles"), cached[text].reset_index(drop=True)], axis=1)
    test["log1p_n_berita"] = np.log1p(test.n_berita)
    assert dev.target_date.max() < test.Date.min() and test.Date.is_unique
    joblib.dump(test, OUTPUT / "test_features.joblib", compress=3)
    predictions, metrics, classes = [], [], []
    for (method, seed), (model, path) in models.items():
        p, m, c = evaluate(model, method, test, seed, path, features)
        predictions.append(p)
        metrics.append(m)
        classes.extend(c)
    for method, source_id in [("training_prior", "L1-a"), ("last_direction", "L1-b")]:
        new = next(frame for frame in predictions if frame.method.iloc[0] == method).sort_values("Date")
        old = pd.read_csv(OLD / source_id / "predictions.csv", parse_dates=["Date", "target_date"]).sort_values("Date")
        np.testing.assert_array_equal(new.y, old.y)
        np.testing.assert_array_equal(new.Date, old.Date)
        np.testing.assert_allclose(new.p_up, old.p_up, atol=1e-15, rtol=0)
    original = pd.read_csv(OLD / "L1-d/predictions.csv", parse_dates=["Date", "target_date"])
    for seed, part in original.groupby("seed"):
        part = part.sort_values("Date").reset_index(drop=True)
        pd.testing.assert_frame_equal(part[["Date", "target_date", "y"]], test[["Date", "target_date", "y"]])
        original_metrics = {"method": "finbert_original", "seed": seed, "n": len(part), **score_binary(part.y, part.p_up)}
        predictions.append(part[["Date", "target_date", "y", "n_berita", "has_news", "seed", "p_up", "y_pred"]].assign(method="finbert_original"))
        metrics.append(original_metrics)
    predictions, metrics = pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)
    summary = metrics.groupby("method")[METRICS].agg(["mean", "std"])
    rows = [{"method": method, "metric": metric, "mean": summary.loc[method, (metric, "mean")], "std": summary.loc[method, (metric, "std")]} for method in summary.index for metric in METRICS]
    for name, frame in {"predictions": predictions, "seed_metrics": metrics, "per_class": pd.DataFrame(classes), "summary": pd.DataFrame(rows)}.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    indices = circular_indices(len(test), repeats=1000, block_length=10, seed=20261110)
    draws = {}
    for method, part in predictions.groupby("method"):
        draws[method] = np.mean([bootstrap_metrics(group.sort_values("Date").y, group.sort_values("Date").p_up, indices) for _, group in part.groupby("seed")], axis=0)
    draws.update({f"improved_minus_{reference}": draws["finbert_improved"] - draws[reference] for reference in ["training_prior", "last_direction", "price_matched", "price_tuned", "finbert_original"]})
    intervals = []
    for comparison, values in draws.items():
        low, high = np.nanquantile(values, [.025, .975], axis=0)
        intervals.extend({"comparison": comparison, "metric": metric, "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
    pd.DataFrame(intervals).to_csv(OUTPUT / "intervals.csv", index=False)
    np.savez_compressed(OUTPUT / "bootstrap_draws.npz", metrics=np.asarray(METRICS), **draws)
    methods = ["training_prior", "last_direction", "finbert_original", "price_matched", "price_tuned", "finbert_improved"]
    lines = ["# Frozen improvement candidate: exploratory test", "", "Evaluate the single fused-model winner already selected by three-fold validation log loss: frozen FinBERT plus a 16-unit MLP, alpha 0.01, dropout 0.2, uncentered text. Refit uses train+val; epochs are the per-seed medians from validation, without rescaling.", "",
             "The validation success gate failed. This requested test follow-up does not change that result or confer a new untouched holdout. No candidate, threshold, preprocessing or epoch is selected using test results.", "",
             "Matched price control shares alpha/dropout and selects its own epochs by the same validation rule. Tuned price control is the price-only validation winner. Naive baselines are refitted and match their earlier predictions. Original FinBERT test predictions are reused as a historical reference.", "",
             "| Test model | MCC | Macro-F1 | Balanced accuracy | Accuracy | ROC-AUC | Brier | Log loss | ECE-10 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in methods:
        lines.append("| " + " | ".join([method, *[f"{summary.loc[method, (metric, 'mean')]:.4f}" for metric in METRICS]]) + " |")
    lines += ["", f"Refit decisions: **{len(dev)}**. Test decisions: **{len(test)}**. Fixed probability threshold: **0.5**. Epochs: `{json.dumps(epochs, sort_keys=True)}`.", "",
              "| Improved model minus | Metric | 95% lower | 95% upper |", "| --- | --- | ---: | ---: |"]
    interval_frame = pd.DataFrame(intervals)
    for reference in ["training_prior", "price_matched", "price_tuned", "finbert_original"]:
        for metric in ["mcc", "roc_auc", "log_loss"]:
            row = interval_frame.loc[interval_frame.comparison.eq(f"improved_minus_{reference}") & interval_frame.metric.eq(metric)].iloc[0]
            lines.append(f"| {reference} | {metric} | {row.low_95:.4f} | {row.high_95:.4f} |")
    lines += ["", "Reported neural scores average individual seeds, without ensembling. Summary.csv includes sample SD across seeds. Paired intervals use 1,000 circular block resamples of 10 decisions, conditional on the fitted models. Candidate-selection uncertainty and repeated test access are not accounted for; all comparisons are exploratory.", "",
              "The earlier large sklearn FinBERT model and the new small Torch model differ in several implementation/regularization choices. The matched Torch price control is the relevant text comparison. Hard 0/1 momentum probabilities are not a calibrated log-loss baseline. Existing news-timestamp and article-version limitations remain.", "",
              "Original validation reports, failed-gate flags, main notebooks and data exports remain unchanged."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, method, title in zip(axes, ["finbert_original", "price_tuned", "finbert_improved"], ["Original FinBERT", "Tuned price only", "Improved FinBERT"]):
        matrix = np.mean([confusion_matrix(group.y, group.y_pred, labels=[0, 1]) for _, group in predictions.loc[predictions.method.eq(method)].groupby("seed")], axis=0)
        share = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.1f}\n({share[i, j]:.0%})", ha="center", va="center", color="white" if share[i, j] > .6 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], xlabel="Predicted", title=title)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"test_confusion_matrices.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    for relative, expected in hashes.items():
        assert sha256(ROOT / relative) == expected, relative
    artifacts = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    write_json("run_metadata.json", {"completed_at_wib": now(), "input_sha256": hashes,
               "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts}, "fit_rows": len(dev), "test_rows": len(test),
               "refit_neural_models": len(fits), "refit_naive_models": 2, "source_representation": rep, "validation_gate_passed": selection["test_eligible"],
               "test_evaluated": True, "test_used_for_fitting_or_stopping": bool(pd.DataFrame(fits).test_used_for_fitting_or_stopping.any()),
               "test_used_for_selection": False, "fixed_threshold": .5, "saved_predictions_verified": True,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}})
    print(metrics.groupby("method")[METRICS].mean().to_string(), flush=True)
    print("Results:", OUTPUT.relative_to(ROOT), flush=True)

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        main()
