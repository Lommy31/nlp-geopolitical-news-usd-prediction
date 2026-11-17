from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
sys.path.insert(0, str(ROOT))
from src.task_3.preprocessing.splitting import load_split
from src.task_3.sanity_check.level1_improvement_models import fit_small
from src.task_3.sanity_check.level3_improvement_models import fit_improved
from src.task_3.training.controlled_fusion import (
    MODEL_CONFIG, WORD_FEATURES, encode_articles, fit_fusion, score_binary,
)
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

OUTPUT = ROOT / "src/task_3/output/model/sanity_check/level3_improvement_v1"
FIGURES = ROOT / "src/task_3/output/figure/sanity_check/level3_improvement_v1"
CONTRACT = ROOT / "src/task_3/output/model/level_3/level_3_contract.json"
L1 = ROOT / "src/task_3/output/model/sanity_check/level1_improvement_v1"
L2 = ROOT / "src/task_3/output/model/sanity_check/level2_improvement_v1"
PRIOR_TEST = ROOT / "src/task_3/output/model/sanity_check/level2_improvement_test_v1"
OLD_A000 = ROOT / "src/task_3/output/model/level_3/refit_test_v1/A000"
SEEDS = [42, 123, 2026]
MARKET = json.loads((ROOT / "data/final/metadata.json").read_text())["features"]["market"]
AVAILABILITY = ["log1p_n_berita", "has_news"]

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def read_json(name):
    return json.loads((OUTPUT / name).read_text())

def prepare():
    if OUTPUT.exists():
        raise FileExistsError("Use a new version instead of overwriting the experiment.")
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text())
    assert contract["model_config"] == MODEL_CONFIG and contract["representation"] == "R2 pretrained Word2Vec"
    assert contract["implementation_sha256"] == sha256(ROOT / "src/task_3/training/controlled_fusion.py")
    assert json.loads((L2 / "verification.json").read_text())["verified"]
    assert json.loads((PRIOR_TEST / "verification.json").read_text())["verified"]
    winner = json.loads((L2 / "frozen_selection.json").read_text())["winners"]["R2"]
    assert winner["alpha"] == 1. and winner["dropout"] == .5
    sources = [Path(__file__), ROOT / "src/task_3/sanity_check/level3_improvement_models.py", ROOT / "src/task_3/sanity_check/level1_improvement_models.py",
               ROOT / "src/task_3/improvement_plan.md", CONTRACT, ROOT / "data/final/metadata.json", ROOT / "data/final/train.csv", ROOT / "data/final/val.csv"]
    sources += [L2 / name for name in ["verification.json", "run_metadata.json", "frozen_selection.json", "permutation_result.json"]]
    sources += [PRIOR_TEST / name for name in ["verification.json", "run_metadata.json", "predictions.csv", "seed_metrics.csv"]]
    sources += [OLD_A000 / name for name in ["run_metadata.json", "predictions.csv", "seed_metrics.csv"]]
    sources += [p for p in (ROOT / "src/task_3/training").glob("*.py")]
    sources += [p for level in ["level_1", "level_2", "level_3"] for p in (ROOT / "src/task_3/training" / level).glob("*") if p.is_file()]
    sources += [L1 / "candidates/small_mlp_price_a1_d0p5_center0" / name for name in ["complete.json", "fits.csv", "predictions.csv", "seed_metrics.csv"]]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    write_json("protocol.json", {"experiment": OUTPUT.name, "frozen_at_wib": now(), "target": "Original daily next-fixing UP/DOWN, original folds and final CSV dates",
               "representation": "Frozen Word2Vec 300, matching the main Level 3 contract", "hyperparameter_source": winner,
               "conditions": {"old_A111": {"flags": [1, 1, 1], "recipe": "Original Level 3 torch 64/32, alpha .001, no dropout, max200/patience25, MCC selection"},
                              "A111_uncentered": {"flags": [1, 1, 1], "centered": False}, "A111_centered": {"flags": [1, 1, 1], "centered": True},
                              "A000_improved": {"flags": [0, 0, 0], "centered": "The validation-selected A111 centering option"}},
               "improved_recipe": {"hidden": [16], "dropout": .5, "alpha": 1., "native_fusion_width": 916,
                                   "remove_zero_padding": "Only constant padding removed; this alone does not reduce effective capacity",
                                   "selection": "Trailing five-epoch mean validation log loss, min_delta .001, max60, patience10",
                                   "training": "Original Adam, batch32, float64, persistent seeded shuffle; dropout seed matches Level 2"},
               "selection": "Only two A111 centering options; minimum three-fold mean validation log loss after averaging seeds within folds",
               "refit_epochs": "Median selected epoch per condition and seed across three validation folds, without scaling",
               "primary_comparisons": ["Improved A111 minus old-recipe A111", "Improved A111 minus matched improved A000"],
               "old_A111_status": "Trained here as an original-recipe reference; no earlier daily A111 result is assumed",
               "controls": ["New matched price-only alpha1/dropout.5 with its validation-selected epochs", "Reused prior, last direction and validation-tuned price baseline", "Reused historical original A000"],
               "test_followup": "Requested exploratory evaluation regardless of validation gate; freeze all choices before final test scoring",
               "threshold": .5, "seeds": SEEDS, "bootstrap": {"repeats": 1000, "block_length": 10, "seed": 20261110},
               "no_new_permutation_claim": "Existing Level 2 Word2Vec permutation is context only; it is not a null distribution for Level 3 attention/gating",
               "limitations": ["Test period already accessed repeatedly; exploratory only", "Validation still selects and scores checkpoints",
                               "Combined modifications do not isolate one improvement", "No encoder fine-tuning or new PCA/sentiment/rolling-delta branch"], "input_sha256": hashes})
    raw = pd.concat([load_split(ROOT / "data/final", part) for part in ["train", "val"]], ignore_index=True).sort_values("Date").reset_index(drop=True)
    frame, vectors, path, spec, info, _ = encode_articles(raw, ROOT / "src/task_3/output/model/encoders/word2vec")
    joblib.dump(frame, OUTPUT / "development_features.joblib", compress=3)
    write_json("prepared.json", {"completed_at_wib": now(), "protocol_sha256": sha256(OUTPUT / "protocol.json"), "development_rows": len(frame),
               "features_sha256": sha256(OUTPUT / "development_features.joblib"), "article_cache": str(path.relative_to(ROOT)), "article_cache_sha256": sha256(path),
               "subset_cache": str(Path(info["subset_cache"]).relative_to(ROOT)), "subset_sha256": info["subset_sha256"], "text_spec": spec})
    print(f"Prepared {len(frame)} development decisions and {len(vectors)} frozen article vectors", flush=True)

def development_data():
    prepared = read_json("prepared.json")
    assert prepared["features_sha256"] == sha256(OUTPUT / "development_features.joblib")
    path = ROOT / prepared["article_cache"]
    assert sha256(path) == prepared["article_cache_sha256"]
    with np.load(path, allow_pickle=False) as data:
        vectors = dict(zip(data["article_ids"].tolist(), data["vectors"]))
    return joblib.load(OUTPUT / "development_features.joblib"), vectors

def evaluate(model, frame, vectors, condition, seed, fold=None):
    p = model.predict_proba(frame, vectors)[:, 1]
    result = frame[["Date", "target_date", "y", "n_berita", "has_news"]].copy()
    result["condition"], result["seed"], result["fold"] = condition, seed, fold
    result["p_up"], result["y_pred"] = p, (p >= .5).astype(int)
    metrics = {"condition": condition, "fold": fold, "seed": seed, "n": len(frame), **score_binary(frame.y, p)}
    precision, recall, f1, support = precision_recall_fscore_support(frame.y, result.y_pred, labels=[0, 1], zero_division=0)
    classes = [{"condition": condition, "seed": seed, "fold": fold, "class": name, "support": int(support[i]),
                "precision": precision[i], "recall": recall[i], "f1": f1[i]} for i, name in enumerate(["DOWN", "UP"])]
    return result, metrics, classes

def run_cv(condition, flags, centered=False, original=False):
    directory = OUTPUT / "candidates" / condition
    if (directory / "complete.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    frame, vectors = development_data()
    predictions, metrics, classes, fits, history = [], [], [], [], []
    for fold in [1, 2, 3]:
        role = frame.fold_roles.map(lambda value: value[str(fold)])
        train, val = frame.loc[role.eq("train")], frame.loc[role.eq("val")]
        assert train.target_date.max() < val.Date.min()
        for seed in SEEDS:
            if original:
                predictor, best, ran, trace, seconds = fit_fusion(train, val, vectors, seed, flags, MARKET, AVAILABILITY)
                changes = {}
            else:
                predictor, best, ran, trace, seconds, changes = fit_improved(train, val, vectors, seed, flags, MARKET, AVAILABILITY, centered=centered)
            path = directory / f"fold_{fold}_seed_{seed}.joblib"
            joblib.dump(predictor, path)
            p, m, c = evaluate(predictor, val, vectors, condition, seed, fold)
            np.testing.assert_array_equal(joblib.load(path).predict_proba(val, vectors)[:, 1], p.p_up)
            predictions.append(p)
            metrics.append(m)
            classes.extend(c)
            fits.append({"condition": condition, "fold": fold, "seed": seed, "train_n": len(train), "eval_n": len(val), "best_epoch": best,
                         "epochs_run": ran, "fit_seconds": seconds, "centered": centered, "original_recipe": original,
                         "parameters": sum(v.numel() for v in predictor.model.parameters()), **changes})
            history.extend({"condition": condition, "fold": fold, "seed": seed, **row} for row in trace)
            for name, table in {"predictions": pd.concat(predictions, ignore_index=True), "seed_metrics": pd.DataFrame(metrics),
                                "fits": pd.DataFrame(fits), "history": pd.DataFrame(history), "per_class": pd.DataFrame(classes)}.items():
                table.to_csv(directory / f"{name}.csv", index=False)
            print(f"{condition}, fold {fold}, seed {seed}: epoch {best}, MCC {m['mcc']:.4f}, LL {m['log_loss']:.4f}, {seconds:.1f}s", flush=True)
    summary = pd.DataFrame(metrics).groupby("fold")[METRICS].mean().agg(["mean", "std"]).rename_axis("statistic")
    summary.to_csv(directory / "summary.csv")
    (directory / "complete.json").write_text(json.dumps({"completed_at_wib": now(), "condition": condition, "flags": flags, "centered": centered,
        "original_recipe": original, "models": len(fits), "epochs": {str(int(seed)): int(v) for seed, v in pd.DataFrame(fits).groupby("seed").best_epoch.median().items()}}, indent=2) + "\n")

def improved_validation():
    run_cv("A111_uncentered", [1, 1, 1])
    run_cv("A111_centered", [1, 1, 1], centered=True)
    losses = {name: pd.read_csv(OUTPUT / "candidates" / name / "summary.csv").set_index("statistic").loc["mean", "log_loss"] for name in ["A111_uncentered", "A111_centered"]}
    chosen = min(losses, key=losses.get)
    centered = chosen == "A111_centered"
    write_json("selected_centering.json", {"frozen_at_wib": now(), "chosen": chosen, "centered": centered, "validation_mean_log_loss": losses,
               "test_used_for_selection": False, "source_summaries_sha256": {name: sha256(OUTPUT / "candidates" / name / "summary.csv") for name in losses}})
    run_cv("A000_improved", [0, 0, 0], centered=centered)

def final_test():
    dev, dev_vectors = development_data()
    selected = read_json("selected_centering.json")
    names = {"A111_original_recipe": "old_A111", "A111_improved": selected["chosen"], "A000_improved": "A000_improved"}
    conditions = {name: json.loads((OUTPUT / "candidates" / source / "complete.json").read_text()) for name, source in names.items()}
    price_source = L1 / "candidates/small_mlp_price_a1_d0p5_center0"
    price_config = json.loads((price_source / "complete.json").read_text())["candidate"]
    price_epochs = {str(int(seed)): int(v) for seed, v in pd.read_csv(price_source / "fits.csv").groupby("seed").best_epoch.median().items()}
    write_json("frozen_final_config.json", {"frozen_at_wib": now(), "protocol_sha256": sha256(OUTPUT / "protocol.json"),
               "selected_centering_sha256": sha256(OUTPUT / "selected_centering.json"), "conditions": conditions, "source_names": names,
               "candidate_fits_sha256": {name: sha256(OUTPUT / "candidates" / source / "fits.csv") for name, source in names.items()},
               "price_config": price_config, "price_epochs": price_epochs, "refit_rows": len(dev), "probability_threshold": .5, "test_used_for_selection": False,
               "interpretation": "Requested exploratory follow-up on a previously accessed test period; no test-based choices"})
    directory = OUTPUT / "final_test"
    directory.mkdir(exist_ok=True)
    models, fits, histories = {}, [], []
    for name, config in conditions.items():
        for seed in SEEDS:
            epochs = config["epochs"][str(seed)]
            evaluation = None
            if config["original_recipe"]:
                predictor, best, ran, trace, seconds = fit_fusion(dev, evaluation, dev_vectors, seed, config["flags"], MARKET, AVAILABILITY, fixed_epochs=epochs)
            else:
                predictor, best, ran, trace, seconds, _ = fit_improved(dev, evaluation, dev_vectors, seed, config["flags"], MARKET, AVAILABILITY, centered=config["centered"], fixed_epochs=epochs)
            path = directory / f"{name}_seed_{seed}.joblib"
            joblib.dump(predictor, path)
            models[name, seed] = predictor, path
            fits.append({"condition": name, "seed": seed, "train_n": len(dev), "epochs": epochs, "epochs_run": ran, "evaluation_is_none": evaluation is None,
                         "fixed_epochs": epochs, "test_used_for_fitting_or_stopping": bool(evaluation is not None or epochs is None), "fit_seconds": seconds})
            histories.extend({"condition": name, "seed": seed, **row} for row in trace)
            print(f"Refit {name}, seed {seed}: {epochs} epochs, {seconds:.1f}s", flush=True)
    for seed in SEEDS:
        predictor, best, ran, trace = fit_small(dev, None, seed, price_config, MARKET, WORD_FEATURES, AVAILABILITY, epochs=price_epochs[str(seed)])
        path = directory / f"price_matched_seed_{seed}.joblib"
        joblib.dump(predictor, path)
        models["price_matched", seed] = predictor, path
        fits.append({"condition": "price_matched", "seed": seed, "train_n": len(dev), "epochs": best, "epochs_run": ran, "evaluation_is_none": True,
                     "fixed_epochs": best, "test_used_for_fitting_or_stopping": False})
        histories.extend({"condition": "price_matched", "seed": seed, **row} for row in trace)
    pd.DataFrame(fits).to_csv(directory / "fits.csv", index=False)
    pd.DataFrame(histories).to_csv(directory / "history.csv", index=False)
    write_json("test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(OUTPUT / "frozen_final_config.json"), "final_fits_sha256": sha256(directory / "fits.csv")})
    raw = load_split(ROOT / "data/final", "test")
    test, test_vectors, path, _, info, _ = encode_articles(raw, ROOT / "src/task_3/output/model/encoders/word2vec")
    assert dev.target_date.max() < test.Date.min()
    joblib.dump(test, OUTPUT / "test_features.joblib", compress=3)
    write_json("test_encoder.json", {"article_cache": str(path.relative_to(ROOT)), "article_cache_sha256": sha256(path),
               "subset_cache": str(Path(info["subset_cache"]).relative_to(ROOT)), "subset_sha256": info["subset_sha256"]})
    predictions, metrics, classes = [], [], []
    for (name, seed), (predictor, path) in models.items():
        if name == "price_matched":
            p = predictor.predict_proba(test)[:, 1]
            np.testing.assert_array_equal(joblib.load(path).predict_proba(test)[:, 1], p)
            frame = test[["Date", "target_date", "y", "n_berita", "has_news"]].copy()
            frame["condition"], frame["seed"], frame["p_up"], frame["y_pred"] = name, seed, p, (p >= .5).astype(int)
            m = {"condition": name, "seed": seed, "n": len(test), **score_binary(test.y, p)}
            c = []
        else:
            frame, m, c = evaluate(predictor, test, test_vectors, name, seed)
            np.testing.assert_array_equal(joblib.load(path).predict_proba(test, test_vectors)[:, 1], frame.p_up)
        predictions.append(frame)
        metrics.append(m)
        classes.extend(c)
    previous = pd.read_csv(PRIOR_TEST / "predictions.csv", parse_dates=["Date", "target_date"])
    references = previous.loc[previous.method.isin(["training_prior", "last_direction", "price_tuned"])].rename(columns={"method": "condition"})
    old = pd.read_csv(OLD_A000 / "predictions.csv", parse_dates=["Date", "target_date"])
    old["condition"] = "A000_original"
    for name, part in pd.concat([references, old], ignore_index=True).groupby("condition"):
        for seed, group in part.groupby("seed"):
            group = group.sort_values("Date").reset_index(drop=True)
            pd.testing.assert_frame_equal(group[["Date", "target_date", "y"]], test[["Date", "target_date", "y"]])
            predictions.append(group[["Date", "target_date", "y", "n_berita", "has_news", "condition", "seed", "p_up", "y_pred"]])
            metrics.append({"condition": name, "seed": seed, "n": len(test), **score_binary(group.y, group.p_up)})
    pd.concat(predictions, ignore_index=True).to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(directory / "seed_metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(directory / "per_class.csv", index=False)
    write_json("test_complete.json", {"completed_at_wib": now(), "refit_rows": len(dev), "test_rows": len(test), "new_models": len(models), "test_used_for_selection": False})

def report():
    predictions = pd.read_csv(OUTPUT / "final_test/predictions.csv", parse_dates=["Date", "target_date"])
    metrics = pd.read_csv(OUTPUT / "final_test/seed_metrics.csv")
    rows = []
    for condition, frame in metrics.groupby("condition"):
        rows.extend({"condition": condition, "metric": metric, "mean": frame[metric].mean(), "std": frame[metric].std(ddof=1)} for metric in METRICS)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "test_summary.csv", index=False)
    indices = circular_indices(predictions.Date.nunique(), repeats=1000, block_length=10, seed=20261110)
    draws = {name: np.mean([bootstrap_metrics(frame.sort_values("Date").y, frame.sort_values("Date").p_up, indices) for _, frame in group.groupby("seed")], axis=0)
             for name, group in predictions.groupby("condition")}
    comparisons = ["A111_original_recipe", "A000_improved", "training_prior", "price_matched", "price_tuned"]
    draws.update({f"improved_minus_{reference}": draws["A111_improved"] - draws[reference] for reference in comparisons})
    intervals = []
    for comparison, values in draws.items():
        low, high = np.nanquantile(values, [.025, .975], axis=0)
        intervals.extend({"comparison": comparison, "metric": metric, "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
    pd.DataFrame(intervals).to_csv(OUTPUT / "test_intervals.csv", index=False)
    np.savez_compressed(OUTPUT / "test_bootstrap.npz", metrics=np.asarray(METRICS), **draws)
    order = ["training_prior", "last_direction", "price_tuned", "price_matched", "A000_original", "A111_original_recipe", "A000_improved", "A111_improved"]
    lines = ["# Level 3 improvement: exploratory daily-direction test", "", "The main Level 3 Word2Vec representation is retained. A111 is query + price-to-news cross-attention + gate. The original-recipe A111 reference is trained in this trial; it is not presented as an earlier completed notebook result.", "",
             "The improved recipe fixes hidden16, alpha1 and dropout0.5 from the earlier R2 validation selection. Only centering versus no centering is compared for A111 in validation. Its winning option is used by the matched A000 control. Checkpoints use trailing-five-epoch log loss with min_delta0.001, max60 and patience10; final durations are per-seed median epochs, without rescaling.", "",
             "A111 versus matched A000 measures the combined components under the same improved recipe. Improved versus original-recipe A111 compares the full improvement bundle, not one isolated change. Native zero padding removal reduces computation, not effective capacity by itself.", "",
             "| Test model | MCC | Macro-F1 | Balanced accuracy | Accuracy | ROC-AUC | Brier | Log loss | ECE-10 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition in order:
        row = summary.loc[summary.condition.eq(condition)].set_index("metric")
        lines.append("| " + " | ".join([condition, *[f"{row.loc[metric, 'mean']:.4f}" for metric in METRICS]]) + " |")
    lines += ["", "| Improved A111 minus | Metric | 95% lower | 95% upper |", "| --- | --- | ---: | ---: |"]
    intervals = pd.DataFrame(intervals)
    for reference in comparisons:
        for metric in ["mcc", "macro_f1", "roc_auc", "log_loss"]:
            row = intervals.loc[intervals.comparison.eq(f"improved_minus_{reference}") & intervals.metric.eq(metric)].iloc[0]
            lines.append(f"| {reference} | {metric} | {row.low_95:.4f} | {row.high_95:.4f} |")
    lines += ["", f"Selected centering and final epochs: `{json.dumps(read_json('frozen_final_config.json')['conditions'], sort_keys=True)}`.", "",
              "Scores average three individual seeds for neural models, without ensembling. Naive baselines are deterministic. Intervals use 1,000 paired circular block resamples of length10, conditional on fitted models; they omit selection uncertainty. Prior, last-direction, tuned-price and historical A000 predictions are reused on identical dates and labels.", "",
              "The test period has already been accessed; this is a requested exploratory comparison, not a new untouched holdout. No test-based choice, threshold tuning, new encoder, extra hyperparameter grid or new null-distribution claim is made. The learned task query is not an externally anchored USD/IDR token. Date-only news timing and historical-body limitations remain.", "",
              "Original main notebooks, data exports, model selection and earlier sanity checks remain unchanged."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, condition, title in zip(axes, ["A111_original_recipe", "A000_improved", "A111_improved"], ["A111 original recipe", "A000 improved", "A111 improved"]):
        matrix = np.mean([confusion_matrix(group.y, group.y_pred, labels=[0, 1]) for _, group in predictions.loc[predictions.condition.eq(condition)].groupby("seed")], axis=0)
        shares = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.1f}\n({shares[i, j]:.0%})", ha="center", va="center", color="white" if shares[i, j] > .6 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], xlabel="Predicted", title=title)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"test_confusion_matrices.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    hashes = read_json("protocol.json")["input_sha256"]
    for relative, expected in hashes.items():
        assert sha256(ROOT / relative) == expected, relative
    artifacts = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    final_fits = pd.read_csv(OUTPUT / "final_test/fits.csv")
    write_json("run_metadata.json", {"completed_at_wib": now(), "input_sha256": hashes,
               "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts},
               "validation_models": sum(len(pd.read_csv(path)) for path in (OUTPUT / "candidates").glob("*/fits.csv")),
               "new_test_models": len(final_fits), "refit_rows": read_json("test_complete.json")["refit_rows"], "test_rows": predictions.Date.nunique(),
               "test_used_for_selection": False, "test_used_for_fitting_or_stopping": bool(final_fits.test_used_for_fitting_or_stopping.any()),
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}})
    print(metrics.groupby("condition")[METRICS].mean().to_string(), flush=True)
    print("Results:", OUTPUT.relative_to(ROOT), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "original", "improved", "test", "report"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        if args.phase == "original":
            run_cv("old_A111", [1, 1, 1], original=True)
        else:
            {"prepare": prepare, "improved": improved_validation, "test": final_test, "report": report}[args.phase]()
