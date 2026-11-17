from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy
import json
import platform
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
sys.path.insert(0, str(ROOT))
from src.task_3.training.fusion import BlockScaler, MarketAlignedMLP
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256
from src.task_3.training.word2vec import FrozenWord2Vec

NAME = "r2_logloss_selection_v1"
OUTPUT = ROOT / "src/task_3/output/model/sanity_check" / NAME
FIGURES = ROOT / "src/task_3/output/figure/sanity_check" / NAME
BASELINE = ROOT / "src/task_3/output/model/level_2/02_pretrained_word2vec"
BASELINE_TEST = ROOT / "src/task_3/output/model/level_2/refit_test_v1/R2"
PRIOR = ROOT / "src/task_3/output/model/level_1/01_training_majority"
PRIOR_TEST = ROOT / "src/task_3/output/model/level_1/refit_test_v1/L1-a"
SEEDS = [42, 123, 2026]

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def score(y, probability):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    predicted = probability >= .5
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = sum((bins == b).mean() * abs(probability[bins == b].mean() - y[bins == b].mean()) for b in np.unique(bins))
    return {"mcc": matthews_corrcoef(y, predicted),
            "macro_f1": f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, predicted), "accuracy": accuracy_score(y, predicted),
            "roc_auc": roc_auc_score(y, probability), "brier": brier_score_loss(y, probability),
            "log_loss": log_loss(y, np.clip(probability, 1e-15, 1 - 1e-15), labels=[0, 1]), "ece_10": ece}

def read_data(names):
    columns = ["Date", "target_date", "y", "n_berita", "has_news", "fold_roles", "articles", *MARKET]
    frame = pd.concat([pd.read_csv(ROOT / "data/final" / name, usecols=columns,
                                   parse_dates=["Date", "target_date"]) for name in names], ignore_index=True)
    for column in ["fold_roles", "articles"]:
        frame[column] = frame[column].map(json.loads)
    return frame.sort_values("Date").reset_index(drop=True)

def fit_model(train, validation, seed, fold=None, fixed_epochs=None):
    scaler = BlockScaler(MARKET, TEXT, AVAILABILITY).fit(train)
    x_train = scaler.transform(train[FEATURES])
    x_val = scaler.transform(validation[FEATURES]) if validation is not None else None
    model = MarketAlignedMLP(**MLP_CONFIG, random_state=seed)
    best_model, best_loss, best_epoch, stale, history = None, np.inf, 0, 0, []
    limit = MAX_EPOCHS if fixed_epochs is None else int(fixed_epochs)
    assert (validation is None) == (fixed_epochs is not None) and limit > 0
    for epoch in range(1, limit + 1):
        model.partial_fit(x_train, train.y.to_numpy(), classes=np.array([0, 1]))
        row = {"fold": fold, "seed": seed, "epoch": epoch,
               "train_log_loss": log_loss(train.y, np.clip(model.predict_proba(x_train)[:, 1], 1e-15, 1 - 1e-15), labels=[0, 1])}
        if validation is not None:
            metrics = score(validation.y, model.predict_proba(x_val)[:, 1])
            row.update({f"val_{key}": value for key, value in metrics.items()})
            if metrics["log_loss"] < best_loss:
                best_model, best_loss, best_epoch, stale = copy.deepcopy(model), metrics["log_loss"], epoch, 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= PATIENCE:
            break
    if validation is None:
        best_model, best_epoch = model, limit
    assert best_model.t_ == len(train) * best_epoch
    return Pipeline([("scaler", scaler), ("mlp", best_model)]), best_epoch, epoch, history

def evaluate(model, frame, path, phase, seed, fold=None):
    joblib.dump(model, path)
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict_proba(frame[FEATURES])[:, 1], probability)
    result = frame[["Date", "target_date", "y", "n_berita", "has_news"]].copy()
    result["phase"], result["seed"], result["fold"] = phase, seed, fold
    result["p_up"], result["y_pred"] = probability, (probability >= .5).astype(int)
    metrics = {"phase": phase, "fold": fold, "seed": seed, "n": len(frame), **score(frame.y, probability)}
    precision, recall, f1, support = precision_recall_fscore_support(frame.y, result.y_pred, labels=[0, 1], zero_division=0)
    per_class = [{"phase": phase, "fold": fold, "seed": seed, "class": name, "support": int(support[i]),
                  "precision": precision[i], "recall": recall[i], "f1": f1[i]} for i, name in enumerate(["DOWN", "UP"])]
    return result, metrics, per_class

def save_figure(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_figures(history, fits, validation, test):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), sharey=True)
    for fold, ax in zip([1, 2, 3], axes):
        trace = history.loc[history.fold.eq(fold) & history.seed.eq(42)]
        selected = fits.loc[fits.fold.eq(fold) & fits.seed.eq(42), "best_epoch"].iloc[0]
        ax.plot(trace.epoch, trace.train_log_loss, color="#767676", linestyle="--", label="Training")
        ax.plot(trace.epoch, trace.val_log_loss, color="#3775BA", label="Validation")
        ax.axvline(selected, color="black", linestyle=":", linewidth=1, label="Selected epoch")
        ax.set(xlabel="Epoch", title=f"Fold {fold}")
    axes[0].set_ylabel("Log loss")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.04), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save_figure(fig, "logloss_selection")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))
    for frame, ax, title in [(validation, axes[0], "Validation"), (test, axes[1], "Test")]:
        matrix = np.mean([confusion_matrix(group.y, group.y_pred, labels=[0, 1]) for _, group in frame.groupby("seed")], axis=0)
        proportions = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.1f}\n({proportions[i, j]:.0%})", ha="center", va="center",
                        color="white" if proportions[i, j] > .6 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"],
               xlabel="Predicted", ylabel="Actual", title=title)
        ax.tick_params(length=0)
    fig.tight_layout()
    save_figure(fig, "confusion_matrices")

def main():
    global MARKET, TEXT, AVAILABILITY, FEATURES, MLP_CONFIG, MAX_EPOCHS, PATIENCE
    if (OUTPUT / "protocol.json").exists():
        raise FileExistsError("Use a new version instead of overwriting the existing experiment.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    source = json.loads((BASELINE / "run_metadata.json").read_text())
    MARKET, TEXT, AVAILABILITY = [source["feature_blocks"][key] for key in ["market", "text", "availability"]]
    FEATURES = [*MARKET, *TEXT, *AVAILABILITY]
    assert FEATURES == source["predictors"] and source["training_seeds"] == SEEDS
    MLP_CONFIG = {**source["model_config"], "hidden_layer_sizes": tuple(source["model_config"]["hidden_layer_sizes"])}
    MAX_EPOCHS, PATIENCE = source["max_epochs"], source["patience"]
    inputs = [*list((ROOT / "data/final").glob("*")), Path(__file__),
              *[ROOT / "src/task_3/training" / name for name in ["fusion.py", "word2vec.py", "testing.py"]],
              *[p for level in ["level_2", "level_3"] for p in (ROOT / "src/task_3/training" / level).glob("*")],
              *[directory / name for directory in [BASELINE, BASELINE_TEST, PRIOR, PRIOR_TEST]
                for name in ["run_metadata.json", "predictions.csv", "summary.csv"]]]
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in inputs if path.is_file()}
    protocol = {"experiment": NAME, "frozen_at_wib": now(), "representation": "R2 pretrained Word2Vec",
                "selection": "Minimum validation log loss; exact ties retain earliest epoch",
                "training_loss": "Original sklearn binary cross-entropy with L2 regularization",
                "selection_only_change": True, "hyperparameter_search": False, "inner_validation": False,
                "seeds": SEEDS, "model_config": MLP_CONFIG, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "threshold": .5, "scaling": "Original BlockScaler, including uncentered text and availability",
                "shuffle": "Original integer random_state behavior retained",
                "final_epochs": "Per-seed median log-loss-selected epoch across three development folds",
                "references": ["Original MCC-selected R2", "Training-prior probability baseline"],
                "input_sha256": input_hashes,
                "interpretation": "One prespecified exploratory trial on a previously accessed test period; validation still selects and scores checkpoints"}
    write_json(OUTPUT / "protocol.json", protocol)
    encoder = FrozenWord2Vec(str(ROOT / "src/task_3/output/model/encoders/word2vec"))
    development = encoder.transform(read_data(["train.csv", "val.csv"]))
    input_hashes[str(Path(encoder.cache_path_).relative_to(ROOT))] = sha256(encoder.cache_path_)
    joblib.dump(encoder, OUTPUT / "encoder.joblib", compress=3)
    fits, histories, frames, scores, classes = [], [], [], [], []
    for fold in [1, 2, 3]:
        roles = development.fold_roles.map(lambda value: value[str(fold)])
        train, val = development.loc[roles.eq("train")].copy(), development.loc[roles.eq("val")].copy()
        assert train.target_date.max() < val.Date.min()
        for seed in SEEDS:
            model, selected, ran, trace = fit_model(train, val, seed, fold)
            prediction, metrics, per_class = evaluate(model, val, OUTPUT / f"fold_{fold}_seed_{seed}.joblib", "validation", seed, fold)
            frames.append(prediction)
            scores.append(metrics)
            classes.extend(per_class)
            histories.extend(trace)
            fits.append({"fold": fold, "seed": seed, "train_n": len(train), "val_n": len(val),
                         "best_epoch": selected, "epochs_run": ran, "selected_log_loss": metrics["log_loss"]})
            print(f"Fold {fold}, seed {seed}: epoch {selected}; val log loss {metrics['log_loss']:.4f}; MCC {metrics['mcc']:.4f}", flush=True)
        write_json(OUTPUT / "status.json", {"phase": "development", "completed_fold": fold, "updated_at_wib": now()})
    fits, history = pd.DataFrame(fits), pd.DataFrame(histories)
    validation_predictions, validation_metrics = pd.concat(frames, ignore_index=True), pd.DataFrame(scores)
    fold_metrics = validation_metrics.groupby("fold")[METRICS].mean().reset_index()
    validation_summary = fold_metrics[METRICS].agg(["mean", "std"]).rename_axis("statistic")
    for filename, frame in {"fits.csv": fits, "history.csv": history, "validation_predictions.csv": validation_predictions,
                            "validation_seed_metrics.csv": validation_metrics, "fold_metrics.csv": fold_metrics,
                            "validation_summary.csv": validation_summary.reset_index()}.items():
        frame.to_csv(OUTPUT / filename, index=False)
    final_epochs = {str(seed): int(value) for seed, value in fits.groupby("seed").best_epoch.median().items()}
    write_json(OUTPUT / "frozen_final_config.json", {"frozen_at_wib": now(), "fit_rows": len(development), "epochs": final_epochs,
               "threshold": .5, "test_used_for_selection": False, "protocol_sha256": sha256(OUTPUT / "protocol.json"),
               "development_fits_sha256": sha256(OUTPUT / "fits.csv")})
    final_models, final_history = {}, []
    for seed in SEEDS:
        model, selected, ran, trace = fit_model(development, None, seed, fixed_epochs=final_epochs[str(seed)])
        final_models[seed] = model
        final_history.extend(trace)
        joblib.dump(model, OUTPUT / f"final_seed_{seed}.joblib")
    pd.DataFrame(final_history).to_csv(OUTPUT / "final_history.csv", index=False)
    write_json(OUTPUT / "test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(OUTPUT / "frozen_final_config.json"),
                                            "purpose": "One exploratory evaluation; no threshold or model selection"})
    test = encoder.transform(read_data(["test.csv"]))
    input_hashes[str(Path(encoder.cache_path_).relative_to(ROOT))] = sha256(encoder.cache_path_)
    assert len(development) == 930 and len(test) == 235 and development.target_date.max() < test.Date.min()
    test_frames, test_scores = [], []
    for seed, model in final_models.items():
        prediction, metrics, per_class = evaluate(model, test, OUTPUT / f"final_seed_{seed}.joblib", "test", seed)
        test_frames.append(prediction)
        test_scores.append(metrics)
        classes.extend(per_class)
    test_predictions, test_metrics = pd.concat(test_frames, ignore_index=True), pd.DataFrame(test_scores)
    test_summary = test_metrics[METRICS].agg(["mean", "std"]).rename_axis("statistic")
    for filename, frame in {"test_predictions.csv": test_predictions, "test_seed_metrics.csv": test_metrics,
                            "test_summary.csv": test_summary.reset_index(), "per_class.csv": pd.DataFrame(classes)}.items():
        frame.to_csv(OUTPUT / filename, index=False)
    intervals, draws_by_phase = [], {}
    for phase, actual, baseline in [("validation", validation_predictions, BASELINE), ("test", test_predictions, BASELINE_TEST)]:
        original = pd.read_csv(baseline / "predictions.csv", parse_dates=["Date", "target_date"])
        new_draws, old_draws = [], []
        for fold in ([1, 2, 3] if phase == "validation" else [None]):
            new = actual.loc[actual.fold.eq(fold)] if fold else actual
            old = original.loc[original.fold.eq(fold)] if fold else original
            new, old = new.sort_values(["seed", "Date"]).reset_index(drop=True), old.sort_values(["seed", "Date"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(new[["seed", "Date", "target_date", "y"]], old[["seed", "Date", "target_date", "y"]])
            indices = circular_indices(new.Date.nunique(), seed=20261110 + (fold or 0))
            new_draws.append(np.mean([bootstrap_metrics(g.y, g.p_up, indices) for _, g in new.groupby("seed")], axis=0))
            old_draws.append(np.mean([bootstrap_metrics(g.y, g.p_up, indices) for _, g in old.groupby("seed")], axis=0))
        new_values, old_values = np.mean(new_draws, axis=0), np.mean(old_draws, axis=0)
        draws_by_phase[f"{phase}_new"], draws_by_phase[f"{phase}_original"] = new_values, old_values
        for label, values in [("logloss_selection", new_values), ("mcc_selection", old_values), ("new_minus_original", new_values - old_values)]:
            low, high = np.nanquantile(values, [.025, .975], axis=0)
            intervals.extend({"phase": phase, "comparison": label, "metric": metric,
                              "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
    pd.DataFrame(intervals).to_csv(OUTPUT / "intervals.csv", index=False)
    np.savez_compressed(OUTPUT / "bootstrap_draws.npz", metrics=np.asarray(METRICS), **draws_by_phase)
    summaries = []
    lines = ["# R2 log-loss checkpoint selection", "",
             "Select the minimum validation log loss instead of maximum validation MCC. Model architecture, training objective, scaling, integer seed behavior, split dates, patience, and threshold follow the original R2 pipeline.", "",
             "This is a checkpoint-selection experiment, not a broader hyperparameter search. Validation still selects and scores the same checkpoints; the previously accessed test period is exploratory.", "",
             "| Phase | Method | " + " | ".join(METRICS) + " |", "| --- | --- | " + " | ".join(["---:"] * len(METRICS)) + " |"]
    for phase, summary, baseline, prior in [("validation", validation_summary, BASELINE, PRIOR), ("test", test_summary, BASELINE_TEST, PRIOR_TEST)]:
        references = [("mcc_selection", pd.read_csv(baseline / "summary.csv").set_index("statistic")),
                      ("logloss_selection", summary), ("training_prior", pd.read_csv(prior / "summary.csv").set_index("statistic"))]
        for label, frame in references:
            summaries.append(frame.reset_index().assign(phase=phase, method=label))
            values = []
            for metric in METRICS:
                value, spread = frame.loc["mean", metric], frame.loc["std", metric]
                values.append(f"{value:.4f} ± {spread:.4f}" if pd.notna(spread) else f"{value:.4f}")
            lines.append("| " + " | ".join([phase, label, *values]) + " |")
    lines += ["", "Validation SD is across folds; test SD is across seeds. Bootstrap comparisons use paired circular blocks of 10 decisions with fitted models fixed; validation resampling stays within each fold.", "",
              f"Final epochs: `{json.dumps(final_epochs, sort_keys=True)}`. Original ablation notebooks, models, and result tables remain unchanged."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    pd.concat(summaries, ignore_index=True).to_csv(OUTPUT / "comparison.csv", index=False)
    make_figures(history, fits, validation_predictions, test_predictions)
    for relative, expected in input_hashes.items():
        assert sha256(ROOT / relative) == expected, relative
    write_json(OUTPUT / "status.json", {"phase": "complete", "updated_at_wib": now()})
    artifacts = [path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    write_json(OUTPUT / "run_metadata.json", {"experiment": NAME, "completed_at_wib": now(), "input_sha256": input_hashes,
               "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts}, "features": FEATURES,
               "final_epochs": final_epochs, "development_models": 9, "final_models": 3,
               "validation_decisions_per_seed": 475, "test_decisions_per_seed": 235, "threshold": .5,
               "selection": protocol["selection"], "test_used_for_selection": False, "hyperparameter_search": False,
               "saved_model_predictions_verified": True, "interpretation": protocol["interpretation"],
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}})
    print("Validation:", validation_summary.loc["mean"].to_dict(), flush=True)
    print("Test:", test_summary.loc["mean"].to_dict(), flush=True)
    print("Results:", OUTPUT.relative_to(ROOT), flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
