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

NAME = "r2_inner_validation_v1"
OUTPUT = ROOT / "src/task_3/output/model/sanity_check" / NAME
FIGURES = ROOT / "src/task_3/output/figure/sanity_check" / NAME
BASELINE = ROOT / "src/task_3/output/model/level_2/02_pretrained_word2vec"
BASELINE_TEST = ROOT / "src/task_3/output/model/level_2/refit_test_v1/R2"
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

def make_model(train, seed):
    scaler = BlockScaler(MARKET, TEXT, AVAILABILITY).fit(train)
    assert scaler.n_samples_seen_ == len(train)
    model = MarketAlignedMLP(**MLP_CONFIG, random_state=seed)
    return scaler, model

def fit_inner(train, val, seed, fold):
    scaler, model = make_model(train, seed)
    x_train, x_val = scaler.transform(train[FEATURES]), scaler.transform(val[FEATURES])
    best_model, best_key, best_epoch, stale, history = None, None, 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(x_train, train.y.to_numpy(), classes=np.array([0, 1]))
        metrics = score(val.y, model.predict_proba(x_val)[:, 1])
        history.append({"fold": fold, "seed": seed, "epoch": epoch,
                        "train_log_loss": score(train.y, model.predict_proba(x_train)[:, 1])["log_loss"],
                        **{f"inner_{key}": value for key, value in metrics.items()}})
        key = metrics["mcc"], metrics["macro_f1"], -metrics["log_loss"]
        if best_key is None or key > best_key:
            best_model, best_key, best_epoch, stale = copy.deepcopy(model), key, epoch, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return Pipeline([("scaler", scaler), ("mlp", best_model)]), best_epoch, history

def fit_fixed(train, seed, epochs):
    scaler, model = make_model(train, seed)
    values = scaler.transform(train[FEATURES])
    for _ in range(epochs):
        model.partial_fit(values, train.y.to_numpy(), classes=np.array([0, 1]))
    assert model.t_ == len(train) * epochs
    return Pipeline([("scaler", scaler), ("mlp", model)])

def predict_and_save(model, frame, path, phase, seed, fold=None):
    joblib.dump(model, path)
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict_proba(frame[FEATURES])[:, 1], probability)
    result = frame[["Date", "target_date", "y", "n_berita", "has_news"]].copy()
    result["phase"], result["seed"], result["fold"] = phase, seed, fold
    result["p_up"], result["y_pred"] = probability, (probability >= .5).astype(int)
    metrics = {"phase": phase, "fold": fold, "seed": seed, "n": len(frame), **score(frame.y, probability)}
    precision, recall, f1, support = precision_recall_fscore_support(frame.y, result.y_pred, labels=[0, 1], zero_division=0)
    classes = [{"phase": phase, "fold": fold, "seed": seed, "class": name,
                "precision": precision[i], "recall": recall[i], "f1": f1[i], "support": int(support[i])}
               for i, name in enumerate(["DOWN", "UP"])]
    return result, metrics, classes

def aligned_baseline(new, old, keys):
    actual = new.sort_values(keys).reset_index(drop=True)
    expected = old.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[[*keys, "target_date", "y"]], expected[[*keys, "target_date", "y"]])
    return actual, expected

def make_figures(history, fits, outer, test, original_folds, new_folds):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), sharey=True)
    for fold, ax in zip([1, 2, 3], axes):
        data = history.loc[history.fold.eq(fold) & history.seed.eq(42)]
        selected = fits.loc[fits.fold.eq(fold) & fits.seed.eq(42), "inner_best_epoch"].iloc[0]
        ax.plot(data.epoch, data.train_log_loss, color="#767676", linestyle="--", label="Inner train")
        ax.plot(data.epoch, data.inner_log_loss, color="#3775BA", label="Inner validation")
        ax.axvline(selected, color="black", linestyle=":", linewidth=1, label="Selected epoch")
        ax.set(xlabel="Epoch", title=f"Fold {fold}")
    axes[0].set_ylabel("Log loss")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.04), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save_figure(fig, "inner_selection")
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.plot(original_folds.fold, original_folds.mcc, "o--", color="#767676", label="Original selection")
    ax.plot(new_folds.fold, new_folds.mcc, "o-", color="#3775BA", label="Inner selection")
    ax.axhline(0, color="black", linewidth=.7)
    ax.set(xlabel="Outer fold", ylabel="Validation MCC", xticks=[1, 2, 3])
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "outer_validation")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))
    for frame, ax, title in [(outer, axes[0], "Outer validation"), (test, axes[1], "Test")]:
        matrix = np.mean([confusion_matrix(group.y, group.y_pred, labels=[0, 1]) for _, group in frame.groupby("seed")], axis=0)
        proportion = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(proportion, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.1f}\n({proportion[i, j]:.0%})", ha="center", va="center",
                        color="white" if proportion[i, j] > .6 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"],
               xlabel="Predicted", ylabel="Actual", title=title)
        ax.tick_params(length=0)
    fig.tight_layout()
    save_figure(fig, "confusion_matrices")

def save_figure(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    global MARKET, TEXT, AVAILABILITY, FEATURES, MLP_CONFIG, MAX_EPOCHS, PATIENCE
    if (OUTPUT / "protocol.json").exists():
        raise FileExistsError("Use a new experiment version instead of overwriting an existing run.")
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
              *list((ROOT / "src/task_3/training/level_2").glob("*")),
              *[directory / name for directory in [BASELINE, BASELINE_TEST]
                for name in ["run_metadata.json", "predictions.csv", "summary.csv", "fits.csv"]]]
    inputs = [path for path in inputs if path.is_file()]
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    protocol = {"experiment": NAME, "frozen_at_wib": now(), "representation": "R2 pretrained Word2Vec",
                "inner_train_fraction": .8, "inner_validation": "Chronological trailing 20% of each outer training subset",
                "purge": "Keep inner training rows only when target_date < first inner validation Date",
                "selection": ["inner validation MCC", "inner validation macro-F1", "negative inner validation log loss"],
                "outer_evaluation": "Fresh fit on full outer train for inner-selected epoch count; one outer validation evaluation",
                "final_epochs": "Per-seed median inner-selected epoch across three folds", "threshold": .5,
                "seeds": SEEDS, "model_config": MLP_CONFIG, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "scaling": "Original BlockScaler; text and availability remain uncentered",
                "shuffle": "Original integer random_state behavior retained", "input_sha256": input_hashes,
                "interpretation": "One prespecified diagnostic. Representation and hyperparameters were selected earlier; the test period was already accessed."}
    write_json(OUTPUT / "protocol.json", protocol)
    encoder = FrozenWord2Vec(str(ROOT / "src/task_3/output/model/encoders/word2vec"))
    development = encoder.transform(read_data(["train.csv", "val.csv"]))
    input_hashes[str(Path(encoder.cache_path_).relative_to(ROOT))] = sha256(encoder.cache_path_)
    joblib.dump(encoder, OUTPUT / "encoder.joblib", compress=3)
    fits, histories, memberships, predictions, metrics, per_class, inner_metrics = [], [], [], [], [], [], []
    for fold in [1, 2, 3]:
        roles = development.fold_roles.map(lambda value: value[str(fold)])
        train, val = development.loc[roles.eq("train")].copy(), development.loc[roles.eq("val")].copy()
        split_at = int(len(train) * .8)
        inner_val = train.iloc[split_at:].copy()
        inner_train = train.iloc[:split_at].loc[lambda frame: frame.target_date.lt(inner_val.Date.min())].copy()
        assert inner_train.target_date.max() < inner_val.Date.min()
        assert train.target_date.max() < val.Date.min()
        assert set(inner_train.y) == set(inner_val.y) == {0, 1}
        for role, frame in [("inner_train", inner_train), ("inner_val", inner_val), ("outer_val", val)]:
            membership = frame[["Date", "target_date", "y"]].copy()
            membership["fold"], membership["role"] = fold, role
            memberships.append(membership)
        for seed in SEEDS:
            inner_model, epochs, trace = fit_inner(inner_train, inner_val, seed, fold)
            joblib.dump(inner_model, OUTPUT / f"inner_fold_{fold}_seed_{seed}.joblib")
            inner_metrics.append({"fold": fold, "seed": seed, **score(inner_val.y, inner_model.predict_proba(inner_val[FEATURES])[:, 1])})
            model = fit_fixed(train, seed, epochs)
            result, scores, classes = predict_and_save(model, val, OUTPUT / f"outer_fold_{fold}_seed_{seed}.joblib", "outer_validation", seed, fold)
            predictions.append(result); metrics.append(scores); per_class.extend(classes); histories.extend(trace)
            fits.append({"fold": fold, "seed": seed, "inner_train_n": len(inner_train), "inner_val_n": len(inner_val),
                         "inner_boundary_purged": split_at - len(inner_train), "outer_train_n": len(train),
                         "outer_val_n": len(val), "inner_best_epoch": epochs, "inner_epochs_run": len(trace)})
            print(f"Fold {fold}, seed {seed}: inner epoch {epochs}; outer MCC {scores['mcc']:.4f}", flush=True)
        write_json(OUTPUT / "status.json", {"phase": "outer_validation", "completed_fold": fold, "updated_at_wib": now()})
    fits, history = pd.DataFrame(fits), pd.DataFrame(histories)
    outer_predictions, outer_metrics = pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)
    folds = outer_metrics.groupby("fold")[METRICS].mean().reset_index()
    outer_summary = folds[METRICS].agg(["mean", "std"]).rename_axis("statistic")
    for filename, frame in {"fits.csv": fits, "inner_history.csv": history, "inner_metrics.csv": pd.DataFrame(inner_metrics),
                            "split_membership.csv": pd.concat(memberships, ignore_index=True),
                            "outer_predictions.csv": outer_predictions, "outer_seed_metrics.csv": outer_metrics,
                            "outer_fold_metrics.csv": folds, "outer_summary.csv": outer_summary.reset_index()}.items():
        frame.to_csv(OUTPUT / filename, index=False)
    final_epochs = {str(seed): int(value) for seed, value in fits.groupby("seed").inner_best_epoch.median().items()}
    write_json(OUTPUT / "frozen_final_config.json", {"frozen_at_wib": now(), "fit_rows": len(development),
               "epochs": final_epochs, "test_used_for_selection": False, "protocol_sha256": sha256(OUTPUT / "protocol.json"),
               "inner_fits_sha256": sha256(OUTPUT / "fits.csv"), "threshold": .5})
    final_models = {}
    for seed in SEEDS:
        final_models[seed] = fit_fixed(development, seed, final_epochs[str(seed)])
        joblib.dump(final_models[seed], OUTPUT / f"final_seed_{seed}.joblib")
    write_json(OUTPUT / "test_access.json", {"accessed_at_wib": now(), "settings_frozen_sha256": sha256(OUTPUT / "frozen_final_config.json"),
                                            "purpose": "One exploratory comparison; no configuration or threshold selection"})
    test = encoder.transform(read_data(["test.csv"]))
    input_hashes[str(Path(encoder.cache_path_).relative_to(ROOT))] = sha256(encoder.cache_path_)
    assert len(development) == 930 and len(test) == 235 and development.target_date.max() < test.Date.min()
    test_frames, test_scores = [], []
    for seed, model in final_models.items():
        result, scores, classes = predict_and_save(model, test, OUTPUT / f"final_seed_{seed}.joblib", "test", seed)
        test_frames.append(result); test_scores.append(scores); per_class.extend(classes)
    test_predictions, test_metrics = pd.concat(test_frames, ignore_index=True), pd.DataFrame(test_scores)
    test_summary = test_metrics[METRICS].agg(["mean", "std"]).rename_axis("statistic")
    for filename, frame in {"test_predictions.csv": test_predictions, "test_seed_metrics.csv": test_metrics,
                            "test_summary.csv": test_summary.reset_index(), "per_class.csv": pd.DataFrame(per_class)}.items():
        frame.to_csv(OUTPUT / filename, index=False)
    original_outer = pd.read_csv(BASELINE / "predictions.csv", parse_dates=["Date", "target_date"])
    original_test = pd.read_csv(BASELINE_TEST / "predictions.csv", parse_dates=["Date", "target_date"])
    paired_draws, intervals = {}, []
    for phase, actual, original in [("outer_validation", outer_predictions, original_outer), ("test", test_predictions, original_test)]:
        draws, previous_draws = [], []
        fold_ids = [1, 2, 3] if phase == "outer_validation" else [None]
        for fold in fold_ids:
            new = actual.loc[actual.fold.eq(fold)] if fold else actual
            old = original.loc[original.fold.eq(fold)] if fold else original
            new, old = aligned_baseline(new, old, ["seed", "Date"])
            indices = circular_indices(new.Date.nunique(), seed=20261110 + (fold or 0))
            draws.append(np.mean([bootstrap_metrics(group.y, group.p_up, indices) for _, group in new.groupby("seed")], axis=0))
            previous_draws.append(np.mean([bootstrap_metrics(group.y, group.p_up, indices) for _, group in old.groupby("seed")], axis=0))
        current, previous = np.mean(draws, axis=0), np.mean(previous_draws, axis=0)
        paired_draws[f"{phase}_new"], paired_draws[f"{phase}_original"] = current, previous
        for label, values in [("inner_selection", current), ("original_selection", previous), ("new_minus_original", current - previous)]:
            low, high = np.nanquantile(values, [.025, .975], axis=0)
            intervals.extend({"phase": phase, "comparison": label, "metric": metric,
                              "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
    np.savez_compressed(OUTPUT / "bootstrap_draws.npz", metrics=np.asarray(METRICS), **paired_draws)
    pd.DataFrame(intervals).to_csv(OUTPUT / "intervals.csv", index=False)
    original_folds = pd.read_csv(BASELINE / "fold_metrics.csv")
    make_figures(history, fits, outer_predictions, test_predictions, original_folds, folds)
    summaries = []
    lines = ["# R2 inner-validation sanity check", "",
             "Epochs are selected on a purged chronological 80/20 split inside each outer training fold. Fresh outer models use the selected duration and are evaluated once on the original outer validation dates.", "",
             "Word2Vec, scaling, integer seeds, optimizer, maximum epochs, patience, and threshold follow the original R2 pipeline. The representation was selected in the earlier study; this diagnostic does not create a new untouched test period.", "",
             "| Phase | Method | " + " | ".join(METRICS) + " |", "| --- | --- | " + " | ".join(["---:"] * len(METRICS)) + " |"]
    for phase, summary, baseline in [("validation", outer_summary, BASELINE), ("test", test_summary, BASELINE_TEST)]:
        original = pd.read_csv(baseline / "summary.csv").set_index("statistic")
        for label, frame in [("original_selection", original), ("inner_selection", summary)]:
            summaries.append(frame.reset_index().assign(phase=phase, method=label))
            lines.append("| " + " | ".join([phase, label, *[f"{frame.loc['mean', m]:.4f} ± {frame.loc['std', m]:.4f}" for m in METRICS]]) + " |")
    lines += ["", "Validation SD is across folds; test SD is across training seeds. Bootstrap intervals are conditional on fitted models and use paired circular blocks of 10 decisions; validation resampling stays within each fold.", "",
              f"Final epochs by seed: `{json.dumps(final_epochs, sort_keys=True)}`. Original notebooks, models, and ablation tables are unchanged."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    pd.concat(summaries, ignore_index=True).to_csv(OUTPUT / "comparison.csv", index=False)
    for relative, expected in input_hashes.items():
        assert sha256(ROOT / relative) == expected, relative
    write_json(OUTPUT / "status.json", {"phase": "complete", "updated_at_wib": now()})
    artifacts = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    write_json(OUTPUT / "run_metadata.json", {"experiment": NAME, "completed_at_wib": now(), "input_sha256": input_hashes,
               "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in artifacts},
               "models": {"inner_checkpoints": 9, "outer_refits": 9, "final_refits": 3},
               "features": FEATURES, "final_epochs": final_epochs, "outer_rows_per_seed": 475, "test_rows_per_seed": 235,
               "outer_used_for_epoch_selection": False, "test_used_for_selection": False,
               "saved_model_predictions_verified": True, "interpretation": "Exploratory; fixed R2 representation was selected in the earlier study",
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}})
    print("Outer validation:", outer_summary.loc["mean"].to_dict(), flush=True)
    print("Test:", test_summary.loc["mean"].to_dict(), flush=True)
    print("Results:", OUTPUT.relative_to(ROOT), flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
