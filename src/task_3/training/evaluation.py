from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.task_3.training.controlled_fusion import score_binary
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def scored(frame, probability, model_id, seed, fold):
    predictions = frame[["Date", "target_date", "y", "n_berita", "has_news"]].copy()
    predictions["model_id"], predictions["seed"], predictions["fold"] = model_id, seed, fold
    predictions["p_up"], predictions["y_pred"] = probability, (probability >= .5).astype(int)
    metrics = {"model_id": model_id, "seed": seed, "fold": fold, "n": len(frame), **score_binary(frame.y, probability)}
    precision, recall, f1, support = precision_recall_fscore_support(frame.y, predictions.y_pred, labels=[0, 1], zero_division=0)
    classes = [{"model_id": model_id, "seed": seed, "fold": fold, "class": name, "support": int(support[i]), "precision": precision[i], "recall": recall[i], "f1": f1[i]} for i, name in enumerate(["DOWN", "UP"])]
    return predictions, metrics, classes

def metric_draws(predictions):
    fold_draws = []
    for _, group in predictions.groupby("fold"):
        indices = circular_indices(group.Date.nunique(), repeats=1000, block_length=10, seed=20261110)
        fold_draws.append(np.mean([bootstrap_metrics(rows.sort_values("Date").y, rows.sort_values("Date").p_up, indices) for _, rows in group.groupby("seed")], axis=0))
    return np.mean(fold_draws, axis=0)

def write_phase(directory, predictions, metrics, fits, history, classes, metadata):
    directory = Path(directory)
    summary = metrics.groupby("fold")[METRICS].mean().agg(["mean", "std"]).rename_axis("statistic")
    if metadata["phase"] == "test":
        summary = metrics[METRICS].agg(["mean", "std"]).rename_axis("statistic")
    draws = metric_draws(predictions)
    low, high = np.nanquantile(draws, [.025, .975], axis=0)
    intervals = pd.DataFrame({"metric": METRICS, "mean": summary.loc["mean"].to_numpy(), "low_95": low, "high_95": high})
    for name, frame in {"predictions": predictions, "seed_metrics": metrics, "fits": fits, "history": history, "per_class": classes,
                        "fold_metrics": metrics.groupby("fold")[METRICS].mean().reset_index(), "summary": summary.reset_index(), "intervals": intervals}.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    np.savez_compressed(directory / "bootstrap.npz", metrics=np.asarray(METRICS), draws=draws)
    training_counts = fits.train_n.unique().tolist()
    evaluation_counts = metrics.groupby("seed").n.sum().unique().tolist()
    fit_usage = fits.apply(lambda row: row.evaluation_role == "test" or (metadata["phase"] == "test" and (not row.evaluation_is_none or pd.isna(row.fixed_epochs))), axis=1)
    metadata = {**metadata, "written_at_wib": now(), "training_row_counts": training_counts, "evaluation_row_counts": evaluation_counts,
                "fit_rows": int(training_counts[0]) if len(training_counts) == 1 else None,
                "evaluation_rows": int(predictions.Date.nunique()), "test_rows": int(predictions.Date.nunique()) if metadata["phase"] == "test" else 0,
                "test_used_for_fitting_or_stopping": bool(fit_usage.any()), "threshold": .5, "seeds": sorted(metrics.seed.unique().tolist()),
                "aggregation": "Equal fold mean of individual-seed metrics; final-test mean across individual seeds; no ensemble",
                "uncertainty": "Paired temporal blocks of ten decisions, conditional on fitted models; selection uncertainty omitted"}
    assert not metadata["test_used_for_fitting_or_stopping"]
    artifacts = [p for p in directory.iterdir() if p.is_file() and p.name not in {"run_metadata.json", "verification.json"}]
    metadata["artifact_sha256"] = {p.name: sha256(p) for p in artifacts}
    write_json(directory / "run_metadata.json", metadata)
    return summary, intervals

def plot_results(predictions, fits, history, figure_directory, title):
    figure_directory = Path(figure_directory)
    figure_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    matrix = np.mean([confusion_matrix(rows.y, rows.y_pred, labels=[0, 1]) for _, rows in predictions.groupby("seed")], axis=0)
    shares = matrix / matrix.sum(1, keepdims=True)
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i,j]:.1f}\n({shares[i,j]:.0%})", ha="center", va="center", color="white" if shares[i,j] > .6 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], xlabel="Predicted", ylabel="Actual", title=title)
    ax.tick_params(length=0)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(figure_directory / f"confusion_matrix.{extension}", dpi=300, bbox_inches="tight")
    figures = [fig]
    if not history.empty and "val_log_loss" in history:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        for ax, fold in zip(axes, [1, 2, 3]):
            trace = history.loc[history.fold.eq(fold) & history.seed.eq(42)]
            selected = fits.loc[fits.fold.eq(fold) & fits.seed.eq(42)].iloc[0]
            ax.plot(trace.epoch, trace.train_log_loss, color=".55", linestyle="--", label="Train")
            ax.plot(trace.epoch, trace.val_log_loss, color="#3775BA", label="Validation")
            ax.plot(trace.epoch, trace.smoothed_log_loss, color="black", linewidth=1, label="Five-epoch mean")
            ax.axhline(selected.prior_log_loss, color=".3", linestyle="-.", linewidth=1, label="Train prior")
            ax.axvline(selected.best_epoch, color="black", linestyle=":", linewidth=1)
            ax.set(xlabel="Epoch", title=f"Fold {fold}")
        axes[0].set_ylabel("Log loss")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.05))
        fig.tight_layout(rect=(0, 0, 1, .9))
        for extension in ["png", "pdf"]:
            fig.savefig(figure_directory / f"training_curves.{extension}", dpi=300, bbox_inches="tight")
        figures.append(fig)
    return figures

def paired_comparison(candidate, reference):
    keys = ["fold", "Date"]
    a = candidate.drop_duplicates(keys).sort_values(keys)
    b = reference.drop_duplicates(keys).sort_values(keys)
    pd.testing.assert_frame_equal(a[keys + ["y"]].reset_index(drop=True), b[keys + ["y"]].reset_index(drop=True))
    delta = metric_draws(candidate) - metric_draws(reference)
    low, high = np.nanquantile(delta, [.025, .975], axis=0)
    rows = []
    folds_better = 0
    means = []
    for fold, group in candidate.groupby("fold"):
        ref = reference.loc[reference.fold.eq(fold)]
        left = np.mean([score_binary(g.y, g.p_up)["log_loss"] for _, g in group.groupby("seed")])
        right = np.mean([score_binary(g.y, g.p_up)["log_loss"] for _, g in ref.groupby("seed")])
        means.append(left - right)
        folds_better += left < right
    for i, metric in enumerate(METRICS):
        rows.append({"metric": metric, "low_95": low[i], "high_95": high[i]})
    gate = {"folds_better": int(folds_better), "mean_log_loss_delta": float(np.mean(means)), "low_95": float(low[6]), "high_95": float(high[6]), "passes": bool(folds_better >= 2 and high[6] < 0)}
    return pd.DataFrame(rows), gate
