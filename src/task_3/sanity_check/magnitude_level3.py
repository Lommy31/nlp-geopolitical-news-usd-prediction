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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
sys.path.insert(0, str(ROOT))
from src.task_3.sanity_check.magnitude_roberta import article_windows
from src.task_3.sanity_check.roberta_fusion import TEXT_FEATURES, RobertaFusion, fit_fusion
from src.task_3.training.controlled_fusion import MODEL_CONFIG, score_binary
from src.task_3.training.roberta import embedding_spec
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

NAME = "magnitude_level3_roberta_v1"
OUTPUT = ROOT / "src/task_3/output/model/sanity_check" / NAME
FIGURES = ROOT / "src/task_3/output/figure/sanity_check" / NAME
REFERENCE = ROOT / "src/task_3/output/model/sanity_check/magnitude_roberta_v1"
CONTRACT = ROOT / "src/task_3/output/model/level_3/level_3_contract.json"
SEEDS = [42, 123, 2026]
VARIANTS = {"A000": (0, 0, 0), "A111": (1, 1, 1)}
OLD_METHODS = ["training_prior", "last_magnitude", "finbert_concat", "roberta_concat"]
METHODS = OLD_METHODS + list(VARIANTS)

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def load_data(phase, windows):
    frame = joblib.load(REFERENCE / f"{phase}_features.joblib")
    with np.load(REFERENCE / f"{phase}_article_vectors.npz", allow_pickle=False) as cached:
        ids = cached["article_ids"].tolist()
        matrix = cached["vectors"].astype(np.float64)
        spec = json.loads(str(cached["spec"].item()))
    texts = {a["article_id"]: a["text_transformer"] for date in frame.Date for a in windows[date]}
    assert ids == sorted(texts) and spec == embedding_spec(ids, [texts[key] for key in ids])
    vectors = dict(zip(ids, matrix))
    frame["articles"] = [windows[date] for date in frame.Date]
    means = np.zeros((len(frame), 768))
    for i, row in enumerate(frame.itertuples()):
        assert len(row.articles) == row.n_berita
        if row.articles:
            means[i] = np.stack([vectors[a["article_id"]] for a in row.articles]).mean(0)
    np.testing.assert_allclose(means, frame[TEXT_FEATURES].to_numpy(), atol=5e-7, rtol=2e-6)
    frame[TEXT_FEATURES] = means
    joblib.dump(frame, OUTPUT / f"{phase}_features.joblib", compress=3)
    return frame, vectors

def label(frame, threshold):
    result = frame.copy()
    result["y"] = result.abs_return_next.gt(threshold).astype(int)
    return result

def evaluate(predictor, frame, vectors, method, phase, seed, threshold, fold=None):
    path = OUTPUT / f"{phase}_{method}_fold_{fold}_seed_{seed}.joblib"
    joblib.dump(predictor, path)
    p = predictor.predict_proba(frame, vectors)[:, 1]
    np.testing.assert_array_equal(joblib.load(path).predict_proba(frame, vectors)[:, 1], p)
    result = frame[["Date", "target_date", "y", "abs_return_next", "abs_return_t", "target_gap_days", "was_flat", "n_berita"]].copy()
    result["method"], result["phase"], result["seed"], result["fold"] = method, phase, seed, fold
    result["label_threshold"], result["p_large"], result["y_pred"] = threshold, p, (p >= .5).astype(int)
    metrics = {"method": method, "phase": phase, "seed": seed, "fold": fold, "n": len(frame), **score_binary(frame.y, p)}
    precision, recall, f1, support = precision_recall_fscore_support(frame.y, result.y_pred, labels=[0, 1], zero_division=0)
    classes = [{"method": method, "phase": phase, "seed": seed, "fold": fold, "class": name, "support": int(support[i]),
                "precision": precision[i], "recall": recall[i], "f1": f1[i]} for i, name in enumerate(["SMALL", "LARGE"])]
    return result, metrics, classes

def resample(predictions):
    intervals = []
    for phase in ["validation", "test"]:
        folds = {method: [] for method in METHODS}
        for fold in ([1, 2, 3] if phase == "validation" else [None]):
            frame = predictions.loc[predictions.phase.eq(phase)]
            if fold is not None:
                frame = frame.loc[frame.fold.eq(fold)]
            indices = circular_indices(frame.Date.nunique(), seed=20261110 + (fold or 0))
            for method in METHODS:
                values = []
                for _, group in frame.loc[frame.method.eq(method)].groupby("seed"):
                    group = group.sort_values("Date")
                    values.append(bootstrap_metrics(group.y, group.p_large, indices))
                folds[method].append(np.mean(values, axis=0))
        draws = {method: np.mean(values, axis=0) for method, values in folds.items()}
        draws.update({f"A111_minus_{method}": draws["A111"] - draws[method] for method in ["A000", "roberta_concat", "training_prior", "last_magnitude"]})
        for comparison, values in draws.items():
            low, high = np.nanquantile(values, [.025, .975], axis=0)
            intervals.extend({"phase": phase, "comparison": comparison, "metric": metric, "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
        np.savez_compressed(OUTPUT / f"{phase}_bootstrap.npz", metrics=np.asarray(METRICS), **draws)
    return pd.DataFrame(intervals)

def figures(predictions):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    test = predictions.loc[predictions.phase.eq("test")]
    for ax, method, title in zip(axes, ["roberta_concat", "A000", "A111"], ["Previous concat", "A000", "A111"]):
        matrix = np.mean([confusion_matrix(group.y, group.y_pred, labels=[0, 1]) for _, group in test.loc[test.method.eq(method)].groupby("seed")], axis=0)
        shares = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.1f}\n({shares[i, j]:.0%})", ha="center", va="center", color="white" if shares[i, j] > .6 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Small", "Large"], yticklabels=["Small", "Large"], xlabel="Predicted", title=title)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"test_confusion_matrices.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    global MARKET, AVAILABILITY
    if OUTPUT.exists():
        raise FileExistsError("Use a new version instead of overwriting an existing trial.")
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    reference_metadata = json.loads((REFERENCE / "run_metadata.json").read_text())
    assert json.loads((REFERENCE / "verification.json").read_text())["verified"]
    for name in ["development_features.joblib", "test_features.joblib", "development_article_vectors.npz", "test_article_vectors.npz", "predictions.csv", "seed_metrics.csv"]:
        assert sha256(REFERENCE / name) == reference_metadata["artifact_sha256"][str((REFERENCE / name).relative_to(ROOT))]
    contract = json.loads(CONTRACT.read_text())
    assert contract["model_config"] == MODEL_CONFIG
    assert sha256(ROOT / "src/task_3/training/controlled_fusion.py") == contract["implementation_sha256"]
    r4 = json.loads((ROOT / "src/task_3/output/model/level_2/04_roberta_base/run_metadata.json").read_text())
    MARKET, AVAILABILITY = r4["feature_blocks"]["market"], r4["feature_blocks"]["availability"]
    sources = [Path(__file__), CONTRACT, ROOT / "src/task_3/sanity_check/roberta_fusion.py", ROOT / "src/task_3/sanity_check/magnitude_roberta.py"]
    sources += [REFERENCE / name for name in ["run_metadata.json", "verification.json", "protocol.json", "development_features.joblib", "test_features.joblib",
                                           "development_article_vectors.npz", "test_article_vectors.npz", "predictions.csv", "seed_metrics.csv"]]
    sources += [ROOT / "src/task_3/output/model/sanity_check/movement_magnitude_v1/restored_flat_windows.jsonl"]
    sources += [p for p in (ROOT / "data/final").glob("*") if p.is_file()]
    sources += [ROOT / "data/processed/usd_idr_jisdor_clean.csv", ROOT / "src/task_3/output/model/level_2/04_roberta_base/run_metadata.json"]
    sources += list((ROOT / "src/task_3/training").glob("*.py"))
    sources += [p for level in ["level_1", "level_2", "level_3"] for p in (ROOT / "src/task_3/training" / level).glob("*") if p.is_file()]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    write_json("protocol.json", {"experiment": NAME, "frozen_at_wib": now(), "target": "abs((J_next - J_t) / J_t) > training median",
               "variants": VARIANTS, "primary_comparison": "A111 - A000", "representation": "Frozen RoBERTa-base article embeddings, dimension 768",
               "architecture": "Original Level 3 mean/task/cross slots; task query 768, price-to-query 14-to-768; shared scalar gate from 14 prices plus log news count",
               "query_semantics": "Learned query for the USD/IDR magnitude task; no externally anchored USD/IDR token or guaranteed semantic relevance",
               "difference_from_main_level3": "RoBERTa 768 replaces Word2Vec 300; separate experiment, not a change to the main contract",
               "mean_pooling": "Article vectors promoted to float64 before pooling/scaling; previous sklearn concat window means were pooled in float32",
               "model_config": MODEL_CONFIG, "seeds": SEEDS, "label_threshold_fit": "Fold training only; final train+val only", "threshold_quantile": .5,
               "probability_threshold": .5, "selection": contract["checkpoint_selection"], "final_epochs": contract["final_epochs"],
               "scaling": "Train-only, centered market, uncentered text fitted on news-present window means, uncentered availability",
               "reference_interpretation": "A111 versus A000 isolates the combined components in the same torch backend; comparison with prior sklearn concat also changes backend/shuffle/regularization implementation",
               "bootstrap": {"repeats": 1000, "block_length": 10, "paired": True, "scope": "Fixed fitted models; validation resampling within folds"},
               "input_sha256": hashes, "test_used_for_selection": False, "fine_tuned_encoder": False,
               "limitations": ["Previously accessed test period; exploratory trial", "Validation still selects and scores checkpoints", "Date-only news timestamps and historical-body limitations remain"]})
    windows = article_windows()
    dev, dev_vectors = load_data("development", windows)
    predictions, metrics, classes, fits, histories = [], [], [], [], []
    for fold in [1, 2, 3]:
        roles = dev.fold_roles.map(lambda value: value[str(fold)])
        raw_train, raw_val = dev.loc[roles.eq("train")], dev.loc[roles.eq("val")]
        assert raw_train.target_date.max() < raw_val.Date.min()
        threshold = float(raw_train.abs_return_next.median())
        train, val = label(raw_train, threshold), label(raw_val, threshold)
        for method, flags in VARIANTS.items():
            for seed in SEEDS:
                predictor, best, ran, trace, seconds, changes = fit_fusion(train, val, dev_vectors, seed, flags, MARKET, AVAILABILITY)
                p, m, c = evaluate(predictor, val, dev_vectors, method, "validation", seed, threshold, fold)
                predictions.append(p)
                metrics.append(m)
                classes.extend(c)
                histories.extend({"phase": "validation", "method": method, "fold": fold, "seed": seed, **row} for row in trace)
                fits.append({"phase": "validation", "method": method, "fold": fold, "seed": seed, "train_n": len(train), "val_n": len(val), "threshold": threshold,
                             "best_epoch": best, "epochs_run": ran, "fit_seconds": seconds, "parameters": sum(v.numel() for v in predictor.model.parameters()), **changes})
                print(f"Fold {fold}, {method}, seed {seed}: epoch {best}, MCC {m['mcc']:.4f}, AUC {m['roc_auc']:.4f}, {seconds:.1f}s", flush=True)
                pd.DataFrame(fits).to_csv(OUTPUT / "development_fits.csv", index=False)
                write_json("status.json", {"phase": "development", "completed_fits": len(fits), "total_fits": 18, "updated_at_wib": now()})
    epochs = {method: {str(seed): int(value) for seed, value in pd.DataFrame(fits).loc[lambda d: d.method.eq(method)].groupby("seed").best_epoch.median().items()} for method in VARIANTS}
    threshold = float(dev.abs_return_next.median())
    write_json("frozen_final_config.json", {"frozen_at_wib": now(), "train_n": len(dev), "epochs": epochs, "label_threshold": threshold,
               "probability_threshold": .5, "test_used_for_selection": False, "protocol_sha256": sha256(OUTPUT / "protocol.json"),
               "development_fits_sha256": sha256(OUTPUT / "development_fits.csv")})
    final_train = label(dev, threshold)
    final_models = {}
    for method, flags in VARIANTS.items():
        for seed in SEEDS:
            predictor, best, ran, trace, seconds, changes = fit_fusion(final_train, None, dev_vectors, seed, flags, MARKET, AVAILABILITY, fixed_epochs=epochs[method][str(seed)])
            final_models[method, seed] = predictor
            histories.extend({"phase": "refit", "method": method, "fold": None, "seed": seed, **row} for row in trace)
            fits.append({"phase": "test", "method": method, "fold": None, "seed": seed, "train_n": len(final_train), "threshold": threshold,
                         "best_epoch": best, "epochs_run": ran, "fit_seconds": seconds, "parameters": sum(v.numel() for v in predictor.model.parameters()), **changes})
            print(f"Refit {method}, seed {seed}: {best} epochs, {seconds:.1f}s", flush=True)
            write_json("status.json", {"phase": "refit", "completed_refits": len(final_models), "total_refits": 6, "updated_at_wib": now()})
    write_json("test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(OUTPUT / "frozen_final_config.json"), "purpose": "Prespecified exploratory component comparison"})
    test, test_vectors = load_data("test", windows)
    test = label(test, threshold)
    assert len(dev) == 938 and len(test) == 240 and dev.target_date.max() < test.Date.min()
    for (method, seed), predictor in final_models.items():
        p, m, c = evaluate(predictor, test, test_vectors, method, "test", seed, threshold)
        predictions.append(p)
        metrics.append(m)
        classes.extend(c)
    old_predictions = pd.read_csv(REFERENCE / "predictions.csv", parse_dates=["Date", "target_date"])
    new_predictions = pd.concat(predictions, ignore_index=True)
    for (phase, method, seed), frame in new_predictions.groupby(["phase", "method", "seed"]):
        old = old_predictions.loc[old_predictions.phase.eq(phase) & old_predictions.method.eq("roberta_concat") & old_predictions.seed.eq(seed)].sort_values("Date").reset_index(drop=True)
        current = frame.sort_values("Date").reset_index(drop=True)
        pd.testing.assert_frame_equal(current[["Date", "target_date", "y"]], old[["Date", "target_date", "y"]])
        np.testing.assert_allclose(current.label_threshold, old.label_threshold, atol=1e-16, rtol=0)
    predictions = pd.concat([old_predictions, new_predictions], ignore_index=True)
    metrics = pd.concat([pd.read_csv(REFERENCE / "seed_metrics.csv"), pd.DataFrame(metrics)], ignore_index=True)
    rows = []
    for (phase, method), frame in metrics.groupby(["phase", "method"]):
        units = frame.groupby("fold")[METRICS].mean() if phase == "validation" else frame[METRICS]
        rows.extend({"phase": phase, "method": method, "metric": metric, "mean": units[metric].mean(), "std": units[metric].std(ddof=1)} for metric in METRICS)
    summary = pd.DataFrame(rows)
    for name, frame in {"predictions": predictions, "seed_metrics": metrics, "summary": summary, "fits": pd.DataFrame(fits),
                        "history": pd.DataFrame(histories), "per_class": pd.DataFrame(classes)}.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    intervals = resample(predictions)
    intervals.to_csv(OUTPUT / "intervals.csv", index=False)
    figures(predictions)
    lines = ["# RoBERTa Level 3 components for movement magnitude", "", "Compare A111 (learned target query, price-to-news cross-attention, scalar gate) with a matched A000 control. Both use frozen 768-dimensional RoBERTa article vectors and the original Level 3 torch training recipe. This separate experiment does not replace the main Word2Vec Level 3 contract.", "",
             "Targets, training-only median thresholds, zero-return policy, decision dates, news windows, market features and seeds match the earlier magnitude experiments. Article vectors are promoted to float64 before sequence pooling; earlier sklearn concat pooled window means in float32.", "",
             "A111 versus A000 is the component comparison. Previous sklearn RoBERTa concat is contextual: backend, shuffle, numerical pooling and regularization implementation also differ. All classifiers and scalers are fitted again; the encoder stays frozen.", "",
             "| Phase | Method | " + " | ".join(METRICS) + " |", "| --- | --- | " + " | ".join(["---:"] * len(METRICS)) + " |"]
    for phase in ["validation", "test"]:
        for method in METHODS:
            table = summary.loc[summary.phase.eq(phase) & summary.method.eq(method)].set_index("metric")
            lines.append("| " + " | ".join([phase, method, *[f"{table.loc[metric, 'mean']:.4f}" for metric in METRICS]]) + " |")
    lines += ["", f"Refit decisions: **938**. Test decisions: **240**. Magnitude threshold: **{threshold * 100:.4f}%**. Final epochs: `{json.dumps(epochs, sort_keys=True)}`.", "",
              "| Test comparison | Metric | 95% lower | 95% upper |", "| --- | --- | ---: | ---: |"]
    for comparison in ["A111", "A111_minus_A000", "A111_minus_roberta_concat"]:
        for metric in ["mcc", "macro_f1", "roc_auc"]:
            row = intervals.loc[intervals.phase.eq("test") & intervals.comparison.eq(comparison) & intervals.metric.eq(metric)].iloc[0]
            lines.append(f"| {comparison} | {metric} | {row.low_95:.4f} | {row.high_95:.4f} |")
    lines += ["", "Validation averages individual seeds within folds and then folds equally; test averages individual seeds without ensembling. Paired circular block bootstrap uses 1,000 resamples and block length 10 decisions, conditional on fitted models.", "",
              "The learned query is specific to this USD/IDR task through training; it is not an externally anchored semantic USD/IDR token. This A111-versus-A000 trial cannot identify which individual component matters.", "",
              "The test period has already been accessed, and validation still selects and scores checkpoints. Results are exploratory. All prior experiments, fixed data exports and main ablation notebooks/results remain unchanged."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    for relative, expected in hashes.items():
        assert sha256(ROOT / relative) == expected, relative
    write_json("status.json", {"phase": "complete", "updated_at_wib": now()})
    artifacts = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    write_json("run_metadata.json", {"completed_at_wib": now(), "input_sha256": hashes, "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts},
               "model_config": MODEL_CONFIG, "variants": VARIANTS, "encoder_fine_tuned": False, "new_models": 24, "final_epochs": epochs,
               "development_rows": 938, "test_rows": 240, "validation_dates": 477, "test_used_for_selection": False,
               "saved_model_predictions_verified": True, "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}})
    print(summary.pivot(index=["phase", "method"], columns="metric", values="mean").to_string(), flush=True)
    print("Results:", OUTPUT.relative_to(ROOT), flush=True)

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        main()
