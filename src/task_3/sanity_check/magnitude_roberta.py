from datetime import datetime
from pathlib import Path
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
sys.path.insert(0, str(ROOT))
from src.task_3.preprocessing.splitting import load_split
from src.task_3.sanity_check.r2_logloss_selection import score
from src.task_3.training import roberta
from src.task_3.training.fusion import BlockScaler, MarketAlignedMLP
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

NAME = "magnitude_roberta_v1"
OUTPUT = ROOT / "src/task_3/output/model/sanity_check" / NAME
FIGURES = ROOT / "src/task_3/output/figure/sanity_check" / NAME
REFERENCE = ROOT / "src/task_3/output/model/sanity_check/movement_magnitude_v1"
R4 = ROOT / "src/task_3/output/model/level_2/04_roberta_base/run_metadata.json"
ENCODERS = ROOT / "src/task_3/output/model/encoders/roberta"
SEEDS = [42, 123, 2026]
METHODS = ["training_prior", "last_magnitude", "finbert_concat", "roberta_concat"]

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def article_windows():
    rows = pd.concat([load_split(ROOT / "data/final", part) for part in ["train", "val", "test"]])
    windows = dict(zip(rows.Date, rows.articles))
    for line in (REFERENCE / "restored_flat_windows.jsonl").read_text().splitlines():
        row = json.loads(line)
        windows[pd.Timestamp(row["Date"])] = row["articles"]
    return windows

def encode(phase, windows, hashes, reference_metadata):
    source = REFERENCE / f"{phase}_features.joblib"
    assert sha256(source) == reference_metadata["artifact_sha256"][str(source.relative_to(ROOT))]
    old = joblib.load(source)
    columns = ["Date", "target_date", "calendar_index", "USD_IDR", "n_berita", "has_news", "fold_roles", "split",
               "abs_return_next", "abs_return_t", "target_gap_days", "was_flat", "restored_flat", *MARKET]
    frame = old[columns].copy()
    lookup = {a["article_id"]: a["text_transformer"] for date in frame.Date for a in windows[date]}
    all_texts = {a["article_id"]: a["text_transformer"] for items in windows.values() for a in items}
    available = {}
    for path in sorted(ENCODERS.glob("articles_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            ids = data["article_ids"].tolist()
            if not any(key in lookup for key in ids):
                continue
            spec, info = json.loads(str(data["spec"].item())), json.loads(str(data["encoder_info"].item()))
            assert spec == roberta.embedding_spec(ids, [all_texts[key] for key in ids])
            assert info["checkpoint_sha256"] == roberta.CHECKPOINT_SHA256
            vectors = data["vectors"]
            assert roberta.array_sha256(vectors) == info["vectors_sha256"]
            available.update((key, vectors[i].copy()) for i, key in enumerate(ids) if key in lookup)
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    missing = sorted(set(lookup) - set(available))
    if missing:
        print(f"{phase}: encode {len(missing)} additional RoBERTa articles", flush=True)
        directory = OUTPUT / "encoder"
        directory.mkdir(exist_ok=True)
        link = directory / "huggingface"
        if not link.exists():
            link.symlink_to(ENCODERS / "huggingface", target_is_directory=True)
        roberta.THREADS, roberta.BATCH_SIZE = 1, 4
        vectors, path, spec, info = roberta.load_or_encode(missing, [lookup[key] for key in missing], directory)
        available.update(zip(missing, vectors))
    ids = sorted(lookup)
    matrix = np.stack([available[key] for key in ids])
    spec = roberta.embedding_spec(ids, [lookup[key] for key in ids])
    np.savez_compressed(OUTPUT / f"{phase}_article_vectors.npz", article_ids=np.asarray(ids), vectors=matrix, spec=json.dumps(spec))
    values = np.zeros((len(frame), len(TEXT)))
    for i, row in enumerate(frame.itertuples()):
        items = windows[row.Date]
        assert len(items) == row.n_berita and all(pd.Timestamp(a["source_date"]) < row.Date for a in items)
        if items:
            values[i] = np.stack([available[a["article_id"]] for a in items]).mean(axis=0)
    frame = pd.concat([frame, pd.DataFrame(values, columns=TEXT, index=frame.index)], axis=1)
    frame["log1p_n_berita"] = np.log1p(frame.n_berita)
    assert np.isfinite(frame[FEATURES]).all().all()
    assert not frame.loc[frame.has_news.eq(0), TEXT].to_numpy().any()
    joblib.dump(frame, OUTPUT / f"{phase}_features.joblib", compress=3)
    write_json(f"{phase}_encoder.json", {"spec": spec, "articles": len(ids), "reused": len(ids) - len(missing), "newly_encoded": len(missing),
               "checkpoint_sha256": roberta.CHECKPOINT_SHA256, "frozen": True, "fine_tuned": False, "new_encoding_threads": 1, "new_encoding_batch_size": 4})
    return frame

def label(frame, threshold):
    result = frame.copy()
    result["y"] = result.abs_return_next.gt(threshold).astype(int)
    return result

def fit_model(train, val, seed, fold=None, epochs=None):
    scaler = BlockScaler(MARKET, TEXT, AVAILABILITY).fit(train)
    x_train = scaler.transform(train[FEATURES])
    x_val = scaler.transform(val[FEATURES]) if val is not None else None
    classifier = MarketAlignedMLP(**CONFIG, random_state=seed)
    best_model, best_key, best_epoch, stale, history = None, None, 0, 0, []
    for epoch in range(1, (MAX_EPOCHS if epochs is None else epochs) + 1):
        classifier.partial_fit(x_train, train.y.to_numpy(), classes=np.array([0, 1]))
        row = {"phase": "validation" if val is not None else "refit", "fold": fold, "seed": seed, "epoch": epoch,
               "train_log_loss": score(train.y, classifier.predict_proba(x_train)[:, 1])["log_loss"]}
        if val is not None:
            metrics = score(val.y, classifier.predict_proba(x_val)[:, 1])
            row.update({f"val_{key}": value for key, value in metrics.items()})
            key = metrics["mcc"], metrics["macro_f1"], -metrics["log_loss"]
            if best_key is None or key > best_key:
                best_model, best_key, best_epoch, stale = copy.deepcopy(classifier), key, epoch, 0
            else:
                stale += 1
        history.append(row)
        if val is not None and stale >= PATIENCE:
            break
    if val is None:
        best_model, best_epoch = classifier, epoch
    assert best_model.t_ == len(train) * best_epoch
    return Pipeline([("scaler", scaler), ("mlp", best_model)]), best_epoch, epoch, history

def evaluate(model, frame, phase, seed, threshold, fold=None):
    path = OUTPUT / f"{phase}_fold_{fold}_seed_{seed}.joblib"
    joblib.dump(model, path)
    p = model.predict_proba(frame[FEATURES])[:, 1]
    np.testing.assert_array_equal(joblib.load(path).predict_proba(frame[FEATURES])[:, 1], p)
    result = frame[["Date", "target_date", "y", "abs_return_next", "abs_return_t", "target_gap_days", "was_flat", "n_berita"]].copy()
    result["method"], result["phase"], result["seed"], result["fold"] = "roberta_concat", phase, seed, fold
    result["label_threshold"], result["p_large"], result["y_pred"] = threshold, p, (p >= .5).astype(int)
    metrics = {"method": "roberta_concat", "phase": phase, "seed": seed, "fold": fold, "n": len(frame), **score(frame.y, p)}
    precision, recall, f1, support = precision_recall_fscore_support(frame.y, result.y_pred, labels=[0, 1], zero_division=0)
    classes = [{"phase": phase, "seed": seed, "fold": fold, "class": name, "support": int(support[i]),
                "precision": precision[i], "recall": recall[i], "f1": f1[i]} for i, name in enumerate(["SMALL", "LARGE"])]
    return result, metrics, classes

def bootstrap(predictions):
    intervals = []
    for phase in ["validation", "test"]:
        fold_draws = {method: [] for method in METHODS}
        for fold in ([1, 2, 3] if phase == "validation" else [None]):
            group = predictions.loc[predictions.phase.eq(phase)]
            if fold is not None:
                group = group.loc[group.fold.eq(fold)]
            indices = circular_indices(group.Date.nunique(), seed=20261110 + (fold or 0))
            for method in METHODS:
                selected = group.loc[group.method.eq(method)]
                seed_draws = []
                for _, stream in selected.groupby("seed"):
                    stream = stream.sort_values("Date")
                    seed_draws.append(bootstrap_metrics(stream.y, stream.p_large, indices))
                fold_draws[method].append(np.mean(seed_draws, axis=0))
        draws = {method: np.mean(values, axis=0) for method, values in fold_draws.items()}
        draws.update({f"roberta_minus_{method}": draws["roberta_concat"] - draws[method] for method in METHODS[:-1]})
        for comparison, values in draws.items():
            low, high = np.nanquantile(values, [.025, .975], axis=0)
            intervals.extend({"phase": phase, "comparison": comparison, "metric": metric, "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
        np.savez_compressed(OUTPUT / f"{phase}_bootstrap.npz", metrics=np.asarray(METRICS), **draws)
    return pd.DataFrame(intervals)

def figures(predictions):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    test = predictions.loc[predictions.phase.eq("test")]
    for ax, method, title in zip(axes, ["finbert_concat", "roberta_concat"], ["Price + FinBERT", "Price + RoBERTa"]):
        matrix = np.mean([confusion_matrix(g.y, g.y_pred, labels=[0, 1]) for _, g in test.loc[test.method.eq(method)].groupby("seed")], axis=0)
        shares = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.1f}\n({shares[i, j]:.0%})", ha="center", va="center", color="white" if shares[i, j] > .6 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Small", "Large"], yticklabels=["Small", "Large"], xlabel="Predicted", ylabel="Actual", title=title)
        ax.tick_params(length=0)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"test_confusion_matrices.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    global MARKET, TEXT, AVAILABILITY, FEATURES, CONFIG, MAX_EPOCHS, PATIENCE
    if OUTPUT.exists():
        raise FileExistsError("Use a new version instead of overwriting an existing trial.")
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    r4 = json.loads(R4.read_text())
    reference_protocol = json.loads((REFERENCE / "protocol.json").read_text())
    reference_metadata = json.loads((REFERENCE / "run_metadata.json").read_text())
    assert json.loads((REFERENCE / "verification.json").read_text())["verified"]
    MARKET, TEXT, AVAILABILITY = [r4["feature_blocks"][key] for key in ["market", "text", "availability"]]
    FEATURES = MARKET + TEXT + AVAILABILITY
    assert len(TEXT) == 768 and r4["model_config"] == reference_protocol["model_config"]
    CONFIG = {**r4["model_config"], "hidden_layer_sizes": tuple(r4["model_config"]["hidden_layer_sizes"])}
    MAX_EPOCHS, PATIENCE = r4["max_epochs"], r4["patience"]
    assert MAX_EPOCHS == reference_protocol["max_epochs"] and PATIENCE == reference_protocol["patience"]
    sources = [p for p in (ROOT / "data/final").glob("*") if p.is_file()]
    sources += [REFERENCE / name for name in ["protocol.json", "run_metadata.json", "verification.json", "predictions.csv", "seed_metrics.csv", "thresholds.csv",
                                           "frozen_final_config.json", "development_features.joblib", "test_features.joblib", "restored_flat_windows.jsonl"]]
    sources += [Path(__file__), R4, ROOT / "src/task_3/sanity_check/r2_logloss_selection.py", ROOT / "data/processed/usd_idr_jisdor_clean.csv"]
    sources += list((ROOT / "src/task_3/training").glob("*.py"))
    sources += [p for level in ["level_1", "level_2", "level_3"] for p in (ROOT / "src/task_3/training" / level).glob("*") if p.is_file()]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    write_json("protocol.json", {"experiment": NAME, "frozen_at_wib": now(), "target": reference_protocol["target"],
               "only_experimental_change": "Frozen FinBERT article embeddings replaced by frozen RoBERTa-base article embeddings",
               "encoder": roberta.ENCODER_ID, "revision": roberta.REVISION, "fine_tuned": False,
               "seeds": SEEDS, "model_config": CONFIG, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
               "selection": reference_protocol["checkpoint_selection"], "threshold_quantile": .5, "probability_threshold": .5,
               "label_threshold_fit": "Fold training only; final train+val only", "split": "Exact same decisions, fold roles and zero-return policy as movement_magnitude_v1",
               "final_epochs": "Per-seed median across development folds", "features": FEATURES,
               "scaling": "Original BlockScaler; uncentered text on news-present training windows",
               "comparison": "Reuse verified FinBERT and naive predictions on identical dates; fit all RoBERTa classifiers from scratch",
               "bootstrap": {"repeats": 1000, "block_length": 10, "paired": True, "scope": "Conditional on fitted models; within-fold resampling for validation"},
               "input_sha256": hashes, "test_used_for_selection": False,
               "interpretation": "One exploratory representation trial on a previously accessed test period; validation still selects and scores checkpoints"})
    windows = article_windows()
    development = encode("development", windows, hashes, reference_metadata)
    predictions, metrics, classes, fits, histories, thresholds = [], [], [], [], [], []
    for fold in [1, 2, 3]:
        roles = development.fold_roles.map(lambda value: value[str(fold)])
        raw_train, raw_val = development.loc[roles.eq("train")], development.loc[roles.eq("val")]
        assert raw_train.target_date.max() < raw_val.Date.min()
        threshold = float(raw_train.abs_return_next.median())
        train, val = label(raw_train, threshold), label(raw_val, threshold)
        thresholds.append({"phase": "validation", "fold": fold, "threshold": threshold, "train_n": len(train), "eval_n": len(val),
                           "train_large_share": train.y.mean(), "eval_large_share": val.y.mean()})
        for seed in SEEDS:
            model, best, ran, trace = fit_model(train, val, seed, fold)
            p, m, c = evaluate(model, val, "validation", seed, threshold, fold)
            predictions.append(p)
            metrics.append(m)
            classes.extend(c)
            histories.extend(trace)
            fits.append({"phase": "validation", "fold": fold, "seed": seed, "train_n": len(train), "val_n": len(val), "threshold": threshold, "best_epoch": best, "epochs_run": ran})
            print(f"Fold {fold}, seed {seed}: epoch {best}, MCC {m['mcc']:.4f}, AUC {m['roc_auc']:.4f}", flush=True)
        write_json("status.json", {"phase": "development", "completed_fold": fold, "updated_at_wib": now()})
    pd.DataFrame(fits).to_csv(OUTPUT / "development_fits.csv", index=False)
    epochs = {str(seed): int(value) for seed, value in pd.DataFrame(fits).groupby("seed").best_epoch.median().items()}
    threshold = float(development.abs_return_next.median())
    write_json("frozen_final_config.json", {"frozen_at_wib": now(), "train_n": len(development), "epochs": epochs, "label_threshold": threshold,
               "probability_threshold": .5, "test_used_for_selection": False, "protocol_sha256": sha256(OUTPUT / "protocol.json"),
               "development_fits_sha256": sha256(OUTPUT / "development_fits.csv")})
    train = label(development, threshold)
    final_models = {}
    for seed in SEEDS:
        model, best, ran, trace = fit_model(train, None, seed, epochs=epochs[str(seed)])
        final_models[seed] = model
        histories.extend(trace)
        fits.append({"phase": "test", "fold": None, "seed": seed, "train_n": len(train), "threshold": threshold, "best_epoch": best, "epochs_run": ran})
    write_json("test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(OUTPUT / "frozen_final_config.json"), "purpose": "One prespecified exploratory representation comparison"})
    test = label(encode("test", windows, hashes, reference_metadata), threshold)
    assert len(development) == 938 and len(test) == 240 and development.target_date.max() < test.Date.min()
    thresholds.append({"phase": "test", "fold": None, "threshold": threshold, "train_n": len(train), "eval_n": len(test),
                       "train_large_share": train.y.mean(), "eval_large_share": test.y.mean()})
    for seed, model in final_models.items():
        p, m, c = evaluate(model, test, "test", seed, threshold)
        predictions.append(p)
        metrics.append(m)
        classes.extend(c)
    new_predictions, new_metrics = pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)
    old_predictions = pd.read_csv(REFERENCE / "predictions.csv", parse_dates=["Date", "target_date"])
    for phase in ["validation", "test"]:
        for seed in SEEDS:
            new = new_predictions.loc[new_predictions.phase.eq(phase) & new_predictions.seed.eq(seed)].sort_values("Date").reset_index(drop=True)
            old = old_predictions.loc[old_predictions.phase.eq(phase) & old_predictions.method.eq("finbert_concat") & old_predictions.seed.eq(seed)].sort_values("Date").reset_index(drop=True)
            pd.testing.assert_frame_equal(new[["Date", "target_date", "y"]], old[["Date", "target_date", "y"]])
            np.testing.assert_allclose(new.label_threshold, old.label_threshold, atol=1e-16, rtol=0)
    predictions = pd.concat([old_predictions, new_predictions], ignore_index=True)
    metrics = pd.concat([pd.read_csv(REFERENCE / "seed_metrics.csv"), new_metrics], ignore_index=True)
    summaries = []
    for (phase, method), group in metrics.groupby(["phase", "method"]):
        units = group.groupby("fold")[METRICS].mean() if phase == "validation" else group[METRICS]
        summaries.extend({"phase": phase, "method": method, "metric": metric, "mean": units[metric].mean(), "std": units[metric].std(ddof=1)} for metric in METRICS)
    summary = pd.DataFrame(summaries)
    for name, frame in {"predictions": predictions, "seed_metrics": metrics, "per_class": pd.DataFrame(classes), "fits": pd.DataFrame(fits),
                        "history": pd.DataFrame(histories), "thresholds": pd.DataFrame(thresholds), "summary": summary}.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    intervals = bootstrap(predictions)
    intervals.to_csv(OUTPUT / "intervals.csv", index=False)
    figures(predictions)
    lines = ["# RoBERTa for next-fixing movement magnitude", "", "Replace frozen FinBERT article embeddings with frozen RoBERTa-base embeddings. This is a transformer representation comparison, with no end-to-end encoder fine-tuning. All classifiers are trained again for LARGE/SMALL.", "",
             "The target, training-median thresholds, original-plus-CNBC corpus, restored flat days, fold roles, 938 refit decisions, 240 test decisions, classifier architecture, scaling, seeds, checkpoint selection, and 0.5 prediction threshold match movement_magnitude_v1. Only the text encoder changes; final epoch choices can differ because each representation selects its own checkpoints.", "",
             "| Phase | Method | " + " | ".join(METRICS) + " |", "| --- | --- | " + " | ".join(["---:"] * len(METRICS)) + " |"]
    for phase in ["validation", "test"]:
        for method in METHODS:
            table = summary.loc[summary.phase.eq(phase) & summary.method.eq(method)].set_index("metric")
            lines.append("| " + " | ".join([phase, method, *[f"{table.loc[metric, 'mean']:.4f}" for metric in METRICS]]) + " |")
    lines += ["", f"Final median threshold: **{threshold * 100:.4f}%**. Final RoBERTa epochs: `{json.dumps(epochs, sort_keys=True)}`.", "",
              "| Test comparison | Metric | 95% lower | 95% upper |", "| --- | --- | ---: | ---: |"]
    for comparison in ["roberta_concat", "roberta_minus_finbert_concat", "roberta_minus_training_prior", "roberta_minus_last_magnitude"]:
        for metric in ["mcc", "macro_f1", "roc_auc"]:
            row = intervals.loc[intervals.phase.eq("test") & intervals.comparison.eq(comparison) & intervals.metric.eq(metric)].iloc[0]
            lines.append(f"| {comparison} | {metric} | {row.low_95:.4f} | {row.high_95:.4f} |")
    lines += ["", "Validation averages seed metrics within folds and then folds equally. Test averages individual seed metrics, without ensembling. Intervals use 1,000 paired circular block resamples of 10 decisions, conditional on fitted models; validation resampling remains inside each fold.", "",
              "Naive and FinBERT predictions are reused from the verified prior magnitude run on identical dates and labels. The previous-movement baseline has hard 0/1 probabilities and is not a calibrated-probability log-loss benchmark. This previously accessed test period remains exploratory; validation still selects and scores checkpoints. Existing news publication-time and historical-body limitations remain.", "",
              "Original data/final files, main ablation notebooks/results, representation selection, and prior sanity-check results remain unchanged."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    for relative, expected in hashes.items():
        assert sha256(ROOT / relative) == expected, relative
    write_json("status.json", {"phase": "complete", "updated_at_wib": now()})
    artifacts = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    write_json("run_metadata.json", {"completed_at_wib": now(), "input_sha256": hashes,
               "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts}, "features": FEATURES,
               "final_epochs": epochs, "final_threshold": threshold, "development_rows": len(development), "test_rows": len(test),
               "validation_decisions": int(new_predictions.loc[new_predictions.phase.eq("validation"), "Date"].nunique()),
               "new_models": 12, "encoder_fine_tuned": False, "test_used_for_selection": False, "saved_model_predictions_verified": True,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}})
    print(summary.pivot(index=["phase", "method"], columns="metric", values="mean").to_string(), flush=True)
    print("Results:", OUTPUT.relative_to(ROOT), flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
