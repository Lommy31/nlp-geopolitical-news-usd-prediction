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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
sys.path.insert(0, str(ROOT))
from src.task_3.preprocessing import merging, splitting
from src.task_3.sanity_check.r2_logloss_selection import score
from src.task_3.training.finbert import CHECKPOINT_SHA256, embedding_spec, load_or_encode
from src.task_3.training.fusion import BlockScaler, MarketAlignedMLP
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

NAME = "movement_magnitude_v1"
OUTPUT = ROOT / "src/task_3/output/model/sanity_check" / NAME
FIGURES = ROOT / "src/task_3/output/figure/sanity_check" / NAME
REFERENCE = ROOT / "src/task_3/output/model/level_1/04_market_news_concat/run_metadata.json"
ENCODERS = ROOT / "src/task_3/output/model/encoders/finbert"
SEEDS = [42, 123, 2026]
METHODS = ["training_prior", "last_magnitude", "finbert_concat"]

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def prepare_data():
    calendar, articles, provenance = merging.build_dataset(ROOT, progress=lambda line: print(line, flush=True))
    direction_assignments = splitting.make_split_assignments(calendar)
    magnitude_calendar = calendar.copy()
    magnitude_calendar["eligible_binary"] = calendar.features_ready & calendar.next_jisdor.notna()
    magnitude_calendar["y"] = 0
    assignments = splitting.make_split_assignments(magnitude_calendar)
    records = splitting.make_records(assignments, articles)
    frames = []
    exported = []
    restored = []
    for split, rows in records.items():
        frame = pd.DataFrame(rows)
        frame["Date"] = pd.to_datetime(frame.Date)
        frame["target_date"] = pd.to_datetime(frame.target_date)
        frame["split"] = split
        frames.append(frame)
        original = splitting.load_split(ROOT / "data/final", split)
        match = frame.set_index("Date").loc[original.Date].reset_index()
        pd.testing.assert_frame_equal(match[original.columns.drop("y")], original.drop(columns="y"), check_exact=False, atol=1e-12, rtol=1e-12)
        added = frame.loc[~frame.Date.isin(original.Date)]
        assert added.Date.isin(calendar.loc[calendar.target_status.eq("FLAT"), "Date"]).all()
        restored.append({"split": split, "original_n": len(original), "restored_flat_n": len(added), "n": len(frame)})
        exported.extend(original.Date.tolist())
    frame = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    rates = calendar.set_index("Date")
    source = rates.loc[frame.Date]
    frame["abs_return_next"] = np.abs((source.next_jisdor.to_numpy() - source.USD_IDR.to_numpy()) / source.USD_IDR.to_numpy())
    frame["abs_return_t"] = np.abs(np.expm1(frame.log_return_t))
    frame["target_gap_days"] = (frame.target_date - frame.Date).dt.days
    frame["was_flat"] = source.target_status.eq("FLAT").to_numpy()
    frame["restored_flat"] = ~frame.Date.isin(exported)
    assert frame.Date.is_unique and frame.abs_return_next.notna().all()
    pd.DataFrame(restored).to_csv(OUTPUT / "cohort.csv", index=False)
    write_json("data_preparation.json", {"source_provenance": provenance, "cohort": restored,
               "split_calendar_unchanged": True, "original_rows_match": True,
               "original_excluded_boundaries": direction_assignments.loc[direction_assignments.eligible_binary & direction_assignments.split.eq("excluded"), "Date"].dt.strftime("%Y-%m-%d").tolist(),
               "restored_dates": frame.loc[frame.restored_flat, "Date"].dt.strftime("%Y-%m-%d").tolist()})
    flat_windows = frame.loc[frame.restored_flat, ["Date", "articles"]].copy()
    flat_windows["Date"] = flat_windows.Date.dt.strftime("%Y-%m-%d")
    flat_windows.to_json(OUTPUT / "restored_flat_windows.jsonl", orient="records", lines=True)
    return frame

def encode(frame, phase, source_hashes):
    lookup = {a["article_id"]: a["text_transformer"] for row in frame.articles for a in row}
    ids = sorted(lookup)
    texts = [lookup[key] for key in ids]
    expected = dict(zip(ids, texts))
    available = {}
    for path in sorted(ENCODERS.glob("articles_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            old_ids = data["article_ids"].tolist()
            if not any(key in expected for key in old_ids):
                continue
            info, spec = json.loads(str(data["encoder_info"])), json.loads(str(data["spec"]))
            assert info["checkpoint_sha256"] == CHECKPOINT_SHA256
            fixed = embedding_spec([], [])
            assert all(spec[key] == value for key, value in fixed.items() if key != "article_text_sha256")
            # Verify the cached text fingerprint before reusing any vectors.
            original_texts = ORIGINAL_TEXTS
            assert all(key in original_texts for key in old_ids)
            assert embedding_spec(old_ids, [original_texts[key] for key in old_ids]) == spec
            matrix = data["vectors"]
            for i, key in enumerate(old_ids):
                if key in expected:
                    assert original_texts[key] == expected[key]
                    available[key] = matrix[i].copy()
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    missing = [key for key in ids if key not in available]
    if missing:
        print(f"{phase}: encoding {len(missing)} additional articles for restored flat days", flush=True)
        vectors, path, spec, info = load_or_encode(missing, [lookup[key] for key in missing], ENCODERS)
        available.update(zip(missing, vectors))
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    vectors = np.stack([available[key] for key in ids])
    np.savez_compressed(OUTPUT / f"{phase}_article_vectors.npz", article_ids=np.asarray(ids), vectors=vectors,
                        spec=json.dumps(embedding_spec(ids, texts)))
    windows = np.zeros((len(frame), len(TEXT)))
    for i, items in enumerate(frame.articles):
        if items:
            windows[i] = np.stack([available[a["article_id"]] for a in items]).mean(axis=0)
    result = pd.concat([frame.drop(columns="articles").reset_index(drop=True), pd.DataFrame(windows, columns=TEXT)], axis=1)
    result["log1p_n_berita"] = np.log1p(result.n_berita)
    assert np.isfinite(result[FEATURES]).all().all()
    joblib.dump(result, OUTPUT / f"{phase}_features.joblib", compress=3)
    write_json(f"{phase}_encoder.json", {"spec": embedding_spec(ids, texts), "articles": len(ids),
               "reused_articles": len(ids) - len(missing), "newly_encoded_articles": len(missing),
               "frozen": True, "aggregation": "Uniform mean of article embeddings per fixing window"})
    return result

def label(frame, threshold):
    result = frame.copy()
    result["y"] = result.abs_return_next.gt(threshold).astype(int)
    return result

def fit_model(train, validation, seed, fold=None, fixed_epochs=None):
    scaler = BlockScaler(MARKET, TEXT, AVAILABILITY).fit(train)
    x_train = scaler.transform(train[FEATURES])
    x_val = scaler.transform(validation[FEATURES]) if validation is not None else None
    model = MarketAlignedMLP(**CONFIG, random_state=seed)
    best, best_key, best_epoch, stale, history = None, None, 0, 0, []
    limit = MAX_EPOCHS if fixed_epochs is None else int(fixed_epochs)
    for epoch in range(1, limit + 1):
        model.partial_fit(x_train, train.y.to_numpy(), classes=np.array([0, 1]))
        row = {"fold": fold, "seed": seed, "epoch": epoch,
               "train_log_loss": score(train.y, model.predict_proba(x_train)[:, 1])["log_loss"]}
        if validation is not None:
            metrics = score(validation.y, model.predict_proba(x_val)[:, 1])
            row.update({f"val_{key}": value for key, value in metrics.items()})
            key = (metrics["mcc"], metrics["macro_f1"], -metrics["log_loss"])
            if best_key is None or key > best_key:
                best, best_key, best_epoch, stale = copy.deepcopy(model), key, epoch, 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= PATIENCE:
            break
    if validation is None:
        best, best_epoch = model, limit
    assert best.t_ == len(train) * best_epoch
    return Pipeline([("scaler", scaler), ("mlp", best)]), best_epoch, epoch, history

def baseline_models(train, threshold):
    prior = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), train.y)
    return {"training_prior": {"estimator": prior, "threshold": threshold},
            "last_magnitude": {"rule": "abs_return_t > threshold", "threshold": threshold}}

def probability(model, method, frame):
    if method == "finbert_concat":
        return model.predict_proba(frame[FEATURES])[:, 1]
    if method == "training_prior":
        return model["estimator"].predict_proba(np.zeros((len(frame), 1)))[:, 1]
    return frame.abs_return_t.gt(model["threshold"]).to_numpy(dtype=float)

def evaluate(model, method, frame, phase, seed, threshold, fold=None):
    path = OUTPUT / f"{phase}_fold_{fold}_{method}_seed_{seed}.joblib"
    joblib.dump(model, path)
    p = probability(model, method, frame)
    np.testing.assert_array_equal(probability(joblib.load(path), method, frame), p)
    result = frame[["Date", "target_date", "y", "abs_return_next", "abs_return_t", "target_gap_days", "was_flat", "n_berita"]].copy()
    result["method"], result["phase"], result["seed"], result["fold"] = method, phase, seed, fold
    result["label_threshold"], result["p_large"], result["y_pred"] = threshold, p, (p >= .5).astype(int)
    metric = {"method": method, "phase": phase, "seed": seed, "fold": fold, "n": len(frame), **score(frame.y, p)}
    precision, recall, f1, support = precision_recall_fscore_support(frame.y, result.y_pred, labels=[0, 1], zero_division=0)
    classes = [{"method": method, "phase": phase, "seed": seed, "fold": fold, "class": name,
                "support": int(support[i]), "precision": precision[i], "recall": recall[i], "f1": f1[i]}
               for i, name in enumerate(["SMALL", "LARGE"])]
    return result, metric, classes

def figures(predictions, thresholds):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    test = predictions.loc[predictions.phase.eq("test")]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, method, title in zip(axes, METHODS, ["Training prior", "Previous movement", "Price + FinBERT"]):
        matrix = np.mean([confusion_matrix(g.y, g.y_pred, labels=[0, 1]) for _, g in test.loc[test.method.eq(method)].groupby("seed")], axis=0)
        shares = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.1f}\n({shares[i, j]:.0%})", ha="center", va="center",
                        color="white" if shares[i, j] > .6 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Small", "Large"], yticklabels=["Small", "Large"],
               xlabel="Predicted", title=title)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    save_figure(fig, "test_confusion_matrices")
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    one = test.loc[test.method.eq("training_prior")].sort_values("Date")
    ax.plot(one.Date, one.abs_return_next * 100, color="#767676", linewidth=.9)
    ax.axhline(thresholds[-1]["threshold"] * 100, color="#3775BA", linestyle="--", label="Training median")
    ax.set(xlabel="Decision date", ylabel="Absolute next-fixing return (%)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, "test_return_magnitude")

def save_figure(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    global MARKET, TEXT, AVAILABILITY, FEATURES, CONFIG, MAX_EPOCHS, PATIENCE, ORIGINAL_TEXTS
    if OUTPUT.exists():
        raise FileExistsError("Use a new version instead of overwriting an existing trial.")
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(REFERENCE.read_text())
    MARKET, TEXT, AVAILABILITY = [metadata["feature_blocks"][key] for key in ["market", "text", "availability"]]
    FEATURES = MARKET + TEXT + AVAILABILITY
    CONFIG = {**metadata["model_config"], "hidden_layer_sizes": tuple(metadata["model_config"]["hidden_layer_sizes"])}
    MAX_EPOCHS, PATIENCE = metadata["max_epochs"], metadata["patience"]
    sources = [p for p in (ROOT / "data/final").glob("*") if p.is_file()]
    sources += [ROOT / p for p in json.loads((ROOT / "data/final/metadata.json").read_text())["input_sha256"]]
    sources += [Path(__file__), REFERENCE, ROOT / "src/task_3/sanity_check/r2_logloss_selection.py"]
    sources += list((ROOT / "src/task_3/preprocessing").glob("*.py")) + list((ROOT / "src/task_3/training").glob("*.py"))
    sources += [p for level in ["level_1", "level_2", "level_3"] for p in (ROOT / "src/task_3/training" / level).glob("*") if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    write_json("protocol.json", {"experiment": NAME, "frozen_at_wib": now(), "target": "abs((J_next - J_t) / J_t) > median(training absolute next-fixing returns)",
               "threshold_quantile": .5, "threshold_fit": "Each fold training set only; final refit train+val only",
               "ties": "Equal to the median is SMALL", "flat_returns": "Restored before applying unchanged calendar boundaries and purges",
               "methods": METHODS, "seeds": SEEDS, "model_config": CONFIG, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
               "checkpoint_selection": ["validation MCC", "validation macro-F1", "negative validation log loss"],
               "final_epochs": "Per-seed median selected epoch across three development folds", "probability_threshold": .5,
               "predictors": FEATURES, "future_columns_excluded": ["abs_return_next", "target_gap_days", "target_date", "was_flat", "restored_flat", "y"],
               "persistence": "Repeat whether the current known absolute return exceeds the same training threshold; hard 0/1 probabilities are uncalibrated",
               "input_sha256": hashes, "test_used_for_selection": False,
               "interpretation": "Exploratory new target on a previously accessed test period; not directly comparable to direction labels; validation also selects checkpoints"})
    original = pd.concat([splitting.load_split(ROOT / "data/final", name) for name in ["train", "val", "test"]])
    ORIGINAL_TEXTS = {a["article_id"]: a["text_transformer"] for row in original.articles for a in row}
    raw = prepare_data()
    ORIGINAL_TEXTS.update({a["article_id"]: a["text_transformer"] for row in raw.articles for a in row})
    dev = encode(raw.loc[raw.split.ne("test")].copy(), "development", hashes)
    predictions, metrics, classes, fits, histories, thresholds = [], [], [], [], [], []
    for fold in [1, 2, 3]:
        roles = dev.fold_roles.map(lambda value: value[str(fold)])
        raw_train, raw_val = dev.loc[roles.eq("train")], dev.loc[roles.eq("val")]
        assert raw_train.target_date.max() < raw_val.Date.min()
        threshold = float(raw_train.abs_return_next.median())
        train, val = label(raw_train, threshold), label(raw_val, threshold)
        thresholds.append({"phase": "validation", "fold": fold, "threshold": threshold, "train_n": len(train), "eval_n": len(val),
                           "train_large_share": train.y.mean(), "eval_large_share": val.y.mean()})
        for method, model in baseline_models(train, threshold).items():
            p, m, c = evaluate(model, method, val, "validation", -1, threshold, fold)
            predictions.append(p); metrics.append(m); classes.extend(c)
        for seed in SEEDS:
            model, best, ran, trace = fit_model(train, val, seed, fold)
            p, m, c = evaluate(model, "finbert_concat", val, "validation", seed, threshold, fold)
            predictions.append(p); metrics.append(m); classes.extend(c); histories.extend(trace)
            fits.append({"phase": "validation", "fold": fold, "seed": seed, "train_n": len(train), "val_n": len(val), "threshold": threshold,
                         "best_epoch": best, "epochs_run": ran})
            print(f"Fold {fold}, seed {seed}: threshold {threshold:.6f}, epoch {best}, MCC {m['mcc']:.4f}, AUC {m['roc_auc']:.4f}", flush=True)
        write_json("status.json", {"phase": "development", "completed_fold": fold, "updated_at_wib": now()})
    pd.DataFrame(fits).to_csv(OUTPUT / "development_fits.csv", index=False)
    epochs = {str(seed): int(value) for seed, value in pd.DataFrame(fits).groupby("seed").best_epoch.median().items()}
    threshold = float(dev.abs_return_next.median())
    write_json("frozen_final_config.json", {"frozen_at_wib": now(), "train_n": len(dev), "epochs": epochs, "label_threshold": threshold,
               "probability_threshold": .5, "test_used_for_selection": False, "protocol_sha256": sha256(OUTPUT / "protocol.json")})
    final_train = label(dev, threshold)
    final_models = baseline_models(final_train, threshold)
    for seed in SEEDS:
        model, best, ran, trace = fit_model(final_train, None, seed, fixed_epochs=epochs[str(seed)])
        final_models[f"finbert_{seed}"] = model
        histories.extend(trace)
        fits.append({"phase": "test", "fold": None, "seed": seed, "train_n": len(final_train), "threshold": threshold, "best_epoch": best, "epochs_run": ran})
    write_json("test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(OUTPUT / "frozen_final_config.json"),
               "purpose": "One exploratory evaluation of prespecified methods"})
    test = label(encode(raw.loc[raw.split.eq("test")].copy(), "test", hashes), threshold)
    assert final_train.target_date.max() < test.Date.min()
    thresholds.append({"phase": "test", "fold": None, "threshold": threshold, "train_n": len(final_train), "eval_n": len(test),
                       "train_large_share": final_train.y.mean(), "eval_large_share": test.y.mean()})
    for key, model in final_models.items():
        method, seed = ("finbert_concat", int(key.split("_")[1])) if key.startswith("finbert_") else (key, -1)
        p, m, c = evaluate(model, method, test, "test", seed, threshold)
        predictions.append(p); metrics.append(m); classes.extend(c)
    predictions, metrics = pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)
    summaries = []
    for phase, part in metrics.groupby("phase"):
        for method, group in part.groupby("method"):
            units = group.groupby("fold")[METRICS].mean() if phase == "validation" else group[METRICS]
            for metric in METRICS:
                summaries.append({"phase": phase, "method": method, "metric": metric, "mean": units[metric].mean(), "std": units[metric].std(ddof=1)})
    summary = pd.DataFrame(summaries)
    for name, table in {"predictions": predictions, "seed_metrics": metrics, "per_class": pd.DataFrame(classes),
                        "fits": pd.DataFrame(fits), "history": pd.DataFrame(histories), "thresholds": pd.DataFrame(thresholds), "summary": summary}.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)
    raw.drop(columns="articles").assign(y="threshold-dependent").to_csv(OUTPUT / "targets.csv", index=False)
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
                fold_draws[method].append(np.mean([bootstrap_metrics(g.sort_values("Date").y, g.sort_values("Date").p_large, indices)
                                                       for _, g in selected.groupby("seed")], axis=0))
        draws = {method: np.mean(values, axis=0) for method, values in fold_draws.items()}
        draws.update({f"finbert_minus_{baseline}": draws["finbert_concat"] - draws[baseline] for baseline in METHODS[:2]})
        for method, values in draws.items():
            low, high = np.nanquantile(values, [.025, .975], axis=0)
            intervals.extend({"phase": phase, "comparison": method, "metric": metric, "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
        np.savez_compressed(OUTPUT / f"{phase}_bootstrap.npz", metrics=np.asarray(METRICS), **draws)
    pd.DataFrame(intervals).to_csv(OUTPUT / "intervals.csv", index=False)
    gaps = []
    for (method, seed, gap), group in predictions.loc[predictions.phase.eq("test")].groupby(["method", "seed", "target_gap_days"]):
        gaps.append({"method": method, "seed": seed, "gap_days": gap, "n": len(group), "large_share": group.y.mean(),
                     **(score(group.y, group.p_large) if group.y.nunique() == 2 else {})})
    pd.DataFrame(gaps).to_csv(OUTPUT / "test_gap_metrics.csv", index=False)
    figures(predictions, thresholds)
    lines = ["# Next-fixing movement magnitude", "", "Predict LARGE versus SMALL using absolute simple next-fixing returns. LARGE means strictly above the training median; ties and zero returns are SMALL.", "",
             "Calendar split boundaries and target purges match the direction study. Thirteen previously excluded flat decisions are restored. Features and news for every existing row match the immutable final export.", "",
             "FinBERT concat reuses Level 1 architecture, preprocessing and MCC checkpoint selection, with three seeds. Final epochs are the per-seed median across development folds; refit uses train+val. The final label threshold is recomputed on train+val before test evaluation.", "",
             "| Phase | Method | " + " | ".join(METRICS) + " |", "| --- | --- | " + " | ".join(["---:"] * len(METRICS)) + " |"]
    for phase in ["validation", "test"]:
        for method in METHODS:
            table = summary.loc[summary.phase.eq(phase) & summary.method.eq(method)].set_index("metric")
            values = [f"{table.loc[metric, 'mean']:.4f}" for metric in METRICS]
            lines.append("| " + " | ".join([phase, method, *values]) + " |")
    lines += ["", f"Final threshold: **{threshold * 100:.4f}%** absolute return. Final epochs: `{json.dumps(epochs, sort_keys=True)}`.", "",
              "Training prior uses the training LARGE frequency. Previous movement repeats the current known magnitude class and supplies hard 0/1 probabilities; its log loss is not a calibrated probability benchmark. Naive ties at probability 0.5 predict LARGE consistently with the neural threshold.", "",
              "Validation metrics average seeds within each fold, then folds equally. Test metrics average individual seeds; deterministic baselines have one prediction stream. Intervals use 1,000 paired circular block resamples of 10 decisions, conditional on fitted models. Validation still selects and scores checkpoints.", "",
              "This is a different prediction task, not an improvement of UP/DOWN forecasting. The test period has already been accessed, so results are exploratory. Beating naive baselines alone does not isolate the effect of text; that would require a matched price-only control. Date-only news timing and historical article-version limitations remain. No target gap length is supplied as a feature.", "",
              "Original final CSVs, notebooks, ablation results and representation selection remain unchanged. See cohort.csv, thresholds.csv, intervals.csv and test_gap_metrics.csv for supporting checks."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    for relative, expected in hashes.items():
        assert sha256(ROOT / relative) == expected, relative
    write_json("status.json", {"phase": "complete", "updated_at_wib": now()})
    artifacts = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    write_json("run_metadata.json", {"completed_at_wib": now(), "input_sha256": hashes,
               "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in artifacts}, "features": FEATURES,
               "final_epochs": epochs, "final_threshold": threshold, "development_rows": len(dev), "test_rows": len(test),
               "saved_model_predictions_verified": True, "test_used_for_selection": False,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}})
    print(summary.pivot(index=["phase", "method"], columns="metric", values="mean").to_string(), flush=True)
    print("Results:", OUTPUT.relative_to(ROOT), flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
