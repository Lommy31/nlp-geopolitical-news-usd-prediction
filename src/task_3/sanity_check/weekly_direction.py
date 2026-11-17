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
from src.task_3.preprocessing.merging import build_rate_features
from src.task_3.sanity_check.r2_logloss_selection import score
from src.task_3.training.fusion import BlockScaler, MarketAlignedMLP
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

NAME = "weekly_direction_v1"
OUTPUT = ROOT / "src/task_3/output/model/sanity_check" / NAME
FIGURES = ROOT / "src/task_3/output/figure/sanity_check" / NAME
CACHE = ROOT / "src/task_3/output/model/sanity_check/movement_magnitude_v1"
REFERENCE = ROOT / "src/task_3/output/model/level_1/04_market_news_concat/run_metadata.json"
SEEDS = [42, 123, 2026]
METHODS = ["training_prior", "previous_5_fixings", "finbert_concat"]
HORIZON = 5
BLOCK_LENGTHS = [20, 10, 40]

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def make_calendar():
    calendar = build_rate_features(pd.read_csv(ROOT / "data/processed/usd_idr_jisdor_clean.csv"))
    calendar["target_date"] = calendar.Date.shift(-HORIZON)
    calendar["target_calendar_index"] = calendar.calendar_index.shift(-HORIZON)
    calendar["future_jisdor"] = calendar.USD_IDR.shift(-HORIZON)
    calendar["weekly_return"] = np.log(calendar.future_jisdor / calendar.USD_IDR)
    calendar["previous_5_return"] = np.log(calendar.USD_IDR / calendar.USD_IDR.shift(HORIZON))
    calendar["weekly_status"] = np.select([calendar.future_jisdor.gt(calendar.USD_IDR), calendar.future_jisdor.lt(calendar.USD_IDR),
                                            calendar.future_jisdor.eq(calendar.USD_IDR)], ["UP", "DOWN", "FLAT"], default="UNAVAILABLE")
    calendar["weekly_eligible"] = calendar.features_ready & calendar.weekly_status.isin(["UP", "DOWN"])
    calendar["y"] = calendar.future_jisdor.gt(calendar.USD_IDR).astype(int)
    test_start = int(len(calendar) * .8)
    development_end = test_start - 1
    bounds = np.linspace(development_end // 2, development_end, 4, dtype=int)
    boundaries, purges = [], []
    for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        train_region = calendar.calendar_index.lt(start)
        val_region = calendar.calendar_index.ge(start) & calendar.calendar_index.lt(end)
        train = train_region & calendar.target_calendar_index.lt(start) & calendar.weekly_eligible
        val = val_region & calendar.target_calendar_index.lt(end) & calendar.weekly_eligible
        calendar[f"fold_{fold}"] = np.select([train, val], ["train", "val"], default="unused")
        assert calendar.loc[train, "target_date"].max() < calendar.loc[val, "Date"].min()
        boundaries.append({"fold": fold, "train_before_index": int(start), "validation_start_index": int(start),
                           "validation_end_exclusive_index": int(end), "train_n": int(train.sum()), "val_n": int(val.sum()),
                           "last_train_target": calendar.loc[train, "target_date"].max().strftime("%Y-%m-%d"),
                           "first_val_date": calendar.loc[val, "Date"].min().strftime("%Y-%m-%d")})
        for role, region, cutoff in [("train", train_region, start), ("val", val_region, end)]:
            removed = calendar.loc[region & calendar.features_ready & calendar.future_jisdor.notna() & calendar.target_calendar_index.ge(cutoff)]
            purges.extend({"fold": fold, "role": role, "Date": row.Date.strftime("%Y-%m-%d"),
                           "target_date": row.target_date.strftime("%Y-%m-%d"), "calendar_index": row.calendar_index,
                           "target_calendar_index": int(row.target_calendar_index)} for row in removed.itertuples())
    calendar["split"] = calendar.fold_3.map({"train": "train", "val": "val"}).fillna("excluded")
    calendar.loc[calendar.calendar_index.ge(test_start) & calendar.weekly_eligible, "split"] = "test"
    for fold in [1, 2, 3]:
        assert not (calendar[f"fold_{fold}"].ne("unused") & ~calendar.split.isin(["train", "val"])).any()
    calendar["fold_roles"] = [{str(fold): row[f"fold_{fold}"] for fold in [1, 2, 3]} for _, row in calendar.iterrows()]
    calendar["target_gap_days"] = (calendar.target_date - calendar.Date).dt.days
    calendar["nonoverlap_offset"] = calendar.calendar_index % HORIZON
    pd.DataFrame(purges).to_csv(OUTPUT / "purged_boundaries.csv", index=False)
    write_json("calendar_contract.json", {"horizon_fixings": HORIZON, "calendar_rows": len(calendar), "fold_bounds": bounds.tolist(),
               "test_start_index": test_start, "test_start_date": calendar.loc[calendar.calendar_index.eq(test_start), "Date"].iloc[0].strftime("%Y-%m-%d"),
               "folds": boundaries, "flat_five_fixing_returns": int((calendar.features_ready & calendar.weekly_status.eq("FLAT")).sum()),
               "final_refit": "Union of exported weekly train and validation rows; boundary-purged rows stay excluded"})
    audit = calendar[["Date", "calendar_index", "USD_IDR", "features_ready", "target_date", "target_calendar_index", "future_jisdor",
                      "weekly_return", "previous_5_return", "weekly_status", "weekly_eligible", "y", "split", "target_gap_days", "nonoverlap_offset",
                      "fold_1", "fold_2", "fold_3"]].copy()
    audit.loc[~audit.weekly_eligible, "y"] = np.nan
    audit.to_csv(OUTPUT / "calendar_labels.csv", index=False)
    return calendar

def load_features(calendar, phase, source_metadata):
    path = CACHE / f"{phase}_features.joblib"
    assert sha256(path) == source_metadata["artifact_sha256"][str(path.relative_to(ROOT))]
    cached = joblib.load(path)
    columns = list(dict.fromkeys(["Date", "calendar_index", "USD_IDR", "n_berita", "has_news", *FEATURES]))
    cached = cached[columns].copy()
    selected = calendar.loc[calendar.split.eq("test") if phase == "test" else calendar.split.isin(["train", "val"])].copy()
    assert selected.Date.isin(cached.Date).all()
    market = cached.set_index("Date").loc[selected.Date, MARKET].to_numpy()
    np.testing.assert_allclose(market, selected[MARKET].to_numpy(), atol=1e-12, rtol=1e-12)
    targets = selected[["Date", "target_date", "target_calendar_index", "weekly_return", "previous_5_return", "weekly_status",
                        "y", "split", "fold_roles", "target_gap_days", "nonoverlap_offset"]]
    frame = cached.merge(targets, on="Date", how="inner", validate="one_to_one").sort_values("Date").reset_index(drop=True)
    assert np.isfinite(frame[FEATURES]).all().all()
    assert frame.y.nunique() == 2 and (frame.target_calendar_index - frame.calendar_index).eq(HORIZON).all()
    joblib.dump(frame, OUTPUT / f"{phase}_features.joblib", compress=3)
    return frame

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

def baselines(train):
    prior = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), train.y)
    return {"training_prior": {"estimator": prior},
            "previous_5_fixings": {"rule": "Sign of log(J_t / J_t_minus_5); zero uses training UP frequency", "fallback_p_up": float(train.y.mean())}}

def probability(model, method, frame):
    if method == "finbert_concat":
        return model.predict_proba(frame[FEATURES])[:, 1]
    if method == "training_prior":
        return model["estimator"].predict_proba(np.zeros((len(frame), 1)))[:, 1]
    r = frame.previous_5_return.to_numpy()
    return np.where(r > 0, 1., np.where(r < 0, 0., model["fallback_p_up"]))

def evaluate(model, method, frame, phase, seed, fold=None):
    path = OUTPUT / f"{phase}_fold_{fold}_{method}_seed_{seed}.joblib"
    joblib.dump(model, path)
    p = probability(model, method, frame)
    np.testing.assert_array_equal(probability(joblib.load(path), method, frame), p)
    columns = ["Date", "target_date", "calendar_index", "target_calendar_index", "y", "weekly_return", "previous_5_return",
               "nonoverlap_offset", "target_gap_days", "n_berita"]
    output = frame[columns].copy()
    output["method"], output["phase"], output["seed"], output["fold"] = method, phase, seed, fold
    output["p_up"], output["y_pred"] = p, (p >= .5).astype(int)
    metrics = {"method": method, "phase": phase, "seed": seed, "fold": fold, "n": len(frame), **score(frame.y, p)}
    precision, recall, f1, support = precision_recall_fscore_support(frame.y, output.y_pred, labels=[0, 1], zero_division=0)
    classes = [{"method": method, "phase": phase, "seed": seed, "fold": fold, "class": name, "support": int(support[i]),
                "precision": precision[i], "recall": recall[i], "f1": f1[i]} for i, name in enumerate(["DOWN", "UP"])]
    return output, metrics, classes

def resample(predictions):
    intervals = []
    for phase in ["validation", "test"]:
        for block_length in BLOCK_LENGTHS:
            by_fold = {method: [] for method in METHODS}
            for fold in ([1, 2, 3] if phase == "validation" else [None]):
                frame = predictions.loc[predictions.phase.eq(phase)]
                if fold is not None:
                    frame = frame.loc[frame.fold.eq(fold)]
                indices = circular_indices(frame.Date.nunique(), block_length=block_length, seed=20261110 + (fold or 0))
                for method in METHODS:
                    selected = frame.loc[frame.method.eq(method)]
                    values = []
                    for _, group in selected.groupby("seed"):
                        group = group.sort_values("Date")
                        values.append(bootstrap_metrics(group.y, group.p_up, indices))
                    by_fold[method].append(np.mean(values, axis=0))
            draws = {method: np.mean(values, axis=0) for method, values in by_fold.items()}
            draws.update({f"finbert_minus_{method}": draws["finbert_concat"] - draws[method] for method in METHODS[:2]})
            for method, values in draws.items():
                low, high = np.nanquantile(values, [.025, .975], axis=0)
                intervals.extend({"phase": phase, "block_length": block_length, "comparison": method, "metric": metric,
                                  "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
            np.savez_compressed(OUTPUT / f"{phase}_bootstrap_block_{block_length}.npz", metrics=np.asarray(METRICS), **draws)
    return pd.DataFrame(intervals)

def nonoverlap_check(predictions):
    rows = []
    for (phase, method, seed, fold, offset), group in predictions.groupby(["phase", "method", "seed", "fold", "nonoverlap_offset"], dropna=False):
        group = group.sort_values("Date")
        assert (group.calendar_index.to_numpy()[1:] >= group.target_calendar_index.to_numpy()[:-1]).all()
        values = score(group.y, group.p_up) if group.y.nunique() == 2 else {key: np.nan for key in METRICS}
        rows.append({"phase": phase, "method": method, "seed": seed, "fold": fold, "offset": offset, "n": len(group), **values})
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "nonoverlap_seed_metrics.csv", index=False)
    result.loc[result.phase.eq("test")].groupby(["method", "offset"])[["n", *METRICS]].mean().reset_index().to_csv(OUTPUT / "test_nonoverlap_summary.csv", index=False)

def figures(predictions, history, fits):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    test = predictions.loc[predictions.phase.eq("test")]
    for ax, method, title in zip(axes, METHODS, ["Training prior", "Previous 5 fixings", "Price + FinBERT"]):
        matrix = np.mean([confusion_matrix(group.y, group.y_pred, labels=[0, 1]) for _, group in test.loc[test.method.eq(method)].groupby("seed")], axis=0)
        shares = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.1f}\n({shares[i, j]:.0%})", ha="center", va="center", color="white" if shares[i, j] > .6 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], xlabel="Predicted", title=title)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    save_figure(fig, "test_confusion_matrices")
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), sharey=True)
    for ax, fold in zip(axes, [1, 2, 3]):
        trace = history.loc[history.fold.eq(fold) & history.seed.eq(42)]
        selected = fits.loc[fits.phase.eq("validation") & fits.fold.eq(fold) & fits.seed.eq(42), "best_epoch"].iloc[0]
        ax.plot(trace.epoch, trace.train_log_loss, color="#767676", linestyle="--", label="Training")
        ax.plot(trace.epoch, trace.val_log_loss, color="#3775BA", label="Validation")
        ax.axvline(selected, color="black", linestyle=":", linewidth=1, label="Selected epoch")
        ax.set(xlabel="Epoch", title=f"Fold {fold}")
    axes[0].set_ylabel("Log loss")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.04), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .9))
    save_figure(fig, "training_curves")

def save_figure(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    global MARKET, TEXT, AVAILABILITY, FEATURES, CONFIG, MAX_EPOCHS, PATIENCE
    if OUTPUT.exists():
        raise FileExistsError("Use a new version instead of overwriting an existing trial.")
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(REFERENCE.read_text())
    MARKET, TEXT, AVAILABILITY = [metadata["feature_blocks"][key] for key in ["market", "text", "availability"]]
    FEATURES = MARKET + TEXT + AVAILABILITY
    CONFIG = {**metadata["model_config"], "hidden_layer_sizes": tuple(metadata["model_config"]["hidden_layer_sizes"])}
    MAX_EPOCHS, PATIENCE = metadata["max_epochs"], metadata["patience"]
    source_metadata = json.loads((CACHE / "run_metadata.json").read_text())
    assert json.loads((CACHE / "verification.json").read_text())["verified"]
    assert source_metadata["features"] == FEATURES
    sources = [p for p in (ROOT / "data/final").glob("*") if p.is_file()]
    sources += [ROOT / "data/processed/usd_idr_jisdor_clean.csv", Path(__file__), REFERENCE,
                ROOT / "src/task_3/sanity_check/r2_logloss_selection.py"]
    sources += [CACHE / name for name in ["run_metadata.json", "verification.json", "development_features.joblib", "test_features.joblib"]]
    sources += list((ROOT / "src/task_3/preprocessing").glob("*.py")) + list((ROOT / "src/task_3/training").glob("*.py"))
    sources += [p for level in ["level_1", "level_2", "level_3"] for p in (ROOT / "src/task_3/training" / level).glob("*") if p.is_file()]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    write_json("protocol.json", {"experiment": NAME, "frozen_at_wib": now(), "horizon_fixings": HORIZON,
               "target": "J_t_plus_5 > J_t; computed on full JISDOR calendar before filtering", "flat_policy": "Exclude only zero five-fixing returns; retain daily-flat decisions with a nonzero weekly target",
               "split": "Original calendar boundaries; each training/evaluation segment requires target index strictly before its end boundary",
               "methods": METHODS, "seeds": SEEDS, "model_config": CONFIG, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
               "checkpoint_selection": ["validation MCC", "validation macro-F1", "negative validation log loss"],
               "final_epochs": "Per-seed median across development folds", "probability_threshold": .5,
               "text_input": "Original fixing-t news window; no news after decision t and no new five-day news aggregation",
               "predictors": FEATURES, "baseline_zero_previous_return": "Training UP frequency",
               "bootstrap": {"repeats": 1000, "primary_block_length": BLOCK_LENGTHS[0], "sensitivity_block_lengths": BLOCK_LENGTHS[1:],
                             "method": "Paired circular blocks of observed decisions; within folds for validation; fixed fitted models"},
               "nonoverlap_check": "Report all five calendar_index modulo 5 offsets; never select an offset by performance",
               "input_sha256": hashes, "test_used_for_selection": False,
               "limitations": ["Previously accessed test period; exploratory horizon experiment", "Validation still selects and scores checkpoints",
                               "Nonoverlapping returns can remain serially dependent; effective sample size is not exactly n/5",
                               "Frozen daily FinBERT feature cache is reused without retraining an encoder", "Naive comparison does not isolate text contribution"]})
    calendar = make_calendar()
    development = load_features(calendar, "development", source_metadata)
    predictions, metrics, classes, fits, histories, cohorts = [], [], [], [], [], []
    for fold in [1, 2, 3]:
        roles = development.fold_roles.map(lambda value: value[str(fold)])
        train, val = development.loc[roles.eq("train")], development.loc[roles.eq("val")]
        assert train.target_date.max() < val.Date.min()
        cohorts.extend({"phase": "validation", "fold": fold, "role": role, "n": len(frame), "up": int(frame.y.sum()), "down": int((1 - frame.y).sum()),
                        "first_date": frame.Date.min().strftime("%Y-%m-%d"), "last_date": frame.Date.max().strftime("%Y-%m-%d")} for role, frame in [("train", train), ("val", val)])
        for method, model in baselines(train).items():
            p, m, c = evaluate(model, method, val, "validation", -1, fold)
            predictions.append(p)
            metrics.append(m)
            classes.extend(c)
        for seed in SEEDS:
            model, best, ran, trace = fit_model(train, val, seed, fold)
            p, m, c = evaluate(model, "finbert_concat", val, "validation", seed, fold)
            predictions.append(p)
            metrics.append(m)
            classes.extend(c)
            histories.extend(trace)
            fits.append({"phase": "validation", "fold": fold, "seed": seed, "train_n": len(train), "val_n": len(val), "best_epoch": best, "epochs_run": ran})
            print(f"Fold {fold}, seed {seed}: train {len(train)}, val {len(val)}, epoch {best}, MCC {m['mcc']:.4f}, AUC {m['roc_auc']:.4f}", flush=True)
        write_json("status.json", {"phase": "development", "completed_fold": fold, "updated_at_wib": now()})
    pd.DataFrame(fits).to_csv(OUTPUT / "development_fits.csv", index=False)
    epochs = {str(seed): int(value) for seed, value in pd.DataFrame(fits).groupby("seed").best_epoch.median().items()}
    write_json("frozen_final_config.json", {"frozen_at_wib": now(), "train_n": len(development), "epochs": epochs, "horizon_fixings": HORIZON,
               "probability_threshold": .5, "test_used_for_selection": False, "protocol_sha256": sha256(OUTPUT / "protocol.json"),
               "development_fits_sha256": sha256(OUTPUT / "development_fits.csv")})
    final_models = baselines(development)
    for seed in SEEDS:
        model, best, ran, trace = fit_model(development, None, seed, epochs=epochs[str(seed)])
        final_models[f"finbert_{seed}"] = model
        histories.extend(trace)
        fits.append({"phase": "test", "fold": None, "seed": seed, "train_n": len(development), "best_epoch": best, "epochs_run": ran})
    write_json("test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(OUTPUT / "frozen_final_config.json"),
               "purpose": "One exploratory evaluation of the prespecified five-fixing horizon"})
    test = load_features(calendar, "test", source_metadata)
    assert development.target_date.max() < test.Date.min()
    cohorts.extend({"phase": "test", "fold": None, "role": role, "n": len(frame), "up": int(frame.y.sum()), "down": int((1 - frame.y).sum()),
                    "first_date": frame.Date.min().strftime("%Y-%m-%d"), "last_date": frame.Date.max().strftime("%Y-%m-%d")} for role, frame in [("refit", development), ("test", test)])
    for key, model in final_models.items():
        method, seed = ("finbert_concat", int(key.split("_")[1])) if key.startswith("finbert_") else (key, -1)
        p, m, c = evaluate(model, method, test, "test", seed)
        predictions.append(p)
        metrics.append(m)
        classes.extend(c)
    predictions, metrics = pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)
    summaries = []
    for (phase, method), group in metrics.groupby(["phase", "method"]):
        units = group.groupby("fold")[METRICS].mean() if phase == "validation" else group[METRICS]
        summaries.extend({"phase": phase, "method": method, "metric": metric, "mean": units[metric].mean(), "std": units[metric].std(ddof=1)} for metric in METRICS)
    summary = pd.DataFrame(summaries)
    history, fits = pd.DataFrame(histories), pd.DataFrame(fits)
    for name, frame in {"predictions": predictions, "seed_metrics": metrics, "per_class": pd.DataFrame(classes), "fits": fits,
                        "history": history, "cohort": pd.DataFrame(cohorts), "summary": summary}.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    intervals = resample(predictions)
    intervals.to_csv(OUTPUT / "intervals.csv", index=False)
    nonoverlap_check(predictions)
    figures(predictions, history, fits)
    lines = ["# Five-fixing direction", "", "Predict whether JISDOR at t+5 exceeds JISDOR at t. The horizon is computed on the full fixing calendar before filtering. Zero five-fixing returns are excluded; daily-flat rows remain when their five-fixing target is nonzero.", "",
             "Original calendar boundaries are retained. Every training and validation segment is purged using its five-fixing target index; target dates must be strictly before the next segment. Final refit uses the union of the weekly train/validation exports, retaining their boundary exclusions.", "",
             "Price + FinBERT concat keeps the original Level 1 architecture, features, MCC checkpoint selection, three seeds and fixed 0.5 threshold. News inputs still end at decision t. Frozen article/window vectors are reused from the verified magnitude experiment; all classifiers and scalers are fitted again for weekly labels.", "",
             "| Phase | Method | " + " | ".join(METRICS) + " |", "| --- | --- | " + " | ".join(["---:"] * len(METRICS)) + " |"]
    for phase in ["validation", "test"]:
        for method in METHODS:
            table = summary.loc[summary.phase.eq(phase) & summary.method.eq(method)].set_index("metric")
            lines.append("| " + " | ".join([phase, method, *[f"{table.loc[metric, 'mean']:.4f}" for metric in METRICS]]) + " |")
    lines += ["", f"Refit decisions: **{len(development)}**. Test decisions: **{len(test)}**. Final epochs: `{json.dumps(epochs, sort_keys=True)}`.", "",
              "Validation averages individual-seed scores within folds, then averages folds equally. Test averages individual seeds; the naive baselines each have one prediction stream. See seed_metrics.csv and summary.csv for seed/fold variation.", "",
              "| Test comparison | Metric | 95% lower | 95% upper |", "| --- | --- | ---: | ---: |"]
    for comparison in ["finbert_concat", "finbert_minus_training_prior", "finbert_minus_previous_5_fixings"]:
        for metric in ["mcc", "roc_auc"]:
            row = intervals.loc[intervals.phase.eq("test") & intervals.block_length.eq(BLOCK_LENGTHS[0]) & intervals.comparison.eq(comparison) & intervals.metric.eq(metric)].iloc[0]
            lines.append(f"| {comparison} | {metric} | {row.low_95:.4f} | {row.high_95:.4f} |")
    lines += ["", "Intervals use 1,000 paired circular block resamples with blocks of 20 observed decisions, conditional on fitted models. Blocks of 10 and 40 are prespecified sensitivity checks in intervals.csv. Label overlap and longer dependence mean an IID interval would be inappropriate; these intervals do not include training/checkpoint-selection uncertainty.", "",
              "All five calendar-index modulo-5 offsets are reported in test_nonoverlap_summary.csv. Returns within an offset share no fixing increments; they can still have temporal dependence. Each offset contains only about one-fifth of the test decisions. No offset is selected as the main result.", "",
              "Training prior predicts the training weekly majority using its UP frequency. Previous 5 fixings repeats the sign of the known past-five-fixing return; a zero past return uses training UP frequency. Hard 0/1 persistence probabilities are uncalibrated, so their log loss is not a calibrated-probability benchmark.", "",
              "This is an exploratory horizon change on a previously accessed test period. Daily and weekly metrics describe different targets and cohorts. Higher weekly scores alone would not establish improvement or a text contribution; naive baselines and paired intervals must be considered. A matched price-only model would be required to isolate text. Existing news-timestamp and article-version limitations remain.", "",
              "Original data/final files, ablation notebooks, result tables, and representation selection remain unchanged."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    for relative, expected in hashes.items():
        assert sha256(ROOT / relative) == expected, relative
    write_json("status.json", {"phase": "complete", "updated_at_wib": now()})
    artifacts = [path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    write_json("run_metadata.json", {"completed_at_wib": now(), "input_sha256": hashes,
               "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts}, "features": FEATURES,
               "final_epochs": epochs, "horizon_fixings": HORIZON, "development_rows": len(development), "test_rows": len(test),
               "validation_rows": int(predictions.loc[predictions.phase.eq("validation"), "Date"].nunique()),
               "saved_model_predictions_verified": True, "test_used_for_selection": False,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}})
    print(summary.pivot(index=["phase", "method"], columns="metric", values="mean").to_string(), flush=True)
    print("Results:", OUTPUT.relative_to(ROOT), flush=True)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
