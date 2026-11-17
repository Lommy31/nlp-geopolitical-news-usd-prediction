from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import json
import platform
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
sys.path.insert(0, str(ROOT))
from src.task_3.preprocessing.splitting import load_split
from src.task_3.sanity_check.level1_improvement_models import SEEDS, fit_legacy, fit_small, fit_logistic
from src.task_3.training.controlled_fusion import score_binary
from src.task_3.training.fusion import BlockScaler
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

OUTPUT = ROOT / "src/task_3/output/model/sanity_check/level1_improvement_v1"
FIGURES = ROOT / "src/task_3/output/figure/sanity_check/level1_improvement_v1"
BASE = ROOT / "src/task_3/output/model/level_1"
CACHE = ROOT / "src/task_3/output/model/sanity_check/movement_magnitude_v1"
CONDITIONS = {"original_prior": "01_training_majority", "original_last_direction": "02_last_direction", "original_price": "03_market_only",
              "original_text": "04_market_news_concat", "original_availability": "05_market_news_availability"}

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def read_json(name):
    return json.loads((OUTPUT / name).read_text())

def effective_hashes():
    hashes = read_json("protocol.json")["input_sha256"].copy()
    path = OUTPUT / "implementation_amendment.json"
    if path.exists():
        amendment = json.loads(path.read_text())
        for change in amendment["changes"]:
            assert hashes[change["source"]] == change["original_sha256"]
            assert sha256(ROOT / change["snapshot"]) == change["original_sha256"]
            hashes[change["source"]] = change["updated_sha256"]
            hashes[change["snapshot"]] = change["original_sha256"]
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes

def blocks():
    meta = json.loads((BASE / "04_market_news_concat/run_metadata.json").read_text())
    return [meta["feature_blocks"][key] for key in ["market", "text", "availability"]]

def legacy_config():
    config = json.loads((BASE / "04_market_news_concat/run_metadata.json").read_text())["model_config"]
    return {**config, "hidden_layer_sizes": tuple(config["hidden_layer_sizes"])}

def fold_data(dev, fold):
    roles = dev.fold_roles.map(lambda value: value[str(fold)])
    train, val = dev.loc[roles.eq("train")], dev.loc[roles.eq("val")]
    assert train.target_date.max() < val.Date.min()
    return train, val

def prepare():
    if OUTPUT.exists():
        raise FileExistsError("Use a new version instead of overwriting this experiment.")
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    market, text, availability = blocks()
    sources = [ROOT / "data/final" / name for name in ["train.csv", "val.csv", "metadata.json"]]
    sources += [Path(__file__), ROOT / "src/task_3/sanity_check/level1_improvement_models.py", ROOT / "src/task_3/improvement_plan.md"]
    sources += [CACHE / name for name in ["development_features.joblib", "development_article_vectors.npz", "run_metadata.json", "verification.json"]]
    sources += list((ROOT / "src/task_3/training").glob("*.py"))
    sources += [p for level in ["level_1", "level_2", "level_3"] for p in (ROOT / "src/task_3/training" / level).glob("*") if p.is_file()]
    for directory in CONDITIONS.values():
        sources += [BASE / directory / name for name in ["run_metadata.json", "predictions.csv", "summary.csv"]]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    protocol = {"experiment": OUTPUT.name, "frozen_at_wib": now(), "target": "Original daily UP/DOWN; zero returns excluded as in data/final",
                "evaluation_phase": "Validation only; no test CSV, test feature cache, or test predictions parsed by this run",
                "representation": "Frozen FinBERT window means, identical to original L1-d", "seeds": SEEDS,
                "original_selection_diagnostic": {"training_label_block_permutations": 50, "block_length": 20, "folds": 3, "training_seeds": SEEDS,
                    "selection": "Original MCC, macro-F1, negative log loss; max 200, patience 25", "validation_labels": "Fixed original labels",
                    "interpretation": "Conditional training-label permutation diagnostic, not an exact permutation p-value or causal estimate of all validation bias"},
                "smooth_selection": {"metric": "Trailing 5-epoch mean validation log loss", "min_delta": .001, "max_epochs": 60, "patience": 10,
                                     "warmup": "First checkpoint eligible at epoch 5; keep current epoch when smoothed loss improves by more than min_delta"},
                "regularization_grid": {"alpha": [.01, .1, 1.], "dropout": [.2, .5], "hidden": [16], "adam_lr": .001,
                                        "batch_size": 32, "dtype": "float64", "l2": "alpha / (2 * batch_n) times non-bias squared weights"},
                "logistic_grid": {"C": [.01, .1, 1.], "solver": "lbfgs", "max_iter": 2000, "tol": 1e-8},
                "views": {"price": "14 market predictors", "availability": "Market + log1p news count + news mask", "text": "Market + frozen FinBERT + availability"},
                "centering_gate": "Mean off-diagonal training cosine > .95 in at least two folds, or median absolute text mean/std > 2 in at least two folds",
                "centering_branch": "Centered text versions of the best uncentered small-MLP text configuration and all three logistic C values; preserve empty-news zeros",
                "dimension_branch": "If a regularized text candidate passes the success gate, evaluate PCA 16 and 32 on the best small-MLP and logistic text settings; PCA/scaling fit on training news windows only",
                "success": {"probability_quality": "Validation log loss below training-prior baseline in at least two folds AND paired block-bootstrap 95% upper bound of model-minus-prior log loss < 0",
                            "text_contribution": "Additionally beat the same-family/hyperparameter price-only control in at least two folds, with paired upper log-loss bound < 0",
                            "selection": "Lowest mean log loss over three folds; seeds averaged within folds; no test selection",
                            "other_metrics": METRICS, "probability_threshold": .5},
                "bootstrap": {"repeats": 1000, "block_length": 10, "seed": 20261110, "scope": "Paired dates within each validation fold, conditional on fitted and selected models"},
                "follow_up": "Freeze one final configuration after validation; a separate final-test phase is permitted only after the success gate; the already accessed test is never a fresh holdout",
                "metadata": "Sample counts from actual frames, stopping/evaluation scope from fit records; shared experiment writer for all new candidates",
                "limitations": ["Hyperparameter selection and scoring still share validation folds; intervals omit selection uncertainty",
                                "The audit did not prove checkpoint selection is the sole cause of the validation/test gap",
                                "Embedding similarity alone cannot establish absence of label signal", "Existing publication-time and historical-body limitations remain"],
                "input_sha256": hashes}
    write_json("protocol.json", protocol)
    original = pd.concat([load_split(ROOT / "data/final", part) for part in ["train", "val"]], ignore_index=True).sort_values("Date").reset_index(drop=True)
    old_metadata = json.loads((CACHE / "run_metadata.json").read_text())
    path = CACHE / "development_features.joblib"
    assert sha256(path) == old_metadata["artifact_sha256"][str(path.relative_to(ROOT))]
    cached = joblib.load(path).set_index("Date").loc[original.Date].reset_index()
    pd.testing.assert_frame_equal(cached[["Date", "target_date", "n_berita", "has_news", *market]], original[["Date", "target_date", "n_berita", "has_news", *market]])
    dev = pd.concat([original.drop(columns=["USD_IDR", "calendar_index"]), cached[text].reset_index(drop=True)], axis=1)
    dev["log1p_n_berita"] = np.log1p(dev.n_berita)
    joblib.dump(dev, OUTPUT / "development_features.joblib", compress=3)
    baseline_predictions, baseline_metrics = [], []
    for name, directory in CONDITIONS.items():
        prediction = pd.read_csv(BASE / directory / "predictions.csv", parse_dates=["Date", "target_date"])
        if "seed" not in prediction:
            prediction["seed"] = -1
        prediction["candidate"] = name
        baseline_predictions.append(prediction[["Date", "target_date", "y", "fold", "seed", "candidate", "p_up", "y_pred"]])
        for (fold, seed), group in prediction.groupby(["fold", "seed"], dropna=False):
            baseline_metrics.append({"candidate": name, "fold": fold, "seed": -1 if pd.isna(seed) else int(seed), "n": len(group), **score_binary(group.y, group.p_up)})
    pd.concat(baseline_predictions, ignore_index=True).fillna({"seed": -1}).to_csv(OUTPUT / "baseline_predictions.csv", index=False)
    pd.DataFrame(baseline_metrics).to_csv(OUTPUT / "baseline_metrics.csv", index=False)
    with np.load(CACHE / "development_article_vectors.npz", allow_pickle=False) as data:
        vectors = dict(zip(data["article_ids"].tolist(), data["vectors"].astype(float)))
    diagnostics = []
    for fold in [1, 2, 3]:
        train, val = fold_data(dev, fold)
        news = train.loc[train.has_news.eq(1), text].to_numpy()
        unit = news / np.linalg.norm(news, axis=1, keepdims=True)
        cosine = (np.square(unit.sum(0)).sum() - len(news)) / (len(news) * (len(news) - 1))
        principal = PCA(n_components=1, svd_solver="full").fit(news)
        between = np.var(news, axis=0).mean()
        within = []
        for items in train.articles:
            if items:
                matrix = np.stack([vectors[a["article_id"]] for a in items])
                within.append(np.var(matrix, axis=0).mean())
        std = news.std(0)
        ratios = np.abs(news.mean(0)[std > 0]) / std[std > 0]
        diagnostics.append({"fold": fold, "train_n": len(train), "val_n": len(val), "mean_pair_cosine": cosine,
                            "pc1_variance_share": principal.explained_variance_ratio_[0], "between_day_variance": between,
                            "mean_within_day_variance": np.mean(within), "between_to_within_variance": between / np.mean(within),
                            "median_abs_mean_over_std": np.median(ratios), "center_supported": bool(cosine > .95 or np.median(ratios) > 2)})
    diagnostic = pd.DataFrame(diagnostics)
    diagnostic.to_csv(OUTPUT / "embedding_diagnostics.csv", index=False)
    write_json("prepared.json", {"completed_at_wib": now(), "protocol_sha256": sha256(OUTPUT / "protocol.json"), "development_rows": len(dev),
               "development_features_sha256": sha256(OUTPUT / "development_features.joblib"), "center_supported": bool(diagnostic.center_supported.sum() >= 2),
               "test_parsed": False})
    print(diagnostic.to_string(index=False), flush=True)

def load_development():
    prepared = read_json("prepared.json")
    assert prepared["protocol_sha256"] == sha256(OUTPUT / "protocol.json")
    assert prepared["development_features_sha256"] == sha256(OUTPUT / "development_features.joblib")
    return joblib.load(OUTPUT / "development_features.joblib")

def permutation():
    dev = load_development()
    market, text, availability = blocks()
    config = legacy_config()
    path = OUTPUT / "permutation_seed_metrics.csv"
    records = pd.read_csv(path).to_dict("records") if path.exists() else []
    transforms = {}
    for fold in [1, 2, 3]:
        train, val = fold_data(dev, fold)
        transform = BlockScaler(market, text, availability).fit(train)
        transforms[fold] = transform.transform(train), transform.transform(val)
    for repeat in range(50):
        for fold in [1, 2, 3]:
            train, val = fold_data(dev, fold)
            labels = train.y.to_numpy()
            chunks = [labels[start:start + 20] for start in range(0, len(labels), 20)]
            order = np.random.default_rng(20261116 + 1000 * repeat + fold).permutation(len(chunks))
            shuffled = np.concatenate([chunks[i] for i in order])
            assert shuffled.sum() == labels.sum() and len(shuffled) == len(labels)
            for seed in SEEDS:
                if any(row["repeat"] == repeat and row["fold"] == fold and row["seed"] == seed for row in records):
                    continue
                started = time.perf_counter()
                model, best, ran, trace = fit_legacy(train, val, seed, market, text, availability, config,
                                                   shuffled_y=shuffled, scaled=transforms[fold])
                metrics = score_binary(val.y, model.predict_proba(val)[:, 1])
                records.append({"repeat": repeat, "fold": fold, "seed": seed, "train_n": len(train), "val_n": len(val), "best_epoch": best,
                                "epochs_run": ran, "fit_seconds": time.perf_counter() - started, **metrics})
                pd.DataFrame(records).to_csv(path, index=False)
        write_json("permutation_status.json", {"phase": "running", "completed_repeats": repeat + 1, "total_repeats": 50, "updated_at_wib": now()})
        mean = pd.DataFrame(records).loc[lambda d: d.repeat.eq(repeat), "mcc"].mean()
        print(f"Permutation {repeat + 1}/50: mean validation MCC {mean:.4f}", flush=True)
    results = pd.DataFrame(records).groupby(["repeat", "fold"])[METRICS].mean().groupby("repeat").mean()
    results.to_csv(OUTPUT / "permutation_summary.csv")
    observed = pd.read_csv(BASE / "04_market_news_concat/summary.csv").set_index("statistic").loc["mean", "mcc"]
    write_json("permutation_result.json", {"completed_at_wib": now(), "repeats": len(results), "fits": len(records), "observed_original_mcc": observed,
               "null_mean_mcc": results.mcc.mean(), "null_mcc_2p5": results.mcc.quantile(.025), "null_mcc_97p5": results.mcc.quantile(.975),
               "fraction_at_least_observed": float(results.mcc.ge(observed).mean()),
               "interpretation": "Descriptive conditional training-label permutation reference; not an exact permutation p-value"})
    write_json("permutation_status.json", {"phase": "complete", "completed_repeats": len(results), "updated_at_wib": now()})

def identifier(family, view, alpha=None, dropout=None, C=None, centered=False, pca=None):
    result = f"{family}_{view}"
    if alpha is not None:
        result += f"_a{alpha:g}_d{dropout:g}"
    if C is not None:
        result += f"_C{C:g}"
    return (result + f"_center{int(centered)}" + (f"_pca{pca}" if pca else "")).replace(".", "p")

def new_candidate(family, view, **kwargs):
    candidate = {"family": family, "view": view, "centered": False, **kwargs}
    candidate["id"] = identifier(**candidate)
    return candidate

def evaluate_candidate(candidate, dev):
    directory = OUTPUT / "candidates" / candidate["id"]
    if (directory / "complete.json").exists() or (directory / "failure.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    market, text, availability = blocks()
    predictions, metrics, fits, histories = [], [], [], []
    for fold in [1, 2, 3]:
        train, val = fold_data(dev, fold)
        for seed in ([-1] if candidate["family"] == "logistic" else SEEDS):
            started = time.perf_counter()
            if candidate["family"] == "logistic":
                try:
                    model = fit_logistic(train, candidate, market, text, availability)
                except ValueError as error:
                    if str(error) != "Logistic regression did not converge.":
                        raise
                    (directory / "failure.json").write_text(json.dumps({"candidate": candidate, "fold": fold, "seed": seed,
                        "reason": str(error), "max_iter": 2000, "excluded_from_selection": True,
                        "partial_models": [p.name for p in directory.glob("*.joblib")], "recorded_at_wib": now()}, indent=2) + "\n")
                    print(f"{candidate['id']}: excluded, solver did not converge in fold {fold}", flush=True)
                    return
                best, ran, trace = None, None, []
            elif candidate["family"] == "small_mlp":
                model, best, ran, trace = fit_small(train, val, seed, candidate, market, text, availability)
            else:
                model, best, ran, trace = fit_legacy(train, val, seed, market, text, availability, legacy_config(), selection="smooth")
            probability = model.predict_proba(val)[:, 1]
            path = directory / f"fold_{fold}_seed_{seed}.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(joblib.load(path).predict_proba(val)[:, 1], probability)
            result = val[["Date", "target_date", "y"]].copy()
            result["fold"], result["seed"], result["candidate"] = fold, seed, candidate["id"]
            result["p_up"], result["y_pred"] = probability, (probability >= .5).astype(int)
            predictions.append(result)
            metrics.append({"candidate": candidate["id"], "fold": fold, "seed": seed, "n": len(val), **score_binary(val.y, probability)})
            fits.append({"candidate": candidate["id"], "fold": fold, "seed": seed, "train_n": len(train), "eval_n": len(val), "best_epoch": best,
                         "epochs_run": ran, "fit_seconds": time.perf_counter() - started, "evaluation_role": "development_validation", "fixed_epochs": None,
                         "test_used_for_fitting_or_stopping": False, "has_test_evaluation": False})
            histories.extend({"candidate": candidate["id"], "fold": fold, "seed": seed, **row} for row in trace)
    pd.concat(predictions, ignore_index=True).to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(directory / "seed_metrics.csv", index=False)
    pd.DataFrame(fits).to_csv(directory / "fits.csv", index=False)
    pd.DataFrame(histories).to_csv(directory / "history.csv", index=False)
    (directory / "complete.json").write_text(json.dumps({"candidate": candidate, "completed_at_wib": now(), "saved_models": len(fits), "evaluation_phase": "validation"}, indent=2) + "\n")
    mean = pd.DataFrame(metrics).groupby("fold")[METRICS].mean().mean()
    print(f"{candidate['id']}: validation LL {mean['log_loss']:.4f}, MCC {mean['mcc']:.4f}, AUC {mean['roc_auc']:.4f}", flush=True)

def all_results():
    configurations, predictions, metrics = {}, [], []
    for path in sorted((OUTPUT / "candidates").glob("*/complete.json")):
        candidate = json.loads(path.read_text())["candidate"]
        configurations[candidate["id"]] = candidate
        predictions.append(pd.read_csv(path.parent / "predictions.csv", parse_dates=["Date", "target_date"]))
        metrics.append(pd.read_csv(path.parent / "seed_metrics.csv"))
    return configurations, pd.concat(predictions, ignore_index=True), pd.concat(metrics, ignore_index=True)

def comparison_draws(prediction, reference):
    values, others = [], []
    for fold in [1, 2, 3]:
        part = prediction.loc[prediction.fold.eq(fold)]
        base = reference.loc[reference.fold.eq(fold)]
        left = part.loc[part.seed.eq(part.seed.iloc[0])].sort_values("Date")
        right = base.loc[base.seed.eq(base.seed.iloc[0])].sort_values("Date")
        np.testing.assert_array_equal(left.Date, right.Date)
        np.testing.assert_array_equal(left.y, right.y)
        indices = circular_indices(len(left), seed=20261110 + fold)
        values.append(np.mean([bootstrap_metrics(group.sort_values("Date").y, group.sort_values("Date").p_up, indices) for _, group in part.groupby("seed")], axis=0))
        others.append(np.mean([bootstrap_metrics(group.sort_values("Date").y, group.sort_values("Date").p_up, indices) for _, group in base.groupby("seed")], axis=0))
    return np.mean(values, axis=0), np.mean(others, axis=0)

def success(candidate_id, predictions, reference):
    selected = predictions.loc[predictions.candidate.eq(candidate_id)]
    a, b = comparison_draws(selected, reference)
    low, high = np.nanquantile(a - b, [.025, .975], axis=0)
    fold_losses = selected.groupby(["fold", "seed"]).apply(lambda g: score_binary(g.y, g.p_up)["log_loss"], include_groups=False).groupby("fold").mean()
    base_losses = reference.groupby(["fold", "seed"]).apply(lambda g: score_binary(g.y, g.p_up)["log_loss"], include_groups=False).groupby("fold").mean()
    wins = int(fold_losses.lt(base_losses).sum())
    index = METRICS.index("log_loss")
    return {"passes": bool(wins >= 2 and high[index] < 0), "folds_better": wins, "log_loss_delta": float((fold_losses - base_losses).mean()),
            "low_95": float(low[index]), "high_95": float(high[index])}

def development():
    dev = load_development()
    registry = []
    evaluate_candidate(new_candidate("legacy_mlp", "text"), dev)
    for view in ["price", "availability", "text"]:
        for alpha in [.01, .1, 1.]:
            for dropout in [.2, .5]:
                candidate = new_candidate("small_mlp", view, alpha=alpha, dropout=dropout)
                registry.append(candidate)
                evaluate_candidate(candidate, dev)
        for C in [.01, .1, 1.]:
            candidate = new_candidate("logistic", view, C=C)
            registry.append(candidate)
            evaluate_candidate(candidate, dev)
    configs, predictions, metrics = all_results()
    fold_metrics = metrics.groupby(["candidate", "fold"])[METRICS].mean()
    ranking = fold_metrics.groupby("candidate").mean().sort_values("log_loss")
    if read_json("prepared.json")["center_supported"]:
        neural = [key for key, candidate in configs.items() if candidate["family"] == "small_mlp" and candidate["view"] == "text"]
        best = ranking.loc[neural].sort_values("log_loss").index[0]
        candidate = {**configs[best], "centered": True}
        candidate.pop("id")
        evaluate_candidate(new_candidate(**candidate), dev)
        for C in [.01, .1, 1.]:
            evaluate_candidate(new_candidate("logistic", "text", C=C, centered=True), dev)
    configs, predictions, metrics = all_results()
    ranking = metrics.groupby(["candidate", "fold"])[METRICS].mean().groupby("candidate").mean().sort_values("log_loss")
    eligible = [key for key, candidate in configs.items() if candidate["family"] in {"small_mlp", "logistic"} and candidate["view"] == "text"]
    best_text = ranking.loc[eligible].sort_values("log_loss").index[0]
    prior = pd.read_csv(OUTPUT / "baseline_predictions.csv", parse_dates=["Date", "target_date"]).loc[lambda d: d.candidate.eq("original_prior")]
    gate = success(best_text, predictions, prior)
    dimension_run = False
    if gate["passes"]:
        dimension_run = True
        for family in ["small_mlp", "logistic"]:
            keys = [key for key in eligible if configs[key]["family"] == family]
            best = ranking.loc[keys].sort_values("log_loss").index[0]
            template = {key: value for key, value in configs[best].items() if key != "id"}
            for width in [16, 32]:
                evaluate_candidate(new_candidate(**{**template, "centered": True, "pca": width}), dev)
    write_json("development_complete.json", {"completed_at_wib": now(), "centering_run": read_json("prepared.json")["center_supported"],
               "dimension_branch_run": dimension_run, "dimension_gate": gate,
               "sentiment_branch": "Not selected; PCA is the prespecified optional dimension-reduction branch",
               "delta_text_branch": "Not introduced in this bounded trial; raw recent-window novelty is a separate feature-engineering question",
               "test_evaluated": False})

def report():
    configs, predictions, metrics = all_results()
    base_pred = pd.read_csv(OUTPUT / "baseline_predictions.csv", parse_dates=["Date", "target_date"])
    base_metrics = pd.read_csv(OUTPUT / "baseline_metrics.csv")
    full_metrics = pd.concat([base_metrics, metrics], ignore_index=True)
    fold_metrics = full_metrics.groupby(["candidate", "fold"])[METRICS].mean().reset_index()
    ranking = fold_metrics.groupby("candidate")[METRICS].mean().sort_values("log_loss")
    ranking.to_csv(OUTPUT / "validation_ranking.csv")
    fold_metrics.to_csv(OUTPUT / "validation_fold_metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "validation_predictions.csv", index=False)
    metrics.to_csv(OUTPUT / "validation_seed_metrics.csv", index=False)
    text_keys = [key for key, cfg in configs.items() if cfg["view"] == "text" and cfg["family"] in {"small_mlp", "logistic"}]
    best_text = ranking.loc[text_keys].sort_values("log_loss").index[0]
    price_keys = [key for key, cfg in configs.items() if cfg["view"] == "price"]
    best_price = ranking.loc[price_keys].sort_values("log_loss").index[0]
    available_keys = [key for key, cfg in configs.items() if cfg["view"] == "availability"]
    best_availability = ranking.loc[available_keys].sort_values("log_loss").index[0]
    text_config = {key: value for key, value in configs[best_text].items() if key not in {"id", "pca"}}
    matched = new_candidate(**{**text_config, "view": "price", "centered": False})["id"]
    prior = base_pred.loc[base_pred.candidate.eq("original_prior")]
    comparisons = {"text_minus_prior": success(best_text, predictions, prior),
                   "text_minus_matched_price": success(best_text, predictions, predictions.loc[predictions.candidate.eq(matched)]),
                   "text_minus_tuned_price": success(best_text, predictions, predictions.loc[predictions.candidate.eq(best_price)]),
                   "price_minus_prior": success(best_price, predictions, prior),
                   "availability_minus_prior": success(best_availability, predictions, prior)}
    intervals = []
    for candidate, reference_name in [(best_text, "original_prior"), (best_text, matched), (best_text, best_price), (best_text, best_availability), (best_price, "original_prior")]:
        source = base_pred if reference_name.startswith("original_") else predictions
        a, b = comparison_draws(predictions.loc[predictions.candidate.eq(candidate)], source.loc[source.candidate.eq(reference_name)])
        low, high = np.nanquantile(a - b, [.025, .975], axis=0)
        intervals.extend({"candidate": candidate, "reference": reference_name, "metric": metric, "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
    pd.DataFrame(intervals).to_csv(OUTPUT / "comparison_intervals.csv", index=False)
    for candidate in [best_text, best_price, best_availability]:
        cfg = configs[candidate]
        fits = pd.read_csv(OUTPUT / "candidates" / candidate / "fits.csv")
        epochs = {str(int(seed)): int(value) for seed, value in fits.groupby("seed").best_epoch.median().dropna().items()} if cfg["family"] == "small_mlp" else None
        configs[candidate]["final_median_epochs"] = epochs
    winner = min([best_text, best_price, best_availability], key=lambda name: ranking.loc[name, "log_loss"])
    gate_name = "text_minus_prior" if winner == best_text else "price_minus_prior" if winner == best_price else "availability_minus_prior"
    write_json("frozen_selection.json", {"frozen_at_wib": now(), "protocol_sha256": sha256(OUTPUT / "protocol.json"), "criterion": "Minimum three-fold mean validation log loss",
               "best_text": configs[best_text], "best_price": configs[best_price], "best_availability": configs[best_availability], "matched_price_id": matched,
               "overall_winner": winner, "comparisons": comparisons, "test_eligible": comparisons[gate_name]["passes"], "test_used_for_selection": False})
    null = read_json("permutation_result.json")
    shown = [*CONDITIONS, new_candidate("legacy_mlp", "text")["id"], best_price, best_availability, best_text]
    lines = ["# Level 1 improvement trial", "", "Daily UP/DOWN on the original combined-corpus Level 1 folds. This run evaluates validation only. All original notebooks and result tables are preserved.", "",
             "Train-only diagnostics precede 50 training-label block permutations, smoothed-log-loss checkpoint selection, a six-setting small-MLP regularization grid and a three-setting L2-logistic grid. Three seeds are used for MLPs; logistic regression is deterministic. Conditions are price only, price plus news availability, and price plus FinBERT plus availability.", "",
             "Text centering is enabled by the frozen diagnostic rule. PCA 16/32 is evaluated only if the regularized text candidate passes the frozen prior-comparison gate. Sentiment-head extraction and rolling text-delta features are not selected in this bounded trial.", "",
             "| Validation model | MCC | Macro-F1 | Balanced accuracy | Accuracy | ROC-AUC | Brier | Log loss | ECE-10 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for candidate in dict.fromkeys(shown):
        lines.append("| " + " | ".join([candidate, *[f"{ranking.loc[candidate, metric]:.4f}" for metric in METRICS]]) + " |")
    lines += ["", f"Original text validation MCC: **{null['observed_original_mcc']:.4f}**. Training-label-permuted MCC mean: **{null['null_mean_mcc']:.4f}**, 2.5–97.5% range **{null['null_mcc_2p5']:.4f}–{null['null_mcc_97p5']:.4f}**. Fraction of 50 permutation runs at least as high as observed: **{null['fraction_at_least_observed']:.1%}**.", "",
              "Validation labels remain fixed while only training labels are shuffled in blocks of 20 observed decisions inside each training fold. This is a conditional diagnostic of scores attainable under disrupted training associations, not an exact permutation p-value or proof that selection explains the entire test gap.", "",
              "| Comparison | Mean log-loss difference | 95% lower | 95% upper | Folds better | Passes frozen gate |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, value in comparisons.items():
        lines.append(f"| {name} | {value['log_loss_delta']:.4f} | {value['low_95']:.4f} | {value['high_95']:.4f} | {value['folds_better']}/3 | {value['passes']} |")
    lines += ["", "Negative log-loss differences favor the candidate. The success rule requires improvement in at least two folds and a paired 95% upper bound below zero. Text attribution also requires beating the matched price-only control. All eight agreed metrics are reported, but selection uses log loss only.", "",
              "Bootstrap uses 1,000 circular block resamples of length 10, paired within validation folds and averaged across individual seeds. Intervals are conditional on the selected fitted models and do not account for the candidate search or earlier model/data decisions. Treat all gains as exploratory; no multiplicity-adjusted confirmatory claim is made.", "",
              "The small MLP uses a single 16-unit layer, dropout and stronger L2. It also uses a Torch optimizer/shuffle implementation, so improvements over the original sklearn MLP cannot be assigned solely to one modification. Same-recipe price/text controls are the appropriate text comparison.", "",
              "Non-converged logistic configurations are excluded from selection and recorded in failure.json. The implementation amendment records failure handling and preserves the initial source snapshot; optimizer limits, grid and success criteria remain unchanged.", "",
              f"Final validation selection is frozen in frozen_selection.json. Overall winner: `{winner}`. Test eligibility under the frozen gate: **{comparisons[gate_name]['passes']}**. No test evaluation is performed by this validation report."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    permutation_scores = pd.read_csv(OUTPUT / "permutation_summary.csv")
    ax.hist(permutation_scores.mcc, bins=12, color="#9CA3AF", edgecolor="white")
    ax.axvline(null["observed_original_mcc"], color="#3775BA", linewidth=1.5, label="Original text MCC")
    ax.set(xlabel="Mean selected validation MCC", ylabel="Permutation runs")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "permuted_training_labels")
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for candidate, label, color in [("original_prior", "Prior", "#767676"), (best_price, "Price only", "#3775BA"), (best_text, "Price + FinBERT", "#C17C42")]:
        frame = fold_metrics.loc[fold_metrics.candidate.eq(candidate)]
        ax.plot(frame.fold, frame.log_loss, "o-", color=color, label=label)
    ax.set(xlabel="Validation fold", ylabel="Log loss", xticks=[1, 2, 3])
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "validation_log_loss")
    for relative, expected in effective_hashes().items():
        assert sha256(ROOT / relative) == expected, relative
    artifacts = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    fits = pd.concat([pd.read_csv(path) for path in (OUTPUT / "candidates").glob("*/fits.csv")], ignore_index=True)
    write_json("run_metadata.json", {"completed_at_wib": now(), "input_sha256": effective_hashes(),
               "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts}, "development_rows": len(load_development()),
               "validation_dates": int(predictions.Date.nunique()), "candidate_count": len(configs), "saved_validation_models": len(fits),
               "permutation_fits": null["fits"], "test_evaluated": bool(fits.has_test_evaluation.any()), "test_used_for_fitting_or_stopping": bool(fits.test_used_for_fitting_or_stopping.any()),
               "failed_configurations": len(list((OUTPUT / "candidates").glob("*/failure.json"))),
               "test_rows_evaluated": 0, "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}})
    print(ranking.loc[list(dict.fromkeys(shown))].to_string(), flush=True)
    print(json.dumps(comparisons, indent=2), flush=True)
    print("Results:", OUTPUT.relative_to(ROOT), flush=True)

def save_figure(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "permutation", "development", "report"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        {"prepare": prepare, "permutation": permutation, "development": development, "report": report}[args.phase]()
