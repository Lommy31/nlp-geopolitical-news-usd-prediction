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
from src.task_3.sanity_check.level1_improvement import new_candidate, comparison_draws, success
from src.task_3.sanity_check.level1_improvement_models import SEEDS, fit_legacy
from src.task_3.sanity_check.level2_improvement_models import fit_candidate
from src.task_3.training.controlled_fusion import encode_articles, score_binary
from src.task_3.training.fusion import BlockScaler
from src.task_3.training.testing import METRICS, sha256

OUTPUT = ROOT / "src/task_3/output/model/sanity_check/level2_improvement_v1"
FIGURES = ROOT / "src/task_3/output/figure/sanity_check/level2_improvement_v1"
L1 = ROOT / "src/task_3/output/model/sanity_check/level1_improvement_v1"
BASE = ROOT / "src/task_3/output/model/level_2"
REPRESENTATIONS = {"R1": "01_tfidf_svd", "R2": "02_pretrained_word2vec", "R3": "03_finbert", "R4": "04_roberta_base", "T1": "05_word2vec_text_only"}
ROBER = ROOT / "src/task_3/output/model/sanity_check/magnitude_roberta_v1"

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def read_json(name):
    return json.loads((OUTPUT / name).read_text())

def metadata(rep):
    if rep == "T1":
        rep = "R2"
    return json.loads((BASE / REPRESENTATIONS[rep] / "run_metadata.json").read_text())

def columns(rep):
    meta = metadata(rep)
    market, padded, availability = [meta["feature_blocks"][key] for key in ["market", "text", "availability"]]
    width = 128 if rep == "R1" else 300 if rep in {"R2", "T1"} else 768
    return market, padded[:width], padded, availability

def original_config(rep):
    config = metadata(rep)["model_config"]
    return {**config, "hidden_layer_sizes": tuple(config["hidden_layer_sizes"])}

def load_fold(rep, fold):
    source = "R2" if rep == "T1" else rep
    directory = OUTPUT / "features"
    return [joblib.load(directory / f"{source}_fold_{fold}_{role}.joblib") for role in ["train", "val"]]

def candidate_spec(rep, family, **kwargs):
    candidate = new_candidate(family, "text", **kwargs)
    candidate["local_id"] = candidate["id"]
    candidate["representation"] = rep
    candidate["id"] = rep + "__" + candidate["id"]
    return candidate

def diagnose(rep, fold, train, vectors):
    _, native, _, _ = columns(rep)
    values = train.loc[train.has_news.eq(1), native].to_numpy()
    norms = np.linalg.norm(values, axis=1)
    normalized = values[norms > 0] / norms[norms > 0, None]
    cosine = (np.square(normalized.sum(0)).sum() - len(normalized)) / (len(normalized) * (len(normalized) - 1))
    pc = PCA(n_components=1, svd_solver="full").fit(values)
    within = []
    for items in train.articles:
        if items:
            within.append(np.var(np.stack([vectors[a["article_id"]] for a in items]), axis=0).mean())
    between = values.var(0).mean()
    std = values.std(0)
    ratio = float(np.median(np.abs(values.mean(0)[std > 0]) / std[std > 0]))
    return {"representation": rep, "fold": fold, "train_n": len(train), "native_dimension": len(native), "nonzero_windows": len(normalized),
            "mean_pair_cosine": cosine, "pc1_variance_share": pc.explained_variance_ratio_[0], "between_day_variance": between,
            "within_day_variance": np.mean(within), "between_to_within_variance": between / np.mean(within), "median_abs_mean_over_std": ratio,
            "center_supported": bool(cosine > .95 or ratio > 2)}

def prepare():
    if OUTPUT.exists():
        raise FileExistsError("Use a new version instead of overwriting the experiment.")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "features").mkdir()
    FIGURES.mkdir(parents=True, exist_ok=True)
    assert json.loads((L1 / "verification.json").read_text())["verified"]
    sources = [ROOT / "data/final" / name for name in ["train.csv", "val.csv", "metadata.json"]]
    sources += [ROOT / "src/task_3/improvement_plan.md", Path(__file__), ROOT / "src/task_3/sanity_check/level2_improvement_models.py",
                ROOT / "src/task_3/sanity_check/level1_improvement.py", ROOT / "src/task_3/sanity_check/level1_improvement_models.py"]
    sources += [L1 / name for name in ["protocol.json", "run_metadata.json", "verification.json", "development_features.joblib", "baseline_predictions.csv",
                                     "baseline_metrics.csv", "embedding_diagnostics.csv", "permutation_result.json", "permutation_summary.csv"]]
    sources += [p for p in (L1 / "candidates").rglob("*") if p.is_file()]
    sources += [ROBER / name for name in ["run_metadata.json", "verification.json", "development_features.joblib", "development_article_vectors.npz"]]
    sources += [p for p in (ROOT / "src/task_3/training").glob("*.py")]
    sources += [p for level in ["level_1", "level_2", "level_3"] for p in (ROOT / "src/task_3/training" / level).glob("*") if p.is_file()]
    sources += [BASE / "representation_selection.json"]
    for directory in REPRESENTATIONS.values():
        sources += [BASE / directory / name for name in ["run_metadata.json", "predictions.csv", "summary.csv"]]
    sources += list((BASE / REPRESENTATIONS["R1"]).glob("fold_*_text.joblib"))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    write_json("protocol.json", {"experiment": OUTPUT.name, "frozen_at_wib": now(), "target": "Original daily next-fixing UP/DOWN, unchanged data/final folds",
               "representation_ids": list(REPRESENTATIONS), "fine_tuned_encoders": False,
               "preprocessing": "Reuse the original fold-trained TF-IDF/SVD encoders; frozen Word2Vec/transformer article vectors; all scalers/PCA fit on fold training only",
               "native_dimensions": {"R1": 128, "R2": 300, "R3": 768, "R4": 768, "T1": 300},
               "legacy_selection": "Original 768-wide zero-padded input retained; only selection changes to smoothed log loss",
               "small_models": "Remove all-zero text padding; shared 16-unit head; native text widths can differ across representations",
               "grid": {"alpha": [.01, .1, 1.], "dropout": [.2, .5], "hidden": [16], "logistic_C": [.01, .1, 1.], "logistic_max_iter": 2000,
                        "max_epochs": 60, "patience": 10, "trailing_smoothing_epochs": 5, "min_delta": .001, "threshold": .5},
               "centering_gate": "Training cosine > .95 or median absolute mean/std > 2 in at least two folds; test the best uncentered small-MLP setting centered plus all logistic C settings centered",
               "pca_gate": "If a representation's best regularized model beats prior in at least two folds and paired log-loss CI upper < 0, evaluate PCA16/32 on its best small-MLP and logistic configurations",
               "T1": "Fixed Word2Vec text-only control from the main protocol; prices are zeroed at fit and inference, news count/mask retained",
               "reuse": "R3 FinBERT aliases the verified Level 1 text candidates; price/availability controls alias their identical Level 1 recipes",
               "permutation": {"representation": "R2", "repeats": 50, "block_length": 20, "seeds": SEEDS,
                               "selection": "Original validation MCC, macro-F1, negative log loss; 200 epochs, patience25", "validation_labels": "Fixed; shuffle only training labels within each fold",
                               "interpretation": "Conditional diagnostic, not a formal permutation p-value; existing R3 diagnostic reused"},
               "selection": "Minimum mean validation log loss across three folds; average seed scores within folds", "seeds": SEEDS,
               "success": "Below prior in at least two folds AND paired block-bootstrap upper log-loss difference < 0; text attribution also requires matched and tuned price-only comparisons",
               "final_test_gate": "Only the globally selected fused representation/configuration, after freezing, and only if prior, matched-price and tuned-price gates all pass; T1 is diagnostic and cannot win the fused model selection",
               "bootstrap": {"repeats": 1000, "block_length": 10, "scope": "Paired within validation folds, conditional on selected fitted models"},
               "test_read_in_validation": False, "input_sha256": hashes,
               "limitations": ["Previously accessed test period; all subsequent findings exploratory", "Candidate selection and evaluation share validation; intervals omit selection uncertainty",
                               "No exhaustive search or new sentiment/delta feature branch", "The audit did not prove the sole cause of the validation/test gap"]})
    dev = pd.concat([load_split(ROOT / "data/final", part) for part in ["train", "val"]], ignore_index=True).sort_values("Date").reset_index(drop=True)
    r2, r2_vectors, article_path, _, info, _ = encode_articles(dev, ROOT / "src/task_3/output/model/encoders/word2vec")
    hash_additions = {str(article_path.relative_to(ROOT)): sha256(article_path), str(Path(info["subset_cache"]).relative_to(ROOT)): sha256(info["subset_cache"])}
    finbert = joblib.load(L1 / "development_features.joblib")
    roberta_all = joblib.load(ROBER / "development_features.joblib").set_index("Date").loc[dev.Date]
    r4 = dev.copy()
    _, r4_text, _, _ = columns("R4")
    r4[r4_text] = roberta_all[r4_text].to_numpy()
    r4["log1p_n_berita"] = np.log1p(r4.n_berita)
    with np.load(ROBER / "development_article_vectors.npz", allow_pickle=False) as data:
        r4_vectors = dict(zip(data["article_ids"].tolist(), data["vectors"].astype(float)))
    diagnostics, parity = [], []
    old_predictions = {}
    for rep, directory in REPRESENTATIONS.items():
        old_predictions[rep] = pd.read_csv(BASE / directory / "predictions.csv", parse_dates=["Date", "target_date"])
    for fold in [1, 2, 3]:
        roles = dev.fold_roles.map(lambda value: value[str(fold)])
        raw_train, raw_val = dev.loc[roles.eq("train")], dev.loc[roles.eq("val")]
        encoder = joblib.load(BASE / REPRESENTATIONS["R1"] / f"fold_{fold}_text.joblib")
        ids = sorted({a["article_id"] for items in raw_train.articles for a in items})
        assert encoder.training_ids_ == ids and encoder.training_dates_ == raw_train.Date.dt.strftime("%Y-%m-%d").tolist()
        r1_train, r1_val = encoder.transform(raw_train), encoder.transform(raw_val)
        text_map = {a["article_id"]: a["text_tfidf"] for items in raw_train.articles for a in items}
        r1_matrix = encoder.svd_.transform(encoder.vectorizer_.transform([text_map[key] for key in ids]))
        r1_vectors = dict(zip(ids, r1_matrix))
        for rep in ["R1", "R2", "R3", "R4"]:
            if rep == "R1":
                train, val, vectors = r1_train, r1_val, r1_vectors
            else:
                frame = {"R2": r2, "R3": finbert, "R4": r4}[rep]
                roles = frame.fold_roles.map(lambda value: value[str(fold)])
                train, val = frame.loc[roles.eq("train")].copy(), frame.loc[roles.eq("val")].copy()
                vectors = r2_vectors if rep == "R2" else r4_vectors if rep == "R4" else None
            train, val = train.reset_index(drop=True), val.reset_index(drop=True)
            assert train.target_date.max() < val.Date.min()
            for role, frame in [("train", train), ("val", val)]:
                joblib.dump(frame, OUTPUT / "features" / f"{rep}_fold_{fold}_{role}.joblib", compress=3)
            if rep == "R3":
                diagnostic = pd.read_csv(L1 / "embedding_diagnostics.csv").loc[lambda d: d.fold.eq(fold)].iloc[0].to_dict()
                diagnostic.update(representation=rep, native_dimension=768)
            else:
                diagnostic = diagnose(rep, fold, train, vectors)
            diagnostics.append(diagnostic)
            market, native, padded, availability = columns(rep)
            for seed in SEEDS:
                model = joblib.load(BASE / REPRESENTATIONS[rep] / f"fold_{fold}_seed_{seed}.joblib")
                p = model.predict_proba(val[[*market, *padded, *availability]])[:, 1]
                old = old_predictions[rep].loc[lambda d: d.fold.eq(fold) & d.seed.eq(seed)].sort_values("Date")
                np.testing.assert_array_equal(val.Date, old.Date)
                np.testing.assert_array_equal(val.y, old.y)
                error = float(np.max(np.abs(p - old.p_up)))
                assert error < 1e-12
                parity.append({"representation": rep, "fold": fold, "seed": seed, "maximum_probability_error": error})
    diagnostic = pd.DataFrame(diagnostics)
    diagnostic.to_csv(OUTPUT / "embedding_diagnostics.csv", index=False)
    pd.DataFrame(parity).to_csv(OUTPUT / "original_prediction_parity.csv", index=False)
    write_json("prepared.json", {"completed_at_wib": now(), "protocol_sha256": sha256(OUTPUT / "protocol.json"), "development_rows": len(dev),
               "feature_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (OUTPUT / "features").glob("*.joblib")},
               "additional_input_hashes": hash_additions, "center_supported": {rep: bool(diagnostic.loc[diagnostic.representation.eq(rep), "center_supported"].sum() >= 2) for rep in ["R1", "R2", "R3", "R4"]},
               "test_evaluated": False})
    print(diagnostic[["representation", "fold", "mean_pair_cosine", "median_abs_mean_over_std", "center_supported"]].to_string(index=False), flush=True)

def permutation():
    market, native, padded, availability = columns("R2")
    path = OUTPUT / "permutation_seed_metrics.csv"
    records = pd.read_csv(path).to_dict("records") if path.exists() else []
    for repeat in range(50):
        for fold in [1, 2, 3]:
            train, val = load_fold("R2", fold)
            chunks = [train.y.to_numpy()[start:start + 20] for start in range(0, len(train), 20)]
            order = np.random.default_rng(20261116 + 1000 * repeat + fold).permutation(len(chunks))
            shuffled = np.concatenate([chunks[i] for i in order])
            assert len(shuffled) == len(train) and shuffled.sum() == train.y.sum()
            transform = BlockScaler(market, padded, availability).fit(train)
            scaled = transform.transform(train), transform.transform(val)
            for seed in SEEDS:
                if any(row["repeat"] == repeat and row["fold"] == fold and row["seed"] == seed for row in records):
                    continue
                started = time.perf_counter()
                predictor, best, ran, _ = fit_legacy(train, val, seed, market, padded, availability, original_config("R2"), shuffled_y=shuffled, scaled=scaled)
                records.append({"repeat": repeat, "fold": fold, "seed": seed, "train_n": len(train), "val_n": len(val), "best_epoch": best,
                                "epochs_run": ran, "fit_seconds": time.perf_counter() - started, **score_binary(val.y, predictor.predict_proba(val)[:, 1])})
                pd.DataFrame(records).to_csv(path, index=False)
        write_json("permutation_status.json", {"completed_repeats": repeat + 1, "total_repeats": 50, "updated_at_wib": now()})
        print(f"R2 permutation {repeat + 1}/50: MCC {pd.DataFrame(records).loc[lambda d: d.repeat.eq(repeat), 'mcc'].mean():.4f}", flush=True)
    summary = pd.DataFrame(records).groupby(["repeat", "fold"])[METRICS].mean().groupby("repeat").mean()
    summary.to_csv(OUTPUT / "permutation_summary.csv")
    observed = pd.read_csv(BASE / REPRESENTATIONS["R2"] / "summary.csv").set_index("statistic").loc["mean", "mcc"]
    write_json("permutation_result.json", {"completed_at_wib": now(), "representation": "R2", "fits": len(records), "repeats": len(summary),
               "original_mcc": observed, "permuted_mean_mcc": summary.mcc.mean(), "mcc_2p5": summary.mcc.quantile(.025), "mcc_97p5": summary.mcc.quantile(.975),
               "fraction_at_least_observed": float(summary.mcc.ge(observed).mean()), "interpretation": "Descriptive conditional diagnostic; not a formal permutation p-value"})

def evaluate(candidate):
    directory = OUTPUT / "candidates" / candidate["id"]
    if (directory / "complete.json").exists() or (directory / "failure.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    rep = candidate["representation"]
    market, native, padded, availability = columns(rep)
    predictions, metrics, fits, history = [], [], [], []
    for fold in [1, 2, 3]:
        train, val = load_fold(rep, fold)
        for seed in ([-1] if candidate["family"] == "logistic" else SEEDS):
            started = time.perf_counter()
            try:
                model, best, ran, trace = fit_candidate(train, val, seed, candidate, market, native, padded, availability, original_config(rep))
            except ValueError as error:
                if str(error) != "Logistic regression did not converge.":
                    raise
                (directory / "failure.json").write_text(json.dumps({"candidate": candidate, "fold": fold, "seed": seed, "reason": str(error),
                    "max_iter": 2000, "excluded_from_selection": True, "recorded_at_wib": now()}, indent=2) + "\n")
                print(f"{candidate['id']}: excluded, non-converged fold {fold}", flush=True)
                return
            path = directory / f"fold_{fold}_seed_{seed}.joblib"
            joblib.dump(model, path)
            p = model.predict_proba(val)[:, 1]
            np.testing.assert_array_equal(joblib.load(path).predict_proba(val)[:, 1], p)
            if rep == "T1":
                perturbed = val.copy()
                perturbed[market] = 1e6
                np.testing.assert_array_equal(model.predict_proba(perturbed)[:, 1], p)
            result = val[["Date", "target_date", "y"]].copy()
            result["fold"], result["seed"], result["candidate"] = fold, seed, candidate["id"]
            result["p_up"], result["y_pred"] = p, (p >= .5).astype(int)
            predictions.append(result)
            metrics.append({"candidate": candidate["id"], "fold": fold, "seed": seed, "n": len(val), **score_binary(val.y, p)})
            evaluation_role, fixed_epochs = "development_validation", None
            test_evaluation = bool(evaluation_role == "test")
            fits.append({"candidate": candidate["id"], "fold": fold, "seed": seed, "train_n": len(train), "eval_n": len(val), "best_epoch": best,
                         "epochs_run": ran, "fit_seconds": time.perf_counter() - started, "evaluation_role": evaluation_role, "fixed_epochs": fixed_epochs,
                         "test_used_for_fitting_or_stopping": bool(test_evaluation and fixed_epochs is None), "has_test_evaluation": test_evaluation})
            history.extend({"candidate": candidate["id"], "fold": fold, "seed": seed, **row} for row in trace)
    pd.concat(predictions, ignore_index=True).to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(directory / "seed_metrics.csv", index=False)
    pd.DataFrame(fits).to_csv(directory / "fits.csv", index=False)
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    (directory / "complete.json").write_text(json.dumps({"candidate": candidate, "completed_at_wib": now(), "saved_models": len(fits), "evaluation_phase": "validation"}, indent=2) + "\n")
    scores = pd.DataFrame(metrics).groupby("fold")[METRICS].mean().mean()
    print(f"{candidate['id']}: LL {scores['log_loss']:.4f}, MCC {scores['mcc']:.4f}, AUC {scores['roc_auc']:.4f}", flush=True)

def results(include_reuse=True):
    configs, predictions, metrics = {}, [], []
    for path in sorted((OUTPUT / "candidates").glob("*/complete.json")):
        cfg = json.loads(path.read_text())["candidate"]
        configs[cfg["id"]] = cfg
        predictions.append(pd.read_csv(path.parent / "predictions.csv", parse_dates=["Date", "target_date"]))
        metrics.append(pd.read_csv(path.parent / "seed_metrics.csv"))
    if include_reuse:
        for path in sorted((L1 / "candidates").glob("*/complete.json")):
            cfg = json.loads(path.read_text())["candidate"]
            if cfg["view"] != "text":
                continue
            new_id = "R3__" + cfg["id"]
            configs[new_id] = {**cfg, "representation": "R3", "local_id": cfg["id"], "id": new_id, "reuse_source": str(path.parent.relative_to(ROOT))}
            predictions.append(pd.read_csv(path.parent / "predictions.csv", parse_dates=["Date", "target_date"]).assign(candidate=new_id))
            metrics.append(pd.read_csv(path.parent / "seed_metrics.csv").assign(candidate=new_id))
    return configs, pd.concat(predictions, ignore_index=True), pd.concat(metrics, ignore_index=True)

def development():
    for rep in ["R2", "R1", "R4", "T1"]:
        evaluate(candidate_spec(rep, "legacy_mlp"))
        for alpha in [.01, .1, 1.]:
            for dropout in [.2, .5]:
                evaluate(candidate_spec(rep, "small_mlp", alpha=alpha, dropout=dropout))
        for C in [.01, .1, 1.]:
            evaluate(candidate_spec(rep, "logistic", C=C))
        configs, predictions, metrics = results(False)
        ranking = metrics.groupby(["candidate", "fold"])[METRICS].mean().groupby("candidate").mean().sort_values("log_loss")
        support_rep = "R2" if rep == "T1" else rep
        if read_json("prepared.json")["center_supported"][support_rep]:
            keys = [key for key, cfg in configs.items() if cfg["representation"] == rep and cfg["family"] == "small_mlp"]
            chosen = ranking.loc[keys].sort_values("log_loss").index[0]
            cfg = configs[chosen]
            evaluate(candidate_spec(rep, "small_mlp", alpha=cfg["alpha"], dropout=cfg["dropout"], centered=True))
            for C in [.01, .1, 1.]:
                evaluate(candidate_spec(rep, "logistic", C=C, centered=True))
        configs, predictions, metrics = results(False)
        ranking = metrics.groupby(["candidate", "fold"])[METRICS].mean().groupby("candidate").mean().sort_values("log_loss")
        keys = [key for key, cfg in configs.items() if cfg["representation"] == rep and cfg["family"] in {"small_mlp", "logistic"}]
        best = ranking.loc[keys].sort_values("log_loss").index[0]
        prior = pd.read_csv(L1 / "baseline_predictions.csv", parse_dates=["Date", "target_date"]).loc[lambda d: d.candidate.eq("original_prior")]
        gate = success(best, predictions, prior)
        pca_run = gate["passes"]
        if pca_run:
            for family in ["small_mlp", "logistic"]:
                selected = [key for key in keys if configs[key]["family"] == family]
                chosen = ranking.loc[selected].sort_values("log_loss").index[0]
                cfg = configs[chosen]
                for width in [16, 32]:
                    kwargs = {key: cfg[key] for key in ["alpha", "dropout", "C"] if key in cfg}
                    evaluate(candidate_spec(rep, family, centered=True, pca=width, **kwargs))
        write_json(f"{rep}_development_complete.json", {"completed_at_wib": now(), "representation": rep, "pre_pca_winner": best,
                   "pca_run": pca_run, "pca_gate": gate, "test_evaluated": False})
    write_json("development_complete.json", {"completed_at_wib": now(), "R3_reused": True, "test_evaluated": False})

def report():
    configs, predictions, metrics = results()
    baseline_pred = pd.read_csv(L1 / "baseline_predictions.csv", parse_dates=["Date", "target_date"])
    baseline_metrics = pd.read_csv(L1 / "baseline_metrics.csv")
    controls_pred, controls_metrics, control_configs = [], [], {}
    for path in sorted((L1 / "candidates").glob("*/complete.json")):
        cfg = json.loads(path.read_text())["candidate"]
        if cfg["view"] not in {"price", "availability"}:
            continue
        control_configs[cfg["id"]] = cfg
        controls_pred.append(pd.read_csv(path.parent / "predictions.csv", parse_dates=["Date", "target_date"]))
        controls_metrics.append(pd.read_csv(path.parent / "seed_metrics.csv"))
    controls_pred, controls_metrics = pd.concat(controls_pred, ignore_index=True), pd.concat(controls_metrics, ignore_index=True)
    original_pred, original_metrics = [], []
    for rep, directory in REPRESENTATIONS.items():
        frame = pd.read_csv(BASE / directory / "predictions.csv", parse_dates=["Date", "target_date"])
        frame["candidate"] = "original_" + rep
        original_pred.append(frame[["Date", "target_date", "y", "fold", "seed", "candidate", "p_up", "y_pred"]])
        for (fold, seed), stream in frame.groupby(["fold", "seed"]):
            original_metrics.append({"candidate": "original_" + rep, "fold": fold, "seed": seed, "n": len(stream), **score_binary(stream.y, stream.p_up)})
    full_pred = pd.concat([predictions, controls_pred, baseline_pred, *original_pred], ignore_index=True)
    full_metrics = pd.concat([metrics, controls_metrics, baseline_metrics, pd.DataFrame(original_metrics)], ignore_index=True)
    folded = full_metrics.groupby(["candidate", "fold"])[METRICS].mean()
    ranking = folded.groupby("candidate").mean().sort_values("log_loss")
    winners, comparisons, intervals, selected_rows = {}, {}, [], []
    prior = full_pred.loc[full_pred.candidate.eq("original_prior")]
    price_ids = [key for key, cfg in control_configs.items() if cfg["view"] == "price"]
    best_price = ranking.loc[price_ids].sort_values("log_loss").index[0]
    for rep in REPRESENTATIONS:
        eligible = [key for key, cfg in configs.items() if cfg["representation"] == rep and cfg["family"] in {"small_mlp", "logistic"}]
        winner = ranking.loc[eligible].sort_values("log_loss").index[0]
        cfg = configs[winner]
        matched_kwargs = {key: cfg[key] for key in ["alpha", "dropout", "C"] if key in cfg}
        matched = new_candidate(cfg["family"], "price", **matched_kwargs)["id"]
        source = ROOT / cfg["reuse_source"] if "reuse_source" in cfg else OUTPUT / "candidates" / winner
        fits = pd.read_csv(source / "fits.csv")
        epochs = {str(int(seed)): int(v) for seed, v in fits.groupby("seed").best_epoch.median().dropna().items()} if cfg["family"] == "small_mlp" else None
        winners[rep] = {**cfg, "final_median_epochs": epochs, "matched_price": matched}
        checks = {"prior": success(winner, full_pred, prior), "matched_price": success(winner, full_pred, full_pred.loc[full_pred.candidate.eq(matched)]),
                  "tuned_price": success(winner, full_pred, full_pred.loc[full_pred.candidate.eq(best_price)])}
        comparisons[rep] = checks
        for reference in ["original_prior", matched, best_price, "original_" + rep]:
            a, b = comparison_draws(full_pred.loc[full_pred.candidate.eq(winner)], full_pred.loc[full_pred.candidate.eq(reference)])
            low, high = np.nanquantile(a - b, [.025, .975], axis=0)
            intervals.extend({"representation": rep, "candidate": winner, "reference": reference, "metric": metric,
                              "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
        selected_rows.append({"representation": rep, "candidate": winner, **ranking.loc[winner].to_dict(),
                              "prior_gate": checks["prior"]["passes"], "matched_price_gate": checks["matched_price"]["passes"], "tuned_price_gate": checks["tuned_price"]["passes"]})
    fused = [winners[rep]["id"] for rep in ["R1", "R2", "R3", "R4"]]
    global_winner = ranking.loc[fused].sort_values("log_loss").index[0]
    winner_rep = configs[global_winner]["representation"]
    test_eligible = all(check["passes"] for check in comparisons[winner_rep].values())
    write_json("frozen_selection.json", {"frozen_at_wib": now(), "protocol_sha256": sha256(OUTPUT / "protocol.json"),
               "criterion": "Mean three-fold validation log loss", "winners": winners, "comparisons": comparisons, "best_price": best_price,
               "global_winner": global_winner, "global_winner_representation": winner_rep, "test_eligible": test_eligible, "test_used_for_selection": False})
    for name, frame in {"validation_predictions": full_pred, "validation_seed_metrics": full_metrics, "validation_fold_metrics": folded.reset_index(),
                        "validation_ranking": ranking.reset_index(), "selected_representations": pd.DataFrame(selected_rows), "comparison_intervals": pd.DataFrame(intervals)}.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    null = read_json("permutation_result.json")
    shown = ["original_prior", "original_last_direction", best_price]
    for rep in REPRESENTATIONS:
        shown += ["original_" + rep, winners[rep]["id"]]
    lines = ["# Level 2 improvement trial", "", "Daily next-fixing UP/DOWN with the original combined-corpus splits. Hyperparameters and checkpoint selection use validation only. Main notebooks, original results and representation selection are preserved.", "",
             "R1 reuses encoders fitted on each fold's training articles only. R2 uses frozen Word2Vec. R3 aliases the verified Level 1 FinBERT trial and is not independent evidence. R4 reuses frozen RoBERTa. T1 is the fixed Word2Vec text-only control with the market block zeroed at training and inference.", "",
             "Smoothed log loss, six small-MLP settings, three L2-logistic settings, conditional centering and conditional PCA16/32 follow protocol.json. All-zero padding is removed for the new compact classifiers; legacy selection-only comparisons retain the original 768-wide slots. Non-converged solver configurations are recorded and excluded without expanding the grid.", "",
             "| Validation model | MCC | Macro-F1 | Balanced accuracy | Accuracy | ROC-AUC | Brier | Log loss | ECE-10 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for candidate in shown:
        lines.append("| " + " | ".join([candidate, *[f"{ranking.loc[candidate, metric]:.4f}" for metric in METRICS]]) + " |")
    lines += ["", "| Representation | Versus | Mean log-loss difference | 95% lower | 95% upper | Folds better | Passes |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for rep, checks in comparisons.items():
        for name, check in checks.items():
            lines.append(f"| {rep} | {name} | {check['log_loss_delta']:.4f} | {check['low_95']:.4f} | {check['high_95']:.4f} | {check['folds_better']}/3 | {check['passes']} |")
    lines += ["", f"R2 original MCC: **{null['original_mcc']:.4f}**. Fifty training-label block-permutation runs: mean MCC **{null['permuted_mean_mcc']:.4f}**, 2.5–97.5% range **{null['mcc_2p5']:.4f}–{null['mcc_97p5']:.4f}**; fraction at least as high as observed **{null['fraction_at_least_observed']:.1%}**.", "",
              "Only training labels are shuffled, in blocks of 20 decisions inside each training fold; validation labels stay fixed. This is a conditional reference, not a formal permutation p-value or a causal explanation for the entire validation/test gap. R3's earlier permutation results are reused; this run does not provide new null distributions for R1 or R4.", "",
              "Selection averages seed metrics within folds and folds equally. MLPs use three seeds; logistic regression is deterministic. Paired bootstrap intervals use 1,000 circular block resamples of length 10 within folds, conditional on fitted selected models. Candidate-search uncertainty is omitted; all findings remain exploratory.", "",
              f"Global fused-model winner: `{global_winner}`. Final-test eligibility: **{test_eligible}**. Eligibility requires the prior, matched-price and tuned-price gates all to pass. No final test is evaluated by this report.", "",
              "Existing news-timestamp and historical-body limitations remain. No new sentiment-head extraction or rolling text-delta features are added. The original test period was already accessed in earlier experiments."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    chosen = pd.DataFrame(selected_rows).set_index("representation")
    ax.plot(list(REPRESENTATIONS), [ranking.loc['original_' + rep, 'log_loss'] for rep in REPRESENTATIONS], "o--", color="#767676", label="Original")
    ax.plot(chosen.index, chosen.log_loss, "o-", color="#3775BA", label="Improved selection")
    ax.axhline(ranking.loc["original_prior", "log_loss"], color="black", linestyle=":", linewidth=1, label="Prior")
    ax.set(xlabel="Representation", ylabel="Validation log loss")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "representation_log_loss")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    draws = pd.read_csv(OUTPUT / "permutation_summary.csv")
    ax.hist(draws.mcc, bins=12, color="#9CA3AF", edgecolor="white")
    ax.axvline(null["original_mcc"], color="#3775BA", label="Original R2 MCC")
    ax.set(xlabel="Selected validation MCC", ylabel="Permutation runs")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "r2_permuted_training_labels")
    hashes = {**read_json("protocol.json")["input_sha256"], **read_json("prepared.json")["additional_input_hashes"]}
    for relative, expected in hashes.items():
        assert sha256(ROOT / relative) == expected, relative
    fits = pd.concat([pd.read_csv(path) for path in (OUTPUT / "candidates").glob("*/fits.csv")], ignore_index=True)
    artifacts = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "run_metadata.json"] + list(FIGURES.glob("*"))
    write_json("run_metadata.json", {"completed_at_wib": now(), "input_sha256": hashes,
               "artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in artifacts}, "development_rows": read_json("prepared.json")["development_rows"],
               "validation_dates": int(predictions.Date.nunique()), "new_successful_candidates": len(list((OUTPUT / "candidates").glob("*/complete.json"))),
               "new_saved_validation_models": len(fits), "R3_reused_candidates": sum("reuse_source" in cfg for cfg in configs.values()), "R2_permutation_fits": null["fits"],
               "failed_configurations": len(list((OUTPUT / "candidates").glob("*/failure.json"))), "test_evaluated": bool(fits.has_test_evaluation.any()),
               "test_used_for_fitting_or_stopping": bool(fits.test_used_for_fitting_or_stopping.any()), "test_rows_evaluated": 0,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}})
    print(pd.DataFrame(selected_rows).to_string(index=False), flush=True)
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
