from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
sys.path.insert(0, str(ROOT))
from src.task_3.sanity_check.level3_vader_models import VADER, VaderFusion, fit_vader
from src.task_3.sanity_check.level3_improvement_models import ImprovedFusion
from src.task_3.training.controlled_fusion import make_batch, score_binary
from src.task_3.training.testing import METRICS, bootstrap_metrics, circular_indices, sha256

BASE = ROOT / "src/task_3/output/model/sanity_check/level3_improvement_v1"
OUT = ROOT / "src/task_3/output/model/sanity_check/level3_vader_v1"
FIG = ROOT / "src/task_3/output/figure/sanity_check/level3_vader_v1"
SEEDS = [42, 123, 2026]
CONDITIONS = {"A111_vader": [1, 1, 1], "A000_vader": [0, 0, 0]}
MARKET = json.loads((ROOT / "data/final/metadata.json").read_text())["features"]["market"]
AVAILABILITY = ["log1p_n_berita", "has_news"]

def now():
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def vectors(phase):
    source = BASE / ("prepared.json" if phase == "development" else "test_encoder.json")
    info = json.loads(source.read_text())
    path = ROOT / info["article_cache"]
    assert sha256(path) == info["article_cache_sha256"]
    with np.load(path, allow_pickle=False) as data:
        return dict(zip(data["article_ids"].tolist(), data["vectors"]))

def add_vader(frame, phase, analyzer):
    rows, lookup = [], {}
    total = int(frame.n_berita.sum())
    for row in frame.itertuples():
        for article in row.articles:
            title, separator, lead = article["text_transformer"].partition("\n\n")
            assert separator and title == article["title"] and len(lead.split()) <= 180
            article_id = article["article_id"]
            assert article_id not in lookup and pd.Timestamp(article["source_date"]) < row.Date
            scores = [analyzer.polarity_scores(text) for text in [title, lead]]
            values = [score[key] for score in scores for key in ["compound", "pos", "neg", "neu"]]
            lookup[article_id] = values
            rows.append({"article_id": article_id, "Date": row.Date, "source_date": article["source_date"], **dict(zip(VADER, values))})
            if len(rows) % 1000 == 0:
                print(f"VADER {phase}: {len(rows)}/{total} articles", flush=True)
    means = [np.mean([lookup[a["article_id"]] for a in items], axis=0) if items else np.zeros(8) for items in frame.articles]
    result = frame.copy()
    result[VADER] = np.asarray(means)
    assert np.isfinite(result[VADER]).all().all() and not result.loc[result.has_news.eq(0), VADER].to_numpy().any()
    pd.DataFrame(rows).to_csv(OUT / f"{phase}_article_scores.csv", index=False)
    result[["Date", "n_berita", "has_news", *VADER]].to_csv(OUT / f"{phase}_window_scores.csv", index=False)
    joblib.dump(result, OUT / f"{phase}_features.joblib", compress=3)
    print(f"VADER {phase}: completed {len(rows)} articles", flush=True)
    return result

def evaluate(model, frame, lookup, condition, seed, fold=None):
    p = model.predict_proba(frame, lookup)[:, 1]
    rows = frame[["Date", "target_date", "y", "n_berita", "has_news"]].copy()
    rows["condition"], rows["seed"], rows["fold"] = condition, seed, fold
    rows["p_up"], rows["y_pred"] = p, (p >= .5).astype(int)
    return rows, {"condition": condition, "seed": seed, "fold": fold, **score_binary(frame.y, p)}

def preflight(frame, lookup):
    sample = frame.iloc[:3].copy()
    sample["articles"] = [[], [], []]
    sample[["n_berita", "has_news", "log1p_n_berita", *VADER]] = 0
    from src.task_3.training.controlled_fusion import WORD_FEATURES
    sample[WORD_FEATURES] = 0
    from src.task_3.sanity_check.level3_vader_models import VaderScaler
    scaler = VaderScaler(MARKET, AVAILABILITY, False).fit(frame)
    prepared = scaler.prepare(sample, lookup)
    batch = make_batch(prepared, np.arange(3))
    for flags in CONDITIONS.values():
        model, reference = VaderFusion(42, flags).eval(), ImprovedFusion(42, flags).eval()
        for key, value in reference.state_dict().items():
            assert torch.equal(value, model.state_dict()[key]), key
        with torch.no_grad():
            p = model(*batch)
            original = reference(*batch[:3], batch[3][:, :2])
            np.testing.assert_allclose(p.numpy(), original.numpy(), atol=1e-14, rtol=0)
        assert sum(v.numel() for v in model.parameters()) == (19633 if flags[0] else 14817)
    save_json("preflight.json", {"verified": True, "shared_initialization_identical": True, "zero_vader_logits_match": True, "additional_weights": 128})

def validation(dev, lookup):
    all_predictions, all_metrics, fits, histories = [], [], [], []
    for condition, flags in CONDITIONS.items():
        directory = OUT / condition
        directory.mkdir()
        for fold in [1, 2, 3]:
            roles = dev.fold_roles.map(lambda value: value[str(fold)])
            train, val = dev.loc[roles.eq("train")], dev.loc[roles.eq("val")]
            assert train.target_date.max() < val.Date.min()
            for seed in SEEDS:
                model, best, ran, trace, seconds = fit_vader(train, val, lookup, seed, flags, MARKET, AVAILABILITY)
                path = directory / f"fold_{fold}_seed_{seed}.joblib"
                joblib.dump(model, path)
                predictions, metrics = evaluate(model, val, lookup, condition, seed, fold)
                np.testing.assert_array_equal(joblib.load(path).predict_proba(val, lookup)[:, 1], predictions.p_up)
                all_predictions.append(predictions)
                all_metrics.append(metrics)
                fits.append({"condition": condition, "fold": fold, "seed": seed, "best_epoch": best, "epochs_run": ran,
                             "train_n": len(train), "val_n": len(val), "fit_seconds": seconds})
                histories.extend({"condition": condition, "fold": fold, "seed": seed, **row} for row in trace)
                for name, table in {"validation_predictions": pd.concat(all_predictions), "validation_metrics": pd.DataFrame(all_metrics),
                                    "validation_fits": pd.DataFrame(fits), "validation_history": pd.DataFrame(histories)}.items():
                    table.to_csv(OUT / f"{name}.csv", index=False)
                print(f"{condition} fold{fold} seed{seed}: epoch{best}, MCC{metrics['mcc']:.4f}, AUC{metrics['roc_auc']:.4f}, LL{metrics['log_loss']:.4f}, {seconds:.1f}s", flush=True)
    epochs = {condition: {str(int(seed)): int(v) for seed, v in pd.DataFrame(fits).loc[lambda d: d.condition.eq(condition)].groupby("seed").best_epoch.median().items()} for condition in CONDITIONS}
    save_json("frozen_final_config.json", {"frozen_at_wib": now(), "epochs": epochs, "threshold": .5, "test_used_for_selection": False,
              "validation_fits_sha256": sha256(OUT / "validation_fits.csv"), "conditions": CONDITIONS})
    return epochs

def final_test(dev, lookup, epochs, analyzer):
    models, fits, histories = {}, [], []
    directory = OUT / "final_test"
    directory.mkdir()
    for condition, flags in CONDITIONS.items():
        for seed in SEEDS:
            duration = epochs[condition][str(seed)]
            model, best, ran, trace, seconds = fit_vader(dev, None, lookup, seed, flags, MARKET, AVAILABILITY, fixed_epochs=duration)
            joblib.dump(model, directory / f"{condition}_seed_{seed}.joblib")
            models[condition, seed] = model
            fits.append({"condition": condition, "seed": seed, "train_n": len(dev), "epochs": duration, "epochs_run": ran, "test_used_for_fitting_or_stopping": False})
            histories.extend({"condition": condition, "seed": seed, **row} for row in trace)
            print(f"Refit {condition} seed{seed}: {ran} epochs", flush=True)
    pd.DataFrame(fits).to_csv(directory / "fits.csv", index=False)
    pd.DataFrame(histories).to_csv(directory / "history.csv", index=False)
    save_json("test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(OUT / "frozen_final_config.json"), "final_fits_sha256": sha256(directory / "fits.csv")})
    test = add_vader(joblib.load(BASE / "test_features.joblib"), "test", analyzer)
    assert dev.target_date.max() < test.Date.min()
    test_lookup = vectors("test")
    predictions, metrics = [], []
    for (condition, seed), model in models.items():
        p, m = evaluate(model, test, test_lookup, condition, seed)
        predictions.append(p)
        metrics.append(m)
    reference = pd.read_csv(BASE / "final_test/predictions.csv", parse_dates=["Date", "target_date"])
    reference = reference.loc[reference.condition.isin(["A111_improved", "A000_improved", "training_prior", "price_matched", "price_tuned", "last_direction"])]
    for (condition, seed), rows in reference.groupby(["condition", "seed"]):
        rows = rows.sort_values("Date").reset_index(drop=True)
        pd.testing.assert_frame_equal(rows[["Date", "target_date", "y"]], test[["Date", "target_date", "y"]])
        predictions.append(rows)
        metrics.append({"condition": condition, "seed": seed, **score_binary(rows.y, rows.p_up)})
    pd.concat(predictions, ignore_index=True).to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(directory / "seed_metrics.csv", index=False)

def report():
    metrics = pd.read_csv(OUT / "final_test/seed_metrics.csv")
    summary = metrics.groupby("condition")[METRICS].mean()
    summary.to_csv(OUT / "test_summary.csv")
    validation = pd.read_csv(OUT / "validation_metrics.csv").groupby(["condition", "fold"])[METRICS].mean().groupby("condition").mean()
    validation.to_csv(OUT / "validation_summary.csv")
    predictions = pd.read_csv(OUT / "final_test/predictions.csv")
    indices = circular_indices(predictions.Date.nunique(), repeats=1000, block_length=10, seed=20261110)
    draws = {name: np.mean([bootstrap_metrics(part.sort_values("Date").y, part.sort_values("Date").p_up, indices) for _, part in rows.groupby("seed")], axis=0) for name, rows in predictions.groupby("condition")}
    comparisons = [("A111_vader", "A111_improved"), ("A000_vader", "A000_improved"), ("A111_vader", "A000_vader"), ("A111_vader", "training_prior"), ("A111_vader", "price_matched")]
    intervals = []
    for candidate, reference in comparisons:
        name = f"{candidate}_minus_{reference}"
        draws[name] = draws[candidate] - draws[reference]
        low, high = np.nanquantile(draws[name], [.025, .975], axis=0)
        intervals.extend({"comparison": name, "metric": metric, "delta": summary.loc[candidate, metric] - summary.loc[reference, metric], "low_95": low[i], "high_95": high[i]} for i, metric in enumerate(METRICS))
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    np.savez_compressed(OUT / "bootstrap.npz", metrics=np.asarray(METRICS), **draws)
    lines = ["# VADER addition to improved Level 3", "", "Daily UP/DOWN on the combined original + CNBC corpus. A111 retains the frozen Word2Vec query, cross-attention and gate; A000 retains mean pooling. VADER adds eight auxiliary classifier inputs, outside the text gate: compound/positive/negative/neutral scores for the existing cleaned title and 180-word lead, averaged across articles in each original fixing window.", "",
             "Shared weights have identical initialization to the no-VADER controls; 128 input weights are added. Training remains hidden16, alpha1, dropout0.5, float64 Adam and the same three seeds/folds. No centering or hyperparameter search is repeated. VADER scales fit only news-present training windows; empty windows remain zero. Polarity is a feature, not an assumed mapping to currency direction.", "",
             "Checkpoints use trailing-five-epoch validation log loss, min_delta0.001, max60 and patience10. Final refits use all 930 development rows for each seed's median selected epoch, then score the same 235 test dates at threshold0.5.", "",
             "| Test model | MCC | Macro-F1 | Balanced accuracy | Accuracy | ROC-AUC | Brier | Log loss | ECE-10 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition, row in summary.iterrows():
        lines.append("| " + " | ".join([condition, *[f"{row[metric]:.4f}" for metric in METRICS]]) + " |")
    lines += ["", "| Paired comparison | Metric | Change | 95% lower | 95% upper |", "| --- | --- | ---: | ---: | ---: |"]
    for row in intervals:
        if row["metric"] in ["mcc", "macro_f1", "roc_auc", "log_loss"]:
            lines.append(f"| {row['comparison']} | {row['metric']} | {row['delta']:.4f} | {row['low_95']:.4f} | {row['high_95']:.4f} |")
    lines += ["", "Scores average individual seeds, without ensembling. Paired intervals use 1,000 circular block resamples of length10, conditional on fitted models; selection uncertainty is omitted. The test period was already accessed, so this is exploratory. Existing no-VADER predictions are reused on identical dates. No test-driven selection or threshold tuning is performed. Main notebooks, exported data and earlier results remain unchanged."]
    (OUT / "result.md").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.1))
    for ax, name, title in zip(axes, ["A111_improved", "A111_vader"], ["A111", "A111 + VADER"]):
        matrix = np.mean([confusion_matrix(rows.y, rows.y_pred, labels=[0, 1]) for _, rows in predictions.loc[predictions.condition.eq(name)].groupby("seed")], axis=0)
        shares = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i,j]:.1f}\n({shares[i,j]:.0%})", ha="center", va="center", color="white" if shares[i,j] > .6 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"], xlabel="Predicted", title=title)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIG / f"test_confusion_matrices.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, expected in protocol["input_sha256"].items():
        assert sha256(ROOT / path) == expected, path
    artifacts = [p for p in OUT.rglob("*") if p.is_file()] + list(FIG.glob("*"))
    save_json("run_metadata.json", {"completed_at_wib": now(), "input_sha256": protocol["input_sha256"], "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in artifacts},
              "validation_models": 18, "final_models": 6, "refit_rows": 930, "test_rows": 235, "test_used_for_selection": False})
    print(summary.to_string(), flush=True)

def main():
    if OUT.exists():
        raise FileExistsError("Use a new version instead of overwriting the experiment.")
    OUT.mkdir(parents=True)
    FIG.mkdir(parents=True)
    assert json.loads((BASE / "verification.json").read_text())["verified"]
    analyzer = SentimentIntensityAnalyzer()
    sources = [Path(__file__), ROOT / "src/task_3/sanity_check/level3_vader_models.py", ROOT / "src/task_3/sanity_check/level3_improvement_models.py", ROOT / "src/task_3/sanity_check/level1_improvement_models.py", ROOT / "src/task_3/training/controlled_fusion.py", ROOT / "src/task_3/training/testing.py"]
    sources += [BASE / name for name in ["run_metadata.json", "verification.json", "protocol.json", "prepared.json", "test_encoder.json", "development_features.joblib", "test_features.joblib", "final_test/predictions.csv"]]
    sources += [ROOT / "data/final" / name for name in ["metadata.json", "train.csv", "val.csv", "test.csv"]]
    save_json("protocol.json", {"frozen_at_wib": now(), "conditions": CONDITIONS, "seeds": SEEDS, "vader_features": VADER,
              "recipe": {"hidden": [16], "alpha": 1., "dropout": .5, "centering": False, "max_epochs": 60, "patience": 10, "checkpoint": "Trailing five-epoch validation LL; min_delta .001", "refit": "Per-seed median epoch across original three folds", "threshold": .5},
              "addition": "128 auxiliary classifier weights; shared initialization fixed; VADER inputs are outside the text gate", "aggregation": "Title and existing 180-word lead scored separately; article means per fixing window; empty windows zero; uncentered train-only news-present scaling",
              "test": "User-requested exploratory comparison on already accessed period; no test-based choices", "lexicon_sha256": hashlib.sha256(analyzer.lexicon_file.encode()).hexdigest(), "nltk_version": nltk.__version__,
              "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources}})
    dev = add_vader(joblib.load(BASE / "development_features.joblib"), "development", analyzer)
    lookup = vectors("development")
    preflight(dev, lookup)
    epochs = validation(dev, lookup)
    final_test(dev, lookup, epochs, analyzer)
    report()

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        main()
