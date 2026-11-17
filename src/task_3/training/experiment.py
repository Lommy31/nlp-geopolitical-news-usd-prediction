from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier

from src.task_3.training.attention import fit_fusion
from src.task_3.training.baselines import LastDirectionClassifier
from src.task_3.training.data import ROOT, OUTPUT, MARKET, AVAILABILITY, WIDTHS, raw_data, role_data, fold_data, encoded, final_data, text_columns
from src.task_3.training.evaluation import now, write_json, scored, write_phase, paired_comparison, plot_results
from src.task_3.training.models import SEEDS, fit_small, fit_logistic, TextOnlyPredictor
from src.task_3.training.testing import METRICS, sha256
from src.task_3.training.controlled_fusion import score_binary

FIGURES = ROOT / "src/task_3/output/figure/improved_v1"
STUDIES = {1: {"L1-a": "01_training_majority", "L1-b": "02_last_direction", "L1-c": "03_market_only", "L1-d": "04_market_news_concat", "L1-e": "05_market_news_availability"},
           2: {"R1": "01_tfidf_svd", "R2": "02_pretrained_word2vec", "R3": "03_finbert", "R4": "04_roberta_base", "T1": "05_text_only"},
           3: {"A000": "01_a000_mean_concat", "A100": "02_a100_task_query_pooling", "A010": "03_a010_cross_attention", "A001": "04_a001_learned_gate", "A111": "05_a111_all_components"}}

def read_json(path):
    return json.loads(Path(path).read_text())

def notebook_code_hash(path):
    import hashlib
    content = read_json(path)
    code = "\n\n".join("".join(cell["source"]) for cell in content["cells"] if cell["cell_type"] == "code")
    return hashlib.sha256(code.encode()).hexdigest()

def prepare_run():
    path = OUTPUT / "protocol.json"
    if path.exists():
        return check_inputs()
    source = ROOT / "src/task_3/output/model/sanity_check"
    l1, l2 = source / "level1_improvement_v1", source / "level2_improvement_v1"
    for directory in [l1, l2]:
        assert read_json(directory / "verification.json")["verified"]
    selected1, selected2 = read_json(l1 / "frozen_selection.json"), read_json(l2 / "frozen_selection.json")
    def config(value):
        return {key: value[key] for key in ["family", "view", "centered", "alpha", "dropout", "pca"] if key in value}
    text_config = config(selected1["best_text"])
    configs = {"L1-c": {**text_config, "view": "price"}, "L1-d": text_config, "L1-e": {**text_config, "view": "availability"},
               "price_tuned": config(selected1["best_price"]), **{rep: config(selected2["winners"][rep]) for rep in ["R1", "R2", "R3", "R4"]}}
    inputs = [ROOT / "data/final" / name for name in ["metadata.json", "train.csv", "val.csv", "test.csv"]]
    inputs += [ROOT / "src/task_3/improvement_plan.md", OUTPUT / "legacy_training/manifest.json"]
    inputs += list((ROOT / "src/task_3/training").glob("*.py"))
    inputs += [ROOT / "src/task_3/training/PROTOCOL.md"]
    inputs += [d / name for d in [l1, l2] for name in ["frozen_selection.json", "run_metadata.json", "verification.json", "permutation_result.json", "embedding_diagnostics.csv"]]
    write_json(OUTPUT / "configurations.json", {"configs": configs, "source_validation_selections": [str((d / "frozen_selection.json").relative_to(ROOT)) for d in [l1, l2]],
               "selection_basis": "Previously verified three-fold log-loss search; no new hyperparameter expansion and no test-driven choices",
               "level1_control": "One shared alpha/dropout configuration for L1-c/d/e; independently validation-tuned price is an additional reference",
               "level2_control": "Same 16-unit concat head and prior equal search budgets; each representation retains its validation-selected regularization",
               "conditional_branches": "No prior success gate passed. Do not expand to PCA, scalar sentiment or rolling-delta features."})
    inputs.append(OUTPUT / "configurations.json")
    write_json(path, {"created_at_wib": now(), "version": "improved_v1", "target": "Daily next-fixing UP/DOWN; original data/final rows and expanding folds", "vader": False,
               "selection": "Minimum equal-fold mean validation log loss; three individual seeds; smooth checkpoint over five epochs; min_delta .001; max60/patience10",
               "level3": "Five main variants; choose eligible representation from updated Level 2; inherit its head regularization without a new search",
               "final_test": "Freeze one selected NLP model after all validation runs, then evaluate it and prespecified prior/persistence/matched-price/tuned-price/availability controls",
               "test_history": "Repeated previous test access; exploratory only. Previous comparisons are not independent new holdouts.", "multiplicity": "All contrasts exploratory; no new confirmatory significance claims",
               "success": "Log loss below train prior in at least two of three folds and paired conditional upper 95% bound below zero; text attribution also needs matched-price and availability comparisons",
               "refit": "All development data; per-seed median selected epoch across folds; no duration scaling", "threshold": .5,
               "permutation_evidence": {"R3": read_json(l1 / "permutation_result.json"), "R2": read_json(l2 / "permutation_result.json")},
               "evidence_scope": "Training-label block permutations are descriptive conditional references, not formal p-values; embedding similarity alone cannot disprove signal",
               "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
               "notebook_code_sha256": {str(p.relative_to(ROOT)): notebook_code_hash(p) for level in STUDIES for p in (ROOT / f"src/task_3/training/level_{level}").glob("*.ipynb")}})
    return check_inputs()

def check_inputs():
    protocol = read_json(OUTPUT / "protocol.json")
    extension_path = OUTPUT / "all_conditions_test_v1/frozen_config.json"
    extension = read_json(extension_path) if extension_path.exists() else None
    if extension:
        assert extension["base_protocol_sha256"] == sha256(OUTPUT / "protocol.json")
    for path, digest in protocol["input_sha256"].items():
        if extension and path in extension["input_amendments"]:
            change = extension["input_amendments"][path]
            assert digest == change["before_sha256"] == sha256(ROOT / change["snapshot"])
            digest = change["after_sha256"]
        assert sha256(ROOT / path) == digest, path
    for path, digest in protocol["notebook_code_sha256"].items():
        if extension and path in extension["notebook_amendments"]:
            change = extension["notebook_amendments"][path]
            assert digest == change["before_sha256"] == notebook_code_hash(ROOT / change["snapshot"])
            digest = change["after_sha256"]
        assert notebook_code_hash(ROOT / path) == digest, path
    if extension:
        for path, digest in extension["implementation_sha256"].items():
            assert sha256(ROOT / path) == digest, path
    return protocol

class Experiment:
    def __init__(self, level, model_id):
        self.level, self.model_id = level, model_id
        self.name = STUDIES[level].get(model_id, model_id)
        self.directory = OUTPUT / f"level_{level}" / self.name
        self.figures = FIGURES / f"level_{level}" / self.name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        configs = read_json(OUTPUT / "configurations.json")["configs"]
        self.flags = [int(v) for v in model_id[1:]] if level == 3 else None
        self.text_only = model_id == "T1"
        self.alias = model_id == "R3"
        self.naive = model_id in {"L1-a", "L1-b"}
        if level == 3 or self.text_only:
            selection = read_json(OUTPUT / "representation_selection.json")
            self.rep = selection["eligible"] if level == 3 else selection["overall"]
            self.config = configs[self.rep]
        else:
            self.rep = model_id if level == 2 else "R3"
            self.config = configs.get(model_id)
            if model_id in {"price_matched", "availability_matched"}:
                self.rep = read_json(OUTPUT / "representation_selection.json")["eligible"]
                self.config = {**configs[self.rep], "view": "price" if model_id == "price_matched" else "availability"}
        self.text = text_columns(self.rep)
        self.seeds = [-1] if self.naive else SEEDS

    def frames(self, fold):
        if self.naive or (self.level == 1 and self.model_id != "L1-d"):
            return *role_data(raw_data(), fold), None, {}
        return fold_data(self.rep, fold)

    def mask(self, frame):
        if not self.text_only:
            return frame
        result = frame.copy()
        result[MARKET] = 0.
        return result

    def fit(self, train, val, vectors, seed, epochs=None):
        if self.naive:
            model = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), train.y) if self.model_id == "L1-a" else LastDirectionClassifier().fit(train[["log_return_t"]], train.y)
            return model, 0, 0, []
        if self.level == 3:
            return fit_fusion(train, val, vectors, seed, self.flags, self.config, MARKET, self.text, AVAILABILITY, epochs=epochs)
        result = fit_small(self.mask(train), self.mask(val) if val is not None else None, seed, self.config, MARKET, self.text, AVAILABILITY, epochs=epochs)
        if self.text_only:
            return TextOnlyPredictor(result[0], MARKET), *result[1:]
        return result

    def predict(self, model, frame, vectors=None):
        if self.model_id == "L1-a":
            return model.predict_proba(np.zeros((len(frame), 1)))[:, 1]
        if self.model_id == "L1-b":
            return model.predict_proba(frame[["log_return_t"]])[:, 1]
        return model.predict_proba(frame, vectors)[:, 1] if self.level == 3 else model.predict_proba(frame)[:, 1]

    def train(self):
        check_inputs()
        if (self.directory / "run_metadata.json").exists():
            return self.results()
        if self.alias:
            source = Experiment(1, "L1-d").directory
            metadata = read_json(source / "run_metadata.json")
            write_json(self.directory / "alias.json", {"source": str(source.relative_to(ROOT)), "source_metadata_sha256": sha256(source / "run_metadata.json"), "new_fits": 0})
            for name in ["predictions", "seed_metrics", "fits", "history", "per_class", "fold_metrics", "summary", "intervals", "logistic_comparison", "logistic_metrics", "diagnostics", "splits"]:
                path = source / f"{name}.csv"
                if path.exists():
                    import shutil
                    shutil.copy2(path, self.directory / path.name)
            write_json(self.directory / "run_metadata.json", {**metadata, "model_id": self.model_id, "alias": str(source.relative_to(ROOT)), "new_fits": 0,
                       "artifact_sha256": {p.name: sha256(p) for p in self.directory.iterdir() if p.is_file() and p.name != "run_metadata.json"}})
            return self.results()
        predictions, metrics, classes, fits, history, splits, diagnostics, sources = [], [], [], [], [], [], [], {}
        for fold in [1, 2, 3]:
            train, val, vectors, source = self.frames(fold)
            sources[str(fold)] = source
            for role, frame in [("train", train), ("val", val)]:
                splits.append({"fold": fold, "role": role, "rows": len(frame), "first_date": str(frame.Date.min().date()), "last_date": str(frame.Date.max().date()), "up": int(frame.y.sum()),
                               "empty": int(frame.n_berita.eq(0).sum()), "singleton": int(frame.n_berita.eq(1).sum()), "multi_article": int(frame.n_berita.gt(1).sum())})
            if not self.naive and self.config["view"] == "text":
                values = train.loc[train.has_news.eq(1), self.text].to_numpy()
                norms = np.linalg.norm(values, axis=1)
                unit = values[norms > 0] / norms[norms > 0, None]
                cosine = (np.square(unit.sum(0)).sum() - len(unit)) / (len(unit) * (len(unit) - 1))
                pc1 = PCA(n_components=1, svd_solver="full").fit(values).explained_variance_ratio_[0]
                scale = values.std(0)
                diagnostics.append({"fold": fold, "mean_pair_cosine": cosine, "pc1_variance_share": pc1, "median_abs_mean_over_std": float(np.median(np.abs(values.mean(0)[scale > 0]) / scale[scale > 0]))})
            prior = score_binary(val.y, np.full(len(val), train.y.mean()))["log_loss"]
            for seed in self.seeds:
                started = time.perf_counter()
                model, best, ran, trace = self.fit(train, val, vectors, seed)
                path = self.directory / f"fold_{fold}_seed_{seed}.joblib"
                joblib.dump(model, path)
                p = self.predict(model, val, vectors)
                np.testing.assert_array_equal(p, self.predict(joblib.load(path), val, vectors))
                a, b, c = scored(val, p, self.model_id, seed, fold)
                predictions.append(a); metrics.append(b); classes.extend(c)
                core = model.predictor if self.text_only else model
                parameters = sum(v.numel() for v in core.model.parameters()) if not self.naive else 0
                fits.append({"fold": fold, "seed": seed, "train_n": len(train), "evaluation_n": len(val), "best_epoch": best, "epochs_run": ran,
                             "evaluation_role": "validation", "evaluation_is_none": False, "fixed_epochs": None,
                             "parameters": parameters, "prior_log_loss": prior, "fit_seconds": time.perf_counter() - started})
                history.extend({"fold": fold, "seed": seed, **row} for row in trace)
                write_json(OUTPUT / "status.json", {"updated_at_wib": now(), "phase": "validation", "level": self.level, "model_id": self.model_id, "fold": fold, "seed": seed, "log_loss": b["log_loss"]})
                print(f"{self.model_id} fold{fold} seed{seed}: epoch{best}, LL{b['log_loss']:.4f}, MCC{b['mcc']:.4f}", flush=True)
        for name, data in [("splits", splits), ("diagnostics", diagnostics)]:
            pd.DataFrame(data if data else [], columns=None if data else ["fold"]).to_csv(self.directory / f"{name}.csv", index=False)
        if not self.naive and self.level != 3:
            self.logistic()
        write_phase(self.directory, pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics), pd.DataFrame(fits), pd.DataFrame(history, columns=None if history else ["epoch"]), pd.DataFrame(classes),
                    {"model_id": self.model_id, "level": self.level, "phase": "validation", "config": self.config, "representation": self.rep, "flags": self.flags, "text_only": self.text_only,
                     "protocol_sha256": sha256(OUTPUT / "protocol.json"), "encoder_sources": sources, "alias": False, "new_fits": len(fits)})
        return self.results()

    def logistic(self):
        rows, failures = [], []
        for fold in [1, 2, 3]:
            train, val, _, _ = self.frames(fold)
            for C in [.01, .1, 1.]:
                config = {**self.config, "family": "logistic", "C": C}
                try:
                    model = fit_logistic(self.mask(train), config, MARKET, self.text, AVAILABILITY)
                except ValueError as error:
                    failures.append({"fold": fold, "C": C, "reason": str(error)})
                    continue
                path = self.directory / f"logistic_C{C:g}_fold_{fold}.joblib"
                joblib.dump(model, path)
                p = model.predict_proba(self.mask(val))[:, 1]
                np.testing.assert_array_equal(p, joblib.load(path).predict_proba(self.mask(val))[:, 1])
                rows.append({"fold": fold, "C": C, "n_iter": int(model.model.n_iter_.max()), **score_binary(val.y, p)})
        frame = pd.DataFrame(rows)
        frame.to_csv(self.directory / "logistic_metrics.csv", index=False)
        complete = frame.groupby("C").filter(lambda group: len(group) == 3)
        complete.groupby("C")[METRICS].mean().sort_values("log_loss").to_csv(self.directory / "logistic_comparison.csv")
        write_json(self.directory / "logistic_failures.json", failures)

    def results(self):
        return pd.read_csv(self.directory / "summary.csv").set_index("statistic")

    def plot(self):
        return plot_results(pd.read_csv(self.directory / "predictions.csv"), pd.read_csv(self.directory / "fits.csv"), pd.read_csv(self.directory / "history.csv"), self.figures, f"{self.model_id}: validation")

    def finalize_level(self):
        return summarize_level(self.level)

    def final_test(self):
        if (OUTPUT / "all_conditions_test_v1/frozen_config.json").exists():
            from src.task_3.training.all_tests import evaluate
            return evaluate(self.level, self.model_id)
        directory = OUTPUT / "final_test"
        if not (directory / "frozen_config.json").exists():
            return "Final test is deferred until all validation conditions are complete."
        return evaluate_final(self.model_id, self.level)

def summarize_level(level):
    rows = []
    for model_id in STUDIES[level]:
        if model_id == "T1" and not (OUTPUT / "representation_selection.json").exists():
            continue
        experiment = Experiment(level, model_id)
        if (experiment.directory / "summary.csv").exists():
            rows.append({"model_id": model_id, **experiment.results().loc["mean"].to_dict()})
    table = pd.DataFrame(rows).set_index("model_id")
    table.to_csv(OUTPUT / f"level_{level}/validation_summary.csv")
    labels = ["Model", *METRICS]
    lines = ["| " + " | ".join(labels) + " |", "| --- | " + " | ".join(["---:"] * len(METRICS)) + " |"]
    for model_id, row in table.iterrows():
        lines.append("| " + " | ".join([model_id, *[f"{row[m]:.4f}" for m in METRICS]]) + " |")
    (ROOT / f"src/task_3/training/level_{level}/level_{level}_result.md").write_text("\n".join(lines) + "\n")
    if level == 2 and all(rep in table.index for rep in WIDTHS):
        candidates = table.loc[list(WIDTHS)]
        best = candidates.log_loss.idxmin()
        eligible = candidates.loc[["R2", "R3", "R4"]].log_loss.idxmin()
        path = OUTPUT / "representation_selection.json"
        record = {"selected_at_wib": now(), "criterion": "Minimum equal-fold validation log loss", "overall": best, "eligible": eligible,
                  "test_used_for_selection": False, "scores": candidates.reset_index().to_dict("records"),
                  "interpretation": "Validation selects and scores checkpoints and representations; exploratory estimates"}
        if path.exists():
            old = read_json(path)
            assert old["overall"] == best and old["eligible"] == eligible
        else:
            write_json(path, record)
    return table

def freeze_final():
    path = OUTPUT / "final_test/frozen_config.json"
    if path.exists():
        return read_json(path)
    tables = {level: summarize_level(level) for level in STUDIES}
    assert all(len(tables[level]) == len(STUDIES[level]) for level in STUDIES)
    selected = tables[3].log_loss.idxmin()
    final = Experiment(3, selected)
    matched_price, matched_availability = Experiment(1, "price_matched"), Experiment(1, "availability_matched")
    matched_price.train()
    matched_availability.train()
    candidates = [final, Experiment(1, "L1-a"), Experiment(1, "L1-b"), matched_price, Experiment(1, "price_tuned"), matched_availability]
    selected_predictions = pd.read_csv(final.directory / "predictions.csv")
    gates, contrasts = {}, []
    for control in candidates[1:]:
        frame, gate = paired_comparison(selected_predictions, pd.read_csv(control.directory / "predictions.csv"))
        gates[control.model_id] = gate
        frame["comparison"] = f"{selected}_minus_{control.model_id}"
        contrasts.append(frame)
    conditions = []
    for experiment in candidates:
        fits = pd.read_csv(experiment.directory / "fits.csv")
        epochs = {str(int(seed)): int(value) for seed, value in fits.groupby("seed").best_epoch.median().items()}
        conditions.append({"level": experiment.level, "model_id": experiment.model_id, "directory": str(experiment.directory.relative_to(ROOT)), "config": experiment.config,
                           "representation": experiment.rep, "flags": experiment.flags, "epochs": epochs, "fits_sha256": sha256(experiment.directory / "fits.csv"),
                           "metadata_sha256": sha256(experiment.directory / "run_metadata.json")})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(contrasts, ignore_index=True).to_csv(path.parent / "validation_contrasts.csv", index=False)
    record = {"frozen_at_wib": now(), "selected_level3_model": selected, "representation": final.rep, "criterion": "Minimum equal-fold validation log loss among all five Level 3 variants",
              "protocol_sha256": sha256(OUTPUT / "protocol.json"), "representation_selection_sha256": sha256(OUTPUT / "representation_selection.json"),
              "conditions": conditions, "fit_rows": len(raw_data()), "validation_gates": gates, "success_gate_passed": bool(all(gates[key]["passes"] for key in ["L1-a", "price_matched", "availability_matched"])),
              "evaluation_reason": "Complete requested exploratory final pipeline; failed validation gates remain failures rather than cancelling planned levels",
              "threshold": .5, "test_used_for_selection": False, "test_history": read_json(OUTPUT / "protocol.json")["test_history"]}
    write_json(path, record)
    return record

def evaluate_final(model_id, level):
    check_inputs()
    frozen_path = OUTPUT / "final_test/frozen_config.json"
    frozen = read_json(frozen_path)
    conditions = [cfg for cfg in frozen["conditions"] if cfg["model_id"] == model_id and cfg["level"] == level]
    if not conditions:
        return "Not selected for final test. This condition reports validation only."
    cfg = conditions[0]
    experiment = Experiment(level, model_id)
    assert cfg["config"] == experiment.config and cfg["representation"] == experiment.rep
    assert cfg["fits_sha256"] == sha256(experiment.directory / "fits.csv")
    assert cfg["metadata_sha256"] == sha256(experiment.directory / "run_metadata.json")
    directory = OUTPUT / "final_test" / model_id
    directory.mkdir(exist_ok=True)
    if (directory / "run_metadata.json").exists():
        return pd.read_csv(directory / "summary.csv").set_index("statistic")
    dev, vectors, dev_source = final_data(experiment.rep) if not experiment.naive and experiment.config["view"] == "text" else (raw_data(), None, {})
    models, fits, histories = {}, [], []
    for seed in experiment.seeds:
        duration = cfg["epochs"][str(seed)]
        evaluation = None
        model, best, ran, trace = experiment.fit(dev, evaluation, vectors, seed, epochs=duration)
        path = directory / f"seed_{seed}.joblib"
        joblib.dump(model, path)
        models[seed] = path
        fits.append({"seed": seed, "fold": 0, "train_n": len(dev), "epochs": duration, "epochs_run": ran, "best_epoch": best,
                     "evaluation_role": "none", "evaluation_is_none": evaluation is None, "fixed_epochs": duration})
        histories.extend({"fold": 0, "seed": seed, **row} for row in trace)
    pd.DataFrame(fits).to_csv(directory / "refit_records.csv", index=False)
    write_json(directory / "test_access.json", {"accessed_at_wib": now(), "frozen_config_sha256": sha256(frozen_path), "refit_records_sha256": sha256(directory / "refit_records.csv")})
    test, test_vectors, test_source = final_data(experiment.rep, "test") if not experiment.naive and experiment.config["view"] == "text" else (raw_data("test"), None, {})
    assert dev.target_date.max() < test.Date.min()
    predictions, metrics, classes = [], [], []
    for seed, path in models.items():
        p = experiment.predict(joblib.load(path), test, test_vectors)
        a, b, c = scored(test, p, model_id, seed, 0)
        predictions.append(a); metrics.append(b); classes.extend(c)
    p = pd.concat(predictions, ignore_index=True)
    summary, _ = write_phase(directory, p, pd.DataFrame(metrics), pd.DataFrame(fits), pd.DataFrame(histories, columns=None if histories else ["epoch"]), pd.DataFrame(classes),
                             {"model_id": model_id, "level": level, "phase": "test", "config": experiment.config, "representation": experiment.rep, "flags": experiment.flags,
                              "protocol_sha256": sha256(OUTPUT / "protocol.json"), "frozen_config_sha256": sha256(frozen_path), "encoder_sources": {"development": dev_source, "test": test_source},
                              "exploratory": True, "test_used_for_selection": False})
    plot_results(p, pd.DataFrame(fits), pd.DataFrame(histories), FIGURES / "final_test" / model_id, f"{model_id}: test")
    return summary

def final_report():
    if (OUTPUT / "all_conditions_test_v1/frozen_config.json").exists():
        from src.task_3.training.all_tests import report
        return report()
    frozen = read_json(OUTPUT / "final_test/frozen_config.json")
    rows, predictions = [], {}
    for cfg in frozen["conditions"]:
        model_id = cfg["model_id"]
        directory = OUTPUT / "final_test" / model_id
        if not (directory / "summary.csv").exists():
            return
        summary = pd.read_csv(directory / "summary.csv").set_index("statistic")
        predictions[model_id] = pd.read_csv(directory / "predictions.csv")
        rows.append({"model_id": model_id, **summary.loc["mean"].to_dict()})
    table = pd.DataFrame(rows).set_index("model_id")
    table.to_csv(OUTPUT / "final_test/test_summary.csv")
    contrasts = []
    for reference in predictions:
        if reference == frozen["selected_level3_model"]:
            continue
        intervals, _ = paired_comparison(predictions[frozen["selected_level3_model"]], predictions[reference])
        intervals["comparison"] = f"{frozen['selected_level3_model']}_minus_{reference}"
        contrasts.append(intervals)
    pd.concat(contrasts, ignore_index=True).to_csv(OUTPUT / "final_test/paired_contrasts.csv", index=False)
    header = ["| Model | " + " | ".join(METRICS) + " |", "| --- | " + " | ".join(["---:"] * len(METRICS)) + " |"]
    lines = header + ["| " + " | ".join([name, *[f"{row[m]:.4f}" for m in METRICS]]) + " |" for name, row in table.iterrows()]
    (ROOT / "src/task_3/training/final_test_result.md").write_text("\n".join(lines) + "\n")
    for level in STUDIES:
        ids = [cfg["model_id"] for cfg in frozen["conditions"] if cfg["level"] == level]
        rows = ["| " + " | ".join([name, *[f"{table.loc[name,m]:.4f}" for m in METRICS]]) + " |" for name in ids]
        if not rows:
            (ROOT / f"src/task_3/training/level_{level}/level_{level}_test_result.md").write_text("No condition from this level was opened separately in the final test. The selected final model and controls are reported in ../final_test_result.md.\n")
        else:
            (ROOT / f"src/task_3/training/level_{level}/level_{level}_test_result.md").write_text("\n".join(header + rows) + "\n")
    selected = frozen["selected_level3_model"]
    count = predictions[selected].groupby("seed").y_pred.agg(["sum", "count"])
    lines = ["# Improved three-level training", "", f"All five Level 1, five Level 2 and five main Level 3 conditions completed without VADER. Level 2 selects {frozen['representation']}; Level 3 selects {selected} by validation log loss. Prior verified development grids supplied regularization; all active MLPs were refitted in the training pipeline.", "",
             "See each level's result table for validation. Only one selected NLP configuration and five prespecified controls were refitted for final test. Other notebook test sections explicitly defer evaluation. Test has been accessed before; all comparisons are exploratory.", "", *lines, "",
             f"Validation success gate passed: {frozen['success_gate_passed']}. Failure remains a limitation even though all planned levels and the requested final evaluation completed.", "",
             "UP prediction counts per seed:", "", count.to_string(), "", "Models include training-fitted preprocessing; frozen article encoders are shared separately. No-news windows remain masked zero vectors. Classifier threshold remains0.5; metrics average individual seeds without ensembling.", "",
             "The earlier audit did not establish that checkpoint selection is the sole cause of the validation/test gap. Training-label permutation results and text geometry are conditional diagnostics. Date-only publication times and historical article bodies still limit forecasting claims."]
    (OUTPUT / "result.md").write_text("\n".join(lines) + "\n")
    write_json(OUTPUT / "status.json", {"phase": "completed", "completed_at_wib": now(), "selected_model": selected, "representation": frozen["representation"]})
    return table
