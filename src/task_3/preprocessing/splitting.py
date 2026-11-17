import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

if __package__:
    from . import cleaning, merging
else:
    import cleaning
    import merging

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILES = ["train.csv", "val.csv", "test.csv", "metadata.json"]
ARTICLE_COLUMNS = {
    "article_id": "article_id", "source_date": "source_date", "title": "title_main",
    "source": "Source", "url": "URL", "corpus": "corpus",
    "text_transformer": "text_transformer", "text_tfidf": "text_tfidf",
    "tokens_word2vec": "tokens_word2vec",
}

def make_split_assignments(calendar):
    d = calendar.copy()
    test_start = int(len(d) * 0.8)
    development_end = test_start - 1
    bounds = np.linspace(development_end // 2, development_end, 4, dtype=int)
    if bounds[0] < 25 or len(set(bounds)) != 4:
        raise ValueError("Insufficient calendar history for three expanding folds.")
    for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        train = d.calendar_index.lt(start - 1) & d.eligible_binary
        val = d.calendar_index.ge(start) & d.calendar_index.lt(end - 1) & d.eligible_binary
        d[f"fold_{fold}"] = np.select([train, val], ["train", "val"], default="unused")
        if d.loc[train, "target_date"].max() >= d.loc[val, "Date"].min():
            raise ValueError(f"Target leakage at fold {fold} boundary.")
    d["split"] = d.fold_3.map({"train": "train", "val": "val"}).fillna("excluded")
    d.loc[d.calendar_index.ge(test_start) & d.eligible_binary, "split"] = "test"
    train = d.loc[d.split.eq("train")]
    val = d.loc[d.split.eq("val")]
    test = d.loc[d.split.eq("test")]
    if train.target_date.max() >= val.Date.min() or val.target_date.max() >= test.Date.min():
        raise ValueError("Target leakage across exported split boundaries.")
    for fold in range(1, 4):
        if (d[f"fold_{fold}"].ne("unused") & ~d.split.isin(["train", "val"])).any():
            raise ValueError("An exported split omits a development-fold member.")
    return d

def make_records(assignments, articles):
    grouped = {}
    for date, group in articles.sort_values(["anchor_date", "source_date", "article_id"]).groupby("anchor_date"):
        items = []
        for article in group.to_dict("records"):
            item = {output: article[source] for output, source in ARTICLE_COLUMNS.items()}
            item["source_date"] = item["source_date"].strftime("%Y-%m-%d")
            items.append(item)
        grouped[date] = items
    records = {split: [] for split in ["train", "val", "test"]}
    for row in assignments.loc[assignments.split.ne("excluded")].sort_values("Date").to_dict("records"):
        record = {
            "Date": row["Date"].strftime("%Y-%m-%d"),
            "target_date": row["target_date"].strftime("%Y-%m-%d"),
            "calendar_index": int(row["calendar_index"]), "USD_IDR": float(row["USD_IDR"]),
            "y": int(row["y"]),
            **{name: float(row[name]) for name in merging.MARKET_FEATURES},
            "n_berita": int(row["n_berita"]), "has_news": int(row["has_news"]),
            "articles": grouped.get(row["Date"], []),
            "fold_roles": {str(fold): row[f"fold_{fold}"] for fold in range(1, 4)},
        }
        records[row["split"]].append(record)
    return records

def verify_records(records, assignments, articles):
    dates, ids, urls = set(), set(), set()
    by_id = articles.set_index("article_id")
    schema = None
    for split, rows in records.items():
        expected = assignments.loc[assignments.split.eq(split)].sort_values("Date")
        if [row["Date"] for row in rows] != expected.Date.dt.strftime("%Y-%m-%d").tolist():
            raise ValueError(f"Incorrect {split} decision dates.")
        for row, target in zip(rows, expected.itertuples()):
            if row["Date"] in dates or row["y"] != int(target.y) or row["target_date"] != target.target_date.strftime("%Y-%m-%d"):
                raise ValueError("Overlapping dates or incorrect labels.")
            dates.add(row["Date"])
            schema = set(row) if schema is None else schema
            if set(row) != schema or not all(np.isfinite(row[name]) for name in merging.MARKET_FEATURES):
                raise ValueError("Inconsistent row schema or nonfinite market features.")
            if row["n_berita"] != len(row["articles"]) or row["has_news"] != int(bool(row["articles"])):
                raise ValueError("Incorrect news counts or masks.")
            for article in row["articles"]:
                article_id = article["article_id"]
                if article_id in ids or article["url"] in urls or article_id in cleaning.QUARANTINED_ARTICLE_IDS:
                    raise ValueError("Duplicate or quarantined article in exported data.")
                ids.add(article_id)
                urls.add(article["url"])
                if article["source_date"] >= row["Date"] or by_id.loc[article_id, "anchor_date"] != target.Date:
                    raise ValueError("Incorrect article-to-fixing assignment.")
    expected_ids = set(articles.loc[articles.anchor_date.isin(pd.to_datetime(list(dates))), "article_id"])
    if ids != expected_ids:
        raise ValueError("The export lost or added articles.")
    return {"passed": True, "unique_decisions": len(dates), "unique_articles": len(ids),
            "shared_schema": True, "chronological_boundaries": True, "all_fold_members_preserved": True,
            "article_assignments_counts_and_masks": True, "finite_market_features": True,
            "quarantine_preserved": True}

def summarize(rows):
    return {"rows": len(rows), "first_date": rows[0]["Date"], "last_date": rows[-1]["Date"],
            "up": sum(row["y"] for row in rows), "down": sum(1 - row["y"] for row in rows),
            "articles": sum(row["n_berita"] for row in rows),
            "news_days": sum(row["has_news"] for row in rows)}

def export_dataset(calendar, articles, provenance, output_dir, root=ROOT, overwrite=False):
    root, output_dir = Path(root), Path(output_dir)
    if not overwrite and any((output_dir / name).exists() for name in OUTPUT_FILES):
        raise FileExistsError("Output already exists; use --overwrite to rebuild it.")
    assignments = make_split_assignments(calendar)
    records = make_records(assignments, articles)
    verification = verify_records(records, assignments, articles)
    numeric = merging.MARKET_FEATURES + ["n_berita", "has_news"]
    non_predictors = [name for name in records["train"][0] if name not in numeric]
    code_paths = [Path(__file__), Path(cleaning.__file__), Path(merging.__file__)]
    code_hashes = {str(path.resolve().relative_to(root.resolve())): merging.sha256(path) for path in code_paths}
    folds = {}
    development = records["train"] + records["val"]
    for fold in range(1, 4):
        folds[str(fold)] = {role: summarize([row for row in development if row["fold_roles"][str(fold)] == role])
                            for role in ["train", "val"]}
    omitted = assignments.loc[assignments.eligible_binary & assignments.split.eq("excluded"), "Date"]
    historical_fit_rows = int((calendar.calendar_index.lt(int(len(calendar) * 0.8) - 1) & calendar.eligible_binary).sum())
    metadata = {
        "schema_version": 2, "format": "csv", "nested_json_columns": ["articles", "fold_roles"],
        "nested_column_usage": "Decode articles and fold_roles with json.loads after pandas.read_csv, or use load_split/load_fold from splitting.py.",
        "corpus": "original_plus_cnbc", "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "unit": "one eligible JISDOR fixing decision per row", "label": {"column": "y", "1": "USD_IDR_UP", "0": "USD_IDR_DOWN", "flat": "excluded after full-calendar feature/label calculation"},
        "features": {"market": merging.MARKET_FEATURES, "availability": ["n_berita", "has_news"],
                     "numeric_predictors": numeric, "exclude_from_numeric_predictors": non_predictors,
                     "text_inputs": {"tfidf": "articles[].text_tfidf", "word2vec": "articles[].tokens_word2vec", "transformer": "articles[].text_transformer"}},
        "article_metadata_is_audit_only": ["article_id", "source_date", "title", "source", "url", "corpus"],
        "preprocessing": {"alignment": "first strictly later JISDOR date", "geopolitical_policy": "audit_only",
                          "text_span": "headline plus first 180 cleaned body words", "word2vec_lowercase": False,
                          "empty_news": "articles=[]; n_berita=0; has_news=0; no carry-forward",
                          "aggregation": "mean per article/window; preserve article vectors for attention",
                          "fit_rule": "Fit vocabulary, IDF, SVD, scalers and other learned transformations only on the selected training split/fold.",
                          "quarantine_source_sha256": cleaning.ORIGINAL_SOURCE_SHA256,
                          "quarantined_article_ids": sorted(cleaning.QUARANTINED_ARTICLE_IDS)},
        "splitting": {"method": "last development fold train/val plus existing final-20%-calendar test",
                      "split_by": "JISDOR fixing date, never articles", "files": {split: summarize(rows) for split, rows in records.items()},
                      "folds": folds, "cv_usage": "Concatenate train and val, then select fold_roles[str(fold)] equal to train or val. Test rows are unused in every development fold.",
                      "omitted_eligible_boundary_dates": omitted.dt.strftime("%Y-%m-%d").tolist(),
                      "refit_usage": f"train+val contains {len(development)} decisions; the previous final-fit rule retained {historical_fit_rows}, including additional development boundary decisions not exported here."},
        **provenance, "code_sha256": code_hashes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "build_command": "PYTHONNOUSERSITE=1 nlp_venv/bin/python src/task_3/preprocessing/splitting.py",
        "final_test_status": "Original-corpus test was opened on 2026-11-11; this later corpus revision does not create an untouched holdout.",
        "limitations": ["Date-only alignment does not verify historical publication times or body versions.",
                        "Keyword/anchor screening is not human validation of geopolitical relevance.",
                        "Existing original-corpus model scores are not scores for this export."],
        "verification": verification, "learned_transforms_fitted": False, "models_trained": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preprocessing-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        for split, rows in records.items():
            frame = pd.DataFrame(rows)
            for column in ["articles", "fold_roles"]:
                frame[column] = frame[column].map(lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False))
            frame.to_csv(stage / f"{split}.csv", index=False, encoding="utf-8")
        metadata["output_sha256"] = {f"{split}.csv": merging.sha256(stage / f"{split}.csv") for split in records}
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        for relative, expected in {**provenance["input_sha256"], **code_hashes}.items():
            if merging.sha256(root / relative) != expected:
                raise ValueError(f"Input changed during preparation: {relative}")
        for name in OUTPUT_FILES:
            (stage / name).replace(output_dir / name)
    return metadata

def load_split(directory, split):
    if split not in {"train", "val", "test"}:
        raise ValueError("Choose train, val, or test.")
    frame = pd.read_csv(Path(directory) / f"{split}.csv", parse_dates=["Date", "target_date"])
    for column in ["articles", "fold_roles"]:
        frame[column] = frame[column].map(json.loads)
    return frame

def load_fold(directory, fold):
    if fold not in {1, 2, 3}:
        raise ValueError("Choose fold 1, 2, or 3.")
    development = pd.concat([load_split(directory, "train"), load_split(directory, "val")], ignore_index=True)
    role = development.fold_roles.map(lambda value: value[str(fold)])
    return development.loc[role.eq("train")].copy(), development.loc[role.eq("val")].copy()

def main():
    parser = argparse.ArgumentParser(description="Build shared train, validation, and test datasets.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir or root / "data/final"
    if not args.overwrite and any((output / name).exists() for name in OUTPUT_FILES):
        parser.error("Output already exists; use --overwrite to rebuild it.")
    print("Cleaning and merging original + CNBC articles", flush=True)
    calendar, articles, provenance = merging.build_dataset(root, progress=lambda message: print(message, flush=True))
    metadata = export_dataset(calendar, articles, provenance, output, root, args.overwrite)
    print(json.dumps(metadata["splitting"]["files"], indent=2), flush=True)
    print(f"Saved verified datasets to {output}", flush=True)

if __name__ == "__main__":
    main()
