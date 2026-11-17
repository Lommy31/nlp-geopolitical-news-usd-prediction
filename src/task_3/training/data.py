from functools import lru_cache
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.task_3.preprocessing.splitting import load_split
from src.task_3.training.controlled_fusion import encode_articles
from src.task_3.training.tfidf import FoldTfidfSVD
from src.task_3.training.testing import sha256

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data/final/metadata.json").exists())
OUTPUT = ROOT / "src/task_3/output/model/improved_v1"
WIDTHS = {"R1": 128, "R2": 300, "R3": 768, "R4": 768}
PREFIXES = {"R1": "svd", "R2": "word2vec", "R3": "finbert", "R4": "roberta"}
MARKET = json.loads((ROOT / "data/final/metadata.json").read_text())["features"]["market"]
AVAILABILITY = ["log1p_n_berita", "has_news"]

def text_columns(rep):
    return [f"{PREFIXES[rep]}_{i:03d}" for i in range(WIDTHS[rep])]

@lru_cache(maxsize=2)
def raw_data(phase="development"):
    parts = ["train", "val"] if phase == "development" else ["test"]
    frames = []
    metadata = json.loads((ROOT / "data/final/metadata.json").read_text())
    for part in parts:
        assert sha256(ROOT / "data/final" / f"{part}.csv") == metadata["output_sha256"][f"{part}.csv"]
        frames.append(load_split(ROOT / "data/final", part))
    frame = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    frame["log1p_n_berita"] = np.log1p(frame.n_berita)
    assert frame.Date.is_unique and frame.articles.map(len).eq(frame.n_berita).all()
    assert not set(MARKET + AVAILABILITY) & {"y", "target_date", "USD_IDR", "calendar_index", "articles", "fold_roles"}
    return frame

def role_data(frame, fold):
    roles = frame.fold_roles.map(lambda value: value[str(fold)])
    train, val = frame.loc[roles.eq("train")].copy(), frame.loc[roles.eq("val")].copy()
    assert train.target_date.max() < val.Date.min()
    return train.reset_index(drop=True), val.reset_index(drop=True)

@lru_cache(maxsize=3)
def encoded(rep, phase="development"):
    raw = raw_data(phase)
    directory = ROOT / "src/task_3/output/model/encoders" / {"R2": "word2vec", "R3": "finbert", "R4": "roberta"}[rep]
    if rep == "R2":
        frame, vectors, path, spec, info, _ = encode_articles(raw.drop(columns="log1p_n_berita"), directory)
        frame = frame.drop(columns=[c for c in frame if c.startswith("zero_text_")])
    else:
        if rep == "R3":
            from src.task_3.training.finbert import load_or_encode
        else:
            from src.task_3.training.roberta import load_or_encode
        articles = {a["article_id"]: a["text_transformer"] for items in raw.articles for a in items}
        ids = sorted(articles)
        values, path, spec, info = load_or_encode(ids, [articles[key] for key in ids], directory)
        vectors = dict(zip(ids, values.astype(np.float64)))
        means = np.stack([np.mean([vectors[a["article_id"]] for a in items], axis=0) if items else np.zeros(WIDTHS[rep]) for items in raw.articles])
        frame = pd.concat([raw.copy(), pd.DataFrame(means, columns=text_columns(rep))], axis=1)
    source = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "spec": spec, "encoder_info": info}
    return frame, vectors, source

def fold_data(rep, fold):
    if rep != "R1":
        frame, vectors, source = encoded(rep)
        return *role_data(frame, fold), vectors, source
    raw_train, raw_val = role_data(raw_data(), fold)
    directory = OUTPUT / "encoders"
    directory.mkdir(exist_ok=True)
    path = directory / f"tfidf_fold_{fold}.joblib"
    if path.exists():
        encoder = joblib.load(path)
        assert encoder.training_dates_ == raw_train.Date.dt.strftime("%Y-%m-%d").tolist()
    else:
        encoder = FoldTfidfSVD(n_components=128, padded_width=128).fit(raw_train)
        joblib.dump(encoder, path)
    return encoder.transform(raw_train.drop(columns="log1p_n_berita")), encoder.transform(raw_val.drop(columns="log1p_n_berita")), None, {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "fingerprint": encoder.fingerprint()}

def final_data(rep, phase="development"):
    if rep != "R1":
        return encoded(rep, phase)
    path = OUTPUT / "encoders/tfidf_final.joblib"
    raw = raw_data()
    if path.exists():
        encoder = joblib.load(path)
    else:
        assert phase == "development"
        encoder = FoldTfidfSVD(n_components=128, padded_width=128).fit(raw)
        path.parent.mkdir(exist_ok=True)
        joblib.dump(encoder, path)
    return encoder.transform(raw_data(phase).drop(columns="log1p_n_berita")), None, {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "fingerprint": encoder.fingerprint()}
