"""Positive-control probes on frozen FinBERT day-mean vectors (development days only; test untouched).
Targets: UP/DOWN direction, block-permuted direction (control), and keyword-defined easy day properties."""
import sys, re
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.task_3.preprocessing.splitting import load_split
MODEL = ROOT / "src/task_3/output/model"
def load_enc(name):
    vec = {}
    for f in sorted((MODEL / "encoders" / name).glob("articles_*.npz")):
        z = np.load(f, allow_pickle=False); vec.update(zip(z["article_ids"].tolist(), z["vectors"].astype(np.float64)))
    return vec
dev = pd.concat([load_split(ROOT / "data/final", s) for s in ["train", "val"]], ignore_index=True)
def day_mean(vec): return np.stack([np.mean([vec[a["article_id"]] for a in w], 0) for w in dev.articles])
embs = {"FinBERT": day_mean(load_enc("finbert")), "RoBERTa": day_mean(load_enc("roberta")), "Word2Vec": day_mean(load_enc("word2vec"))}
X = embs["FinBERT"]
rng = np.random.default_rng(0)
KW = {"conflict": r"\b(war|attack|strike|missile|troops|military|ceasefire|gaza|ukraine|israel|iran|hamas)\b",
      "energy": r"\b(oil|crude|opec|gas|energy)\b",
      "rates/Fed": r"\b(fed|rate|rates|inflation|treasury|powell|yields?)\b",
      "China": r"\b(china|chinese|beijing|taiwan)\b"}
def share(p): return np.array([np.mean([bool(re.search(p, a["title"].lower())) for a in w]) for w in dev.articles])
def block_perm(y, b=10):
    n = len(y); idx = np.arange(0, n, b); blocks = [y[i:i+b] for i in idx]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
targets = {"Direction (UP/DOWN)": dev.y.to_numpy()}
for k, p in KW.items():
    s = share(p); targets[f"Keyword: {k}"] = (s > np.median(s)).astype(int) if np.median(s) > 0 else (s > 0).astype(int)
embs["Random Gaussian"] = rng.normal(size=X.shape)
def cv_auc(X, y):
    out = []
    for tr, va in TimeSeriesSplit(5, gap=1).split(X):
        if len(set(y[tr])) < 2 or len(set(y[va])) < 2: continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(X[tr], y[tr])
        out.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    return np.mean(out)
rows = [{"target": t, **{e: round(cv_auc(E, y), 3) for e, E in embs.items()}} for t, y in targets.items()]
# control: direction shuffled in blocks of 10, repeated 100 times (mean and 2.5-97.5% range)
ctrl = {e: np.array([cv_auc(E, block_perm(dev.y.to_numpy())) for _ in range(100)]) for e, E in embs.items()}
rows.append({"target": "Shuffled direction (mean of 100)", **{e: round(c.mean(), 3) for e, c in ctrl.items()}})
rows.append({"target": "Shuffled direction (95% range)", **{e: f"{np.percentile(c,2.5):.2f}-{np.percentile(c,97.5):.2f}" for e, c in ctrl.items()}})
res = pd.DataFrame(rows); print(res.to_string(index=False)); res.to_csv(ROOT / "report/eda-visualization/probe_auc.csv", index=False)
