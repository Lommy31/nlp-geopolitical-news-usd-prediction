"""Learning curve and PCA-dimension sweep with a linear probe on frozen FinBERT day means.
Train on the last n training days before the validation block; score on the 158 validation days. Test untouched."""
import sys, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.task_3.preprocessing.splitting import load_split
MODEL = ROOT / "src/task_3/output/model"
BLUE, BLUE2, RED, TEAL, GREY = "#0F4D92", "#3775BA", "#B64342", "#42949E", "#CFCECE"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans", "sans-serif"], "font.size": 7.5, "axes.linewidth": 1.0,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
vec = {}
for f in sorted((MODEL / "encoders/finbert").glob("articles_*.npz")):
    z = np.load(f, allow_pickle=False); vec.update(zip(z["article_ids"].tolist(), z["vectors"].astype(np.float64)))
tr, va = load_split(ROOT / "data/final", "train"), load_split(ROOT / "data/final", "val")
def dm(d): return np.stack([np.mean([vec[a["article_id"]] for a in w], 0) for w in d.articles])
Xtr, Xva = dm(tr), dm(va)
pat = r"\b(war|attack|strike|missile|troops|military|ceasefire|gaza|ukraine|israel|iran|hamas)\b"
def kw(d):
    s = np.array([np.mean([bool(re.search(pat, a["title"].lower())) for a in w]) for w in d.articles]); return s
s_tr, s_va = kw(tr), kw(va); med = np.median(s_tr)
Y = {"Direction": (tr.y.to_numpy(), va.y.to_numpy()), "Conflict keyword (control)": ((s_tr > med).astype(int), (s_va > med).astype(int))}
def fit_auc(Xa, ya, Xb, yb, C=0.01, k=None):
    steps = [StandardScaler()] + ([PCA(k, random_state=0)] if k else []) + [LogisticRegression(C=C, max_iter=3000)]
    m = make_pipeline(*steps).fit(Xa, ya); return roc_auc_score(yb, m.predict_proba(Xb)[:, 1])
sizes = [50, 100, 200, 400, 600, len(tr)]
fig, a0 = plt.subplots(figsize=(3.45, 1.75)); ax = [a0]
curves = {}
for (name, (ya, yb)), c in zip(Y.items(), [RED, BLUE]):
    curves[name] = [fit_auc(Xtr[-n:], ya[-n:], Xva, yb) for n in sizes]
    ax[0].plot(sizes, curves[name], "o-", color=c, ms=3, lw=1.2, label=name); print(name, dict(zip(sizes, np.round(curves[name], 3))))
ax[0].axhline(.5, color="black", ls="--", lw=.7); ax[0].set_xlabel("Training days"); ax[0].set_ylabel("Validation AUC")
ax[0].axhspan(.41, .59, color=GREY if "GREY" in globals() else "#CFCECE", alpha=.5, lw=0); ax[0].legend(fontsize=6, loc="center right")
ks = [2, 5, 10, 20, 50, 100, 200]
dir_k = [fit_auc(Xtr, tr.y.to_numpy(), Xva, va.y.to_numpy(), C=1.0, k=k) for k in ks]
ctl_k = [fit_auc(Xtr, Y["Conflict keyword (control)"][0], Xva, Y["Conflict keyword (control)"][1], C=1.0, k=k) for k in ks]
print("PCA sweep direction", dict(zip(ks, np.round(dir_k, 3)))); print("PCA sweep control", dict(zip(ks, np.round(ctl_k, 3))))
print("val n", len(va), "approx 95% half-width of AUC under null", round(1.96 * np.sqrt((len(va) + 1) / (12 * (len(va) / 2) ** 2)), 3))
fig.tight_layout(); fig.savefig(ROOT / "report/figures/assets/learning_curve.pdf"); fig.savefig(ROOT / "report/eda-visualization/learning_curve.png", dpi=200)
