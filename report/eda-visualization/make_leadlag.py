"""Lead-lag cross-correlation between daily news features and USD/IDR returns (development days only).
corr(news_t, ret_{t+k}): k>0 news leads the market, k<0 the market leads news. Null band: circular shifts of the news series."""
import sys, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.task_3.preprocessing.splitting import load_split
BLUE, BLUE2, RED, TEAL, GREY = "#0F4D92", "#3775BA", "#B64342", "#42949E", "#CFCECE"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans", "sans-serif"], "font.size": 7.5, "axes.linewidth": 1.0,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
dev = pd.concat([load_split(ROOT / "data/final", s) for s in ["train", "val"]], ignore_index=True)
print("rows", len(dev), "consecutive-day gap median", dev.Date.diff().dt.days.median(), dev[["Date","target_date","log_return_t"]].head(3).to_string())
KW = {"conflict": r"\b(war|attack|strike|missile|troops|military|ceasefire|gaza|ukraine|israel|iran|hamas)\b",
      "rates/Fed": r"\b(fed|rate|rates|inflation|treasury|powell|yields?)\b"}
def share(p): return np.array([np.mean([bool(re.search(p, a["title"].lower())) for a in w]) for w in dev.articles])
news = {"Articles per day": np.log1p(dev.n_berita.to_numpy()), "Conflict share": share(KW["conflict"]), "Rates/Fed share": share(KW["rates/Fed"])}
ret = dev.log_return_t.to_numpy(); targets = {"Return": ret, "|Return|": np.abs(ret)}
lags = np.arange(-5, 6)
def xc(a, b, k):
    n = len(a)
    if k >= 0: x, y = a[: n - k], b[k:]
    else: x, y = a[-k:], b[: n + k]
    return np.corrcoef(x, y)[0, 1]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(3, 2, figsize=(3.45, 3.3), sharex=True)
rows = []
for j, (nn, a) in enumerate(news.items()):
    for i, (tn, b) in enumerate(targets.items()):
        r = np.array([xc(a, b, k) for k in lags])
        null = np.array([[xc(np.roll(a, rng.integers(20, len(a) - 20)), b, k) for k in lags] for _ in range(500)])
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
        ax = axes[j, i]; ax.axhspan(lo.min(), hi.max(), color=GREY, alpha=.5, lw=0)
        ax.bar(lags, r, color=[BLUE2 if k >= 0 else TEAL for k in lags], edgecolor="black", lw=.3)
        ax.axhline(0, color="black", lw=.6); ax.set_title(f"{nn} vs {tn}", fontsize=5.5, loc="left", pad=2)
        if j == 2: ax.set_xlabel("Lag k (days)")
        if i == 0: ax.set_ylabel("Corr.")
        out = [(int(k), round(float(x), 3)) for k, x, l, h in zip(lags, r, lo, hi) if x < l or x > h]
        rows.append((nn, tn, round(float(hi.max()), 3), out)); print(nn, "|", tn, "| band +/-", round(float(hi.max()), 3), "| outside:", out)
fig.text(.01, .5, "", rotation=90)
[a.set_xticks([-4,-2,0,2,4]) for a in axes.ravel()]; fig.tight_layout(); fig.savefig(ROOT / "report/figures/assets/lead_lag.pdf"); fig.savefig(ROOT / "report/eda-visualization/lead_lag.png", dpi=200)
