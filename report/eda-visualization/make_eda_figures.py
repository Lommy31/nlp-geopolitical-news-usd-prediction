"""EDA figure for the Dataset subsection. Run from the repository root:
    python report/eda-visualization/make_eda_figures.py
Reads data/final/*.csv and data/processed/usd_idr_jisdor_clean.csv only.
Style follows ChenLiu-1996/figures4papers (palette, no top/right spines, frameless legend),
with font sizes scaled down to a 7.16 in IEEE page width."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures" / "assets"
OUT_PNG = ROOT / "report" / "eda-visualization"

BLUE, BLUE2, RED, TEAL, GREY = "#0F4D92", "#3775BA", "#B64342", "#42949E", "#CFCECE"
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 8, "axes.linewidth": 1.0, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

splits = {s: pd.read_csv(ROOT / "data/final" / f"{s}.csv", usecols=["Date", "y", "n_berita"], parse_dates=["Date"])
          for s in ["train", "val", "test"]}
dec = pd.concat([d.assign(split=s) for s, d in splits.items()]).sort_values("Date")
fx = pd.read_csv(ROOT / "data/processed/usd_idr_jisdor_clean.csv", parse_dates=["Date"])

bounds = {s: (d.Date.min(), d.Date.max()) for s, d in splits.items()}
shade = {"train": GREY, "val": "#AADCA9", "test": "#F6CFCB"}
names = {"train": "Train", "val": "Val", "test": "Test"}

def shade_splits(ax, label=False):
    for s, (a, b) in bounds.items():
        ax.axvspan(a, b, color=shade[s], alpha=0.55, lw=0, label=names[s] if label else None)

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})

# (a) USD/IDR level
ax = axes[0]
shade_splits(ax, label=True)
ax.plot(fx.Date, fx.USD_IDR / 1000, color=BLUE, lw=0.9)
ax.set_ylabel("USD/IDR (thousand)")
ax.set_xlim(fx.Date.min(), fx.Date.max())
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%y"))
ax.legend(loc="upper left", ncol=3, fontsize=6.5, handlelength=1, columnspacing=0.8, borderaxespad=0.2)
ax.set_title("(a) JISDOR fixing", fontsize=8, loc="left")

# (b) articles per decision day, monthly mean
ax = axes[1]
shade_splits(ax)
m = dec.set_index("Date").n_berita.resample("MS").mean()
ax.plot(m.index, m.values, color=RED, lw=0.9)
ax.set_ylabel("Articles per decision day")
ax.set_xlim(fx.Date.min(), fx.Date.max())
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%y"))
ax.set_title("(b) Monthly mean coverage", fontsize=8, loc="left")

# (c) distribution of articles per decision day
ax = axes[2]
bins = np.logspace(np.log10(dec.n_berita.min()), np.log10(dec.n_berita.max()), 24)
ax.hist(dec.n_berita, bins=bins, color=BLUE2, edgecolor="black", linewidth=0.4)
ax.set_xscale("log")
med = dec.n_berita.median()
ax.axvline(med, color="black", lw=0.9, ls="--")
ax.text(med * 0.93, ax.get_ylim()[1] * 0.97, f"median {med:.0f}", fontsize=6.5, va="top", ha="right")
ax.set_xlabel("Articles per decision day (log)")
ax.set_ylabel("Decision days")
ax.set_title("(c) Articles per day", fontsize=8, loc="left")

fig.tight_layout(pad=0.4, w_pad=0.8)
fig.savefig(OUT / "eda_overview.pdf")
fig.savefig(OUT_PNG / "eda_overview.png", dpi=300)

# numbers printed for the report text and for validation
print(dec.groupby("split").agg(start=("Date", "min"), end=("Date", "max"), n=("y", "size"),
                               up=("y", "mean"), art=("n_berita", "mean")))
print("articles/day: median", med, "p90", dec.n_berita.quantile(.9), "max", dec.n_berita.max(),
      "share>16", round((dec.n_berita > 16).mean(), 3))
