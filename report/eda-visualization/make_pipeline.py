"""Study figure: shared input, three ablation lanes, shared evaluation.
Run from repo root: python3 report/eda-visualization/make_pipeline.py
Flat, near-monochrome 'draw.io' look. Numbers from the Methodology chapter."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures" / "assets"
OUT_PNG = ROOT / "report" / "eda-visualization"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans", "sans-serif"], "pdf.fonttype": 42})
EC, LANE, VB, VR = "#222222", "#F3F3F3", "#A9C0DE", "#E2B3B1"

fig, ax = plt.subplots(figsize=(7.16, 3.15))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

def box(x, y, w, h, t="", fc="white", fs=5.0, lw=0.7, ls="-", ec=EC, style="normal", z=2):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if t:
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.2, style=style, zorder=z + 1)

def strip(x, y, w, h, fc, n=4):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=EC, lw=0.6, zorder=3))
    for i in range(1, n):
        ax.plot([x + w * i / n] * 2, [y, y + h], color=EC, lw=0.4, zorder=4)

def arrow(p, q, ls="-", lw=0.7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5, lw=lw, color=EC, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=5))

def route(pts, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=EC, lw=0.7, ls=ls, zorder=5)
    arrow(pts[-2], pts[-1], ls=ls)

def t(x, y, s, fs=5.0, ha="center", b=False, i=False, c=EC):
    ax.text(x, y, s, fontsize=fs, ha=ha, va="center", fontweight="bold" if b else "normal",
            style="italic" if i else "normal", color=c, linespacing=1.2, zorder=6)

# ---- lanes
for y0, y1, title in [(34.6, 43.4, "Level 1  Baselines, no text encoder"),
                      (17.0, 32.6, "Level 2  Text representation (one encoder at a time, frozen)"),
                      (0.8, 15.4, "Level 3  Fusion variants")]:
    ax.add_patch(Rectangle((15.5, y0), 67.5, y1 - y0, fc=LANE, ec="#8C8C8C", lw=0.6, zorder=1))
    t(16.8, y1 - 1.3, title, fs=5.6, ha="left", b=True)

# ---- shared input
box(1, 12.5, 12, 10, "News articles\nheadline + first\n180 words", fs=5.0)
box(1, 34.0, 12, 8.4, "Market features\n14 inputs,\navailability 2", fs=4.8)
route([(13, 38.2), (17.0, 38.2)])
route([(13, 17.5), (14.4, 17.5), (14.4, 23.8), (15.9, 23.8)])
route([(14.4, 17.5), (14.4, 8.2), (17.0, 8.2)])

# ---- Level 1
names = ["Training\nmajority", "Last\ndirection", "Market\nonly", "Market +\navailability", "Market + FinBERT\n+ availability"]
for k, n in enumerate(names):
    box(17 + k * 12.9, 35.6, 11.6, 5.2, n, fs=4.7)
t(49.5, 34.5, "", fs=4)

# ---- Level 2
for k, n in enumerate(["TF-IDF + SVD", "Word2Vec", "FinBERT", "RoBERTa-base"]):
    box(17, 26.8 - k * 2.8, 12.5, 2.4, n, fs=4.7)
ax.plot([15.9, 15.9], [19.6, 28.0], color=EC, lw=0.7, zorder=5)
ax.plot([30.4, 30.4], [19.6, 28.0], color=EC, lw=0.7, zorder=5)
for cy in (28.0, 25.2, 22.4, 19.6):
    ax.plot([15.9, 17], [cy, cy], color=EC, lw=0.7, zorder=5)
    ax.plot([29.5, 30.4], [cy, cy], color=EC, lw=0.7, zorder=5)
route([(30.4, 23.8), (32.5, 23.8)])
box(32.5, 20.8, 10.5, 6.4, "Mean over tokens,\nthen over the\nday's articles", fs=4.5)
route([(43, 24.0), (46.0, 24.0)]); strip(46.0, 23.1, 6.0, 1.8, VB); t(49, 26.8, "text $\\mathbf{t}$", fs=4.6)
strip(46.0, 19.0, 6.0, 1.8, VR); t(49, 17.9, "market $\\mathbf{x}$, $\\mathbf{a}$", fs=4.4)
box(55.5, 18.6, 4.8, 8.6, "Concat", fs=4.7)
ax.texts[-1].set_rotation(90)
route([(52.0, 24.0), (55.5, 24.0)]); route([(52.0, 19.9), (55.5, 19.9)])
route([(60.3, 22.9), (63.0, 22.9)])
box(63.0, 20.4, 10.5, 5.0, "Dense 16, ReLU\nsigmoid", fs=4.6)
t(68.5, 29.0, "text-only: market zeroed", fs=4.3, i=True, c="#444444")

# ---- Level 3
box(17, 4.5, 9.5, 7.0, "FinBERT\n(frozen)", fs=4.8)
route([(26.5, 8.0), (29.0, 8.0)])
for k in range(3):
    strip(29.0, 10.0 - k * 2.6, 6.8, 1.7, VB)
t(32.4, 3.2, "article vectors", fs=4.3, i=True, c="#444444")
route([(35.8, 9.0), (38.2, 9.0)])
for k, n in enumerate(["Task-query attention", "Market-to-article\ncross-attention", "Scalar gate on text"]):
    box(38.2, 10.9 - k * 3.4, 15.5, 3.0, n, fs=4.4)
t(46.0, 2.4, "each alone, or all three", fs=4.3, i=True, c="#444444")
ax.plot([54.8, 54.8], [5.6, 12.4], color=EC, lw=0.7, zorder=5)
for cy in (12.4, 9.0, 5.6):
    ax.plot([53.7, 54.8], [cy, cy], color=EC, lw=0.7, zorder=5)
route([(54.8, 8.8), (56.0, 8.8)]); strip(56.0, 7.9, 5.4, 1.8, VB); t(58.7, 11.0, "$\\mathbf{z}$", fs=4.6)
strip(56.0, 3.2, 5.4, 1.8, VR); t(58.7, 2.0, "$\\mathbf{x}$, $\\mathbf{a}$", fs=4.4)
box(64.6, 3.0, 4.8, 9.2, "Concat", fs=4.7); ax.texts[-1].set_rotation(90)
route([(61.4, 8.8), (64.6, 8.8)]); route([(61.4, 4.1), (64.6, 4.1)])
route([(69.4, 7.6), (71.6, 7.6)])
box(71.6, 5.2, 10.0, 4.8, "Dense 16, ReLU\nsigmoid", fs=4.6)

# ---- shared evaluation
box(85.5, 0.8, 13.5, 42.4, "", fc="white", ls=(0, (4, 2)), lw=0.7)
t(92.2, 41.6, "Evaluation", fs=5.6, b=True)
box(86.8, 28.0, 10.9, 10.0, "Select on 3\nexpanding folds\n(purged), seeds\n42, 123, 2026", fs=4.6)
box(86.8, 15.0, 10.9, 9.0, "Refit on all\n930 development\ndecisions", fs=4.6)
box(86.8, 3.5, 10.9, 9.0, "Test 235 days\nthreshold 0.5\nMCC, AUC,\nlog loss, Acc.", fs=4.6)
arrow((92.2, 28.0), (92.2, 24.0)); arrow((92.2, 15.0), (92.2, 12.5))
for y in (37.6, 24.3, 7.6):
    pass
route([(81.8, 37.6), (85.5, 37.6)])
route([(73.5, 22.9), (85.5, 22.9)])
route([(81.6, 7.6), (85.5, 7.6)])
t(83.6, 39.0, "$P(\\mathrm{UP})$", fs=4.4); t(79.5, 24.2, "$P(\\mathrm{UP})$", fs=4.4); t(83.6, 9.0, "$P(\\mathrm{UP})$", fs=4.4)
route([(40.0, 34.8), (40.0, 33.6)]) if False else None

fig.savefig(OUT / "pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT_PNG / "pipeline.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("ok")
