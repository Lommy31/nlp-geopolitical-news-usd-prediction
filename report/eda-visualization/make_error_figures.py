"""Error analysis and attention figure. Run from the repository root:
    python report/eda-visualization/make_error_figures.py
Uses saved improved_v1 predictions, the cached frozen FinBERT vectors, and the saved
Level 3 (A111) refit models. Nothing is trained. The test period was already used by earlier
experiments, so every output here is descriptive."""
import sys, json, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, joblib, torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.task_3.training.attention import batch
from src.task_3.preprocessing.splitting import load_split

OUT = ROOT / "report" / "figures" / "assets"
OUT_PNG = ROOT / "report" / "eda-visualization"
MODEL = ROOT / "src/task_3/output/model"
TEST = MODEL / "improved_v1/all_conditions_test_v1"
BLUE, BLUE2, RED, TEAL, GREY = "#0F4D92", "#3775BA", "#B64342", "#42949E", "#CFCECE"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans", "sans-serif"], "font.size": 7.5,
                     "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})

# ---------- predictions (already saved, no new test access) ----------
def probs(model_id):
    d = pd.read_csv(TEST / model_id / "predictions.csv", parse_dates=["Date"])
    return d.groupby("Date").agg(y=("y", "first"), n=("n_berita", "first"), p=("p_up", "mean"))
pa, p1 = probs("A000"), probs("A111")
test = load_split(ROOT / "data/final", "test")
fx = pd.read_csv(ROOT / "data/processed/usd_idr_jisdor_clean.csv", parse_dates=["Date"]).set_index("Date").USD_IDR
test["move"] = (test.target_date.map(fx) / test.USD_IDR - 1).abs()
test = test.set_index("Date")
prior = pd.read_csv(ROOT / "data/final/train.csv").y.mean()
print("test days", len(test), "UP share", round(test.y.mean(), 3), "train prior", round(prior, 3))

def summarize(name, d):
    pred = (d.p >= .5).astype(int)
    print(f"{name}: p range {d.p.min():.3f}-{d.p.max():.3f}, std {d.p.std():.4f}, pred UP {pred.mean():.3f}, "
          f"acc {(pred == d.y).mean():.3f}, MCC {matthews_corrcoef(d.y, pred):.3f}")
    tp = ((pred == 1) & (d.y == 1)).sum(); fp = ((pred == 1) & (d.y == 0)).sum()
    fn = ((pred == 0) & (d.y == 1)).sum(); tn = ((pred == 0) & (d.y == 0)).sum()
    print(f"   confusion (rows true DOWN/UP; cols pred DOWN/UP): [[{tn},{fp}],[{fn},{tp}]]")
summarize("A000 (FinBERT concat)", pa); summarize("A111 (full fusion)", p1)

# accuracy by subgroup, A000 and A111 against the always-UP rule
for name, d in [("A000", pa), ("A111", p1)]:
    t = test.join(d[["p"]], how="inner")
    t["ok"] = ((t.p >= .5).astype(int) == t.y)
    t["bucket_n"] = pd.qcut(t.n_berita, 3, labels=["few", "mid", "many"])
    t["bucket_move"] = pd.qcut(t.move, 3, labels=["small", "mid", "large"])
    for col in ["bucket_n", "bucket_move"]:
        g = t.groupby(col, observed=True).agg(days=("y", "size"), up_share=("y", "mean"), acc=("ok", "mean"))
        print(name, col); print(g.round(3).to_string())
    t["month"] = t.index.to_period("M")
    mo = t.groupby("month").agg(days=("y", "size"), up_share=("y", "mean"), acc=("ok", "mean"))
    print(name, "months: acc below always-UP in", int((mo.acc < mo.up_share).sum()), "of", len(mo), "months")

# ---------- day embeddings (development, frozen FinBERT) ----------
def vectors(path):
    z = np.load(path, allow_pickle=False)
    return dict(zip(z["article_ids"].tolist(), z["vectors"].astype(np.float64)))
vec_dev = vectors(MODEL / "encoders/finbert/articles_6699096ab3da0b36a140.npz")
dev = pd.concat([load_split(ROOT / "data/final", s) for s in ["train", "val"]], ignore_index=True)
days = np.stack([np.mean([vec_dev[a["article_id"]] for a in w], 0) for w in dev.articles])
Z = PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(days))
print("silhouette of UP/DOWN on first 10 PCs (0 = no separation):", round(silhouette_score(Z, dev.y), 4))
print("PC1/PC2 variance share:", np.round(PCA(10, random_state=0).fit(StandardScaler().fit_transform(days)).explained_variance_ratio_[:2], 3))

# ---------- attention weights of the saved A111 refit models on the test days ----------
vec_te = vectors(MODEL / "encoders/finbert/articles_564eb8c5d3693dc6fd55.npz")
tf = test.reset_index()
tf["log1p_n_berita"] = np.log1p(tf.n_berita)
means = np.stack([np.mean([vec_te[a["article_id"]] for a in w], 0) for w in tf.articles])
tf = pd.concat([tf, pd.DataFrame(means, columns=[f"finbert_{j:03d}" for j in range(768)])], axis=1)
saved = pd.read_csv(TEST / "A111" / "predictions.csv", parse_dates=["Date"])
rows_task, rows_cross, gates, seed_p = [], [], [], {}
for seed in [42, 123, 2026]:
    pr = joblib.load(TEST / "A111" / f"seed_{seed}.joblib")
    got = pr.predict_proba(tf, vec_te)[:, 1]
    ref = saved[saved.seed == seed].set_index("Date").p_up.reindex(tf.Date).to_numpy()
    print(f"seed {seed}: max |recomputed - saved P(UP)| = {np.abs(got - ref).max():.2e}; predicts UP on {np.mean(got >= .5):.3f} of days")
    seed_p[seed] = got
    prep = pr.transform.prepare(tf, vec_te)
    with torch.no_grad():
        _, info = pr.model.eval()(*batch(prep, np.arange(len(test)), pr.model.width), details=True)
    rows_task.append(info["weights"][1].numpy()); rows_cross.append(info["weights"][2].numpy()); gates.append(info["gate"].numpy().ravel())
W_task, W_cross = np.mean(rows_task, 0), np.mean(rows_cross, 0)     # (days, max articles), zero on padding
n = test.n_berita.to_numpy().astype(int)
def peak(W): return np.array([W[i, :n[i]].max() * n[i] for i in range(len(n))])   # 1 = uniform
def ent(W): return np.array([-(W[i, :n[i]] * np.log(W[i, :n[i]] + 1e-12)).sum() / np.log(n[i]) for i in range(len(n))])
print(f"task attention: mean peak x uniform {peak(W_task).mean():.2f}, mean normalized entropy {ent(W_task).mean():.3f}")
print(f"cross attention: mean peak x uniform {peak(W_cross).mean():.2f}, mean normalized entropy {ent(W_cross).mean():.3f}")
print("gate mean", round(float(np.mean(gates)), 3), "sd", round(float(np.std(gates)), 3))

# example days: largest error for each class
dd = p1.join(test[["articles"]])
wrong_down = dd[dd.y == 0].p.idxmax(); wrong_up = dd[dd.y == 1].p.idxmin()
pos = {d: i for i, d in enumerate(test.index)}
def example(day):
    i = pos[day]; arts = test.articles.iloc[i]; k = n[i]
    order = np.argsort(-W_task[i, :k])[:8]
    tab = pd.DataFrame({"title": [arts[j]["title"] for j in order], "source": [arts[j]["source"] for j in order],
                        "task": W_task[i, order] * k, "cross": W_cross[i, order] * k})
    return i, k, tab
ex = {"wrong_down": wrong_down, "wrong_up": wrong_up}
for name, day in ex.items():
    i, k, tab = example(day)
    print(f"\n{name}: {day.date()} y={int(test.y.iloc[i])} p_up={p1.loc[day,'p']:.3f} articles={k}")
    print(tab.round(2).to_string())

# ---------- figures, one file each (single column) ----------
W, H = 3.45, 1.9
def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT_PNG / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(W, H))
allp = np.concatenate(list(seed_p.values())); bins = np.linspace(min(pa.p.min(), allp.min()), max(pa.p.max(), allp.max()), 30)
ax.hist(pa.p, bins=bins, color=BLUE2, alpha=.85, label="Concatenation")
ax.hist(allp, bins=bins, color=RED, alpha=.6, label="Full fusion (per seed)")
ax.axvline(prior, color="black", ls="--", lw=.9); ax.axvline(.5, color=GREY, lw=1.2)
ax.set_xlabel("Predicted P(UP)"); ax.set_ylabel("Test days"); ax.set_ylim(0, 330)
ax.legend(fontsize=6, loc="upper center", ncol=2); save(fig, "err_probability")

fig, ax = plt.subplots(figsize=(W, H))
raw = {m: pd.read_csv(TEST / m / "predictions.csv").groupby("seed").y_pred.mean() for m in ["A000", "A111"]}
print("predicted UP share per seed:", {m: v.round(3).to_dict() for m, v in raw.items()})
xs = np.arange(3); w = .38
for off, (m, c, lab) in enumerate([("A000", BLUE2, "Concatenation"), ("A111", RED, "Full fusion")]):
    ax.bar(xs + (off - .5) * w, raw[m].reindex([42, 123, 2026]).to_numpy(), w, color=c, edgecolor="black", lw=.4, label=lab)
ax.axhline(test.y.mean(), color="black", ls="--", lw=.9)
ax.text(2.45, test.y.mean() - .08, "true UP share", fontsize=6, ha="right")
ax.set_xticks(xs); ax.set_xticklabels([42, 123, 2026]); ax.set_xlabel("Seed"); ax.set_ylabel("Share predicted UP")
ax.set_ylim(0, 1.3); ax.legend(fontsize=6, loc="upper center", ncol=2); save(fig, "err_up_share")

fig, ax = plt.subplots(figsize=(W, H))
for lab, c, name in [(0, BLUE2, "DOWN"), (1, RED, "UP")]:
    m = dev.y.to_numpy() == lab
    ax.scatter(Z[m, 0], Z[m, 1], s=4, color=c, alpha=.45, lw=0, label=name)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=6, markerscale=2, loc="upper right"); save(fig, "err_pca")

i, kk, tab = example(wrong_down)
fig, ax = plt.subplots(figsize=(W, H + .3))
data = tab[["task", "cross"]].to_numpy()
im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=.5, vmax=max(2, data.max()))
ax.set_xticks([0, 1]); ax.set_xticklabels(["task", "cross"])
ax.set_yticks(range(len(tab))); ax.set_yticklabels([textwrap.shorten(t, 44, placeholder="...") for t in tab.title], fontsize=5.3)
for r in range(data.shape[0]):
    for c in range(2): ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center", fontsize=5.5)
for sp in ax.spines.values(): sp.set_visible(False)
fig.colorbar(im, ax=ax, fraction=.05, pad=.02).ax.tick_params(labelsize=6)
print("attention example day", wrong_down.date(), "n", kk, "p_up", round(p1.loc[wrong_down, "p"], 3)); save(fig, "err_attention_example")

fig, ax = plt.subplots(figsize=(W, H))
ax.hist(np.r_[peak(W_task), peak(W_cross)], bins=20, color=TEAL, edgecolor="black", lw=.3)
ax.axvline(1, color="black", ls="--", lw=.9); ax.set_xlabel("Largest weight / uniform"); ax.set_ylabel("Test days"); save(fig, "err_attention_concentration")
