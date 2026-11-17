from pathlib import Path
import copy
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
    log_loss, matthews_corrcoef, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.task_3.training.word2vec import FrozenWord2Vec, TEXT_CONFIG
from src.task_3.training.testing import sha256

WORD_DIM = 300
SLOT_WIDTH = 768
FUSION_WIDTH = 14 + 3 * SLOT_WIDTH + 2
WORD_FEATURES = [f"word2vec_{i:03d}" for i in range(WORD_DIM)]
MODEL_CONFIG = {
    "backend": "torch", "hidden_layer_sizes": [64, 32], "activation": "relu", "dtype": "float64",
    "adam": {"lr": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0., "foreach": False},
    "batch_size": 32, "alpha": .001, "max_epochs": 200, "patience": 25,
    "regularization": "alpha / (2 * current_batch_n) * squared non-bias parameters",
    "shuffle_seed_offset": 8000003, "dropout": 0., "gradient_clipping": None,
}

def encode_articles(frame, directory):
    directory = Path(directory)
    articles = {}
    for row in frame.itertuples():
        if len(row.articles) != row.n_berita:
            raise ValueError("Article count does not match its fixing window.")
        for article in row.articles:
            if article["article_id"] in articles or article["source_date"] >= row.Date.strftime("%Y-%m-%d"):
                raise ValueError("Duplicate article ID or invalid date alignment.")
            articles[article["article_id"]] = article["tokens_word2vec"]
    ids = sorted(articles)
    spec = {"recipe": TEXT_CONFIG, "pooling_dtype": "float64", "native_dimension": WORD_DIM,
            "ordered_articles_sha256": hashlib.sha256(json.dumps([(key, articles[key]) for key in ids], ensure_ascii=False).encode()).hexdigest()}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    path = directory / f"articles_{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if cached["article_ids"].tolist() != ids or json.loads(str(cached["spec"].item())) != spec:
                raise ValueError("Article cache does not match the current input.")
            vectors = cached["vectors"].copy()
            tokens, known = cached["tokens"].copy(), cached["known_tokens"].copy()
            info = json.loads(str(cached["info"].item()))
    else:
        lookup, subset, pretrained = FrozenWord2Vec(str(directory))._vectors(frame)
        vectors = np.zeros((len(ids), WORD_DIM), dtype=np.float64)
        tokens, known = np.zeros(len(ids), dtype=int), np.zeros(len(ids), dtype=int)
        for index, article_id in enumerate(ids):
            words = articles[article_id]
            values = [lookup[word] for word in words if word in lookup]
            tokens[index], known[index] = len(words), len(values)
            if values:
                vectors[index] = np.mean(values, axis=0, dtype=np.float64)
        info = {"pretrained": pretrained, "subset_cache": str(subset), "subset_sha256": sha256(subset),
                "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(), "trainable_encoder_parameters": 0}
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, article_ids=np.asarray(ids), vectors=vectors, tokens=tokens,
                            known_tokens=known, spec=json.dumps(spec), info=json.dumps(info))
        temporary.replace(path)
    if vectors.shape != (len(ids), WORD_DIM) or not np.isfinite(vectors).all():
        raise ValueError("Invalid article vectors.")
    if hashlib.sha256(vectors.tobytes()).hexdigest() != info["vectors_sha256"]:
        raise ValueError("Article-vector checksum mismatch.")
    if sha256(info["subset_cache"]) != info["subset_sha256"]:
        raise ValueError("Pretrained lookup cache changed.")
    vectors.flags.writeable = False
    lookup = dict(zip(ids, vectors))
    means = np.zeros((len(frame), SLOT_WIDTH))
    for index, window in enumerate(frame.articles):
        if window:
            means[index, :WORD_DIM] = np.mean([lookup[a["article_id"]] for a in window], axis=0)
    columns = WORD_FEATURES + [f"zero_text_{i:03d}" for i in range(WORD_DIM, SLOT_WIDTH)]
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(means, columns=columns)], axis=1)
    result["log1p_n_berita"] = np.log1p(result.n_berita)
    coverage = pd.DataFrame({"article_id": ids, "tokens": tokens, "known_tokens": known})
    return result, lookup, path, spec, info, coverage

class SequenceScaler:
    def __init__(self, market_features, availability_features):
        self.market_features = market_features
        self.availability_features = availability_features

    def fit(self, frame):
        self.market = StandardScaler().fit(frame[self.market_features])
        news = frame.loc[frame.has_news.eq(1), WORD_FEATURES]
        if not len(news):
            raise ValueError("No news-present training windows.")
        self.text = StandardScaler(with_mean=False).fit(news)
        self.availability = StandardScaler(with_mean=False).fit(frame[self.availability_features])
        self.text_scale_ = self.text.scale_.copy()
        self.n_samples_seen_, self.news_n_samples_seen_ = len(frame), len(news)
        return self

    def prepare(self, frame, article_vectors):
        sequences = []
        for row in frame.itertuples():
            sequence = np.stack([article_vectors[a["article_id"]] for a in row.articles]) if row.articles else np.empty((0, WORD_DIM))
            if len(sequence) != row.n_berita:
                raise ValueError("Sequence length differs from the saved article count.")
            sequences.append(sequence / self.text_scale_)
        means = np.stack([sequence.mean(0) if len(sequence) else np.zeros(WORD_DIM) for sequence in sequences])
        np.testing.assert_allclose(means, frame[WORD_FEATURES].to_numpy() / self.text_scale_, atol=1e-12, rtol=1e-12)
        return {"frame": frame, "market": self.market.transform(frame[self.market_features]),
                "availability": self.availability.transform(frame[self.availability_features]),
                "sequences": sequences, "means": means}

def make_batch(prepared, positions):
    sequences = [prepared["sequences"][index] for index in positions]
    width = max(1, max(map(len, sequences), default=0))
    values = np.zeros((len(positions), width, WORD_DIM))
    padding = np.ones((len(positions), width), dtype=bool)
    for index, sequence in enumerate(sequences):
        values[index, :len(sequence)] = sequence
        padding[index, :len(sequence)] = False
    return (torch.as_tensor(prepared["market"][positions], dtype=torch.float64), torch.from_numpy(values),
            torch.from_numpy(padding), torch.as_tensor(prepared["availability"][positions], dtype=torch.float64))

class ControlledFusion(nn.Module):
    def __init__(self, seed, flags):
        super().__init__()
        self.seed, self.flags = int(seed), tuple(flags)
        if len(self.flags) != 3 or not set(self.flags).issubset({0, 1}):
            raise ValueError("Flags must be (task query, cross-attention, gate).")
        self.head = nn.Sequential(nn.Linear(FUSION_WIDTH, 64, dtype=torch.float64), nn.ReLU(),
                                  nn.Linear(64, 32, dtype=torch.float64), nn.ReLU(), nn.Linear(32, 1, dtype=torch.float64))
        self.task_query = nn.Parameter(torch.zeros(WORD_DIM, dtype=torch.float64)) if flags[0] else None
        self.cross_query = nn.Linear(14, WORD_DIM, dtype=torch.float64) if flags[1] else None
        self.gate = nn.Linear(15, 1, dtype=torch.float64) if flags[2] else None
        self.initialize(seed)

    def initialize(self, seed):
        random = np.random.RandomState(seed)
        def draw(rng, fan_in, fan_out):
            bound = np.sqrt(6 / (fan_in + fan_out))
            return rng.uniform(-bound, bound, (fan_in, fan_out)), rng.uniform(-bound, bound, fan_out)
        market, bias = draw(random, 14, 64)
        def block(width, offset):
            bound = np.sqrt(6 / (width + 64))
            return np.random.RandomState(seed + offset).uniform(-bound, bound, (width, 64))
        first = np.vstack([market, block(SLOT_WIDTH, 1000003), block(SLOT_WIDTH, 3000003),
                           block(SLOT_WIDTH, 4000003), block(2, 2000003)])
        arrays = [(first, bias), draw(random, 64, 32), draw(random, 32, 1)]
        with torch.no_grad():
            for layer, (weight, bias) in zip([self.head[0], self.head[2], self.head[4]], arrays):
                layer.weight.copy_(torch.from_numpy(weight.T))
                layer.bias.copy_(torch.from_numpy(bias))
            if self.task_query is not None:
                bound = np.sqrt(6 / (WORD_DIM + 1))
                self.task_query.copy_(torch.from_numpy(np.random.RandomState(seed + 5000003).uniform(-bound, bound, WORD_DIM)))
            for layer, offset in [(self.cross_query, 6000003), (self.gate, 7000003)]:
                if layer is not None:
                    weight, bias = draw(np.random.RandomState(seed + offset), layer.in_features, layer.out_features)
                    layer.weight.copy_(torch.from_numpy(weight.T))
                    layer.bias.copy_(torch.from_numpy(bias))

    def forward(self, market, articles, key_padding_mask, availability, details=False):
        if key_padding_mask.dtype != torch.bool:
            raise ValueError("Expected a boolean padding mask.")
        counts = (~key_padding_mask).sum(1)
        present = counts.gt(0)
        if not torch.equal(present, availability[:, 1].gt(0)):
            raise ValueError("Article presence differs from the availability mask.")
        clean = articles.masked_fill(key_padding_mask.unsqueeze(-1), 0.)
        mean = clean.sum(1) / counts.clamp_min(1).unsqueeze(-1)
        task, cross = torch.zeros_like(mean), torch.zeros_like(mean)
        positions = torch.where(present)[0]
        weights_by_component = {}
        for name, active in [("task", self.flags[0]), ("cross", self.flags[1])]:
            if active and len(positions):
                news, mask = clean[positions], key_padding_mask[positions]
                query = self.task_query.expand(len(positions), -1) if name == "task" else self.cross_query(market[positions])
                scores = (news * query.unsqueeze(1)).sum(-1) / np.sqrt(WORD_DIM)
                weights = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=1)
                summary = torch.zeros_like(mean).index_copy(0, positions, (weights.unsqueeze(-1) * news).sum(1))
                if name == "task":
                    task = summary
                else:
                    cross = summary
                weights_by_component[name] = weights
        gate = torch.sigmoid(self.gate(torch.cat([market, availability[:, :1]], 1))) if self.flags[2] else torch.ones_like(availability[:, :1])
        slots = [nn.functional.pad(values, (0, SLOT_WIDTH - WORD_DIM)) for values in [mean, task, cross]]
        text = torch.cat(slots, 1) * gate * present.unsqueeze(1)
        fused = torch.cat([market, text, availability], 1)
        logits = self.head(fused).squeeze(-1)
        if details:
            return logits, {"mean": mean, "task": task, "cross": cross, "gate": gate,
                            "fused": fused, "weights": weights_by_component}
        return logits

    def penalty(self):
        return sum(value.square().sum() for name, value in self.named_parameters() if not name.endswith("bias"))

def predict_prepared(model, prepared):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(prepared["frame"]), 128):
            positions = np.arange(start, min(start + 128, len(prepared["frame"])))
            output.append(torch.sigmoid(model(*make_batch(prepared, positions))).numpy())
    return np.concatenate(output)

class FusionPredictor:
    def __init__(self, scaler, model, epochs):
        self.scaler, self.model, self.epochs = scaler, model, int(epochs)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, frame, article_vectors):
        probability = predict_prepared(self.model, self.scaler.prepare(frame, article_vectors))
        return np.column_stack([1 - probability, probability])

    def predict(self, frame, article_vectors):
        return (self.predict_proba(frame, article_vectors)[:, 1] >= .5).astype(int)

def score_binary(y, probability):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    predicted = probability >= .5
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = sum((bins == b).mean() * abs(probability[bins == b].mean() - y[bins == b].mean()) for b in np.unique(bins))
    return {"mcc": matthews_corrcoef(y, predicted), "macro_f1": f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, predicted), "accuracy": accuracy_score(y, predicted),
            "roc_auc": roc_auc_score(y, probability), "brier": brier_score_loss(y, probability),
            "log_loss": log_loss(y, np.clip(probability, 1e-15, 1 - 1e-15), labels=[0, 1]), "ece_10": ece}

def fit_fusion(train, validation, article_vectors, seed, flags, market_features, availability_features, fixed_epochs=None):
    started = time.perf_counter()
    scaler = SequenceScaler(market_features, availability_features).fit(train)
    prepared = scaler.prepare(train, article_vectors)
    evaluation = scaler.prepare(validation, article_vectors) if validation is not None else None
    model = ControlledFusion(seed, flags)
    adam = {**MODEL_CONFIG["adam"], "betas": tuple(MODEL_CONFIG["adam"]["betas"])}
    optimizer = torch.optim.Adam(model.parameters(), **adam)
    rng = np.random.RandomState(seed + MODEL_CONFIG["shuffle_seed_offset"])
    y = torch.as_tensor(train.y.to_numpy(float), dtype=torch.float64)
    best_state, best_key, best_epoch, stale, history = None, None, 0, 0, []
    limit = MODEL_CONFIG["max_epochs"] if fixed_epochs is None else int(fixed_epochs)
    if limit < 1 or (fixed_epochs is not None and evaluation is not None):
        raise ValueError("Fixed-duration fits must exclude validation scoring.")
    for epoch in range(1, limit + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), MODEL_CONFIG["batch_size"]):
            positions = order[start:start + MODEL_CONFIG["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*make_batch(prepared, positions))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[positions])
            loss = loss + MODEL_CONFIG["alpha"] / (2 * len(positions)) * model.penalty()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective.")
            loss.backward()
            optimizer.step()
        row = {"epoch": epoch, "train_log_loss": score_binary(train.y, predict_prepared(model, prepared))["log_loss"]}
        if evaluation is not None:
            metrics = score_binary(validation.y, predict_prepared(model, evaluation))
            row.update({"val_log_loss": metrics["log_loss"], "val_mcc": metrics["mcc"], "val_macro_f1": metrics["macro_f1"]})
            key = metrics["mcc"], metrics["macro_f1"], -metrics["log_loss"]
            if best_key is None or key > best_key:
                best_state, best_key, best_epoch, stale = copy.deepcopy(model.state_dict()), key, epoch, 0
            else:
                stale += 1
        history.append(row)
        if evaluation is not None and stale >= MODEL_CONFIG["patience"]:
            break
    if evaluation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = limit
    model.eval()
    return FusionPredictor(scaler, model, best_epoch), best_epoch, epoch, history, time.perf_counter() - started

def bridge_from_sklearn(pipeline, seed, market_features, availability_features):
    original = pipeline.named_steps["scaler"]
    scaler = SequenceScaler(market_features, availability_features)
    scaler.market, scaler.availability = copy.deepcopy(original.market), copy.deepcopy(original.availability)
    scaler.text_scale_ = original.text.scale_[:WORD_DIM].copy()
    scaler.n_samples_seen_, scaler.news_n_samples_seen_ = original.n_samples_seen_, original.news_n_samples_seen_
    model = ControlledFusion(seed, (0, 0, 0))
    reference = pipeline.named_steps["mlp"]
    active = np.r_[np.arange(14 + SLOT_WIDTH), np.arange(FUSION_WIDTH - 2, FUSION_WIDTH)]
    with torch.no_grad():
        first = model.head[0].weight.detach().numpy().T.copy()
        first[active] = reference.coefs_[0]
        model.head[0].weight.copy_(torch.from_numpy(first.T))
        for index, layer in enumerate([model.head[0], model.head[2], model.head[4]]):
            if index:
                layer.weight.copy_(torch.from_numpy(reference.coefs_[index].T))
            layer.bias.copy_(torch.from_numpy(reference.intercepts_[index]))
    return FusionPredictor(scaler, model.eval(), len(reference.loss_curve_))
