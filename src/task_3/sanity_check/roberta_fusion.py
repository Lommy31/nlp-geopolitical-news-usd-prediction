import copy
import time

import numpy as np
import torch
from torch import nn

from src.task_3.training.controlled_fusion import ControlledFusion, MODEL_CONFIG, score_binary
from src.task_3.training.fusion import BlockScaler

DIMENSION = 768
TEXT_FEATURES = [f"roberta_{i:03d}" for i in range(DIMENSION)]

class RobertaFusion(ControlledFusion):
    def __init__(self, seed, flags):
        super().__init__(seed, (0, 0, 0))
        self.flags = tuple(flags)
        if len(self.flags) != 3 or not set(self.flags).issubset({0, 1}):
            raise ValueError("Expected query, cross-attention and gate flags.")
        if self.flags[0]:
            bound = np.sqrt(6 / (DIMENSION + 1))
            values = np.random.RandomState(seed + 5000003).uniform(-bound, bound, DIMENSION)
            self.task_query = nn.Parameter(torch.from_numpy(values))
        self.cross_query = nn.Linear(14, DIMENSION, dtype=torch.float64) if self.flags[1] else None
        self.gate = nn.Linear(15, 1, dtype=torch.float64) if self.flags[2] else None
        with torch.no_grad():
            for layer, offset in [(self.cross_query, 6000003), (self.gate, 7000003)]:
                if layer is not None:
                    rng = np.random.RandomState(seed + offset)
                    bound = np.sqrt(6 / (layer.in_features + layer.out_features))
                    weight = rng.uniform(-bound, bound, (layer.in_features, layer.out_features))
                    bias = rng.uniform(-bound, bound, layer.out_features)
                    layer.weight.copy_(torch.from_numpy(weight.T))
                    layer.bias.copy_(torch.from_numpy(bias))

    def forward(self, market, articles, key_padding_mask, availability, details=False):
        if articles.shape[-1] != DIMENSION or key_padding_mask.dtype != torch.bool:
            raise ValueError("Expected 768-dimensional articles and a boolean padding mask.")
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
                scores = (news * query.unsqueeze(1)).sum(-1) / np.sqrt(DIMENSION)
                weights = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=1)
                summary = torch.zeros_like(mean).index_copy(0, positions, (weights.unsqueeze(-1) * news).sum(1))
                if name == "task":
                    task = summary
                else:
                    cross = summary
                weights_by_component[name] = weights
        gate = torch.sigmoid(self.gate(torch.cat([market, availability[:, :1]], 1))) if self.flags[2] else torch.ones_like(availability[:, :1])
        text = torch.cat([mean, task, cross], 1) * gate * present.unsqueeze(1)
        fused = torch.cat([market, text, availability], 1)
        logits = self.head(fused).squeeze(-1)
        if details:
            return logits, {"mean": mean, "task": task, "cross": cross, "gate": gate, "fused": fused, "weights": weights_by_component}
        return logits

class ArticleScaler:
    def __init__(self, market_features, availability_features):
        self.market_features = market_features
        self.availability_features = availability_features

    def fit(self, frame):
        fitted = BlockScaler(self.market_features, TEXT_FEATURES, self.availability_features).fit(frame)
        self.market, self.text, self.availability = fitted.market, fitted.text, fitted.availability
        self.text_scale_ = self.text.scale_.copy()
        self.n_samples_seen_, self.news_n_samples_seen_ = fitted.n_samples_seen_, fitted.news_n_samples_seen_
        return self

    def prepare(self, frame, article_vectors):
        sequences = []
        for row in frame.itertuples():
            sequence = np.stack([article_vectors[a["article_id"]] for a in row.articles]).astype(np.float64) if row.articles else np.empty((0, DIMENSION))
            if len(sequence) != row.n_berita:
                raise ValueError("Article count differs from sequence length.")
            sequences.append(sequence / self.text_scale_)
        means = np.stack([sequence.mean(0) if len(sequence) else np.zeros(DIMENSION) for sequence in sequences])
        np.testing.assert_allclose(means, frame[TEXT_FEATURES].to_numpy() / self.text_scale_, atol=1e-12, rtol=1e-12)
        return {"frame": frame, "market": self.market.transform(frame[self.market_features]),
                "availability": self.availability.transform(frame[self.availability_features]), "sequences": sequences}

def make_batch(prepared, positions):
    sequences = [prepared["sequences"][index] for index in positions]
    width = max(1, max(map(len, sequences), default=0))
    values = np.zeros((len(positions), width, DIMENSION))
    padding = np.ones((len(positions), width), dtype=bool)
    for i, sequence in enumerate(sequences):
        values[i, :len(sequence)] = sequence
        padding[i, :len(sequence)] = False
    return (torch.as_tensor(prepared["market"][positions], dtype=torch.float64), torch.from_numpy(values), torch.from_numpy(padding),
            torch.as_tensor(prepared["availability"][positions], dtype=torch.float64))

def predict_prepared(model, prepared):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(prepared["frame"]), 128):
            positions = np.arange(start, min(start + 128, len(prepared["frame"])))
            values.append(torch.sigmoid(model(*make_batch(prepared, positions))).numpy())
    return np.concatenate(values)

class Predictor:
    def __init__(self, scaler, model, epochs, optimizer_steps):
        self.scaler, self.model = scaler, model
        self.epochs, self.optimizer_steps = int(epochs), int(optimizer_steps)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, frame, article_vectors):
        p = predict_prepared(self.model, self.scaler.prepare(frame, article_vectors))
        return np.column_stack([1 - p, p])

    def predict(self, frame, article_vectors):
        return (self.predict_proba(frame, article_vectors)[:, 1] >= .5).astype(int)

def fit_fusion(train, val, vectors, seed, flags, market, availability, fixed_epochs=None):
    started = time.perf_counter()
    scaler = ArticleScaler(market, availability).fit(train)
    prepared = scaler.prepare(train, vectors)
    evaluation = scaler.prepare(val, vectors) if val is not None else None
    model = RobertaFusion(seed, flags)
    initial = copy.deepcopy(model.state_dict())
    adam = {**MODEL_CONFIG["adam"], "betas": tuple(MODEL_CONFIG["adam"]["betas"])}
    optimizer = torch.optim.Adam(model.parameters(), **adam)
    rng = np.random.RandomState(seed + MODEL_CONFIG["shuffle_seed_offset"])
    y = torch.as_tensor(train.y.to_numpy(float), dtype=torch.float64)
    best_state, best_key, best_epoch, stale, history = None, None, 0, 0, []
    limit = MODEL_CONFIG["max_epochs"] if fixed_epochs is None else int(fixed_epochs)
    if limit < 1 or (fixed_epochs is not None and val is not None):
        raise ValueError("Fixed-duration fits must exclude validation scoring.")
    for epoch in range(1, limit + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), MODEL_CONFIG["batch_size"]):
            positions = order[start:start + MODEL_CONFIG["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(*make_batch(prepared, positions)), y[positions])
            loss = loss + MODEL_CONFIG["alpha"] / (2 * len(positions)) * model.penalty()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
        row = {"epoch": epoch, "train_log_loss": score_binary(train.y, predict_prepared(model, prepared))["log_loss"]}
        if evaluation is not None:
            metrics = score_binary(val.y, predict_prepared(model, evaluation))
            row.update({f"val_{key}": metrics[key] for key in ["mcc", "macro_f1", "log_loss"]})
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
    steps = int(np.ceil(len(train) / MODEL_CONFIG["batch_size"])) * best_epoch
    changes = {name: float((value.detach() - initial[name]).norm()) for name, value in model.named_parameters()
               if name == "task_query" or name.startswith(("cross_query.", "gate."))}
    return Predictor(scaler, model.eval(), best_epoch, steps), best_epoch, epoch, history, time.perf_counter() - started, changes
