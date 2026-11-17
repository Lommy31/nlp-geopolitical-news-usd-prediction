import copy

import numpy as np
import torch
from torch import nn

from src.task_3.training.models import FeatureTransform, select_checkpoint
from src.task_3.training.controlled_fusion import score_binary

class FusionHead(nn.Module):
    def __init__(self, seed, width, dropout):
        super().__init__()
        self.hidden = nn.Linear(14 + 3 * width + 2, 16, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(16, 1, dtype=torch.float64)
        rng = np.random.RandomState(seed)
        bound = np.sqrt(6 / 30)
        blocks = [rng.uniform(-bound, bound, (14, 16))]
        bias = rng.uniform(-bound, bound, 16)
        for count, offset in [(width, 1000003), (width, 3000003), (width, 4000003), (2, 2000003)]:
            bound = np.sqrt(6 / (count + 16))
            blocks.append(np.random.RandomState(seed + offset).uniform(-bound, bound, (count, 16)))
        bound = np.sqrt(6 / 17)
        output, output_bias = rng.uniform(-bound, bound, (16, 1)), rng.uniform(-bound, bound, 1)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.from_numpy(np.vstack(blocks).T))
            self.hidden.bias.copy_(torch.from_numpy(bias))
            self.output.weight.copy_(torch.from_numpy(output.T))
            self.output.bias.copy_(torch.from_numpy(output_bias))

    def forward(self, values):
        return self.output(self.dropout(torch.relu(self.hidden(values)))).squeeze(-1)

class FusionModel(nn.Module):
    def __init__(self, seed, width, flags, dropout):
        super().__init__()
        self.width, self.flags = width, tuple(flags)
        self.head = FusionHead(seed, width, dropout)
        self.task_query = nn.Parameter(torch.zeros(width, dtype=torch.float64)) if flags[0] else None
        self.cross_query = nn.Linear(14, width, dtype=torch.float64) if flags[1] else None
        self.gate = nn.Linear(15, 1, dtype=torch.float64) if flags[2] else None
        with torch.no_grad():
            if flags[0]:
                bound = np.sqrt(6 / (width + 1))
                self.task_query.copy_(torch.from_numpy(np.random.RandomState(seed + 5000003).uniform(-bound, bound, width)))
            for layer, offset in [(self.cross_query, 6000003), (self.gate, 7000003)]:
                if layer is not None:
                    rng = np.random.RandomState(seed + offset)
                    bound = np.sqrt(6 / (layer.in_features + layer.out_features))
                    layer.weight.copy_(torch.from_numpy(rng.uniform(-bound, bound, (layer.in_features, layer.out_features)).T))
                    layer.bias.copy_(torch.from_numpy(rng.uniform(-bound, bound, layer.out_features)))

    def forward(self, market, articles, padding, availability, details=False):
        if padding.dtype != torch.bool:
            raise ValueError("Expected a boolean padding mask.")
        counts = (~padding).sum(1)
        present = counts.gt(0)
        assert torch.equal(present, availability[:, 1].gt(0))
        clean = articles.masked_fill(padding.unsqueeze(-1), 0.)
        mean = clean.sum(1) / counts.clamp_min(1).unsqueeze(-1)
        summaries = [mean, torch.zeros_like(mean), torch.zeros_like(mean)]
        positions = torch.where(present)[0]
        weights = {}
        for slot, active in [(1, self.flags[0]), (2, self.flags[1])]:
            if active and len(positions):
                news, mask = clean[positions], padding[positions]
                query = self.task_query.expand(len(positions), -1) if slot == 1 else self.cross_query(market[positions])
                scores = (news * query.unsqueeze(1)).sum(-1) / np.sqrt(self.width)
                weights[slot] = torch.softmax(scores.masked_fill(mask, -torch.inf), 1)
                summaries[slot] = torch.zeros_like(mean).index_copy(0, positions, (weights[slot].unsqueeze(-1) * news).sum(1))
        gate = torch.sigmoid(self.gate(torch.cat([market, availability[:, :1]], 1))) if self.flags[2] else torch.ones_like(availability[:, :1])
        text = torch.cat(summaries, 1) * gate * present.unsqueeze(1)
        fused = torch.cat([market, text, availability], 1)
        logits = self.head(fused)
        return (logits, {"fused": fused, "gate": gate, "weights": weights}) if details else logits

class SequenceTransform(FeatureTransform):
    def prepare(self, frame, vectors):
        values = self.transform(frame)
        width = self.pca_components or len(self.text_features)
        sequences = []
        for articles in frame.articles:
            news = np.stack([vectors[a["article_id"]] for a in articles]) if articles else np.empty((0, len(self.text_features)))
            if len(news):
                if self.pca is not None:
                    news = self.pca.transform(news)
                news = (news - self.text.mean_ if self.text.with_mean else news) / self.text.scale_
            else:
                news = np.empty((0, width))
            sequences.append(news)
        means = np.stack([news.mean(0) if len(news) else np.zeros(width) for news in sequences])
        np.testing.assert_allclose(means, values[:, 14:14 + width], atol=1e-11, rtol=1e-11)
        return {"market": values[:, :14], "availability": values[:, -2:], "sequences": sequences}

def batch(prepared, positions, width):
    sequences = [prepared["sequences"][i] for i in positions]
    count = max(1, max(map(len, sequences), default=0))
    news, padding = np.zeros((len(positions), count, width)), np.ones((len(positions), count), dtype=bool)
    for i, values in enumerate(sequences):
        news[i, :len(values)], padding[i, :len(values)] = values, False
    return (torch.as_tensor(prepared["market"][positions], dtype=torch.float64), torch.from_numpy(news), torch.from_numpy(padding),
            torch.as_tensor(prepared["availability"][positions], dtype=torch.float64))

def probability(model, prepared):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(prepared["sequences"]), 128):
            positions = np.arange(start, min(start + 128, len(prepared["sequences"])))
            values.append(torch.sigmoid(model(*batch(prepared, positions, model.width))).numpy())
    return np.concatenate(values)

class FusionPredictor:
    def __init__(self, transform, model, epochs, steps):
        self.transform, self.model = transform, model.eval()
        self.epochs, self.optimizer_steps = epochs, steps

    def predict_proba(self, frame, vectors):
        p = probability(self.model, self.transform.prepare(frame, vectors))
        return np.column_stack([1 - p, p])

def fit_fusion(train, val, vectors, seed, flags, candidate, market, text, availability, epochs=None):
    if (val is None) != (epochs is not None):
        raise ValueError("Fixed-duration refits must exclude validation scoring.")
    transform = SequenceTransform(market, text, availability, "text", candidate["centered"], candidate.get("pca")).fit(train)
    prepared = transform.prepare(train, vectors)
    evaluation = transform.prepare(val, vectors) if val is not None else None
    model = FusionModel(seed, candidate.get("pca") or len(text), flags, candidate["dropout"])
    torch.manual_seed(seed + 9000003)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-8, foreach=False)
    rng = np.random.RandomState(seed + 8000003)
    y = torch.as_tensor(train.y.to_numpy(float), dtype=torch.float64)
    best_state, best_loss, best_epoch, stale, trace = None, np.inf, 0, 0, []
    for epoch in range(1, (60 if epochs is None else epochs) + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), 32):
            positions = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(*batch(prepared, positions, model.width)), y[positions])
            penalty = sum(value.square().sum() for name, value in model.named_parameters() if not name.endswith("bias"))
            loss = loss + candidate["alpha"] / (2 * len(positions)) * penalty
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective.")
            loss.backward()
            optimizer.step()
        row = {"epoch": epoch, "train_log_loss": score_binary(train.y, probability(model, prepared))["log_loss"]}
        if val is not None:
            row.update({f"val_{key}": value for key, value in score_binary(val.y, probability(model, evaluation)).items()})
        trace.append(row)
        if val is not None:
            updated, smoothed = select_checkpoint(trace, best_loss)
            row["smoothed_log_loss"], row["checkpoint_updated"] = smoothed, updated
            if updated:
                best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), smoothed, epoch, 0
            elif smoothed is not None:
                stale += 1
            if stale >= 10:
                break
    if val is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    return FusionPredictor(transform, model, best_epoch, int(np.ceil(len(train) / 32)) * best_epoch), best_epoch, epoch, trace
