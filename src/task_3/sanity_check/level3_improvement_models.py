import copy
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.task_3.sanity_check.level1_improvement_models import select_checkpoint
from src.task_3.training.controlled_fusion import (
    ControlledFusion, FusionPredictor, SequenceScaler, WORD_FEATURES,
    MODEL_CONFIG, FUSION_WIDTH, make_batch, predict_prepared, score_binary,
)

class CompactHead(nn.Module):
    def __init__(self, seed, dropout):
        super().__init__()
        indices = np.r_[np.arange(314), np.arange(782, 1082), np.arange(1550, 1850), np.arange(FUSION_WIDTH - 2, FUSION_WIDTH)]
        self.register_buffer("active_indices", torch.from_numpy(indices))
        self.hidden = nn.Linear(len(indices), 16, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(16, 1, dtype=torch.float64)
        rng = np.random.RandomState(seed)
        bound = np.sqrt(6 / 30)
        blocks = [rng.uniform(-bound, bound, (14, 16))]
        bias = rng.uniform(-bound, bound, 16)
        for width, offset in [(300, 1000003), (300, 3000003), (300, 4000003), (2, 2000003)]:
            bound = np.sqrt(6 / (width + 16))
            blocks.append(np.random.RandomState(seed + offset).uniform(-bound, bound, (width, 16)))
        bound = np.sqrt(6 / 17)
        output, output_bias = rng.uniform(-bound, bound, (16, 1)), rng.uniform(-bound, bound, 1)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.from_numpy(np.vstack(blocks).T))
            self.hidden.bias.copy_(torch.from_numpy(bias))
            self.output.weight.copy_(torch.from_numpy(output.T))
            self.output.bias.copy_(torch.from_numpy(output_bias))

    def forward(self, values):
        selected = values.index_select(1, self.active_indices)
        return self.output(self.dropout(torch.relu(self.hidden(selected))))

class ImprovedFusion(ControlledFusion):
    def __init__(self, seed, flags, dropout=.5):
        super().__init__(seed, flags)
        self.head = CompactHead(seed, dropout)

class CenteredSequenceScaler(SequenceScaler):
    def __init__(self, market_features, availability_features, centered):
        super().__init__(market_features, availability_features)
        self.centered = centered

    def fit(self, frame):
        super().fit(frame)
        news = frame.loc[frame.has_news.eq(1), WORD_FEATURES]
        self.text = StandardScaler(with_mean=self.centered).fit(news)
        self.text_scale_ = self.text.scale_.copy()
        self.text_mean_ = self.text.mean_.copy() if self.centered else np.zeros(300)
        return self

    def prepare(self, frame, article_vectors):
        sequences = []
        for row in frame.itertuples():
            sequence = np.stack([article_vectors[a["article_id"]] for a in row.articles]) if row.articles else np.empty((0, 300))
            if len(sequence) != row.n_berita:
                raise ValueError("Article count differs from its fixing window.")
            sequences.append((sequence - self.text_mean_) / self.text_scale_)
        means = np.stack([sequence.mean(0) if len(sequence) else np.zeros(300) for sequence in sequences])
        present = frame.has_news.eq(1).to_numpy()
        expected = np.zeros_like(means)
        expected[present] = (frame.loc[present, WORD_FEATURES].to_numpy() - self.text_mean_) / self.text_scale_
        np.testing.assert_allclose(means, expected, atol=1e-12, rtol=1e-12)
        return {"frame": frame, "market": self.market.transform(frame[self.market_features]),
                "availability": self.availability.transform(frame[self.availability_features]), "sequences": sequences, "means": means}

class ImprovedPredictor(FusionPredictor):
    def __init__(self, scaler, model, epochs, optimizer_steps):
        super().__init__(scaler, model, epochs)
        self.optimizer_steps = int(optimizer_steps)

def fit_improved(train, val, vectors, seed, flags, market, availability, centered=False, alpha=1., dropout=.5, fixed_epochs=None):
    started = time.perf_counter()
    scaler = CenteredSequenceScaler(market, availability, centered).fit(train)
    prepared = scaler.prepare(train, vectors)
    evaluation = scaler.prepare(val, vectors) if val is not None else None
    model = ImprovedFusion(seed, flags, dropout)
    initial = copy.deepcopy(model.state_dict())
    torch.manual_seed(seed + 9000003)
    adam = {**MODEL_CONFIG["adam"], "betas": tuple(MODEL_CONFIG["adam"]["betas"])}
    optimizer = torch.optim.Adam(model.parameters(), **adam)
    rng = np.random.RandomState(seed + MODEL_CONFIG["shuffle_seed_offset"])
    y = torch.as_tensor(train.y.to_numpy(float), dtype=torch.float64)
    best_state, best_score, best_epoch, stale, steps, best_steps, trace = None, np.inf, 0, 0, 0, 0, []
    limit = 60 if fixed_epochs is None else int(fixed_epochs)
    if limit < 1 or (val is not None and fixed_epochs is not None):
        raise ValueError("Fixed-duration refits must exclude validation scoring.")
    for epoch in range(1, limit + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), 32):
            positions = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(*make_batch(prepared, positions)), y[positions])
            loss = loss + alpha / (2 * len(positions)) * model.penalty()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective.")
            loss.backward()
            optimizer.step()
            steps += 1
        row = {"epoch": epoch, "train_log_loss": score_binary(train.y, predict_prepared(model, prepared))["log_loss"]}
        if evaluation is not None:
            metrics = score_binary(val.y, predict_prepared(model, evaluation))
            row.update({f"val_{key}": value for key, value in metrics.items()})
        trace.append(row)
        if evaluation is not None:
            improved, smooth = select_checkpoint(trace, best_score)
            row["smoothed_log_loss"], row["checkpoint_updated"] = smooth, improved
            if improved:
                best_state, best_score, best_epoch, best_steps, stale = copy.deepcopy(model.state_dict()), smooth, epoch, steps, 0
            elif smooth is not None:
                stale += 1
            if stale >= 10:
                break
    if evaluation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch, best_steps = limit, steps
    changes = {name: float((parameter.detach() - initial[name]).norm()) for name, parameter in model.named_parameters()
               if name == "task_query" or name.startswith(("cross_query.", "gate."))}
    predictor = ImprovedPredictor(scaler, model.eval(), best_epoch, best_steps)
    return predictor, best_epoch, epoch, trace, time.perf_counter() - started, changes
