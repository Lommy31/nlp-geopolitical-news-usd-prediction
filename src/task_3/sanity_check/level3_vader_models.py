import copy
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.task_3.sanity_check.level1_improvement_models import select_checkpoint
from src.task_3.sanity_check.level3_improvement_models import (
    CompactHead, CenteredSequenceScaler, ImprovedFusion, ImprovedPredictor,
)
from src.task_3.training.controlled_fusion import MODEL_CONFIG, FUSION_WIDTH, make_batch, predict_prepared, score_binary

VADER = [f"vader_{span}_{score}" for span in ["title", "lead"] for score in ["compound", "pos", "neg", "neu"]]

class VaderHead(CompactHead):
    def __init__(self, seed):
        super().__init__(seed, .5)
        bound = np.sqrt(6 / (len(VADER) + 16))
        weights = np.random.RandomState(seed + 10000019).uniform(-bound, bound, (len(VADER), 16))
        self.vader_weight = nn.Parameter(torch.from_numpy(weights.T.copy()))

    def forward(self, values):
        shared = self.hidden(values.index_select(1, self.active_indices))
        extra = nn.functional.linear(values[:, FUSION_WIDTH:], self.vader_weight)
        return self.output(self.dropout(torch.relu(shared + extra)))

class VaderFusion(ImprovedFusion):
    def __init__(self, seed, flags):
        super().__init__(seed, flags)
        self.head = VaderHead(seed)

class VaderScaler(CenteredSequenceScaler):
    def fit(self, frame):
        super().fit(frame)
        self.vader = StandardScaler(with_mean=False).fit(frame.loc[frame.has_news.eq(1), VADER])
        return self

    def prepare(self, frame, vectors):
        result = super().prepare(frame, vectors)
        extra = self.vader.transform(frame[VADER])
        assert not extra[frame.has_news.eq(0)].any()
        result["availability"] = np.column_stack([result["availability"], extra])
        return result

def fit_vader(train, val, vectors, seed, flags, market, availability, fixed_epochs=None):
    started = time.perf_counter()
    scaler = VaderScaler(market, availability, centered=False).fit(train)
    prepared = scaler.prepare(train, vectors)
    evaluation = scaler.prepare(val, vectors) if val is not None else None
    model = VaderFusion(seed, flags)
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
            loss = loss + model.penalty() / (2 * len(positions))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective.")
            loss.backward()
            optimizer.step()
            steps += 1
        row = {"epoch": epoch, "train_log_loss": score_binary(train.y, predict_prepared(model, prepared))["log_loss"]}
        if evaluation is not None:
            row.update({f"val_{key}": value for key, value in score_binary(val.y, predict_prepared(model, evaluation)).items()})
        trace.append(row)
        if evaluation is not None:
            updated, smooth = select_checkpoint(trace, best_score)
            row["smoothed_log_loss"], row["checkpoint_updated"] = smooth, updated
            if updated:
                best_state, best_score, best_epoch, best_steps, stale = copy.deepcopy(model.state_dict()), smooth, epoch, steps, 0
            elif smooth is not None:
                stale += 1
            if stale >= 10:
                break
    if evaluation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch, best_steps = limit, steps
    return ImprovedPredictor(scaler, model.eval(), best_epoch, best_steps), best_epoch, epoch, trace, time.perf_counter() - started
