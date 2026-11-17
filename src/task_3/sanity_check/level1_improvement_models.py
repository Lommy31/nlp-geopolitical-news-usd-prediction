import copy

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.task_3.training.controlled_fusion import score_binary
from src.task_3.training.fusion import BlockScaler, MarketAlignedMLP

SEEDS = [42, 123, 2026]
MAX_EPOCHS = 60
PATIENCE = 10
SMOOTHING = 5
MIN_DELTA = .001

class FeatureTransform:
    def __init__(self, market, text, availability, view, centered=False, pca_components=None):
        self.market_features = market
        self.text_features = text
        self.availability_features = availability
        self.view, self.centered, self.pca_components = view, centered, pca_components

    def fit(self, frame):
        self.market = StandardScaler().fit(frame[self.market_features])
        self.availability = StandardScaler(with_mean=False).fit(frame[self.availability_features])
        self.n_samples_seen_ = len(frame)
        self.pca = None
        if self.view == "text":
            news = frame.loc[frame.has_news.eq(1), self.text_features]
            if not len(news):
                raise ValueError("No news-present training rows.")
            self.news_n_samples_seen_ = len(news)
            if self.pca_components is not None:
                self.pca = PCA(n_components=self.pca_components, svd_solver="full").fit(news)
                self.text = StandardScaler().fit(self.pca.transform(news))
            else:
                self.text = StandardScaler(with_mean=self.centered).fit(news)
        values = self.transform(frame)
        self.n_features_out_ = values.shape[1]
        return self

    def transform(self, frame):
        blocks = [self.market.transform(frame[self.market_features])]
        if self.view == "text":
            width = self.pca_components or len(self.text_features)
            text = np.zeros((len(frame), width))
            present = frame.has_news.eq(1).to_numpy()
            if present.any():
                values = frame.loc[present, self.text_features]
                if self.pca is not None:
                    values = self.pca.transform(values)
                text[present] = self.text.transform(values)
            blocks.append(text)
        if self.view in {"availability", "text"}:
            blocks.append(self.availability.transform(frame[self.availability_features]))
        values = np.column_stack(blocks)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite predictors.")
        return values

class SmallMLP(nn.Module):
    def __init__(self, seed, view, text_width, dropout):
        super().__init__()
        width = 14 + (text_width if view == "text" else 0) + (2 if view != "price" else 0)
        self.hidden = nn.Linear(width, 16, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(16, 1, dtype=torch.float64)
        rng = np.random.RandomState(seed)
        bound = np.sqrt(6 / (14 + 16))
        weights = [rng.uniform(-bound, bound, (14, 16))]
        bias = rng.uniform(-bound, bound, 16)
        for count, offset in [(text_width if view == "text" else 0, 1000003), (2 if view != "price" else 0, 2000003)]:
            if count:
                bound = np.sqrt(6 / (count + 16))
                weights.append(np.random.RandomState(seed + offset).uniform(-bound, bound, (count, 16)))
        bound = np.sqrt(6 / 17)
        output = rng.uniform(-bound, bound, (16, 1))
        output_bias = rng.uniform(-bound, bound, 1)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.from_numpy(np.vstack(weights).T))
            self.hidden.bias.copy_(torch.from_numpy(bias))
            self.output.weight.copy_(torch.from_numpy(output.T))
            self.output.bias.copy_(torch.from_numpy(output_bias))

    def forward(self, values):
        return self.output(self.dropout(torch.relu(self.hidden(values)))).squeeze(-1)

class Predictor:
    def __init__(self, transform, model, family, epochs=None, optimizer_steps=None):
        self.transform, self.model, self.family = transform, model, family
        self.epochs, self.optimizer_steps = epochs, optimizer_steps
        self.classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        values = self.transform.transform(frame)
        if self.family == "small_mlp":
            self.model.eval()
            with torch.no_grad():
                p = torch.sigmoid(self.model(torch.as_tensor(values, dtype=torch.float64))).numpy()
            return np.column_stack([1 - p, p])
        return self.model.predict_proba(values)

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] >= .5).astype(int)

def select_checkpoint(trace, best_score):
    if len(trace) < SMOOTHING:
        return False, None
    smoothed = float(np.mean([row["val_log_loss"] for row in trace[-SMOOTHING:]]))
    return smoothed < best_score - MIN_DELTA, smoothed

def fit_small(train, val, seed, candidate, market, text, availability, epochs=None):
    transform = FeatureTransform(market, text, availability, candidate["view"], candidate["centered"], candidate.get("pca")).fit(train)
    x_train = torch.as_tensor(transform.transform(train), dtype=torch.float64)
    x_val = torch.as_tensor(transform.transform(val), dtype=torch.float64) if val is not None else None
    model = SmallMLP(seed, candidate["view"], candidate.get("pca") or len(text), candidate["dropout"])
    torch.manual_seed(seed + 9000003)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-8, foreach=False)
    rng = np.random.RandomState(seed + 8000003)
    y = torch.as_tensor(train.y.to_numpy(float), dtype=torch.float64)
    best_state, best_score, best_epoch, stale, trace = None, np.inf, 0, 0, []
    for epoch in range(1, (MAX_EPOCHS if epochs is None else epochs) + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), 32):
            positions = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            penalty = model.hidden.weight.square().sum() + model.output.weight.square().sum()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x_train[positions]), y[positions])
            loss = loss + candidate["alpha"] / (2 * len(positions)) * penalty
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective.")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_p = torch.sigmoid(model(x_train)).numpy()
            val_p = torch.sigmoid(model(x_val)).numpy() if val is not None else None
        row = {"epoch": epoch, "train_log_loss": score_binary(train.y, train_p)["log_loss"]}
        if val is not None:
            metrics = score_binary(val.y, val_p)
            row.update({f"val_{key}": value for key, value in metrics.items()})
        trace.append(row)
        if val is not None:
            improved, smoothed = select_checkpoint(trace, best_score)
            row["smoothed_log_loss"] = smoothed
            row["checkpoint_updated"] = improved
            if improved:
                best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), smoothed, epoch, 0
            elif smoothed is not None:
                stale += 1
            if stale >= PATIENCE:
                break
    if val is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    steps = int(np.ceil(len(train) / 32)) * best_epoch
    return Predictor(transform, model.eval(), "small_mlp", best_epoch, steps), best_epoch, epoch, trace

def fit_logistic(train, candidate, market, text, availability):
    transform = FeatureTransform(market, text, availability, candidate["view"], candidate["centered"], candidate.get("pca")).fit(train)
    model = LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=2000, tol=1e-8, random_state=42)
    model.fit(transform.transform(train), train.y)
    if model.n_iter_.max() >= model.max_iter:
        raise ValueError("Logistic regression did not converge.")
    return Predictor(transform, model, "logistic")

def fit_legacy(train, val, seed, market, text, availability, config, selection="mcc", shuffled_y=None, scaled=None):
    transform = BlockScaler(market, text, availability).fit(train)
    x_train, x_val = scaled if scaled is not None else (transform.transform(train), transform.transform(val))
    model = MarketAlignedMLP(**config, random_state=seed)
    y = train.y.to_numpy() if shuffled_y is None else shuffled_y
    best, best_key, best_epoch, stale, trace = None, None, 0, 0, []
    limit, patience = (200, 25) if selection == "mcc" else (MAX_EPOCHS, PATIENCE)
    for epoch in range(1, limit + 1):
        model.partial_fit(x_train, y, classes=np.array([0, 1]))
        metrics = score_binary(val.y, model.predict_proba(x_val)[:, 1])
        row = {"epoch": epoch, **{f"val_{key}": value for key, value in metrics.items()}}
        trace.append(row)
        if selection == "mcc":
            key = metrics["mcc"], metrics["macro_f1"], -metrics["log_loss"]
            improved = best_key is None or key > best_key
        else:
            improved, key = select_checkpoint(trace, np.inf if best_key is None else best_key)
            row["smoothed_log_loss"] = key
        if improved:
            best, best_key, best_epoch, stale = copy.deepcopy(model), key, epoch, 0
        elif selection == "mcc" or key is not None:
            stale += 1
        row["checkpoint_updated"] = improved
        if stale >= patience:
            break
    return Predictor(transform, best, "legacy_mlp", best_epoch, int(np.ceil(len(train) / 32)) * best_epoch), best_epoch, epoch, trace
