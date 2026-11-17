import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

class BlockScaler(TransformerMixin, BaseEstimator):
    def __init__(self, market_features, text_features, availability_features):
        self.market_features = market_features
        self.text_features = text_features
        self.availability_features = availability_features

    def fit(self, frame, y=None):
        self.features_ = [*self.market_features, *self.text_features, *self.availability_features]
        self.market = StandardScaler().fit(frame[self.market_features])
        news = frame.loc[frame.has_news.eq(1), self.text_features]
        if not len(news):
            raise ValueError("No news-present training windows are available.")
        self.text = StandardScaler(with_mean=False).fit(news)
        self.availability = StandardScaler(with_mean=False).fit(frame[self.availability_features])
        self.mean_ = np.r_[self.market.mean_, np.zeros(len(self.text_features) + len(self.availability_features))]
        self.scale_ = np.r_[self.market.scale_, self.text.scale_, self.availability.scale_]
        self.n_samples_seen_ = len(frame)
        self.news_n_samples_seen_ = len(news)
        self.n_features_in_ = len(self.features_)
        self.feature_names_in_ = np.asarray(self.features_, dtype=object)
        return self

    def transform(self, frame):
        check_is_fitted(self, "scale_")
        values = (frame[self.features_].to_numpy(dtype=float) - self.mean_) / self.scale_
        empty = frame.has_news.eq(0).to_numpy()
        if values[empty, len(self.market_features):].any():
            raise ValueError("Empty-news windows must have zero text and availability features.")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite scaled features.")
        return values

class MarketAlignedMLP(MLPClassifier):
    def _init_coef(self, fan_in, fan_out, dtype):
        if fan_in != 14 + 768 + 2:
            return super()._init_coef(fan_in, fan_out, dtype)
        market, bias = super()._init_coef(14, fan_out, dtype)
        blocks = [market]
        for width, offset in [(768, 1000003), (2, 2000003)]:
            rng = check_random_state(int(self.random_state) + offset)
            bound = np.sqrt(6.0 / (width + fan_out))
            blocks.append(rng.uniform(-bound, bound, (width, fan_out)).astype(dtype))
        return np.vstack(blocks), bias
