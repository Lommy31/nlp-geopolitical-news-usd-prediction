import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

class AvailabilityScaler(TransformerMixin, BaseEstimator):
    def __init__(self, market_features, availability_features):
        self.market_features = market_features
        self.availability_features = availability_features

    def fit(self, frame, y=None):
        if len(self.market_features) != 14 or len(self.availability_features) != 2:
            raise ValueError("Expected 14 market and 2 availability features.")
        self.features_ = [*self.market_features, *self.availability_features]
        self.market = StandardScaler().fit(frame[self.market_features])
        self.availability = StandardScaler(with_mean=False).fit(frame[self.availability_features])
        self.mean_ = np.r_[self.market.mean_, np.zeros(768 + 2)]
        self.scale_ = np.r_[self.market.scale_, np.ones(768), self.availability.scale_]
        self.n_samples_seen_ = len(frame)
        self.news_days_ = int(frame.has_news.sum())
        self.n_features_in_ = len(self.features_)
        self.feature_names_in_ = np.asarray(self.features_, dtype=object)
        return self

    def transform(self, frame):
        check_is_fitted(self, "scale_")
        market = self.market.transform(frame[self.market_features])
        availability = self.availability.transform(frame[self.availability_features])
        result = np.column_stack([market, np.zeros((len(frame), 768)), availability])
        if availability[frame.has_news.eq(0)].any():
            raise ValueError("Empty-news windows must have zero availability features.")
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite scaled features.")
        return result
