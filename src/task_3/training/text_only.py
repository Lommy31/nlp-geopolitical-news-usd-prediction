import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

class TextOnlyScaler(TransformerMixin, BaseEstimator):
    def __init__(self, text_features, availability_features):
        self.text_features = text_features
        self.availability_features = availability_features

    def fit(self, frame, y=None):
        self.features_ = [*self.text_features, *self.availability_features]
        news = frame.loc[frame.has_news.eq(1), self.text_features]
        if not len(news):
            raise ValueError("No news-present training windows are available.")
        self.text = StandardScaler(with_mean=False).fit(news)
        self.availability = StandardScaler(with_mean=False).fit(frame[self.availability_features])
        self.n_samples_seen_, self.news_n_samples_seen_ = len(frame), len(news)
        self.n_features_in_ = len(self.features_)
        self.feature_names_in_ = np.asarray(self.features_, dtype=object)
        return self

    def transform(self, frame):
        check_is_fitted(self, "text")
        text = self.text.transform(frame[self.text_features])
        availability = self.availability.transform(frame[self.availability_features])
        values = np.column_stack([np.zeros((len(frame), 14)), text, availability])
        if values[frame.has_news.eq(0), 14:].any() or not np.isfinite(values).all():
            raise ValueError("Invalid text-only inputs or empty-news handling.")
        return values
