import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class LastDirectionClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        returns = self._returns(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or not len(labels) or len(returns) != len(labels):
            raise ValueError("Features and labels must contain the same nonzero number of rows.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Labels must be 0 or 1.")
        self.fallback_p_up_ = float(labels.mean())
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = 1
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fallback_p_up_")
        returns = self._returns(X)
        p_up = np.where(returns > 0, 1.0, np.where(returns < 0, 0.0, self.fallback_p_up_))
        return np.column_stack([1 - p_up, p_up])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @staticmethod
    def _returns(X):
        if hasattr(X, "columns") and list(X.columns) != ["log_return_t"]:
            raise ValueError("Pass only the log_return_t column.")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != 1 or np.isinf(values).any():
            raise ValueError("Expected one return column with finite or missing values.")
        return values[:, 0]
