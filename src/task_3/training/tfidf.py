import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

TFIDF_CONFIG = {
    "lowercase": False, "token_pattern": None, "ngram_range": (1, 2),
    "min_df": 2, "max_df": 1.0, "max_features": 20000,
    "sublinear_tf": True, "norm": "l2", "stop_words": None,
}
SVD_CONFIG = {"algorithm": "randomized", "n_iter": 7, "n_oversamples": 10}

class FoldTfidfSVD(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=128, padded_width=768, random_state=20261110):
        self.n_components = n_components
        self.padded_width = padded_width
        self.random_state = random_state

    @staticmethod
    def _articles(frame):
        articles = {}
        for window in frame.articles:
            for article in window:
                key, text = article["article_id"], article["text_tfidf"]
                if key in articles and articles[key] != text:
                    raise ValueError("One article ID has multiple text versions.")
                articles[key] = text
        ids = sorted(articles)
        return ids, [articles[key] for key in ids]

    def fit(self, frame, y=None):
        if not 1 <= self.n_components <= self.padded_width:
            raise ValueError("SVD dimensions must fit inside the padded text width.")
        ids, texts = self._articles(frame)
        if len(ids) < 2:
            raise ValueError("At least two training articles are required.")
        self.training_ids_ = ids
        self.training_dates_ = pd.to_datetime(frame.Date).dt.strftime("%Y-%m-%d").tolist()
        self.vectorizer_ = TfidfVectorizer(**TFIDF_CONFIG, tokenizer=str.split)
        matrix = self.vectorizer_.fit_transform(texts)
        self.rank_ = min(self.n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if self.rank_ < 1:
            raise ValueError("Training vocabulary is too small for SVD.")
        self.svd_ = TruncatedSVD(n_components=self.rank_, random_state=self.random_state, **SVD_CONFIG)
        with threadpool_limits(limits=1):
            self.svd_.fit(matrix)
        self.fit_shape_ = matrix.shape
        self.fit_nnz_ = int(matrix.nnz)
        self.text_features_ = [
            f"svd_{i:03d}" if i < self.n_components else f"zero_text_{i:03d}"
            for i in range(self.padded_width)
        ]
        return self

    def transform(self, frame):
        check_is_fitted(self, "svd_")
        ids, texts = self._articles(frame)
        if ids:
            matrix = self.vectorizer_.transform(texts)
            with threadpool_limits(limits=1):
                dense = self.svd_.transform(matrix)
            lookup = dict(zip(ids, dense))
        else:
            lookup = {}
        vectors = np.zeros((len(frame), self.padded_width), dtype=float)
        for index, window in enumerate(frame.articles):
            if window:
                vectors[index, :self.rank_] = np.mean([lookup[a["article_id"]] for a in window], axis=0)
        if not np.isfinite(vectors).all() or vectors[:, self.rank_:].any():
            raise ValueError("Invalid projected text vectors or zero padding.")
        result = pd.concat([
            frame.reset_index(drop=True), pd.DataFrame(vectors, columns=self.text_features_)
        ], axis=1)
        if result.n_berita.ne(result.articles.map(len)).any():
            raise ValueError("Article counts do not match their windows.")
        if not result.has_news.eq(result.n_berita.gt(0).astype(int)).all():
            raise ValueError("Invalid news-availability mask.")
        result["log1p_n_berita"] = np.log1p(result.n_berita)
        return result

    def coverage(self, frame):
        check_is_fitted(self, "svd_")
        ids, texts = self._articles(frame)
        tokens = [token for text in texts for token in text.split()]
        oov = sum(token not in self.vectorizer_.vocabulary_ for token in tokens)
        matrix = self.vectorizer_.transform(texts) if ids else None
        return {"articles": len(ids), "unigram_tokens": len(tokens), "oov_unigram_tokens": oov,
                "oov_unigram_rate": oov / len(tokens) if tokens else None,
                "zero_tfidf_articles": int(np.count_nonzero(np.diff(matrix.indptr) == 0)) if ids else 0}

    def fingerprint(self):
        check_is_fitted(self, "svd_")
        vocabulary = {term: int(index) for term, index in self.vectorizer_.vocabulary_.items()}
        return {
            "vocabulary_sha256": hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode()).hexdigest(),
            "idf_sha256": hashlib.sha256(np.ascontiguousarray(self.vectorizer_.idf_).tobytes()).hexdigest(),
            "svd_components_sha256": hashlib.sha256(np.ascontiguousarray(self.svd_.components_).tobytes()).hexdigest(),
        }
