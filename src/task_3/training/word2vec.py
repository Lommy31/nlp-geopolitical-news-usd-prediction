from pathlib import Path
import gzip
import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SOURCE_SHA256 = "cdd97c3b73abcd8e1437ad3837b8faf98fc21258f61f031721fe3e2096ed89d9"
SOURCE_MD5 = "a5e5354d40acb95f9ec66d5977d140ef"
SOURCE_URL = "https://github.com/piskvorky/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz"
MIRROR_URL = "https://huggingface.co/akashjd113/morale-ai-models/resolve/da6cb8fda49e823483cbd749ee40db73f7f30fa8/GoogleNews-vectors-negative300.bin.gz"
WORD_DIM = 300
TEXT_WIDTH = 768
TEXT_CONFIG = {
    "model": "word2vec-google-news-300", "source_sha256": SOURCE_SHA256,
    "native_dimension": WORD_DIM, "padded_width": TEXT_WIDTH, "token_view": "tokens_word2vec",
    "case_sensitive": True, "lowercase_fallback": False, "automatic_phrase_joining": False,
    "word_pooling": "Mean of in-vocabulary token occurrences; all-OOV articles remain zero",
    "window_pooling": "Uniform article mean including zero vectors", "project_text_training": False,
}

def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def read_matching_vectors(stream, tokens):
    count, width = map(int, stream.readline().split())
    if count != 3000000 or width != WORD_DIM:
        raise ValueError("Unexpected pretrained vocabulary size or embedding dimension.")
    wanted = {token.encode("utf-8"): token for token in tokens}
    found = {}
    for index in range(count):
        token = bytearray()
        while True:
            char = stream.read(1)
            if not char:
                raise EOFError("Truncated pretrained token record.")
            if char == b" ":
                break
            if char not in (b"\n", b"\r"):
                token.extend(char)
            if len(token) > 4096:
                raise ValueError("Unexpected pretrained token length.")
        raw = stream.read(4 * width)
        if len(raw) != 4 * width:
            raise EOFError("Truncated pretrained vector.")
        key = wanted.get(bytes(token))
        if key is not None and key not in found:
            found[key] = np.frombuffer(raw, dtype="<f4").copy()
        if (index + 1) % 500000 == 0:
            print(f"Scanned {index + 1:,}/{count:,} pretrained records", flush=True)
    if stream.read().strip():
        raise ValueError("Unexpected trailing pretrained records.")
    return found

class FrozenWord2Vec(TransformerMixin, BaseEstimator):
    def __init__(self, directory):
        self.directory = directory

    def _vectors(self, frame):
        tokens = sorted({token for window in frame.articles for article in window for token in article["tokens_word2vec"]})
        spec = {**TEXT_CONFIG, "tokens_sha256": hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode()).hexdigest(),
                "parser": "binary-float32-le-first-entry-v1"}
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"subset_{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if json.loads(str(cached["spec"].item())) != spec:
                    raise ValueError("Pretrained subset does not match the requested tokens.")
                found_tokens, vectors = cached["tokens"].tolist(), cached["vectors"].copy()
                info = json.loads(str(cached["info"].item()))
        else:
            source = directory / "word2vec-google-news-300.gz"
            with source.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != SOURCE_SHA256:
                    raise ValueError("Unexpected pretrained source checksum.")
            started = time.perf_counter()
            with gzip.open(source, "rb") as compressed:
                with io.BufferedReader(compressed, buffer_size=4 * 1024 * 1024) as stream:
                    found = read_matching_vectors(stream, tokens)
            found_tokens = sorted(found)
            vectors = np.stack([found[token] for token in found_tokens]).astype(np.float32)
            info = {"source_sha256": SOURCE_SHA256, "source_md5": SOURCE_MD5,
                    "source_vocabulary": 3000000, "requested_tokens": len(tokens),
                    "matched_tokens": len(found_tokens), "subset_vectors_sha256": array_sha256(vectors),
                    "extraction_seconds": time.perf_counter() - started}
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary, tokens=np.asarray(found_tokens), vectors=vectors,
                                spec=json.dumps(spec), info=json.dumps(info))
            temporary.replace(path)
        if vectors.shape != (len(found_tokens), WORD_DIM) or not np.isfinite(vectors).all():
            raise ValueError("Invalid pretrained subset vectors.")
        if info["source_sha256"] != SOURCE_SHA256 or array_sha256(vectors) != info["subset_vectors_sha256"]:
            raise ValueError("Pretrained subset checksum mismatch.")
        if not set(found_tokens).issubset(tokens):
            raise ValueError("Unexpected tokens in pretrained subset.")
        return dict(zip(found_tokens, vectors)), path, info

    def transform(self, frame):
        lookup, path, info = self._vectors(frame)
        vectors = np.zeros((len(frame), TEXT_WIDTH), dtype=float)
        coverage = []
        for index, row in enumerate(frame.itertuples()):
            article_vectors = []
            for article in row.articles:
                tokens = article["tokens_word2vec"]
                known = [lookup[token] for token in tokens if token in lookup]
                vector = np.mean(known, axis=0, dtype=np.float64) if known else np.zeros(WORD_DIM)
                article_vectors.append(vector)
                coverage.append({"article_id": article["article_id"], "tokens": len(tokens), "known_tokens": len(known)})
            if article_vectors:
                vectors[index, :WORD_DIM] = np.mean(article_vectors, axis=0)
        columns = [f"word2vec_{i:03d}" if i < WORD_DIM else f"zero_text_{i:03d}" for i in range(TEXT_WIDTH)]
        result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(vectors, columns=columns)], axis=1)
        result["log1p_n_berita"] = np.log1p(result.n_berita)
        if not result.articles.map(len).eq(result.n_berita).all() or not result.has_news.eq(result.n_berita.gt(0)).all():
            raise ValueError("Article counts or availability masks do not match their windows.")
        if vectors[result.has_news.eq(0)].any() or vectors[:, WORD_DIM:].any() or not np.isfinite(vectors).all():
            raise ValueError("Invalid word-vector pooling or zero padding.")
        self.coverage_ = pd.DataFrame(coverage, columns=["article_id", "tokens", "known_tokens"])
        self.cache_path_, self.pretrained_info_ = str(path), info
        return result
