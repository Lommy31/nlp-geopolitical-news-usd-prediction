from pathlib import Path
import hashlib
import json

import numpy as np

ENCODER_ID = "ProsusAI/finbert"
REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
DIMENSION = 768
MAX_TOKENS = 512
POOLING = "last_hidden_state_mean_without_special_or_padding_tokens"
CHECKPOINT_SHA256 = "e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00"

def article_fingerprint(ids, texts):
    return hashlib.sha256(json.dumps(list(zip(ids, texts)), ensure_ascii=False).encode()).hexdigest()

def embedding_spec(ids, texts):
    return {"encoder": ENCODER_ID, "revision": REVISION, "dimension": DIMENSION,
            "max_tokens": MAX_TOKENS, "pooling": POOLING,
            "article_text_sha256": article_fingerprint(ids, texts)}

def cache_path(directory, spec):
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return Path(directory) / f"articles_{key}.npz"

def load_or_encode(ids, texts, directory, batch_size=4):
    if len(ids) != len(texts) or len(set(ids)) != len(ids):
        raise ValueError("Article IDs and texts must have matching lengths and unique IDs.")
    spec = embedding_spec(ids, texts)
    path = cache_path(directory, spec)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if data["article_ids"].tolist() != list(ids) or json.loads(str(data["spec"])) != spec:
                raise ValueError("The embedding cache does not match these articles or encoder settings.")
            vectors = data["vectors"].copy()
            info = json.loads(str(data["encoder_info"]))
    else:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from huggingface_hub import hf_hub_download

        torch.set_num_threads(1)
        hf_cache = Path(directory) / "huggingface"
        options = {"revision": REVISION, "cache_dir": hf_cache, "token": False, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID, use_fast=True, **options)
        classifier = AutoModelForSequenceClassification.from_pretrained(ENCODER_ID, use_safetensors=True, **options)
        encoder = classifier.base_model.cpu().eval().requires_grad_(False)
        if encoder.config._commit_hash != REVISION or encoder.config.hidden_size != DIMENSION:
            raise ValueError("Unexpected FinBERT revision or embedding width.")
        lengths = tokenizer(texts, truncation=False, return_length=True)["length"]
        order = np.argsort(lengths, kind="stable")
        vectors = np.zeros((len(ids), DIMENSION), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                positions = order[start:start + batch_size]
                batch = tokenizer([texts[i] for i in positions], padding=True, truncation=True,
                                  max_length=MAX_TOKENS, return_tensors="pt", return_special_tokens_mask=True)
                special = batch.pop("special_tokens_mask").bool()
                valid = batch["attention_mask"].bool() & ~special
                if not valid.sum(1).gt(0).all():
                    raise ValueError("An article has no non-special tokens.")
                hidden = encoder(**batch).last_hidden_state
                vectors[positions] = ((hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)).cpu().numpy()
                if start % (batch_size * 32) == 0:
                    print(f"Encoded {min(start + batch_size, len(order))}/{len(order)} articles", flush=True)
        weight_path = hf_hub_download(ENCODER_ID, "model.safetensors", revision=REVISION,
                                    cache_dir=hf_cache, token=False, local_files_only=True)
        with open(weight_path, "rb") as handle:
            checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        info = {"source": "encoded_locally", "frozen_parameters": sum(p.numel() for p in encoder.parameters()),
                "checkpoint_sha256": checkpoint_hash, "torch": torch.__version__, "transformers": transformers.__version__,
                "batch_size": batch_size, "threads": 1, "truncated_articles": int(np.sum(np.asarray(lengths) > MAX_TOKENS))}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, article_ids=np.asarray(ids), vectors=vectors,
                            spec=json.dumps(spec), encoder_info=json.dumps(info))
    if vectors.shape != (len(ids), DIMENSION) or not np.isfinite(vectors).all():
        raise ValueError("Invalid FinBERT embedding matrix.")
    if info["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Unexpected FinBERT checkpoint checksum.")
    return vectors, path, spec, info
