from pathlib import Path
import hashlib
import json
import time

import numpy as np

ENCODER_ID = "FacebookAI/roberta-base"
REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
CHECKPOINT_SHA256 = "5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81"
DIMENSION = 768
MAX_TOKENS = 512
THREADS = 4
BATCH_SIZE = 16
POOLING = "last_hidden_state_mean_without_special_or_padding_tokens"

def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def embedding_spec(ids, texts):
    fingerprint = hashlib.sha256(json.dumps(list(zip(ids, texts)), ensure_ascii=False).encode()).hexdigest()
    return {"encoder": ENCODER_ID, "revision": REVISION, "dimension": DIMENSION,
            "max_tokens": MAX_TOKENS, "pooling": POOLING, "add_prefix_space": False,
            "pooler_used": False, "article_text_sha256": fingerprint}

def cache_path(directory, spec):
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return Path(directory) / f"articles_{key}.npz"

def load_or_encode(ids, texts, directory):
    if len(ids) != len(texts) or len(set(ids)) != len(ids):
        raise ValueError("Article IDs and texts must have matching lengths and unique IDs.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = embedding_spec(ids, texts)
    path = cache_path(directory, spec)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if cached["article_ids"].tolist() != list(ids) or json.loads(str(cached["spec"].item())) != spec:
                raise ValueError("Embedding cache does not match the articles or encoder settings.")
            vectors = cached["vectors"].copy()
            info = json.loads(str(cached["encoder_info"].item()))
    else:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import hf_hub_download

        started = time.perf_counter()
        torch.set_num_threads(THREADS)
        options = {"revision": REVISION, "cache_dir": directory / "huggingface",
                   "token": False, "trust_remote_code": False, "local_files_only": True}
        tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID, use_fast=True, add_prefix_space=False, **options)
        encoder = AutoModel.from_pretrained(ENCODER_ID, use_safetensors=True,
                                           add_pooling_layer=False, **options).cpu().eval().requires_grad_(False)
        weight = hf_hub_download(ENCODER_ID, "model.safetensors", revision=REVISION,
                                cache_dir=options["cache_dir"], token=False, local_files_only=True)
        with open(weight, "rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != CHECKPOINT_SHA256:
                raise ValueError("Unexpected RoBERTa checkpoint checksum.")
        if encoder.config._commit_hash != REVISION or encoder.config.hidden_size != DIMENSION:
            raise ValueError("Unexpected RoBERTa revision or embedding dimension.")
        if tokenizer.num_special_tokens_to_add(pair=False) != 2 or tokenizer.init_kwargs.get("add_prefix_space", False):
            raise ValueError("Unexpected RoBERTa tokenizer settings.")
        lengths = np.asarray(tokenizer(texts, truncation=False, return_length=True)["length"])
        order = np.argsort(lengths, kind="stable")
        vectors = np.zeros((len(ids), DIMENSION), dtype=np.float32)
        done = np.zeros(len(ids), dtype=bool)
        kept, unknown = np.zeros(len(ids), dtype=int), np.zeros(len(ids), dtype=int)
        partial = path.with_suffix(".partial.npz")
        if partial.exists():
            with np.load(partial, allow_pickle=False) as saved:
                if saved["article_ids"].tolist() != list(ids) or json.loads(str(saved["spec"].item())) != spec:
                    raise ValueError("Partial embedding cache does not match the current input.")
                vectors, done = saved["vectors"].copy(), saved["done"].copy()
                kept, unknown = saved["kept_tokens"].copy(), saved["unknown_tokens"].copy()
            if vectors.shape != (len(ids), DIMENSION) or not np.isfinite(vectors[done]).all():
                raise ValueError("Invalid partial embedding cache.")
        remaining = order[~done[order]]
        progress = directory / "progress.json"
        with torch.inference_mode():
            for start in range(0, len(remaining), BATCH_SIZE):
                positions = remaining[start:start + BATCH_SIZE]
                batch = tokenizer([texts[i] for i in positions], padding=True, truncation=True,
                                  max_length=MAX_TOKENS, return_tensors="pt", return_special_tokens_mask=True)
                special = batch.pop("special_tokens_mask").bool()
                valid = batch["attention_mask"].bool() & ~special
                if not valid.sum(1).gt(0).all():
                    raise ValueError("An article has no content tokens.")
                hidden = encoder(**batch).last_hidden_state
                vectors[positions] = ((hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)).numpy()
                kept[positions] = valid.sum(1).numpy()
                unknown[positions] = ((batch["input_ids"] == tokenizer.unk_token_id) & valid).sum(1).numpy()
                done[positions] = True
                if start % 128 == 0 or done.all():
                    progress.write_text(json.dumps({"cache": path.name, "encoded": int(done.sum()), "total": len(ids),
                                                    "elapsed_seconds": time.perf_counter() - started,
                                                    "threads": THREADS, "batch_size": BATCH_SIZE}) + "\n")
                    print(f"Encoded {done.sum()}/{len(ids)} articles", flush=True)
                if start % 512 == 0 or done.all():
                    temporary = partial.with_suffix(".pending.npz")
                    np.savez(temporary, article_ids=np.asarray(ids), vectors=vectors, done=done,
                             kept_tokens=kept, unknown_tokens=unknown, spec=json.dumps(spec))
                    temporary.replace(partial)
        if not np.array_equal(kept, np.minimum(lengths, MAX_TOKENS) - 2):
            raise ValueError("Token retention does not match truncation settings.")
        info = {"source": "encoded_locally", "checkpoint_sha256": CHECKPOINT_SHA256,
                "frozen_parameters": sum(p.numel() for p in encoder.parameters()), "pooler_used": False,
                "vectors_sha256": array_sha256(vectors), "torch": torch.__version__, "transformers": transformers.__version__,
                "threads": THREADS, "batch_size": BATCH_SIZE, "truncated_articles": int((lengths > MAX_TOKENS).sum()),
                "extraction_seconds": time.perf_counter() - started}
        temporary = path.with_suffix(".pending.npz")
        np.savez_compressed(temporary, article_ids=np.asarray(ids), vectors=vectors, token_lengths=lengths,
                            kept_tokens=kept, unknown_tokens=unknown, spec=json.dumps(spec), encoder_info=json.dumps(info))
        temporary.replace(path)
        partial.unlink(missing_ok=True)
    if vectors.shape != (len(ids), DIMENSION) or not np.isfinite(vectors).all():
        raise ValueError("Invalid RoBERTa embedding matrix.")
    if info["checkpoint_sha256"] != CHECKPOINT_SHA256 or array_sha256(vectors) != info["vectors_sha256"]:
        raise ValueError("Unexpected RoBERTa checkpoint or vector checksum.")
    return vectors, path, spec, info
