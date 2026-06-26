"""Hugging Face dense dual-encoder utilities for Phase 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for dense model training and encoding.") from exc
    return torch


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean-pool token embeddings using the attention mask."""
    torch = import_torch()
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalize(embeddings: Any) -> Any:
    torch = import_torch()
    return torch.nn.functional.normalize(embeddings, p=2, dim=-1)


def in_batch_contrastive_loss(query_embeddings: Any, doc_embeddings: Any, temperature: float, bidirectional: bool = False) -> Any:
    torch = import_torch()
    scores = query_embeddings @ doc_embeddings.T
    scores = scores / temperature
    labels = torch.arange(scores.shape[0], device=scores.device)
    loss = torch.nn.functional.cross_entropy(scores, labels)
    if bidirectional:
        loss = 0.5 * (loss + torch.nn.functional.cross_entropy(scores.T, labels))
    return loss


class DenseBiEncoder:
    """Thin wrapper around ``AutoModel`` for normalized mean-pooled embeddings."""

    def __init__(self, encoder: Any, normalize_embeddings: bool = True) -> None:
        torch = import_torch()

        class _Module(torch.nn.Module):
            def __init__(self, inner_encoder: Any, normalize: bool) -> None:
                super().__init__()
                self.encoder = inner_encoder
                self.normalize_embeddings = normalize

            def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any | None = None) -> Any:
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids
                output = self.encoder(**kwargs)
                embeddings = mean_pool(output.last_hidden_state, attention_mask)
                if self.normalize_embeddings:
                    embeddings = l2_normalize(embeddings)
                return embeddings

        self.module = _Module(encoder, normalize_embeddings)

    def __getattr__(self, name: str) -> Any:
        if name == "module":
            return super().__getattribute__(name)
        return getattr(self.module, name)


def load_tokenizer_and_model(model_name_or_path: str, cache_dir: str | Path | None = None, normalize_embeddings: bool = True) -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for dense model training and encoding.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=str(cache_dir) if cache_dir else None)
    encoder = AutoModel.from_pretrained(model_name_or_path, cache_dir=str(cache_dir) if cache_dir else None)
    return tokenizer, DenseBiEncoder(encoder, normalize_embeddings=normalize_embeddings).module


def save_dense_model(model: Any, tokenizer: Any, output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(target)
    tokenizer.save_pretrained(target)


def load_dense_model(model_dir: str | Path, normalize_embeddings: bool = True) -> tuple[Any, Any]:
    return load_tokenizer_and_model(str(model_dir), cache_dir=None, normalize_embeddings=normalize_embeddings)


def move_batch_to_device(batch: dict[str, Any], device: Any, non_blocking: bool = True) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if hasattr(value, "to") else value
    return moved


def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: Any,
    use_bf16: bool = False,
) -> Any:
    torch = import_torch()
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = move_batch_to_device(batch, device)
            autocast_enabled = bool(use_bf16 and str(device).startswith("cuda"))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                encoded = model(**batch)
            embeddings.append(encoded.detach().float().cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 0), dtype=torch.float32)


def iter_parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
