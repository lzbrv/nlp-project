"""ReProver-style dual encoder: one shared ByT5 encoder, mean pooling, cosine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class DualEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, tokenizer, max_len: int = 1024):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_len = max_len

    @classmethod
    def from_pretrained(cls, name_or_dir: str, max_len: Optional[int] = None) -> "DualEncoder":
        from transformers import AutoTokenizer, T5EncoderModel

        meta_path = Path(name_or_dir) / "dual_encoder.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        enc = T5EncoderModel.from_pretrained(name_or_dir)
        tok = AutoTokenizer.from_pretrained(name_or_dir)
        return cls(enc, tok, max_len or meta.get("max_len", 1024))

    def save(self, out_dir) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        (out / "dual_encoder.json").write_text(json.dumps({"max_len": self.max_len}))

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def tokenize(self, texts: List[str]):
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return F.normalize(pooled.float(), dim=-1)

    def encode(self, texts: List[str]) -> torch.Tensor:
        return self(**self.tokenize(texts))

    @torch.no_grad()
    def encode_many(self, texts: List[str], batch_size: int = 128,
                    progress: bool = False, dtype=torch.float16) -> torch.Tensor:
        """Embeds many texts (sorted by length for less padding), in input order."""
        was_training = self.training
        self.eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = None
        use_amp = self.device.type == "cuda"
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                emb = self.encode([texts[i] for i in idx])
            if out is None:
                out = torch.empty(len(texts), emb.shape[1], dtype=dtype, device=self.device)
            out[torch.as_tensor(idx, device=self.device)] = emb.to(dtype)
            if progress and (start // batch_size) % 50 == 0:
                print(f"  encoded {start + len(idx)}/{len(texts)}", flush=True)
        self.train(was_training)
        if out is None:
            return torch.empty(0, 0, dtype=dtype, device=self.device)
        return out
