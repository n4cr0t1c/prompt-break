"""Optional integration wrappers for Ollama and HuggingFace Inference.

These are soft wrappers: they only activate when the required clients
are available and configured via environment variables. They are useful
for augmenting local heuristics with remote model signals when desired.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional


def ollama_inference(prompt: str, model: Optional[str] = None, timeout: int = 10) -> Optional[str]:
    """Call local Ollama CLI if present. Returns the model output or None.

    Requires the `ollama` CLI to be installed and a model name set via
    `OLLAMA_MODEL` env var or passed as `model`.
    """
    if shutil.which("ollama") is None:
        return None
    model = model or os.environ.get("OLLAMA_MODEL")
    if not model:
        return None
    try:
        # Use the `ollama` CLI predict/run interface where available.
        proc = subprocess.run(["ollama", "predict", model], input=prompt.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        out = proc.stdout.decode(errors="ignore").strip()
        return out or None
    except Exception:
        return None


def hf_inference(prompt: str, model: Optional[str] = None, token: Optional[str] = None, timeout: int = 15) -> Optional[str]:
    """Call HuggingFace Inference API via `huggingface_hub.InferenceApi`.

    Requires `huggingface-hub` to be installed. If `token` is not provided
    the public inference endpoints (if available) will be used.
    """
    try:
        from huggingface_hub import InferenceApi
    except Exception:
        return None
    model = model or os.environ.get("HF_MODEL")
    if not model:
        return None
    token = token or os.environ.get("HF_TOKEN")
    try:
        api = InferenceApi(repo_id=model, token=token)
        resp = api(prompt, timeout=timeout)
        # response can be a string or structured; try to extract text
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            # often returns {"generated_text": "..."}
            if "generated_text" in resp:
                return resp["generated_text"]
            # sometimes a list of items
            if "error" in resp:
                return None
            # fallback to stringification
            return str(resp)
        return None
    except Exception:
        return None
