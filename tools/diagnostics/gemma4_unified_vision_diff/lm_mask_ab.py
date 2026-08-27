"""LM-level A/B: quantify the two TPU-serving divergences on the real task.

Arms (all torch CPU f32, greedy):
  R  : correct bidi masks + clean f32 soft tokens        (HF reference)
  C  : causal-only masks (TPU semantics: no blockwise)   + clean tokens
  N  : correct masks + torchax-bf16 soft tokens          (TPU embedder numerics)
  NC : causal-only masks + torchax-bf16 soft tokens      (both TPU defects)

The question targets the fine-text strings the TPU misreads
(PID 36793 -> 36732 on gemma12b-tpu-r2).
"""
import sys
import types

import numpy as np
import torch
from PIL import Image

import os
CKPT = os.environ["GEMMA4_CKPT"]
IMG = __import__("os").environ["GEMMA4_IMAGE"]
SCRATCH = __import__("os").environ.get("GEMMA4_DIFF_OUT", ".")
MAX_SOFT = 1120
MAX_NEW = 180

from transformers import AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
)

QUESTION = (
    "Look closely at the terminal / agent panel in this screenshot. "
    "Report EXACTLY, character for character: (1) the collector PID, "
    "(2) the output directory path, (3) any git commit hash visible. "
    "Answer as three short lines."
)


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else ["R", "C", "N", "NC"]
    img = Image.open(IMG).convert("RGB")
    proc = AutoProcessor.from_pretrained(CKPT)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QUESTION},
        ]},
    ]
    text = proc.apply_chat_template(messages, add_generation_prompt=True,
                                    tokenize=False)
    inputs = proc(text=[text], images=[img], max_soft_tokens=MAX_SOFT,
                  return_tensors="pt")
    print("input keys:", list(inputs.keys()))
    print("input_ids:", inputs["input_ids"].shape)
    n_img_tok = int((inputs["input_ids"] == 258880).sum())
    print("image tokens in prompt:", n_img_tok)

    print("loading model f32 ...", flush=True)
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    torch.set_num_threads(16)

    noisy = np.load(f"{SCRATCH}/soft_tokens_torchax_bf16.npy")  # (1,1120,3840)
    pos = np.load(f"{SCRATCH}/positions.npy")                   # (1,1120,2)
    valid = pos[0, :, 0] != -1
    noisy_valid = torch.from_numpy(noisy[0][valid]).float()     # (1092,3840)
    assert noisy_valid.shape[0] == n_img_tok, (noisy_valid.shape, n_img_tok)

    real_gif = model.model.get_image_features

    def noisy_gif(pixel_values, image_position_ids=None, **kw):
        return BaseModelOutputWithPooling(
            last_hidden_state=None,
            pooler_output=noisy_valid.to(model.dtype),
        )

    results = {}
    for arm in which:
        kwargs = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        if arm in ("C", "NC"):
            kwargs.pop("mm_token_type_ids", None)   # -> causal-only masks
        if arm in ("N", "NC"):
            model.model.get_image_features = noisy_gif
        else:
            model.model.get_image_features = real_gif

        print(f"\n########## ARM {arm} ##########", flush=True)
        with torch.no_grad():
            out = model.generate(**kwargs, do_sample=False,
                                 max_new_tokens=MAX_NEW)
        new = out[0, inputs["input_ids"].shape[1]:]
        txt = proc.tokenizer.decode(new, skip_special_tokens=True)
        results[arm] = txt
        print(txt, flush=True)

    print("\n\n===== SUMMARY =====")
    for arm, txt in results.items():
        flat = " | ".join(l for l in txt.splitlines() if l.strip())
        print(f"[{arm}] {flat[:400]}")
    for target in ["36793", "b23ce7c"]:
        marks = {a: ("HIT" if target in t else "miss") for a, t in results.items()}
        print(f"target {target}: {marks}")


if __name__ == "__main__":
    main()
