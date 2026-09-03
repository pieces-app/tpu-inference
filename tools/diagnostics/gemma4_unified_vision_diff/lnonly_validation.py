"""LN-ONLY validation: does patching just the three nn.LayerNorm modules to
compute statistics in fp32 recover the fidelity, or is a full-fp32 embedder
required?  This is the script that produced the "LayerNorm-only" table in
README.md, and it is what justifies shipping the narrow patch.

Arms (both bf16 weights, both executed under torchax on CPU-JAX, i.e. TPU
execution semantics), each diffed against the SAME torch-eager fp32
reference:
  bf16         — unpatched (the defect)
  bf16+f32LN   — exactly what gemma4_unified_patcher.py does in production

Env: GEMMA4_CKPT, GEMMA4_IMAGE.  Run in the torch/torchax venv (see README).
"""

from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from frontend_diff import CKPT, IMG, MAX_SOFT, load_weights, stages
from PIL import Image
from transformers.models.gemma4_unified.image_processing_gemma4_unified import \
    Gemma4UnifiedImageProcessor


def f32_ln_forward(self, x):
    od = x.dtype
    w = self.weight.to(torch.float32) if self.weight is not None else None
    b = self.bias.to(torch.float32) if self.bias is not None else None
    return F.layer_norm(x.to(torch.float32), self.normalized_shape, w, b,
                        self.eps).to(od)


img = Image.open(IMG).convert("RGB")
proc = Gemma4UnifiedImageProcessor.from_pretrained(CKPT)
feats = proc(images=[img], max_soft_tokens=MAX_SOFT, return_tensors="pt")
pv, pos = feats["pixel_values"], feats["image_position_ids"]
valid = (pos[0, :, 0] != -1).numpy()

ve32 = load_weights(torch.float32)
with torch.no_grad():
    ref = stages(ve32, pv, pos)

# plain bf16 (the defect) and LN-patched bf16, both under torchax
ve_a = load_weights(torch.bfloat16)
ve_b = load_weights(torch.bfloat16)
for attr in ("patch_ln1", "patch_ln2", "pos_norm"):
    m = getattr(ve_b, attr)
    assert isinstance(m, nn.LayerNorm), attr
    m.forward = MethodType(f32_ln_forward, m)

import torchax

torchax.enable_globally()
outs = {}
for name, mod in (("bf16", ve_a), ("bf16+f32LN", ve_b)):
    mj = mod.to("jax")
    with torch.no_grad():
        o = stages(mj, pv.to("jax"), pos.to("jax"))
    outs[name] = {
        k: torch.from_numpy(np.asarray(v.jax()).astype(np.float32))
        for k, v in o.items()
    }
torchax.disable_globally()

for name, arm in outs.items():
    print(f"\n=== torchax {name} vs f32 eager reference ===")
    print(f"{'stage':<12}{'max|d|':>12}{'rel-mean':>12}{'cosine':>10}")
    for k in ref:
        a = ref[k].float().numpy()[0][valid]
        b = arm[k].float().numpy()[0][valid]
        d = np.abs(a - b)
        rel = d.mean() / (np.abs(a).mean() + 1e-12)
        cos = float(
            (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"{k:<12}{d.max():>12.3e}{rel:>12.3e}{cos:>10.6f}")

# per-token worst-case at the decisive stages
for K in ("S3_ln2", "S7_proj"):
    for name, arm in outs.items():
        a = ref[K].float().numpy()[0][valid]
        b = arm[K].float().numpy()[0][valid]
        pt = np.abs(a - b).max(axis=-1)
        print(
            f"{K} {name:<12} per-token maxerr p50={np.percentile(pt,50):.4f} "
            f"p95={np.percentile(pt,95):.4f} max={pt.max():.4f}")
