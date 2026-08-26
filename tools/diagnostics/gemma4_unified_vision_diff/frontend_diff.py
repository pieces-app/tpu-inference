"""TORCHAX arm: run the HF Gemma4Unified vision embedder under torchax on
CPU JAX in bf16 (the TPU-serving execution semantics) and diff against the
f32 torch eager reference. venv2 (torch 2.10 / torchvision 0.25 / torchax
0.0.13 / transformers 5.10.4 — the tpu-inference image's pins)."""
import numpy as np
import torch
from PIL import Image

import os
CKPT = os.environ["GEMMA4_CKPT"]
IMG = __import__("os").environ["GEMMA4_IMAGE"]
MAX_SOFT = 1120

from safetensors import safe_open
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedConfig,
)
from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
    Gemma4UnifiedImageProcessor,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedVisionEmbedder,
)

cfg = Gemma4UnifiedConfig.from_pretrained(CKPT)


def load_weights(dtype):
    ve = Gemma4UnifiedVisionEmbedder(cfg.vision_config, cfg.text_config)
    sd = {}
    with safe_open(f"{CKPT}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.vision_embedder."):
                sd[k[len("model.vision_embedder."):]] = f.get_tensor(k)
            elif k.startswith("model.embed_vision."):
                sd["multimodal_embedder." + k[len("model.embed_vision."):]] = \
                    f.get_tensor(k)
    missing, unexpected = ve.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return ve.to(dtype).eval()


def stages(ve, pixel_values, positions):
    out = {}
    target_dtype = ve.patch_dense.weight.dtype
    x = pixel_values.to(target_dtype)
    out["S0_pixels"] = x
    h = ve.patch_ln1(x)
    out["S1_ln1"] = h
    h = ve.patch_dense(h)
    out["S2_dense"] = h
    h = ve.patch_ln2(h)
    out["S3_ln2"] = h
    clamped = positions.clamp(min=0).long()
    valid = (positions != -1).to(ve.pos_embedding.dtype).unsqueeze(-1)
    axes = torch.arange(2, device=positions.device)
    pos_embs = (ve.pos_embedding[clamped, axes] * valid).sum(-2)
    h = h + pos_embs
    out["S4_posemb"] = h
    h = ve.pos_norm(h)
    out["S5_posnorm"] = h
    me = ve.multimodal_embedder
    hn = me.embedding_pre_projection_norm(
        h.to(me.embedding_projection.weight.dtype))
    out["S6_rms"] = hn
    proj = me.embedding_projection(hn)
    out["S7_proj"] = proj
    return out


def main():
    img = Image.open(IMG).convert("RGB")
    proc = Gemma4UnifiedImageProcessor.from_pretrained(CKPT)
    feats = proc(images=[img], max_soft_tokens=MAX_SOFT, return_tensors="pt")
    pv, pos = feats["pixel_values"], feats["image_position_ids"]
    valid_mask = (pos[0, :, 0] != -1).numpy()

    ve32 = load_weights(torch.float32)
    with torch.no_grad():
        ref = stages(ve32, pv, pos)

    ve16 = load_weights(torch.bfloat16)
    import torchax
    torchax.enable_globally()
    ve16 = ve16.to("jax")
    pv_j, pos_j = pv.to("jax"), pos.to("jax")
    with torch.no_grad():
        arm = stages(ve16, pv_j, pos_j)
    arm = {k: torch.from_numpy(np.asarray(v.jax()).astype(np.float32))
           for k, v in arm.items()}
    torchax.disable_globally()

    print(f"{'stage':<12}{'max|d|':>12}{'mean|d|':>12}{'rel-mean':>12}{'cosine':>10}")
    for k in ref:
        a = ref[k].float().numpy()[0][valid_mask]
        b = arm[k].float().numpy()[0][valid_mask]
        d = np.abs(a - b)
        rel = d.mean() / (np.abs(a).mean() + 1e-12)
        cos = float((a * b).sum() /
                    (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"{k:<12}{d.max():>12.3e}{d.mean():>12.3e}{rel:>12.3e}{cos:>10.6f}")

    # processor parity across transformers versions (5.10.4 here vs 5.16.1
    # in the reference venv): dump S0 checksum
    print("S0 sha-ish:", float(pv.double().abs().sum()), int(valid_mask.sum()))


if __name__ == "__main__":
    main()
