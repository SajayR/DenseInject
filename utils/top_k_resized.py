#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-patch top-k nearest words from ConceptNet Numberbatch using DenseCLIP text encoder.

Flow:
1) Load DenseCLIP (backbone + text encoder + context decoder).
2) Read numberbatch-en-19.08.txt.gz (word + vector per line).
3) Convert word -> prompt text (replace '_' with ' ', optional --template).
4) Encode all prompts with DenseCLIP text encoder -> CLIP-space matrix T [V,C].
5) For an input image, get per-patch embeddings Vmap [P,C], do bf16 matmul with T^T to get [P,V].
6) Top-k per patch. Save CSV + PT (indices, tokens, scores, numberbatch vectors of the winners).

Usage (example):
python denseclip_nb_topk.py \
  --ckpt /path/to/denseclip_checkpoint.pth \
  --arch vit-b-16 \
  --image /path/to/image.jpg \
  --text "a photo of a dog on the grass" \
  --nb_path /speedy/DenseInject/datasets/numberbatch/numberbatch-en-19.08.txt.gz \
  --nb_cache textbank_nb.pt \
  --out heatmap.png \
  --out_topk_csv patch_topk_nb.csv \
  --out_topk_pt  patch_topk_nb.pt \
  --topk 3 \
  --bf16 \
  --template "a photo of {}"
"""

import os
import io
import gzip
import csv
import json
import math
import argparse
import hashlib
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from denseclip_heatmap import (
    DenseCLIP,
    load_image_clip,
    save_heatmap_overlay,
)

# ----------------------------
# Numberbatch I/O
# ----------------------------

def _nb_clean_token(s: str) -> str:
    """
    Normalize a Numberbatch token to a display token:
    - Handle URIs like '/c/en/new_york' -> 'new_york'
    - Otherwise keep as-is.
    """
    if s.startswith("/c/en/"):
        s = s[len("/c/en/"):]
    return s

def _nb_token_to_prompt_text(s: str) -> str:
    """
    Convert a Numberbatch token to human text for CLIP encoding:
    - Replace '_' with ' '.
    """
    s = _nb_clean_token(s)
    return s.replace("_", " ").strip()

def load_numberbatch_txt_gz(path: str,
                            limit: int = 0,
                            expect_dim: Optional[int] = None) -> Tuple[List[str], torch.Tensor]:
    """
    Reads ConceptNet Numberbatch .txt.gz.
    Returns:
        tokens: list of original tokens (e.g., '/c/en/new_york' or 'new_york')
        nb_vecs: torch.FloatTensor [V, D]
    Notes:
        - Skips comment/header lines starting with '#'
        - First entry per line is the token, remaining are floats
        - If expect_dim is given, enforces dimensionality
    """
    tokens: List[str] = []
    vecs: List[List[float]] = []

    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tok = parts[0]
            # Some distro variants have a count header on the first line like: "516783 300"
            # Detect & skip if token is an integer and only two ints in line.
            if tok.isdigit() and len(parts) == 2 and parts[1].isdigit():
                # dimension header; skip
                continue

            try:
                vec = [float(x) for x in parts[1:]]
            except Exception:
                continue

            if expect_dim is not None and len(vec) != expect_dim:
                # skip mismatched dimensionality lines
                continue

            tokens.append(tok)
            vecs.append(vec)

            if limit > 0 and len(tokens) >= limit:
                break

    if len(tokens) == 0:
        raise RuntimeError(f"No entries read from {path}. Is the path correct?")

    nb_vecs = torch.tensor(np.array(vecs, dtype=np.float32))  # [V,D]
    return tokens, nb_vecs

# ----------------------------
# Text bank builder + cache (Numberbatch -> CLIP text space)
# ----------------------------

def _cache_signature(model: DenseCLIP,
                     nb_dim: int,
                     nb_path: str,
                     template: Optional[str]) -> Dict:
    sig = {
        "arch": model.arch,
        "embed_dim": int(model.text_encoder.text_projection.shape[1]),
        "ctx_len": int(model.text_encoder.context_length),
        "ctx_shape": list(model.contexts.shape),
        "nb_dim": int(nb_dim),
        "nb_path_sha1": hashlib.sha1(nb_path.encode()).hexdigest()[:12],
        "template": template or "",
    }
    return sig

@torch.no_grad()
def build_or_load_nb_textbank(model: DenseCLIP,
                              device: str,
                              nb_path: str,
                              cache_path: Optional[str] = None,
                              template: Optional[str] = None,
                              vocab_limit: int = 0,
                              batch_size: int = 4096,
                              dtype_clip_bank: torch.dtype = torch.bfloat16):
    """
    Returns:
        tokens_raw: list[str]          (original NB tokens)
        tokens_text: list[str]         (prompt texts used for CLIP text encoder)
        nb_vecs: FloatTensor [V, Dnb]  (Numberbatch vectors, CPU float32)
        T_clip: Tensor [V, C]          (normalized CLIP-space text vectors, on device, dtype=dtype_clip_bank)
        cache_used_path: Optional[str]
    """
    # Try load cache first
    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        if "sig" in blob and "tokens_raw" in blob and "tokens_text" in blob and "T_clip" in blob and "nb_vecs" in blob:
            sig_cache = blob["sig"]
            # Minimal consistency check: we compare against a fresh sig below after loading NB header
            # but we need nb_dim now; we'll infer it from stored nb_vecs
            nb_dim_cached = int(blob["nb_vecs"].shape[1])
            sig_now_partial = {
                "arch": model.arch,
                "embed_dim": int(model.text_encoder.text_projection.shape[1]),
                "ctx_len": int(model.text_encoder.context_length),
                "ctx_shape": list(model.contexts.shape),
                "nb_dim": nb_dim_cached,
                "nb_path_sha1": hashlib.sha1(nb_path.encode()).hexdigest()[:12],
                "template": blob.get("template",""),
            }
            if sig_cache == sig_now_partial:
                tokens_raw = blob["tokens_raw"]
                tokens_text = blob["tokens_text"]
                nb_vecs = blob["nb_vecs"]  # CPU float32
                T_clip = blob["T_clip"]    # CPU float16/float32/bf16
                # Move T_clip to device & dtype
                T_clip = T_clip.to(device=device, dtype=dtype_clip_bank, non_blocking=True)
                # Ensure L2 normalized (cached should be already normalized, but normalize defensively)
                T_clip = F.normalize(T_clip.float(), dim=1).to(dtype_clip_bank)
                return tokens_raw, tokens_text, nb_vecs, T_clip, cache_path

    # Load NB fresh
    print(f"🔹 Loading Numberbatch from: {nb_path}")
    tokens_raw, nb_vecs = load_numberbatch_txt_gz(nb_path, limit=vocab_limit if vocab_limit>0 else 0, expect_dim=None)
    V, Dnb = nb_vecs.shape
    print(f"   Loaded {V:,} entries with dim {Dnb}")

    # Build texts for CLIP encoder
    tokens_text = [_nb_token_to_prompt_text(t) for t in tokens_raw]
    if template:
        tokens_text = [template.replace("{}", s) for s in tokens_text]

    # Encode in batches
    embs = []
    print("🔹 Encoding Numberbatch tokens in CLIP text space...")
    for i in range(0, len(tokens_text), batch_size):
        chunk = tokens_text[i:i+batch_size]
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            t = model.encode_text(chunk, device=device)  # [B=1, K, C]
        
        t = F.normalize(t[0].float(), dim=-1)        # [K,C], float32
        embs.append(t.cpu())
        if (i // batch_size) % 2 == 0:
            print(f"   {i}/{len(tokens_text)} ...")

    T_clip_cpu = torch.cat(embs, dim=0)  # [V,C] float32 (normalized)

    # Move to device & dtype
    T_clip = T_clip_cpu.to(device=device, dtype=dtype_clip_bank, non_blocking=True)

    # Save cache
    cache_used = None
    if cache_path:
        sig = _cache_signature(model, Dnb, nb_path, template)
        blob = {
            "sig": sig,
            "tokens_raw": tokens_raw,
            "tokens_text": tokens_text,
            "nb_vecs": nb_vecs,             # CPU float32
            "T_clip": T_clip_cpu.to(torch.float16),  # store compactly on disk; will recast on load
            "template": template or "",
        }
        torch.save(blob, cache_path)
        cache_used = cache_path
        print(f"🗂️  Saved CLIP text-bank cache: {cache_path}")

    return tokens_raw, tokens_text, nb_vecs, T_clip, cache_used

# ----------------------------
# Top-k over patches via bf16 matmul
# ----------------------------

@torch.no_grad()
def per_patch_topk_from_image(model: DenseCLIP,
                              device: str,
                              image_tensor: torch.Tensor,   # [1,3,H,W], CLIP-normalized
                              T_clip: torch.Tensor,         # [V,C], on device, dtype bf16/fp16/fp32
                              topk: int = 3,
                              return_patch_grid: bool = True):
    """
    Compute per-patch top-k tokens from prebuilt CLIP text bank.
    Returns:
        top_vals: numpy [P, k]
        top_idx:  numpy [P, k]
        (h, w):   patch grid size if return_patch_grid
        vmap:     normalized patch embeddings [P, C] (float32, CPU) if needed downstream
    """
    bb_dtype = next(model.backbone.parameters()).dtype
    image_tensor = image_tensor.to(device=device, dtype=bb_dtype, non_blocking=True)

    # Get visual fmap
    g, fmap = model.encode_image_to_embeddings(image_tensor.to(device=device))
    # fmap: [1,C,h,w]
    vmap = F.normalize(fmap, dim=1)[0]            # [C,h,w]
    C, h, w = vmap.shape
    P = h * w

    # Prepare matrices
    # Q: [P,C] on device, dtype matches T_clip
    Q = vmap.reshape(C, P).t().contiguous()       # [P,C]
    Q = Q.to(device=device, dtype=T_clip.dtype, non_blocking=True)

    # Similarity = Q @ T^T -> [P,V]
    # Using bf16/fp16 if provided, otherwise fp32.
    S = Q @ T_clip.t()                            # [P,V]
    # topk over V
    top_vals, top_idx = torch.topk(S, k=min(topk, T_clip.shape[0]), dim=1)

    top_vals_np = top_vals.float().cpu().numpy()
    top_idx_np  = top_idx.int().cpu().numpy()

    if return_patch_grid:
        return top_vals_np, top_idx_np, (h, w), vmap.float().cpu()
    else:
        return top_vals_np, top_idx_np, None, vmap.float().cpu()

# ----------------------------
# CSV / PT writers
# ----------------------------

def save_topk_csv(path: str,
                  top_vals: np.ndarray,   # [P,k]
                  top_idx: np.ndarray,    # [P,k]
                  tokens_text: List[str],
                  grid_hw: Tuple[int,int]):
    h, w = grid_hw
    P, k = top_vals.shape
    assert P == h*w

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["patch_id", "r", "c"] + sum(([f"token{i+1}", f"score{i+1}"] for i in range(k)), [])
        writer.writerow(header)
        for p in range(P):
            r = p // w
            c = p %  w
            row = [p, r, c]
            for i in range(k):
                idx = int(top_idx[p, i])
                tok = tokens_text[idx]
                sc  = float(top_vals[p, i])
                row += [tok, sc]
            writer.writerow(row)

def save_topk_pt(path: str,
                 top_vals: np.ndarray,      # [P,k]
                 top_idx: np.ndarray,       # [P,k]
                 tokens_raw: List[str],
                 tokens_text: List[str],
                 nb_vecs: torch.Tensor,     # [V,Dnb] CPU float32
                 grid_hw: Tuple[int,int],
                 extra: Optional[Dict] = None):
    """
    Saves a compact training-ready package:
        - topk indices & scores
        - tokens (raw + text)
        - selected numberbatch vectors for each patch/topk (gathered)
    """
    h, w = grid_hw
    P, k = top_vals.shape
    V, Dnb = nb_vecs.shape

    # Gather selected NB vectors for convenience: [P,k,Dnb]
    # (keep on CPU float32 for training-time flexibility)
    idx_t = torch.from_numpy(top_idx.reshape(-1))        # [P*k]
    sel_nb = nb_vecs.index_select(0, idx_t).reshape(P, k, Dnb).contiguous()

    pack = {
        "grid_hw": grid_hw,
        "topk_scores": torch.from_numpy(top_vals),       # [P,k] float32
        "topk_indices": torch.from_numpy(top_idx),       # [P,k] int32
        "tokens_raw": tokens_raw,
        "tokens_text": tokens_text,
        "selected_nb_vecs": sel_nb,                      # [P,k,Dnb] float32
    }
    if extra:
        pack["extra"] = extra

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(pack, path)
    print(f"💾 Saved top-k PT package to: {path}")

# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    # DenseCLIP
    p.add_argument('--ckpt', default="/speedy/DenseInject/weights/denseclip.pth", help="DenseCLIP checkpoint .pth")
    p.add_argument('--arch', default='vit-b-16', choices=['rn50','rn101','vit-b-16'])
    p.add_argument('--device', default='cuda')
    # Image + heatmap (optional overlay like your original)
    p.add_argument('--image', default='/speedy/DenseInject/datasets/coco/test2017/000000000016.jpg')
    p.add_argument('--text', default="")  # just for printing heatmap stats/overlay title
    p.add_argument('--out', default="heatmap_nb.png")
    # Numberbatch
    p.add_argument('--nb_path', default="/speedy/DenseInject/datasets/numberbatch/numberbatch-en-19.08.txt.gz")
    p.add_argument('--nb_limit', type=int, default=0, help="For quick tests, limit NB entries (>0).")
    p.add_argument('--nb_cache', default="/speedy/DenseInject/denseclipinference/text_embedding_cache.pt", help="Path to save/load text-bank cache (e.g., textbank_nb.pt).")
    p.add_argument('--template', default="", help="Optional prompt template e.g. 'a photo of {}'.")
    p.add_argument('--batch_size_text', type=int, default=4096)
    # Retrieval
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--bf16', action='store_true', help="Use bfloat16 for T and Q matmul.")
    # Outputs
    p.add_argument('--out_topk_csv', default="patch_topk_nb_resized.csv")
    p.add_argument('--out_topk_pt',  default="patch_topk_nb.pt")
    p.add_argument('--save_overlay', action='store_true', help="Also save heatmap overlay (like your original).")

    args = p.parse_args()

    # Device
    dev = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    print(f"🖥️  Using device: {dev}")

    # Model
    model = DenseCLIP(args.arch, args.ckpt, device=dev).eval().float()
    model = model.to(device=dev, dtype=torch.float32)  # <- cast params & buffers to fp32


    # Load image
    pil, x, original_size = load_image_clip(args.image, device=dev)

    # (Optional) compute/print a heatmap using your similarity routine
    heat = model.similarity_heatmap(x, [args.text if args.text else "object"], original_size)
    print("-" * 40)
    print(f"📊 Heatmap stats for text: '{args.text if args.text else 'object'}'")
    print(f"  - max: {np.max(heat):.4f}")
    print(f"  - min: {np.min(heat):.4f}")
    print(f"  - Original image size: {original_size[0]}x{original_size[1]}")
    print(f"  - Processing size: 256x256")
    print(f"  - Final heatmap size: {heat.shape[1]}x{heat.shape[0]}")
    print("-" * 40)
    if args.save_overlay:
        save_heatmap_overlay(pil, heat, args.out)
        print(f"✅ Saved heatmap overlay to: {args.out}")

    # Build or load Numberbatch CLIP text-bank
    dtype_bank = torch.bfloat16 if args.bf16 else torch.float32
    tokens_raw, tokens_text, nb_vecs, T_clip, cache_used = build_or_load_nb_textbank(
        model=model,
        device=dev,
        nb_path=args.nb_path,
        cache_path=(args.nb_cache if args.nb_cache else None),
        template=(args.template if args.template else None),
        vocab_limit=args.nb_limit if args.nb_limit>0 else 0,
        batch_size=args.batch_size_text,
        dtype_clip_bank=dtype_bank,
    )
    if cache_used:
        print(f"🗂️  Loaded/updated text-bank cache: {cache_used}")
    print(f"📚 Text bank size: {len(tokens_raw):,} ; dtype={T_clip.dtype}; device={T_clip.device}")

    # Per-patch top-k
    top_vals, top_idx, (h, w), vmap_cpu = per_patch_topk_from_image(
        model=model,
        device=dev,
        image_tensor=x,
        T_clip=T_clip,
        topk=args.topk,
        return_patch_grid=True
    )
    P = h * w
    print(f"🔎 Computed top-{args.topk} for {P} patches ({h}x{w}).")

    # Save CSV
    if args.out_topk_csv:
        save_topk_csv(args.out_topk_csv, top_vals, top_idx, tokens_text, (h, w))
        print(f"✅ Wrote CSV: {args.out_topk_csv}")

    # Save PT bundle (includes selected Numberbatch vectors)
    if args.out_topk_pt:
        extra = {
            "image_path": args.image,
            "ckpt": args.ckpt,
            "arch": args.arch,
            "bf16": args.bf16,
            "template": args.template,
        }
        save_topk_pt(args.out_topk_pt, top_vals, top_idx, tokens_raw, tokens_text, nb_vecs, (h, w), extra=extra)

    print("🎯 Done.")

if __name__ == "__main__":
    main()
