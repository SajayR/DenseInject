import math
import os
import argparse
from typing import List, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

# ----------------------------
# Utilities / tokenizer
# ----------------------------
import gzip, html, regex as re
from functools import lru_cache

@lru_cache()
def _default_bpe_path():
    # bpe_simple_vocab_16e6.txt.gz is the same file used in the repo.
    # Put a copy next to this script OR change the path here.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "/speedy/DenseInject/DenseCLIP/segmentation/denseclip/bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def _get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def _basic_clean(text):
    import ftfy
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def _whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = _default_bpe_path()):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(_bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = _get_pairs(word)
        if not pairs:
            return token+'</w>'
        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i]); i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = _get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = _whitespace_clean(_basic_clean(text))
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

_tokenizer = SimpleTokenizer()

def tokenize_texts(texts: Union[str, List[str]], context_length: int = 5, truncate: bool = True) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]
    sot = _tokenizer.encoder["<|startoftext|>"]
    eot = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot] + _tokenizer.encode(t) + [eot] for t in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot
            else:
                raise RuntimeError(f"text too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


# ----------------------------
# Core layers
# ----------------------------
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        return super().forward(x.float()).type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(1.702 * x)

class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p=p
    def forward(self, x):
        if self.p == 0.0 or not self.training: return x
        keep = 1 - self.p
        shape = (x.shape[0],)+(1,)* (x.ndim-1)
        return x.div(keep) * (torch.rand(shape, device=x.device) < keep).float()
from collections import OrderedDict

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, drop_path=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(d_model, d_model*4)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(d_model*4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.dp = DropPath(drop_path) if drop_path>0 else nn.Identity()
    def attention(self, x):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]
    def forward(self, x):
        x = x + self.dp(self.attention(self.ln_1(x)))
        x = x + self.dp(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None, drop_path_rate=0.0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])
    def forward(self, x): return self.resblocks(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim); self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, q, k, v):
        B, N, C = q.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, -1, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, -1, self.num_heads, C // self.num_heads)
        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
    def forward(self, x, mem):
        q = k = v = self.norm1(x); x = x + self.self_attn(q,k,v)
        q = self.norm2(x); x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x

# ----------------------------
# Backbones
# ----------------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False); self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None; self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(nn.AvgPool2d(stride),
                                            nn.Conv2d(inplanes, planes * self.expansion, 1, bias=False),
                                            nn.BatchNorm2d(planes * self.expansion))
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim); self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim); self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads; self.embed_dim = embed_dim; self.spacial_dim = spacial_dim
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(2,0,1)  # (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        cls_pos = self.positional_embedding[0:1, :]
        # interpolate spatial pos to current H x W
        spatial_pos = F.interpolate(self.positional_embedding[1:].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0,3,1,2),
                                    size=(H,W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1,0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight,
            in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
            out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True, training=self.training, need_weights=False
        )
        x = x.permute(1,2,0)  # B C (1+HW)
        global_feat = x[:,:,0]
        fmap = x[:,:,1:].reshape(B,-1,H,W)
        return global_feat, fmap

class CLIPResNetWithAttention(nn.Module):
    def __init__(self, layers, output_dim=1024, input_resolution=512, width=64):
        super().__init__()
        self.output_dim = output_dim; self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width//2, 3, stride=2, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(width//2)
        self.conv2 = nn.Conv2d(width//2, width//2, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(width//2)
        self.conv3 = nn.Conv2d(width//2, width, 3, padding=1, bias=False); self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2); self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(width*4, layers[2], stride=2)
        self.layer4 = self._make_layer(width*8, layers[3], stride=2)
        embed_dim = width * 32
        sd = input_resolution // 32
        self.attnpool = AttentionPool2d(sd, embed_dim, 32, output_dim)
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks): layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        def stem(z):
            for conv, bn in [(self.conv1,self.bn1),(self.conv2,self.bn2),(self.conv3,self.bn3)]:
                z = self.relu(bn(conv(z)))
            return self.avgpool(z)
        x = stem(x)
        x = self.layer1(x); x1=x
        x = self.layer2(x); x2=x
        x = self.layer3(x); x3=x
        x = self.layer4(x); x4=x
        g, f = self.attnpool(x)  # global, fmap
        return (x1,x2,x3,x4,[g,f])

class CLIPVisionTransformer(nn.Module):
    """Minimal CLIP ViT-B/16 that returns (global, fmap) for similarity."""
    def __init__(self, input_resolution=640, patch_size=16, width=768, layers=12, heads=12, output_dim=512, drop_path_rate=0.0):
        super().__init__()
        self.input_resolution = input_resolution; self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)
        # projection like CLIP
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.width = width
    def forward(self, x):
        x = self.conv1(x)  # B, C=width, H/ps, W/ps
        B,C,H,W = x.shape
        n = H*W
        x = x.reshape(B, C, n).permute(0,2,1)  # B, N, C
        cls = self.class_embedding.to(x.dtype) + torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)  # B, N+1, C
        # interpolate pos to current H,W
        pos = self.positional_embedding
        cls_pos = pos[0:1,:]
        spatial_pos = F.interpolate(pos[1:].reshape(1, self.spatial_size, self.spatial_size, C).permute(0,3,1,2),
                                    size=(H,W), mode='bilinear').reshape(1,C,n).permute(0,2,1)
        pos = torch.cat([cls_pos.reshape(1,1,C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1,0,2)  # L,B,C
        x = self.transformer(x)
        x = x.permute(1,0,2)  # B,L,C
        # global (cls)
        xcls = self.ln_post(x[:,0,:]) @ self.proj  # B, output_dim
        # local tokens (exclude cls), project and reshape to fmap
        xtok = self.ln_post(x[:,1:,:]) @ self.proj  # B, N, output_dim
        fmap = xtok.permute(0,2,1).reshape(B, self.output_dim, H, W)  # B,C,H,W
        return (None,None,None,None,[xcls, fmap])


# ----------------------------
# Text encoder and context decoder
# ----------------------------
def _build_causal_mask(L):
    m = torch.empty(L, L); m.fill_(float("-inf")); m.triu_(1); return m

class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=13, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12, embed_dim=1024):
        super().__init__()
        self.context_length = context_length
        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads, _build_causal_mask(context_length))
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # (weights loaded from checkpoint)
    def forward(self, text, context):
        # text: [K, N1] token ids ; context: [B, N2, C=transformer_width]
        x_text = self.token_embedding(text)  # K,N1,C
        K,N1,C = x_text.shape
        B,N2,C2 = context.shape
        assert C2 == C
        eos_idx = text.argmax(dim=-1) + N2  # K
        eos_idx = eos_idx.reshape(1,K).expand(B,K).reshape(-1)
        x_text = x_text.reshape(1,K,N1,C).expand(B,K,N1,C)
        context = context.reshape(B,1,N2,C).expand(B,K,N2,C)
        x = torch.cat([x_text[:,:,0:1], context, x_text[:,:,1:]], dim=2).reshape(B*K, N1+N2, C)
        x = x + self.positional_embedding  # broadcast on seq
        x = x.permute(1,0,2)  # L, BK, C
        x = self.transformer(x)
        x = x.permute(1,0,2)  # BK, L, C
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_idx] @ self.text_projection  # BK, embed_dim
        return x.reshape(B, K, -1)  # B,K,embed_dim

class ContextDecoder(nn.Module):
    def __init__(self, transformer_width=256, transformer_heads=4, transformer_layers=3, visual_dim=1024, dropout=0.1):
        super().__init__()
        self.memory_proj = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, transformer_width), nn.LayerNorm(transformer_width))
        self.text_proj   = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, transformer_width))
        self.decoder = nn.ModuleList([TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)])
        self.out_proj = nn.Sequential(nn.LayerNorm(transformer_width), nn.Linear(transformer_width, visual_dim))
    def forward(self, text, visual):
        # text: B,K,C ; visual: B,N,C
        B,N,C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)
        for layer in self.decoder:
            x = layer(x, visual)
        return self.out_proj(x)  # B,K,C


# ----------------------------
# Minimal wrapper for heatmap
# ----------------------------
def _strict_load(module, prefix, state):
        # grab all ckpt keys that start with prefix
        raw = {k.replace(prefix + '.', ''): v
            for k, v in state.items() if k.startswith(prefix + '.')}

        # keep only intersection with module’s own keys
        mod_keys = set(module.state_dict().keys())
        filt = {k: v for k, v in raw.items() if k in mod_keys}

        # helpful diagnostics if something is off
        missing = sorted(list(mod_keys - set(filt.keys())))
        unexpected = sorted([k for k in raw.keys() if k not in mod_keys])
        if unexpected:
            print(f"[{prefix}] dropping {len(unexpected)} unexpected keys (e.g., {unexpected[:8]})")
        if missing:
            # if you expect zero here, fail early
            raise RuntimeError(f"[{prefix}] checkpoint missing {len(missing)} required params, "
                            f"examples: {missing[:8]}")

        module.load_state_dict(filt, strict=True)

class DenseCLIP(nn.Module):
    
    def __init__(self, arch: str, ckpt_path: str, device='cuda'):
        super().__init__()
        sd_all = torch.load(ckpt_path, map_location='cpu')
        state = sd_all.get('state_dict', sd_all)

        # strip possible "module." prefix
        state = {k.replace('module.', ''): v for k,v in state.items()}

        # Decide backbone type if not obvious from args
        self.arch = arch.lower()
        if self.arch in ['rn50','rn101']:
            # infer input_resolution from attnpool.positional_embedding length
            pos = state[[k for k in state.keys() if k.startswith('backbone.attnpool.positional_embedding')][0]]
            spacial_dim = int(round(math.sqrt(pos.shape[0]-1)))
            input_res = spacial_dim * 32
            layers = [3,4,6,3] if self.arch=='rn50' else [3,4,23,3]
            # infer output_dim from attnpool.c_proj.weight
            output_dim = state['backbone.attnpool.c_proj.weight'].shape[0]
            self.backbone = CLIPResNetWithAttention(layers=layers, output_dim=output_dim, input_resolution=input_res)
        elif self.arch == 'vit-b-16':
            # infer params from backbone shapes
            w = state['backbone.conv1.weight'].shape[0]              # width
            ps = state['backbone.conv1.weight'].shape[-1]            # patch size
            pos_len = state['backbone.positional_embedding'].shape[0]
            grid = int(round(math.sqrt(pos_len-1)))
            input_res = grid * ps
            # infer ViT depth correctly
            layer_idxs = sorted({int(k.split('.')[3]) for k in state
                                if k.startswith('backbone.transformer.resblocks')
                                and k.endswith('.attn.in_proj_weight')})
            layers = max(layer_idxs) + 1

            heads = w // 64
            output_dim = state['backbone.proj'].shape[1]
            self.backbone = CLIPVisionTransformer(input_resolution=input_res, patch_size=ps, width=w, layers=layers, heads=heads, output_dim=output_dim)
        else:
            raise ValueError("arch must be one of: rn50 | rn101 | vit-b-16")

        # Text encoder dims from checkpoint
        txt_proj = state['text_encoder.text_projection']
        transformer_width = state['text_encoder.ln_final.weight'].shape[0]
        embed_dim = txt_proj.shape[1]
        context_length = state['text_encoder.positional_embedding'].shape[0]
        heads = transformer_width // 64
        txt_layer_idxs = sorted({int(k.split('.')[3]) for k in state
                         if k.startswith('text_encoder.transformer.resblocks')
                         and k.endswith('.attn.in_proj_weight')})
        layers = max(txt_layer_idxs) + 1

        self.text_encoder = CLIPTextContextEncoder(context_length=context_length,
                                                   transformer_width=transformer_width,
                                                   transformer_heads=heads,
                                                   transformer_layers=layers,
                                                   embed_dim=embed_dim)

        # Context decoder dims from checkpoint
        t_width = state['context_decoder.memory_proj.1.weight'].shape[0]   # 256
        vis_dim = state['context_decoder.memory_proj.1.weight'].shape[1]   # 512
        # (equivalently: vis_dim = state['context_decoder.out_proj.1.weight'].shape[0])

        # try to infer heads/layers
        dec_layers = sorted({int(k.split('.')[2]) for k in state if k.startswith('context_decoder.decoder.')})
        n_layers = (max(dec_layers)+1) if len(dec_layers)>0 else 3
        # heads heuristic: q_proj is [d,d]; choose divisor of d close to 8/4
        cand = [8,4,2,1,16]
        n_heads = next((h for h in cand if t_width % h == 0), 4)
        self.context_decoder = ContextDecoder(transformer_width=t_width, transformer_heads=n_heads, transformer_layers=n_layers, visual_dim=vis_dim)

        # learned contexts & gamma
        self.contexts = nn.Parameter(state['contexts'].clone())  # [1, N2, token_embed_dim]
        self.gamma = nn.Parameter(state['gamma'].clone())

        # Load weights for submodules
        _strict_load(self.backbone, 'backbone', state)
        _strict_load(self.text_encoder, 'text_encoder', state)
        _strict_load(self.context_decoder, 'context_decoder', state)   

        # Running params
        self.tau = 0.07  # use same scaling as paper
        # after building/loading all submodules
        self.float()                      # make every param/buffer float32

        self.to(device)

    @torch.no_grad()
    def encode_image_to_embeddings(self, image: torch.Tensor):
        """
        image: [B,3,H,W], float in [0,1], CLIP-normalized
        returns: global_feat [B,C], local_fmap [B,C,h,w]
        """
        x = self.backbone(image)
        global_feat, local_fmap = x[4]
        return global_feat, local_fmap

    @torch.no_grad()
    def encode_text(self, texts: List[str], device='cuda'):
        """
        Builds context-conditioned text embedding for a list of strings.
        Returns: [B, K, C] where B is batch (expand), K=len(texts)
        """
        # Use the same short "class token context length" that DenseCLIP expects.
        # This is state-dependent: they trained with N_text tokens (e.g., 5)
        # We pick that from (self.text_encoder.context_length - self.contexts.shape[1])
        n_text = self.text_encoder.context_length - self.contexts.shape[1]
        tok = tokenize_texts(texts, context_length=n_text).to(device)
        '''
        for i, text in enumerate(texts):
            tokens = tok[i].cpu().numpy()
            # Remove padding (zeros)
            tokens = tokens[tokens != 0]
            
            # Decode back to text
            decoded_tokens = []
            for token_id in tokens:
                if token_id in _tokenizer.decoder:
                    decoded_tokens.append(_tokenizer.decoder[token_id])
            
            print(f"Original text {i}: '{text}'")
            print(f"Tokens ({len(tokens)}): {tokens}")
            print(f"Decoded: {decoded_tokens}")
            print(f"Context length limit: {n_text}")
            print("---")
        '''
        B = 1  # we use single-image flow; text encoder expands later
        # Expand learned contexts to batch
        ctx = self.contexts.to(device).expand(B, -1, -1)  # [B, N2, C_emb]
        # text features: [B,K,C]
        t = self.text_encoder(tok, ctx)                  # base text
        # adapt text using visual context per image later (call-conditioned)
        return t  # not adapted yet

