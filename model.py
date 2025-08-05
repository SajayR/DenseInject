from operator import truediv
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast
from denseclip import DenseCLIP           # your minimal DenseCLIP wrapper
import math
from typing import Optional, Tuple
from utils.top_k_resized import build_or_load_nb_textbank



# ---------- Helpers ----------------------------------------------------------



class Q2VCrossAttn(nn.Module):
    """Cross-attention from question tokens (Q) to visual patch tokens (V)."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)       # tiny non-linear pool
        self.tanh = nn.Tanh()

    def forward(self, q, v):                      # q:(B,L,C)  v:(B,P,C)
        q_, v_ = self.norm_q(q), self.norm_v(v)
        h, _   = self.attn(q_, v_, v_)            # (B,L,C)
        cls     = h[:, 0]                         # take “[CLS]”-token of Q
        return self.tanh(self.out_proj(cls))      # (B,C)  question-conditioned

# --- in model.py (top) ---
import math
from typing import Optional, Tuple
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NBInjector(nn.Module):
    """
    Concat-based commonsense injection for per-patch features.
    - bank: T_clip [V, C_v] (GPU, bf16/fp16/fp32, L2-normalized rows)
            nb_vecs [V, D_nb] (CPU float32)
    - retrieve(): top-k over vocab for all patches (chunked if needed)
    - fuse_concat(): concat (k*D_nb [+ k scores]) -> proj(C_v) -> gated residual
    """
    def __init__(self, v_dim: int, nb_dim: int, k: int = 3, include_scores: bool = True,
                 proj_hidden: Optional[int] = None, gate_scalar: bool = False):
        super().__init__()
        self.v_dim = v_dim
        self.nb_dim = nb_dim
        self.k = k
        self.include_scores = include_scores

        in_dim = k * nb_dim + (k if include_scores else 0)
        h = proj_hidden or max(512, v_dim)
        self.nb_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h, v_dim),
        )
        # gate
        self.gate = nn.Linear(v_dim * (1 if gate_scalar else 2), 1 if gate_scalar else v_dim)
        self.gate_scalar = gate_scalar
        # temperature for weighting scores if you later use weighted sum; here we only pass scores to the MLP
        self.tau = nn.Parameter(torch.tensor(0.07))

        # banks
        self.register_buffer("T_clip", torch.empty(0), persistent=False)  # [V, C_v] on device
        self.nb_vecs = None  # CPU FloatTensor [V, D_nb]

    def set_bank(self, T_clip: torch.Tensor, nb_vecs: torch.Tensor):
        # T_clip should already be L2-normalized rows
        self.T_clip = T_clip  # on device, dtype can be bf16/fp16
        self.nb_vecs = nb_vecs.cpu()  # keep on CPU float32

    @torch.no_grad()
    def retrieve(self, v_seq: torch.Tensor, topk: Optional[int] = None,
                 chunk_patches: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        v_seq: (B, P, C_v)  normalized or not; we will normalize
        returns:
          idx: (B, P, k) indices into vocab
          val: (B, P, k) raw sims (same dtype as T_clip)
          nb_sel: (B, P, k, D_nb) CPU float32 NB vectors gathered
        """
        assert self.T_clip.numel() > 0 and self.nb_vecs is not None, "NB bank not set"
        T = self.T_clip  # [V, C_v] (device)
        B, P, C = v_seq.shape
        k = topk or self.k

        # normalize v_seq to match T rows
        q = F.normalize(v_seq, dim=-1)  # (B,P,C); stay in fp32 then cast chunk-wise

        idx_all, val_all = [], []
        # chunk over patches to limit peak memory for (P×V)
        for b in range(B):
            qb = q[b]  # (P,C)
            if chunk_patches is None or chunk_patches >= P:
                qb_chunked = [qb]
            else:
                qb_chunked = torch.split(qb, chunk_patches, dim=0)

            idx_b, val_b = [], []
            for qc in qb_chunked:  # (Pc, C)
                qc = qc.to(dtype=T.dtype, device=T.device, non_blocking=True)
                # S = qc @ T^T : (Pc, V)
                S = qc @ T.t()
                v, i = torch.topk(S, k=min(k, T.shape[0]), dim=1)
                idx_b.append(i.cpu())
                val_b.append(v.float().cpu())  # keep scores in float32 for stability
                del S
            idx_b = torch.cat(idx_b, dim=0)  # (P,k) on CPU
            val_b = torch.cat(val_b, dim=0)  # (P,k) on CPU
            idx_all.append(idx_b)
            val_all.append(val_b)

        idx = torch.stack(idx_all, dim=0)  # (B,P,k) CPU long
        val = torch.stack(val_all, dim=0)  # (B,P,k) CPU float32

        # Gather NB vectors: for speed, one big index_select then reshape
        BPK = idx.reshape(-1)  # (B*P*k,)
        nb_sel = self.nb_vecs.index_select(0, BPK).reshape(B, P, k, self.nb_dim)  # CPU float32
        return idx, val, nb_sel

    def fuse_concat(self, v_seq: torch.Tensor, nb_sel: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        v_seq : (B,P,C_v)
        nb_sel: (B,P,k,D_nb) CPU float32
        scores: (B,P,k) CPU float32  (raw sims)
        returns fused: (B,P,C_v)
        """
        B, P, C = v_seq.shape
        k, Dnb = self.k, self.nb_dim

        # concat NB vectors (and optionally scores), project
        print("nb_sel", nb_sel.shape)
        x = nb_sel.reshape(B, P, k * Dnb).to(v_seq.device, non_blocking=True)
        print("x", x.shape)
        if self.include_scores:
            x = torch.cat([x, scores.to(v_seq.device, non_blocking=True)], dim=-1)  # (B,P, k*Dnb + k)
        nb_feat = self.nb_proj(x)  # (B,P,C_v)

        # gate
        if self.gate_scalar:
            g_in = v_seq  # scalar gate uses only v
        else:
            g_in = torch.cat([v_seq, nb_feat], dim=-1)
        g = torch.sigmoid(self.gate(g_in))  # (B,P,1) or (B,P,C_v)
        if g.shape[-1] == 1: g = g.expand_as(v_seq)

        fused = F.layer_norm(v_seq + g * nb_feat, (C,))
        return fused

# ---------- Main module ------------------------------------------------------

class QCDenseCLIP(nn.Module):
    """
    Question-Conditioned DenseCLIP with a **single** trainable DistilBERT
    text tower for both questions *and* answer options.
    """
    def __init__(self,
                 denseclip_ckpt,
                 arch='vit-b-16',
                 distil_name='distilbert-base-uncased',
                 num_heads=8,
                 device='cuda'):
        super().__init__()
        self.device = device

        # ➊ Frozen DenseCLIP vision trunk (patch tokens only)
        self.vision = DenseCLIP(arch, denseclip_ckpt, device=device).eval()
        for p in self.vision.parameters(): p.requires_grad = False
        self.v_dim = self.vision.gamma.numel()    # e.g. 512 or 1024
        print("v_dim", self.v_dim)
        self.enable_nb = True

        self.nb_k = 1        # default top-k
        self.nb_dim = 300       # Numberbatch dim (19.08 is 300-d)
        self.nb = NBInjector(v_dim=self.v_dim, nb_dim=self.nb_dim, k=self.nb_k, include_scores=True, proj_hidden=None, gate_scalar=False)


        # ➋ Trainable DistilBERT tower  (shared by Q and A)
        self.text_tok = DistilBertTokenizerFast.from_pretrained(distil_name)
        self.text_enc = DistilBertModel.from_pretrained(distil_name).eval()
        for p in self.text_enc.parameters(): p.requires_grad = False
        self.t_dim = self.text_enc.config.hidden_size  # 768

        # If dims don’t match → small projection so everything sits in v_dim
        #self.t2v = nn.Linear(self.t_dim, self.v_dim, bias=False)

        self.t2v_in = nn.Linear(self.t_dim, self.v_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.v_dim, nhead=8, dim_feedforward=self.v_dim*4)
        self.t2v_enc = nn.TransformerEncoder(encoder_layer, num_layers=2)


        # ➌ Cross-attention & pool
        self.q2v = Q2VCrossAttn(self.v_dim, num_heads=num_heads)

        # temperature learnable (init 0.07 like CLIP)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    # --- in QCDenseCLIP ---
    def set_nb_bank(self, T_clip: torch.Tensor, nb_vecs: torch.Tensor):
        """
        T_clip: [V, C_v] L2-normalized text-bank on same device as vision.
        nb_vecs: [V, 300] CPU float32 Numberbatch full table.
        """
        self.nb.set_bank(T_clip, nb_vecs)


    # --------- encoding utilities -------------------------------------------

    def _encode_question(self, questions):
        """
        questions: list[str] length B
        returns   : (B, Lq, v_dim)
        """
        #print(questions)
        toks = self.text_tok(questions,
                             truncation=True,
                             padding='longest',
                             return_tensors='pt').to(self.device)
        out  = self.text_enc(**toks).last_hidden_state      # (B,L,768)
        x = self.t2v_in(out)            # (B, L, v_dim)
        x = self.t2v_enc(x.permute(1,0,2)).permute(1,0,2)  # (B, L, v_dim)
        return x                                # (B,L,C_v)

    def _encode_answers(self, answers):
        """
        answers: list[list[str]]  B × M
        returns: (B, M, v_dim)
        """
        flat = [a for row in answers for a in row]          # B*M
        #print(flat)
        toks = self.text_tok(flat,
                             truncation=True, padding=True,
                             return_tensors='pt').to(self.device)
        out  = self.text_enc(**toks).last_hidden_state[:,0] # CLS only  (B*M,768)
        emb  = self.t2v(out)                                # (B*M,C_v)
        B,M  = len(answers), len(answers[0])
        return emb.view(B, M, -1)                           # (B,M,C_v)

    def _encode_answers(self, answers):  # answers: B×M
    
        all_emb = []
        for opts in answers:
            texts = [f"{o}" for o in opts]
            t = self.vision.encode_text(texts, device=self.device)  # (1, M, C_v)
            all_emb.append(t.squeeze(0))  # → (M, C_v)
        return torch.stack(all_emb, dim=0)  # (B, M, C_v)


    # --------- forward -------------------------------------------------------

    def forward(self, images, questions, answers):
        """
        images    : (B,3,H,W) already CLIP-normalized to match DenseCLIP
        questions : list[str]    length B
        answers   : list[list[str]]  B × M (M=4 for A-OKVQA)
        """
        B = images.size(0)

        # (1) vision → patch tokens (no cls)
        _, fmap      = self.vision.encode_image_to_embeddings(images)
        #print(fmap.shape)
        #simulate fake fmap for ablation
        #fmap = torch.randn(B, self.v_dim, 16, 16, device=images.device) 

        v_seq        = fmap.flatten(2).permute(0,2,1)       # (B,P,C_v)

        if self.enable_nb and self.nb.T_clip.numel() > 0:
            # retrieve top-k for each patch (chunked to cap memory)
            idx, scores, nb_sel = self.nb.retrieve(v_seq, topk=self.nb_k, chunk_patches=2048)
            # fuse by concat+gate → same shape as v_seq
            v_seq = self.nb.fuse_concat(v_seq, nb_sel, scores)

        # (2) Q tokens
        q_seq        = self._encode_question(questions)     # (B,L,C_v)

        # (3) Q→V cross-attention pooled vector
        z_qv         = F.normalize(self.q2v(q_seq, v_seq), dim=-1)  # (B,C_v)

        # (4) answer embeddings
        a_emb        = F.normalize(self._encode_answers(answers), dim=-1)  # (B,M,C_v)

        # (5) cosine similarities → logits
        scale = self.logit_scale.exp()                      # learnable τ^-1
        logits = scale * torch.einsum('bd,bmd->bm', z_qv, a_emb)  # (B,M)

        return logits
