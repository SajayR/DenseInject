import torch, torch.nn as nn, torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizerFast
from denseclip import DenseCLIP           # your minimal DenseCLIP wrapper

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
