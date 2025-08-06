import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from aokvqa_data import AOKVQADataset
from model import QCDenseCLIP
import tqdm
import os
import wandb
import torch, torch.nn.functional as F
from torchvision.utils import save_image
import json
DO_WANDB = False
if DO_WANDB:
    wandb.init(project="DenseInject", name="DenseCLIP+NB")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Debug setup
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
mean = torch.tensor(CLIP_MEAN).view(3,1,1)
std  = torch.tensor(CLIP_STD).view(3,1,1)

def denorm_clip(x):  # x: (3,H,W)
    return (x.cpu()*std + mean).clamp(0,1)

DEBUG_EVERY = 510
OUTDIR = "./nb_peek"
os.makedirs(OUTDIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = QCDenseCLIP('/speedy/DenseInject/weights/denseclip.pth', device=device).to(device)

cache_path = "/speedy/DenseInject/denseclipinference/text_embedding_cache.pt"
blob = torch.load(cache_path, map_location="cpu")
# Pull from cache
T_cpu   = blob["T_clip"]      # stored as fp16 in your saver
nb_vecs = blob["nb_vecs"]     # FloatTensor [V, 300] on CPU
tokens_text = blob.get("tokens_text", blob.get("tokens_raw"))

# Normalize and move to GPU in bf16
T_clip = F.normalize(T_cpu.float(), dim=1).to(device, dtype=torch.bfloat16)

# Hand to the model
model.set_nb_bank(T_clip, nb_vecs, tokens_text)

print(f"Loaded bank: T_clip {T_clip.shape} {T_clip.dtype} on {T_clip.device}; nb_vecs {tuple(nb_vecs.shape)} on CPU")

# Count frozen and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"  Parameter counts:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters: {frozen_params:,}")
print(f"  Trainable ratio: {trainable_params/total_params*100:.1f}%")

if DO_WANDB:
    wandb.log({"train/total_params": total_params,
            "train/trainable_params": trainable_params,
            "train/frozen_params": frozen_params,
            "train/trainable_ratio": trainable_params/total_params*100})

train_set = AOKVQADataset(split='train')
val_set   = AOKVQADataset(split='val')


def collate_fn(batch):
    imgs   = torch.stack([b['image'] for b in batch])
    qs     = [b['question']  for b in batch]      # list of str  (len B)
    choices= [b['choices']   for b in batch]      # list of list (B × 4)
    labels = torch.tensor([b['correct_choice_idx'] for b in batch])
    return {'image': imgs,
            'question': qs,
            'choices': choices,
            'correct_choice_idx': labels}

train_loader = DataLoader(train_set,
                          batch_size=128,
                          shuffle=True,
                          num_workers=12,
                          collate_fn=collate_fn)

val_loader   = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=12, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss()
opt       = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

step = 0
for epoch in range(20):
    model.train()
    for batch in tqdm.tqdm(train_loader):
        imgs = batch['image'].to(device)
        qs   = batch['question']
        opts = batch['choices']                   # list[list[str]]
        gold = batch['correct_choice_idx'].to(device)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(imgs, qs, opts)
        logits = logits.float()
        loss   = criterion(logits, gold)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if DO_WANDB:
            wandb.log({"train/loss": loss.item()})
        
        # Debug: save images and patch-to-word mappings
        if step % DEBUG_EVERY == 0 and getattr(model.nb, "_last_idx", None) is not None:
            # 1) save first image in batch (denorm)
            img_path = os.path.join(OUTDIR, f"e{epoch}_it{step}.png")
            save_image(denorm_clip(batch['image'][0]), img_path)

            # 2) map a few patches → words
            idx0 = model.nb._last_idx    # (P,k)
            sc0  = model.nb._last_scores     # (P,k)
            P, k = idx0.shape
            show_patches = torch.linspace(0, P-1, steps=min(12, P)).long().tolist()

            rows = []
            for p in show_patches:
                ids = idx0[p].tolist()
                scores = sc0[p].tolist()
                words = [model.nb.tokens_text[i] for i in ids] if model.nb.tokens_text else [str(i) for i in ids]
                rows.append({
                    "patch": int(p),
                    "topk": [{"id": int(i), "word": w, "score": float(s)}
                            for i, w, s in zip(ids, words, scores)]
                })

            with open(os.path.join(OUTDIR, f"e{epoch}_it{step}.json"), "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

        step += 1

    # quick val accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader):
            imgs = batch['image'].to(device)
            qs   = batch['question']
            opts = batch['choices']
            gold = batch['correct_choice_idx'].to(device)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                preds = model(imgs, qs, opts).argmax(dim=1)
            correct += (preds == gold).sum().item()
            total   += gold.size(0)
    print(f'E{epoch}: val-acc {(correct/total)*100:.2f}%')
    if DO_WANDB:
        wandb.log({"val/acc": (correct/total)*100})
if DO_WANDB:
    wandb.finish()