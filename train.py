import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from aokvqa_data import AOKVQADataset
from model import QCDenseCLIP
import tqdm
import os
import wandb
wandb.init(project="DenseInject", name="Baseline DenseCLIP")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = QCDenseCLIP('/speedy/DenseInject/weights/denseclip.pth', device=device).to(device)

# Count frozen and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"  Parameter counts:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters: {frozen_params:,}")
print(f"  Trainable ratio: {trainable_params/total_params*100:.1f}%")

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
                          batch_size=256,
                          shuffle=True,
                          num_workers=12,
                          collate_fn=collate_fn)

val_loader   = DataLoader(val_set,   batch_size=512, shuffle=False, num_workers=12, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss()
opt       = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(10):
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

        wandb.log({"train/loss": loss.item()})

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
    wandb.log({"val/acc": (correct/total)*100})
wandb.finish()