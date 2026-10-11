"""Kaggle GPU job: fine-tune cross-encoder/ms-marco-MiniLM-L-12-v2 (Apache-2.0) and score all pairs.

Input: "name | address" [SEP] "name | address" (cleaned text). Output: /kaggle/working/xenc_scores.parquet.
"""
import glob
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# model: cross-encoder/ms-marco-MiniLM-L-12-v2 (Apache-2.0); its weights are loaded from the dataset folder
MAX_LEN, BATCH, SCORE_BATCH, LR, EPOCHS = 64, 128, 1024, 3e-5, 1
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/texts.parquet", recursive=True)[0])
texts = pl.read_parquet(f"{src}/texts.parquet")
tr = pl.read_parquet(f"{src}/train_pairs.parquet").sample(fraction=1.0, shuffle=True, seed=0)
sc = pl.read_parquet(f"{src}/score_pairs.parquet")
look = {sp: dict(zip(g["uid"].to_list(), g["t"].to_list())) for sp, g in texts.group_by("split") for sp in [sp[0]]}
log(f"loaded: {tr.height:,} training pairs, {sc.height:,} pairs to score; GPU {torch.cuda.get_device_name(0)}")

dev = torch.device("cuda")
# pretrained weights come from the dataset (the notebook runs offline)
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ((tr.height + BATCH - 1) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()
t1 = [look["train"][u] for u in tr["s1_uid"].to_list()]
t2 = [look["train"][u] for u in tr["r_uid"].to_list()]
y = tr["y"].to_numpy().astype(np.float32)
model.train()
for ep in range(EPOCHS):
    for i in range(0, len(y), BATCH):
        enc = tok(t1[i:i + BATCH], t2[i:i + BATCH], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(model(**enc).logits.squeeze(-1).float(), torch.from_numpy(y[i:i + BATCH]).to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (i // BATCH) % 2000 == 0:
            log(f"epoch {ep} step {i // BATCH}/{steps} loss {loss.item():.4f}")
model.save_pretrained("/kaggle/working/xenc_model")
tok.save_pretrained("/kaggle/working/xenc_model")
log("training done")

model.eval()
out = []
with torch.no_grad():
    for sp in ("train", "test"):
        part = sc.filter(pl.col("split") == sp)
        a = [look[sp][u] for u in part["s1_uid"].to_list()]
        b = [look[sp][u] for u in part["r_uid"].to_list()]
        scores = np.empty(len(a), dtype=np.float32)
        for i in range(0, len(a), SCORE_BATCH):
            enc = tok(a[i:i + SCORE_BATCH], b[i:i + SCORE_BATCH], truncation=True, max_length=MAX_LEN,
                      padding="max_length", return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                scores[i:i + SCORE_BATCH] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if (i // SCORE_BATCH) % 1000 == 0:
                log(f"scoring {sp}: {i:,}/{len(a):,}")
        out.append(part.with_columns(pl.Series("xenc", scores)))
pl.concat(out).write_parquet("/kaggle/working/xenc_scores.parquet", compression="zstd")
log("all pairs scored -> /kaggle/working/xenc_scores.parquet")
