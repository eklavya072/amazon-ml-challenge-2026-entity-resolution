"""Kaggle GPU job: fine-tune cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (Apache-2.0, multilingual) on raw text.

Trained on folds 1-2, scores folds 0, 3, 4 and the test. Output: /kaggle/working/xenc2_scores.parquet.
"""
import glob
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_LEN, BATCH, SCORE_BATCH, LR, N_TRAIN = 96, 128, 1024, 3e-5, 2_400_000
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
pairs_dir = os.path.dirname(glob.glob("/kaggle/input/**/train_pairs.parquet", recursive=True)[0])
text_files = glob.glob("/kaggle/input/**/texts_raw_*.parquet", recursive=True)
texts = pl.concat([pl.read_parquet(f) for f in text_files])
look = {sp: dict(zip(g["uid"].to_list(), g["t"].to_list())) for (sp,), g in texts.group_by("split")}
del texts
tr = pl.read_parquet(f"{pairs_dir}/train_pairs.parquet").sample(fraction=1.0, shuffle=True, seed=1).head(N_TRAIN)
sc = pl.read_parquet(f"{pairs_dir}/score_pairs.parquet")
log(f"loaded: {tr.height:,} training pairs, {sc.height:,} to score; GPU {torch.cuda.get_device_name(0)}")

dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=1,
                                                           ignore_mismatched_sizes=True).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = (tr.height + BATCH - 1) // BATCH
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()
t1 = [look["train"][u] for u in tr["s1_uid"].to_list()]
t2 = [look["train"][u] for u in tr["r_uid"].to_list()]
y = tr["y"].to_numpy().astype(np.float32)
model.train()
for i in range(0, len(y), BATCH):
    enc = tok(t1[i:i + BATCH], t2[i:i + BATCH], truncation=True, max_length=MAX_LEN, padding="longest",
              return_tensors="pt").to(dev)
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
        log(f"step {i // BATCH}/{steps} loss {loss.item():.4f}")
del t1, t2
model.save_pretrained("/kaggle/working/xenc2_model")
tok.save_pretrained("/kaggle/working/xenc2_model")
log("training done")

model.eval()
out = []
with torch.no_grad():
    for sp in ("test", "train"):
        part = sc.filter(pl.col("split") == sp)
        a = [look[sp][u] for u in part["s1_uid"].to_list()]
        b = [look[sp][u] for u in part["r_uid"].to_list()]
        order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])  # similar lengths per batch -> little padding
        scores = np.empty(len(a), dtype=np.float32)
        for i in range(0, len(a), SCORE_BATCH):
            idx = order[i:i + SCORE_BATCH]
            enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=MAX_LEN,
                      padding="longest", return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                scores[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if (i // SCORE_BATCH) % 1000 == 0:
                log(f"scoring {sp}: {i:,}/{len(a):,}")
        out.append(part.with_columns(pl.Series("xenc2", scores)))
        pl.concat(out).write_parquet("/kaggle/working/xenc2_scores.parquet", compression="zstd")
        log(f"{sp} scored and saved")
log("all pairs scored -> /kaggle/working/xenc2_scores.parquet")
