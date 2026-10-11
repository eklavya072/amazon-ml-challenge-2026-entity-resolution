"""Kaggle GPU job: fine-tune intfloat/multilingual-e5-base (MIT) on raw text and score the uncertain pairs.

Trained on folds 1-2 (kaggle_pack3), scores xscore.parquet. Output: /kaggle/working/xenc3_scores.parquet.
"""
import glob
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "intfloat/multilingual-e5-base"
MAX_LEN, BATCH, SCORE_BATCH, LR, WARMUP = 128, 64, 512, 2e-5, 300
TRAIN_SECONDS = 125 * 60          # training budget; the schedule is fitted to it after a short speed test
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)

pack = os.path.dirname(glob.glob("/kaggle/input/**/xtrain.parquet", recursive=True)[0])
texts = pl.concat([pl.read_parquet(f) for f in glob.glob("/kaggle/input/**/texts_raw_*.parquet", recursive=True)])
look = {sp: dict(zip(g["uid"].to_list(), g["t"].to_list())) for (sp,), g in texts.group_by("split")}
del texts
tr = pl.read_parquet(f"{pack}/xtrain.parquet")
sc = pl.read_parquet(f"{pack}/xscore.parquet")
ngpu = torch.cuda.device_count()
log(f"loaded: {tr.height:,} training pairs, {sc.height:,} to score; {ngpu} x {torch.cuda.get_device_name(0)}")


class Scorer(torch.nn.Module):
    """Wraps the model so mixed precision is switched on inside each GPU replica (DataParallel runs replicas in
    threads, and autocast is thread-local)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, **kw):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.m(**kw).logits.squeeze(-1).float()


dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=1).to(dev)
net = Scorer(base)
if ngpu > 1:
    net = torch.nn.DataParallel(net)
opt = torch.optim.AdamW(base.parameters(), lr=LR, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()
t1 = [look["train"][u] for u in tr["s1_uid"].to_list()]
t2 = [look["train"][u] for u in tr["r_uid"].to_list()]
y = tr["y"].to_numpy().astype(np.float32)
n_steps, total, t_train = (len(y) + BATCH - 1) // BATCH, None, time.time()
base.train()
for step in range(n_steps):
    if step == WARMUP:  # fit the number of steps to the time budget, then decay the learning rate linearly
        per_step = (time.time() - t_train) / WARMUP
        total = min(n_steps, int(TRAIN_SECONDS / per_step))
        log(f"{per_step * 1000:.0f} ms/step -> training {total:,} of {n_steps:,} steps ({total * BATCH:,} pairs)")
    if total is not None and step >= total:
        break
    lr = LR * min(1.0, (step + 1) / WARMUP) * (1.0 if total is None else max(0.0, (total - step) / (total - WARMUP)))
    for g in opt.param_groups:
        g["lr"] = lr
    i = step * BATCH
    enc = tok(t1[i:i + BATCH], t2[i:i + BATCH], truncation=True, max_length=MAX_LEN, padding="longest",
              return_tensors="pt").to(dev)
    loss = lossf(net(**enc), torch.from_numpy(y[i:i + BATCH]).to(dev))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    if step % 2000 == 0:
        log(f"step {step:,} loss {loss.item():.4f} lr {lr:.2e}")
del t1, t2
base.save_pretrained("/kaggle/working/xenc3_model")
tok.save_pretrained("/kaggle/working/xenc3_model")
log("training done")

base.eval()
out = []
with torch.no_grad():
    for sp in ("test", "train"):
        part = sc.filter(pl.col("split") == sp)
        a = [look[sp][u] for u in part["s1_uid"].to_list()]
        b = [look[sp][u] for u in part["r_uid"].to_list()]
        order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])  # similar lengths per batch -> little padding
        scores = np.empty(len(a), dtype=np.float32)
        for k, i in enumerate(range(0, len(a), SCORE_BATCH)):
            idx = order[i:i + SCORE_BATCH]
            enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=MAX_LEN,
                      padding="longest", return_tensors="pt").to(dev)
            scores[idx] = torch.sigmoid(net(**enc)).cpu().numpy()
            if k % 500 == 0:
                log(f"scoring {sp}: {i:,}/{len(a):,}")
        out.append(part.with_columns(pl.Series("xenc3", scores)))
        pl.concat(out).write_parquet("/kaggle/working/xenc3_scores.parquet", compression="zstd")
        log(f"{sp} scored and saved")
log("all pairs scored -> /kaggle/working/xenc3_scores.parquet")
