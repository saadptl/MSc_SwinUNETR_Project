"""
PART 120 — CONTROLLED BACKGROUND-CALIBRATED LOSS RETRAINING
Read-only baseline protection + controlled training experiment.

Only experimental change vs Part98:
background raw CE weight:
    0.015069 -> 0.100000

Everything else is locked to Part98:
architecture, cohorts, preprocessing, crop, augmentation, optimizer,
LR, weight decay, epochs, batch size, accumulation, feature size.

Part104 is loaded only as initialization and is NEVER overwritten.
Targets are development/pseudo-mask labels; this is not clinical validation.
"""

from __future__ import annotations
import hashlib, importlib.util, json, math, random, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
PART98=SRC/"segmentation_rsna_part98_strong_full_cohort_training.py"
TRAIN=ROOT/"outputs/segmentation/rsna_part98_strong_full_cohort_training/part98_train_cohort.csv"
VAL=ROOT/"outputs/segmentation/rsna_part98_strong_full_cohort_training/part98_validation_cohort.csv"
BASE=ROOT/"outputs/segmentation/rsna_part104_final_checkpoint_selection/checkpoints/final_segmentation_model.pth"
OUT=ROOT/"outputs/segmentation/rsna_part120_controlled_background_calibrated_loss_retraining"
CK=OUT/"checkpoints"; REP=ROOT/"reports"
BEST=CK/"best_part120_model.pth"; FINAL=CK/"final_part120_model.pth"
HIST=OUT/"part120_training_history.csv"; MET=OUT/"part120_test_metrics.json"
CMP=OUT/"part120_baseline_vs_experiment.json"
RPT=REP/"part120_controlled_background_calibrated_loss_report.txt"
RPJ=REP/"part120_controlled_background_calibrated_loss_summary.json"

SEED=42; EPOCHS=12; BATCH=1; ACC=4; LR=5e-5; WD=1e-5
NCLS=6; FULL=(64,96,96); CROP=(32,64,64); FEAT=12
WORKERS=0; AMP=True; AUG=0.75; PATIENCE=5
W98=torch.tensor([0.015069,0.722595,0.935995,1.017003,1.178304,1.146102],dtype=torch.float32)
W120=torch.tensor([0.100000,0.722595,0.935995,1.017003,1.178304,1.146102],dtype=torch.float32)
BASE_SHA="fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def jd(x):
    if isinstance(x,Path): return str(x)
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,torch.Tensor): return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=True

def import98():
    spec=importlib.util.spec_from_file_location("part98_p120",PART98)
    if spec is None or spec.loader is None: raise RuntimeError("Part98 import spec failed")
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m

class Loss120(nn.Module):
    def __init__(self):
        super().__init__(); self.raw=W120.clone(); self.w=self.raw/self.raw.mean()
    def forward(self,logits,target):
        target=target.squeeze(1).long()
        ce=F.cross_entropy(logits,target,weight=self.w.to(logits.device))
        p=torch.softmax(logits,1)
        oh=F.one_hot(target,num_classes=NCLS).permute(0,4,1,2,3).float()
        inter=(p*oh).sum((0,2,3,4))
        den=p.sum((0,2,3,4))+oh.sum((0,2,3,4))+1e-6
        dice=(2*inter+1e-6)/den
        return 1-dice[1:].mean()+ce

def model():
    from monai.networks.nets import SwinUNETR
    try:
        m=SwinUNETR(spatial_dims=3,in_channels=1,out_channels=NCLS,feature_size=FEAT,use_checkpoint=False)
    except TypeError:
        m=SwinUNETR(img_size=FULL,spatial_dims=3,in_channels=1,out_channels=NCLS,feature_size=FEAT,use_checkpoint=False)
    return m

@torch.no_grad()
def metrics(logits,masks):
    pred=logits.argmax(1); tar=masks.squeeze(1).long(); probs=logits.softmax(1)
    dices=[]
    for c in range(1,NCLS):
        a=pred==c; b=tar==c; den=int(a.sum())+int(b.sum())
        dices.append(1.0 if den==0 else 2*int((a&b).sum())/den)
    pc=torch.bincount(pred.flatten(),minlength=NCLS).cpu().numpy()
    tc=torch.bincount(tar.flatten(),minlength=NCLS).cpu().numpy()
    pf=int(pc[1:].sum()); tf=int(tc[1:].sum())
    return {
        "dice":float(np.mean(dices)),"class_dice":dices,
        "predicted_class_voxels":pc.tolist(),"target_class_voxels":tc.tolist(),
        "foreground_volume_ratio":float(pf/max(tf,1)),
        "mean_foreground_probability":float((1-probs[:,0]).mean()),
        "mean_background_probability":float(probs[:,0].mean()),
        "mean_foreground_background_margin":float((1-probs[:,0]*2).mean()),
        "mean_entropy":float((-(probs*torch.log(probs.clamp_min(1e-8))).sum(1)).mean())
    }

def epoch(m,loader,crit,opt,scaler,dev,train):
    m.train(train); tl=td=0.; n=0; cd=np.zeros(5); ratios=[]; fps=[]; bps=[]; ms=[]; es=[]
    if train: opt.zero_grad(set_to_none=True)
    ae=AMP and dev.type=="cuda"
    for i,(x,y) in enumerate(loader):
        x=x.to(dev); y=y.to(dev)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=dev.type,dtype=torch.float16 if dev.type=="cuda" else torch.float32,enabled=ae):
                z=m(x); loss=crit(z,y)
            if train:
                q=loss/ACC
                if scaler and ae: scaler.scale(q).backward()
                else: q.backward()
                if (i+1)%ACC==0 or i+1==len(loader):
                    if scaler and ae: scaler.step(opt); scaler.update()
                    else: opt.step()
                    opt.zero_grad(set_to_none=True)
        mm=metrics(z.detach(),y); tl+=float(loss); td+=mm["dice"]; cd+=mm["class_dice"]; n+=1
        ratios.append(mm["foreground_volume_ratio"]); fps.append(mm["mean_foreground_probability"])
        bps.append(mm["mean_background_probability"]); ms.append(mm["mean_foreground_background_margin"]); es.append(mm["mean_entropy"])
    return {"loss":tl/max(n,1),"dice":td/max(n,1),"class_dice":(cd/max(n,1)).tolist(),
            "foreground_volume_ratio_mean_case":float(np.mean(ratios)),
            "foreground_volume_ratio_median_case":float(np.median(ratios)),
            "mean_foreground_probability":float(np.mean(fps)),
            "mean_background_probability":float(np.mean(bps)),
            "mean_foreground_background_margin":float(np.mean(ms)),
            "mean_entropy":float(np.mean(es))}

def save(p,m,opt,sch,sc,ep,best,h):
    torch.save({"model_state_dict":m.state_dict(),"optimizer_state_dict":opt.state_dict(),
                "scheduler_state_dict":sch.state_dict(),"scaler_state_dict":sc.state_dict() if sc else None,
                "epoch":ep,"best_validation_dice":best,"history":h,
                "part120_raw_weights":W120.tolist(),"part98_raw_weights":W98.tolist()},p)

def main():
    seed(); OUT.mkdir(parents=True,exist_ok=True); CK.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
    print("="*96); print("PART 120 — CONTROLLED BACKGROUND-CALIBRATED LOSS RETRAINING"); print("="*96)
    print("CONTROLLED EXPERIMENT — PART104 BASELINE LOCKED")
    print("Python:",sys.executable); print("PyTorch:",torch.__version__); print("CUDA:",torch.cuda.is_available())
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    if not all(p.exists() for p in [PART98,TRAIN,VAL,BASE]): raise FileNotFoundError("Required artifact missing")
    bs=sha(BASE); print("Part104 SHA256:",bs); print("Canonical SHA:",bs==BASE_SHA)
    if bs!=BASE_SHA: raise RuntimeError("Part104 SHA mismatch; aborting")
    print("\nONLY CHANGE: background raw CE weight 0.015069 -> 0.100000")
    print("All other Part98 settings LOCKED.")
    p98=import98()
    tr,va=p98.StrongSegmentationDataset(TRAIN,True),p98.StrongSegmentationDataset(VAL,False)
    trl=DataLoader(tr,batch_size=BATCH,shuffle=True,num_workers=WORKERS,pin_memory=dev.type=="cuda")
    val=DataLoader(va,batch_size=BATCH,shuffle=False,num_workers=WORKERS,pin_memory=dev.type=="cuda")
    m=model().to(dev); pc=sum(x.numel() for x in m.parameters()); print("Model parameters:",pc)
    if pc!=4078116: raise RuntimeError(f"Canonical parameter mismatch: {pc}")
    pay=torch.load(BASE,map_location="cpu",weights_only=False)
    state=pay.get("model_state_dict",pay.get("state_dict",pay))
    m.load_state_dict(state,strict=True); print("Part104 initialization: PASS")
    crit=Loss120().to(dev); print("Normalized Part120 weights:",crit.w.tolist())
    opt=AdamW(m.parameters(),lr=LR,weight_decay=WD); sch=CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=LR*.1)
    sc=torch.amp.GradScaler("cuda") if AMP and dev.type=="cuda" else None
    hist=[]; best=-math.inf; be=-1; patience=0; t0=time.time()
    for ep in range(1,EPOCHS+1):
        t=time.time()
        a=epoch(m,trl,crit,opt,sc,dev,True); b=epoch(m,val,crit,None,None,dev,False); sch.step()
        row={"epoch":ep,"train_loss":a["loss"],"train_fg_dice":a["dice"],"val_loss":b["loss"],"val_fg_dice":b["dice"],
             "learning_rate":opt.param_groups[0]["lr"],"epoch_seconds":time.time()-t,
             **{f"val_class_{i+1}_dice":v for i,v in enumerate(b["class_dice"])},
             "val_foreground_ratio_mean_case":b["foreground_volume_ratio_mean_case"],
             "val_foreground_ratio_median_case":b["foreground_volume_ratio_median_case"],
             "val_mean_foreground_probability":b["mean_foreground_probability"],
             "val_mean_background_probability":b["mean_background_probability"],
             "val_mean_foreground_background_margin":b["mean_foreground_background_margin"],
             "val_mean_entropy":b["mean_entropy"]}
        hist.append(row)
        print(f"Epoch {ep:02d}/{EPOCHS} | train loss {a['loss']:.6f} | train Dice {a['dice']:.6f} | val loss {b['loss']:.6f} | val Dice {b['dice']:.6f} | FG ratio {b['foreground_volume_ratio_mean_case']:.3f}x | P(FG) {b['mean_foreground_probability']:.6f} | {row['epoch_seconds']:.1f}s")
        if b["dice"]>best:
            best=b["dice"]; be=ep; patience=0; save(BEST,m,opt,sch,sc,ep,best,hist)
        else: patience+=1
        if patience>=PATIENCE: print("Early stopping:",ep); break
    save(FINAL,m,opt,sch,sc,len(hist),best,hist); pd.DataFrame(hist).to_csv(HIST,index=False)
    bp=torch.load(BEST,map_location="cpu",weights_only=False); m.load_state_dict(bp["model_state_dict"],strict=True)
    bv=epoch(m,val,crit,None,None,dev,False); bestsha=sha(BEST); finalsha=sha(FINAL)
    cmp={"baseline_part104":{"sha256":bs,"validation_dice":0.2172496851095929,"part114_fg_ratio":67.462,
        "part115_mean_fg_probability":0.513583,"part115_mean_bg_probability":0.486417},
        "part120":{"best_epoch":be,"validation_dice":best,"metrics":bv,"best_sha256":bestsha,"final_sha256":finalsha,
        "raw_weights":W120.tolist(),"normalized_weights":crit.w.tolist()},
        "controlled_change":{"background_raw_weight_from":float(W98[0]),"background_raw_weight_to":float(W120[0]),
        "architecture_changed":False,"cohort_changed":False,"crop_changed":False,"optimizer_changed":False,"lr_changed":False}}
    CMP.write_text(json.dumps(cmp,indent=2,default=jd),encoding="utf-8"); MET.write_text(json.dumps(cmp,indent=2,default=jd),encoding="utf-8")
    ratio=bv["foreground_volume_ratio_mean_case"]; diag="FOREGROUND_OVERSEGMENTATION_REDUCED_VS_PART104" if ratio<67.462 else "NO_FOREGROUND_OVERSEGMENTATION_REDUCTION"
    report=f"""PART 120 — CONTROLLED BACKGROUND-CALIBRATED LOSS RETRAINING

Part104 SHA256: {bs}
Canonical SHA match: {bs==BASE_SHA}
Part104 validation Dice baseline: 0.2172496851095929
Part114 foreground ratio baseline: 67.462x

ONLY EXPERIMENTAL CHANGE:
background raw CE weight: 0.015069 -> 0.100000

Best epoch: {be}
Best validation Dice: {best:.9f}
Best validation foreground ratio: {ratio:.6f}x
Mean foreground probability: {bv['mean_foreground_probability']:.6f}
Mean background probability: {bv['mean_background_probability']:.6f}
Mean FG-BG margin: {bv['mean_foreground_background_margin']:.6f}
Mean entropy: {bv['mean_entropy']:.6f}

Classwise Dice:
""" + "\n".join(f"class {i}: {v:.9f}" for i,v in enumerate(bv["class_dice"],1)) + f"""

Diagnosis: {diag}
Part104 overwritten: NO
Clinical claim: NO — targets are development/pseudo-mask labels.

Outputs:
{BEST}
{FINAL}
{HIST}
{MET}
{CMP}
{RPT}
{RPJ}
"""
    RPT.write_text(report,encoding="utf-8")
    RPJ.write_text(json.dumps({"part":120,"status":"PASS","diagnosis":diag,"best_epoch":be,"best_dice":best,
        "best_fg_ratio":ratio,"best_sha256":bestsha,"final_sha256":finalsha,"part104_overwritten":False},indent=2),encoding="utf-8")
    print("\n"+"="*96); print("PART 120 FINAL RESULT"); print("="*96)
    print("PASS — CONTROLLED BACKGROUND-CALIBRATED LOSS RETRAINING COMPLETED")
    print("Best epoch:",be); print(f"Best validation Dice: {best:.9f}"); print(f"Validation FG ratio: {ratio:.6f}x")
    print("Diagnosis:",diag); print("Part104 overwritten: NO")
    print("Outputs:",BEST,FINAL,HIST,MET,CMP,RPT,RPJ,sep="\n")
    return 0

if __name__=="__main__": raise SystemExit(main())
