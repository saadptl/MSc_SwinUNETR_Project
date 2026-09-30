"""
Part 43 — Improved Controlled Spatial-Sampling Training Pilot.
Compares random vs foreground-centered training crops using the same
Part-15 initialization, cohorts, model, loss, optimizer, and validation crops.
"""
from __future__ import annotations
import csv, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, RandomSampler

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
import segmentation_rsna_part11_controlled_pilot_training as part11
import segmentation_rsna_part9_3d_dataset_loader as part9

P15=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"
CKPT=P15/"checkpoints"/"part15_initialization_from_part11.pth"
TRCSV=P15/"part15_train_cohort.csv"
VACSV=P15/"part15_validation_cohort.csv"
OUT=ROOT/"outputs"/"segmentation"/"rsna_part43_improved_spatial_sampling_training"
BASE=OUT/"random_spatial_crop"
CENTER=OUT/"foreground_centered_spatial_crop"

FULL=(64,96,96); CROP=(32,64,64)
CLASSES=6; FEATURE=12
TRN=200; VALN=100; EPOCHS=5; BS=1; LR=1e-4; WD=1e-5; SEED=42
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def ensure(x,msg):
    if not x: raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def load_case(row):
    if isinstance(row,dict): row=pd.Series(row)
    got=part11.load_tensor_case(row,part9)
    ensure(isinstance(got,(tuple,list)) and len(got)>=2,
           f"Unexpected load_tensor_case return: {type(got)}")
    im,ma=got[0],got[1]
    im=im.detach().cpu(); ma=ma.detach().cpu()
    if im.ndim==4 and im.shape[0]==1: im=im[0]
    if ma.ndim==4 and ma.shape[0]==1: ma=ma[0]
    ensure(tuple(im.shape)==FULL and tuple(ma.shape)==FULL,
           f"Bad shapes: {tuple(im.shape)} {tuple(ma.shape)}")
    return im.float(),ma.long()

def centroid(ma):
    q=torch.nonzero(ma>0,as_tuple=False)
    if q.numel()==0: return tuple((s-1)//2 for s in ma.shape)
    return tuple(int(round(float(v))) for v in q.float().mean(0))

def bounds(center,full,crop):
    out=[]
    for c,f,n in zip(center,full,crop):
        st=max(0,min(int(c)-n//2,f-n)); out.append((st,st+n))
    return out

def crop(im,ma,mode,rng):
    if mode=="random":
        cen=tuple(rng.randint(n//2,f-(n-n//2)) for f,n in zip(FULL,CROP))
    else: cen=centroid(ma)
    b=bounds(cen,FULL,CROP)
    z0,z1=b[0]; y0,y1=b[1]; x0,x1=b[2]
    ci=im[z0:z1,y0:y1,x0:x1]; cm=ma[z0:z1,y0:y1,x0:x1]
    ensure(tuple(ci.shape)==CROP and tuple(cm.shape)==CROP,"Crop shape failure")
    return ci,cm,cen,b

class DS(Dataset):
    def __init__(self,rows,mode,epoch): self.rows,self.mode,self.epoch=rows,mode,epoch
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        im,ma=load_case(self.rows[i])
        rng=random.Random(SEED+self.epoch*1000003+i*1009)
        ci,cm,cen,b=crop(im,ma,self.mode,rng)
        return {"image":ci,"mask":cm,"source_fg":int((ma>0).sum()),
                "crop_fg":int((cm>0).sum()),"bounds":b}
def collate(b): return b[0]

def make_model(state):
    m=part11.create_model(DEVICE); m.load_state_dict(state,strict=True); m.to(DEVICE); return m

@torch.no_grad()
def evaluate(m,rows):
    m.eval(); ls=[]; ds=[]; pfs=[]; tfs=[]; empty=0
    dl=DataLoader(DS(rows,"foreground_centered",9000),batch_size=1,shuffle=False,
                  collate_fn=collate,num_workers=0)
    for i,it in enumerate(dl):
        x=it["image"].to(DEVICE).unsqueeze(0).unsqueeze(0); y=it["mask"].to(DEVICE).unsqueeze(0)
        with autocast(enabled=DEVICE.type=="cuda"):
            lo=m(x); loss=part11.loss_function(lo,y.unsqueeze(1))
        d,_=part11.dice_from_prediction(lo,y); pf=int((lo.argmax(1)>0).sum()); tf=int((y>0).sum())
        ls.append(float(loss.detach())); ds.append(float(d)); pfs.append(pf); tfs.append(tf); empty+=pf==0
        if i in (0,49,99): print(f"  VAL {i+1:03d}/{len(rows)} Loss={loss.item():.5f} Dice={d:.6f} PredFG={pf} TargetFG={tf}")
    return dict(loss=float(np.mean(ls)),dice=float(np.mean(ds)),pred_fg=float(np.mean(pfs)),
                target_fg=float(np.mean(tfs)),empty=int(empty))

def train_epoch(m,rows,mode,epoch,opt,scaler):
    m.train(); ls=[]; ds=[]; pfs=[]; cfs=[]
    dl=DataLoader(DS(rows,mode,epoch),batch_size=1,
                  sampler=RandomSampler(DS(rows,mode,epoch),replacement=False,
                                        generator=torch.Generator().manual_seed(SEED+epoch)),
                  collate_fn=collate,num_workers=0)
    # Recreate sampler dataset issue-free by using the sampler indices with a fresh dataset.
    # RandomSampler length is fixed at 200 and indices map directly.
    for step,it in enumerate(dl):
        x=it["image"].to(DEVICE).unsqueeze(0).unsqueeze(0); y=it["mask"].to(DEVICE).unsqueeze(0)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=DEVICE.type=="cuda"):
            lo=m(x); loss=part11.loss_function(lo,y.unsqueeze(1))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        with torch.no_grad(): d,_=part11.dice_from_prediction(lo,y); pf=int((lo.argmax(1)>0).sum())
        ls.append(float(loss.detach())); ds.append(float(d)); pfs.append(pf); cfs.append(int(it["crop_fg"]))
        if step in (0,49,99,149,199):
            print(f"  TRAIN {step+1:03d}/{len(dl)} Loss={loss.item():.5f} Dice={d:.6f} PredFG={pf} CropFG={it['crop_fg']}")
    return dict(loss=float(np.mean(ls)),dice=float(np.mean(ds)),pred_fg=float(np.mean(pfs)),
                crop_fg=float(np.mean(cfs)),unique=len(rows))

def audit(rows):
    rec=[]
    for i,r in enumerate(rows):
        im,ma=load_case(r); rng=random.Random(SEED+i*1009)
        _,rm,_,rb=crop(im,ma,"random",rng); _,cm,_,cb=crop(im,ma,"foreground_centered",rng)
        rec.append(dict(index=i,source_fg=int((ma>0).sum()),random_crop_fg=int((rm>0).sum()),
                        centered_crop_fg=int((cm>0).sum()),random_bounds=str(rb),centered_bounds=str(cb)))
        if i in (0,49,99,149,199):
            print(f"  COVERAGE {i+1:03d}/{len(rows)} sourceFG={rec[-1]['source_fg']} randomFG={rec[-1]['random_crop_fg']} centeredFG={rec[-1]['centered_crop_fg']}")
    OUT.mkdir(parents=True,exist_ok=True)
    with open(OUT/"part43_crop_coverage_audit.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rec[0].keys()); w.writeheader(); w.writerows(rec)
    r=[x["random_crop_fg"] for x in rec]; c=[x["centered_crop_fg"] for x in rec]
    s={"cases":len(rec),"crop_shape":CROP,"mean_source_fg":float(np.mean([x["source_fg"] for x in rec])),
       "mean_random_crop_fg":float(np.mean(r)),"mean_centered_crop_fg":float(np.mean(c)),
       "random_zero_fg_cases":sum(x==0 for x in r),"centered_zero_fg_cases":sum(x==0 for x in c)}
    (OUT/"part43_crop_coverage_summary.json").write_text(json.dumps(s,indent=2),encoding="utf-8")
    print(f"Mean source foreground voxels       : {s['mean_source_fg']:.2f}")
    print(f"Mean random-crop foreground voxels : {s['mean_random_crop_fg']:.2f}")
    print(f"Mean centered-crop foreground      : {s['mean_centered_crop_fg']:.2f}")
    print(f"Random crops with zero FG           : {s['random_zero_fg_cases']}/{len(rec)}")
    print(f"Centered crops with zero FG         : {s['centered_zero_fg_cases']}/{len(rec)}")
    return s

def run(label,mode,tr,va,state):
    print("\n"+"="*82+"\n"+label+"\n"+"="*82); seed_all(SEED); m=make_model(state)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD); scaler=GradScaler(enabled=DEVICE.type=="cuda")
    v0=evaluate(m,va); print(f"Initialization: loss={v0['loss']:.6f}, Dice={v0['dice']:.6f}, PredFG={v0['pred_fg']:.1f}, empty={v0['empty']}/{len(va)}")
    hist=[dict(epoch=0,train_loss=None,train_dice=None,train_pred_fg=None,train_crop_fg=None,
               val_loss=v0["loss"],val_dice=v0["dice"],val_pred_fg=v0["pred_fg"],
               val_empty=v0["empty"],unique_train_cases=len(tr))]
    out=BASE if mode=="random" else CENTER; out.mkdir(parents=True,exist_ok=True); best=-1
    for ep in range(1,EPOCHS+1):
        print(f"\n{label} — EPOCH {ep}/{EPOCHS}"); t=time.time()
        # Dataset instance used consistently by sampler and loader.
        ds=DS(tr,mode,ep)
        sam=RandomSampler(ds,replacement=False,generator=torch.Generator().manual_seed(SEED+ep))
        dl=DataLoader(ds,batch_size=1,sampler=sam,collate_fn=collate,num_workers=0)
        # Inline training loop to avoid duplicate dataset construction.
        m.train(); ls=[]; dss=[]; pfs=[]; cfs=[]
        for step,it in enumerate(dl):
            x=it["image"].to(DEVICE).unsqueeze(0).unsqueeze(0); y=it["mask"].to(DEVICE).unsqueeze(0)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=DEVICE.type=="cuda"):
                lo=m(x); loss=part11.loss_function(lo,y.unsqueeze(1))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            with torch.no_grad(): d,_=part11.dice_from_prediction(lo,y); pf=int((lo.argmax(1)>0).sum())
            ls.append(float(loss.detach())); dss.append(float(d)); pfs.append(pf); cfs.append(int(it["crop_fg"]))
            if step in (0,49,99,149,199): print(f"  TRAIN {step+1:03d}/{len(dl)} Loss={loss.item():.5f} Dice={d:.6f} PredFG={pf} CropFG={it['crop_fg']}")
        trn=dict(loss=float(np.mean(ls)),dice=float(np.mean(dss)),pred_fg=float(np.mean(pfs)),crop_fg=float(np.mean(cfs)))
        v=evaluate(m,va); mins=(time.time()-t)/60
        row=dict(epoch=ep,train_loss=trn["loss"],train_dice=trn["dice"],train_pred_fg=trn["pred_fg"],
                 train_crop_fg=trn["crop_fg"],val_loss=v["loss"],val_dice=v["dice"],
                 val_pred_fg=v["pred_fg"],val_target_fg=v["target_fg"],val_empty=v["empty"],
                 unique_train_cases=len(tr),time_min=mins); hist.append(row)
        print(f"Epoch {ep}: train_loss={trn['loss']:.6f}, train_Dice={trn['dice']:.6f}, train_CropFG={trn['crop_fg']:.2f}, val_loss={v['loss']:.6f}, val_Dice={v['dice']:.6f}, val_PredFG={v['pred_fg']:.1f}, empty={v['empty']}/{len(va)}, time={mins:.2f} min")
        if v["dice"]>best:
            best=v["dice"]; torch.save({k:x.detach().cpu().clone() for k,x in m.state_dict().items()},out/"best_model_state_only.pth")
            print("✓ New best validation Dice checkpoint saved.")
        if v["empty"]==len(va): print("⚠ VALIDATION: all predictions are background-only.")
    (out/"history.json").write_text(json.dumps(hist,indent=2),encoding="utf-8"); return hist

def main():
    print("="*82+"\nPART 43 PATH VALIDATION\n"+"="*82)
    for path,msg in [(ROOT,"Project root"),(SRC/"segmentation_rsna_part11_controlled_pilot_training.py","Part 11"),
                     (SRC/"segmentation_rsna_part9_3d_dataset_loader.py","Part 9"),(CKPT,"Part 15 initialization checkpoint"),
                     (TRCSV,"Part 15 train cohort"),(VACSV,"Part 15 validation cohort")]:
        ensure(path.exists(),f"{msg} missing: {path}"); print(f"{msg:<43}: FOUND")
    print("\n"+"="*82+"\nPART 43 — IMPROVED CONTROLLED SPATIAL-SAMPLING TRAINING\n"+"="*82)
    for k,v in [("PyTorch",torch.__version__),("Device",DEVICE),("Full volume",FULL),("Training crop",CROP),
                ("Train subset",TRN),("Validation subset",VALN),("Epochs",EPOCHS),("Learning rate",LR),("Weight decay",WD)]:
        print(f"{k:<43}: {v}")
    print(f"{'Validation crop policy':<43}: same foreground-centered crop for both")
    print("Part 15 overwritten                       : NO\nSPIDER used                               : NO\nTest set used                             : NO")
    print(f"Initialization SHA256                     : {sha(CKPT)}")
    tr_all=list(csv.DictReader(open(TRCSV,newline="",encoding="utf-8"))); va_all=list(csv.DictReader(open(VACSV,newline="",encoding="utf-8")))
    ensure(len(tr_all)==500,f"Expected 500 train rows, got {len(tr_all)}"); ensure(len(va_all)==100,f"Expected 100 val rows, got {len(va_all)}")
    tr=tr_all[:TRN]; va=va_all[:VALN]
    print(f"Exact Part 15 train cohort rows           : {len(tr_all)}\nExact Part 15 validation cohort rows      : {len(va_all)}\nControlled train subset                   : {len(tr)}\nControlled validation subset              : {len(va)}")
    raw=torch.load(CKPT,map_location="cpu",weights_only=False)
    state=raw.get("model_state_dict",raw.get("state_dict",raw)) if isinstance(raw,dict) else raw
    ensure(isinstance(state,dict),"Could not locate model state_dict")
    state={k:v.detach().cpu().clone() for k,v in state.items()}
    print("\n"+"="*82+"\nPART 43 CROP COVERAGE AUDIT\n"+"="*82); cov=audit(tr)
    print("\n"+"="*82+"\nPART 43 SHAPE SMOKE TEST\n"+"="*82)
    for i in range(3):
        im,ma=load_case(tr[i]); ri,rm,_,_=crop(im,ma,"random",random.Random(SEED+i)); ci,cm,_,_=crop(im,ma,"foreground_centered",random.Random(SEED+i))
        print(f"Case {i+1}: full={tuple(im.shape)} random={tuple(ri.shape)} randomFG={int((rm>0).sum())} centered={tuple(ci.shape)} centeredFG={int((cm>0).sum())}")
    print("✓ Shape smoke test PASSED.")
    bh=run("RANDOM SPATIAL CROPPING","random",tr,va,state)
    ch=run("FOREGROUND-CENTERED SPATIAL CROPPING","foreground_centered",tr,va,state)
    b=bh[-1]; c=ch[-1]
    comp={"baseline_final_val_loss":b["val_loss"],"foreground_centered_final_val_loss":c["val_loss"],
          "baseline_final_val_dice":b["val_dice"],"foreground_centered_final_val_dice":c["val_dice"],
          "dice_delta_centered_minus_baseline":c["val_dice"]-b["val_dice"],
          "baseline_final_predicted_fg_voxels":b["val_pred_fg"],"foreground_centered_final_predicted_fg_voxels":c["val_pred_fg"],
          "baseline_final_empty_cases":b["val_empty"],"foreground_centered_final_empty_cases":c["val_empty"],
          "same_initialization_sha256":True,"initialization_sha256":sha(CKPT),"train_subset":TRN,"validation_subset":VALN,"epochs":EPOCHS,
          "crop_shape":CROP,"mean_random_crop_fg":cov["mean_random_crop_fg"],"mean_foreground_centered_crop_fg":cov["mean_centered_crop_fg"],
          "random_zero_fg_cases":cov["random_zero_fg_cases"],"centered_zero_fg_cases":cov["centered_zero_fg_cases"],
          "interpretation":"FOREGROUND_CENTERED_SPATIAL_SAMPLING_SHOWS_STABLE_PILOT_BENEFIT" if c["val_dice"]>b["val_dice"] and c["val_empty"]<=b["val_empty"] else "NO_CLEAR_FOREGROUND_CENTERED_ADVANTAGE_IN_THIS_PILOT",
          "caution":"Pilot evidence only; this does not establish clinical performance or prove exclusive causality."}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"part43_controlled_comparison.json").write_text(json.dumps(comp,indent=2),encoding="utf-8")
    print("\n"+"="*82+"\nPART 43 CONTROLLED COMPARISON\n"+"="*82); print(json.dumps(comp,indent=2))
    print("\n"+"="*82+"\nPART 43 COMPLETE\n"+"="*82)
    print(f"Output directory                           : {OUT}\nRandom spatial crop                        : {BASE}\nForeground-centered spatial crop           : {CENTER}")
    print("✓ Part 15 was not modified. ✓ Same initialization/cohort/loss/optimizer/architecture. ✓ Only training spatial crop selection changed.")

if __name__=="__main__": main()
