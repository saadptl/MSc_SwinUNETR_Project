"""Part 42 - Controlled foreground-centered spatial sampling pilot.

A: deterministic random spatial crops.
B: foreground-centered spatial crops.

Only TRAINING crop selection differs. Both conditions use the original
Part 15 initialization-from-Part11 checkpoint, original DiceCELoss,
AdamW, same cohort, architecture, crop size, and number of steps.
Validation uses the same deterministic foreground-centered crop for both.
"""
from __future__ import annotations
import csv, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, RandomSampler
from monai.losses import DiceCELoss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
import segmentation_rsna_part11_controlled_pilot_training as part11
import segmentation_rsna_part9_3d_dataset_loader as part9

P15 = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
CKPT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"
TRAIN_CSV = P15 / "part15_train_cohort.csv"
VAL_CSV = P15 / "part15_validation_cohort.csv"
OUT = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part42_controlled_foreground_centered_spatial_sampling"
BASE_OUT, FG_OUT = OUT / "random_spatial_crop", OUT / "foreground_centered_spatial_crop"

FULL = (64, 96, 96)
CROP = (32, 64, 64)
TRAIN_N, VAL_N, EPOCHS = 100, 50, 3
LR, WD, SEED = 1e-4, 1e-5, 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOSS = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for x in iter(lambda: f.read(1024 * 1024), b""): h.update(x)
    return h.hexdigest()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def ensure(x, msg):
    if not x: raise RuntimeError(msg)


def load_case(row):
    # Part 11's resolver expects a pandas Series (uses row.index).
    # The cohort CSV is read into dictionaries, so normalize each row here.
    if isinstance(row, dict):
        row = pd.Series(row)
    loaded = part11.load_tensor_case(row, part9)
    if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
        raise RuntimeError(f"Unexpected load_tensor_case return: {type(loaded)}")
    image, mask = loaded[0], loaded[1]
    image, mask = image.detach().cpu(), mask.detach().cpu()
    if image.ndim == 4 and image.shape[0] == 1: image = image[0]
    if mask.ndim == 4 and mask.shape[0] == 1: mask = mask[0]
    ensure(image.ndim == 3 and mask.ndim == 3, f"bad shapes {tuple(image.shape)} {tuple(mask.shape)}")
    ensure(tuple(image.shape) == FULL and tuple(mask.shape) == FULL,
           f"expected {FULL}, got {tuple(image.shape)} {tuple(mask.shape)}")
    return image.float(), mask.long()


def bounds(center):
    out=[]
    for c, n, k in zip(center, FULL, CROP):
        s=max(0, min(int(c)-k//2, n-k)); out.append((s, s+k))
    return out


def centroid(mask):
    p=torch.nonzero(mask>0, as_tuple=False)
    if p.numel()==0: return tuple((n-1)//2 for n in FULL)
    return tuple(int(round(float(v))) for v in p.float().mean(0))


def random_center(rng):
    return tuple(rng.randint(k//2, n-(k-k//2)) for n,k in zip(FULL,CROP))


def crop(image, mask, mode, rng):
    c=centroid(mask) if mode=="foreground_centered" else random_center(rng)
    b=bounds(c)
    z0,z1=b[0]; y0,y1=b[1]; x0,x1=b[2]
    a=image[z0:z1,y0:y1,x0:x1]; m=mask[z0:z1,y0:y1,x0:x1]
    ensure(tuple(a.shape)==CROP and tuple(m.shape)==CROP, f"crop shape {tuple(a.shape)} {tuple(m.shape)}")
    return a,m,b


class DS(Dataset):
    def __init__(self, rows, mode, seed): self.rows,self.mode,self.seed=rows,mode,seed
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        im,ma=load_case(self.rows[i]); rng=random.Random(self.seed+i*1009)
        im,ma,b=crop(im,ma,self.mode,rng)
        return im,ma,int((ma>0).sum()),b


def collate(b): return b[0]


def model_from(state):
    m=part11.create_model(DEVICE); m.load_state_dict(state, strict=True); m.to(DEVICE); return m


@torch.no_grad()
def evaluate(model, rows):
    model.eval(); ls=[]; ds=[]; pfs=[]; tfs=[]; empty=0
    dl=DataLoader(DS(rows,"foreground_centered",SEED+90000),batch_size=1,shuffle=False,collate_fn=collate)
    for i,(im,ma,_,_) in enumerate(dl):
        x=im.to(DEVICE).unsqueeze(0).unsqueeze(0); y=ma.to(DEVICE).unsqueeze(0)
        with autocast(enabled=DEVICE.type=="cuda"): logits=model(x); loss=LOSS(logits,y.unsqueeze(1))
        d,_=part11.dice_from_prediction(logits,y)
        pf=int((logits.argmax(1)>0).sum()); tf=int((y>0).sum())
        ls.append(float(loss)); ds.append(float(d)); pfs.append(pf); tfs.append(tf); empty+=pf==0
        if i in (0,24,49): print(f"  VAL {i+1:03d}/{len(rows)} Loss={float(loss):.5f} Dice={d:.6f} PredFG={pf} TargetFG={tf}")
    return dict(loss=float(np.mean(ls)),dice=float(np.mean(ds)),pred_fg=float(np.mean(pfs)),target_fg=float(np.mean(tfs)),empty=int(empty))


def train_epoch(model, dl, opt, scaler):
    model.train(); ls=[]; ds=[]; pfs=[]
    for i,(im,ma,_,_) in enumerate(dl):
        x=im.to(DEVICE).unsqueeze(0).unsqueeze(0); y=ma.to(DEVICE).unsqueeze(0)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=DEVICE.type=="cuda"): logits=model(x); loss=LOSS(logits,y.unsqueeze(1))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        with torch.no_grad(): d,_=part11.dice_from_prediction(logits,y); pf=int((logits.argmax(1)>0).sum())
        ls.append(float(loss)); ds.append(float(d)); pfs.append(pf)
        if i in (0,24,49,74,99): print(f"  TRAIN {i+1:03d}/{len(dl)} Loss={float(loss):.5f} Dice={d:.6f} PredFG={pf} TargetFG={int((y>0).sum())}")
    return float(np.mean(ls)),float(np.mean(ds)),float(np.mean(pfs))


def jwrite(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2),encoding="utf-8")


def coverage(rows):
    rec=[]
    for i,r in enumerate(rows):
        im,ma=load_case(r); rng=random.Random(SEED+i*1009)
        _,rm,rb=crop(im,ma,"random",rng); _,fm,fb=crop(im,ma,"foreground_centered",rng)
        rec.append(dict(index=i,source_fg=int((ma>0).sum()),random_crop_fg=int((rm>0).sum()),foreground_centered_crop_fg=int((fm>0).sum()),random_bounds=str(rb),foreground_centered_bounds=str(fb)))
        if i in (0,24,49,74,99): print(f"  COVERAGE {i+1:03d}/100 sourceFG={rec[-1]['source_fg']} randomFG={rec[-1]['random_crop_fg']} centeredFG={rec[-1]['foreground_centered_crop_fg']}")
    OUT.mkdir(parents=True,exist_ok=True)
    with open(OUT/"part42_crop_coverage_audit.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rec[0].keys()); w.writeheader(); w.writerows(rec)
    s={"cases":100,"crop_shape":CROP,"mean_source_fg":float(np.mean([r['source_fg'] for r in rec])),"mean_random_crop_fg":float(np.mean([r['random_crop_fg'] for r in rec])),"mean_foreground_centered_crop_fg":float(np.mean([r['foreground_centered_crop_fg'] for r in rec])),"random_zero_fg_cases":sum(r['random_crop_fg']==0 for r in rec),"centered_zero_fg_cases":sum(r['foreground_centered_crop_fg']==0 for r in rec)}
    jwrite(OUT/"part42_crop_coverage_summary.json",s); print(json.dumps(s,indent=2)); return s


def run(label,mode,rows,val_rows,state,outdir):
    print("\n"+"="*82+f"\n{label}\n"+"="*82)
    seed_all(SEED); m=model_from(state); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD); scaler=GradScaler(enabled=DEVICE.type=="cuda")
    v0=evaluate(m,val_rows); print(f"Initialization: loss={v0['loss']:.6f}, Dice={v0['dice']:.6f}, PredFG={v0['pred_fg']:.1f}, empty={v0['empty']}/{len(val_rows)}")
    hist=[dict(epoch=0,train_loss=None,train_dice=None,train_pred_fg=None,val_loss=v0['loss'],val_dice=v0['dice'],val_pred_fg=v0['pred_fg'],val_empty=v0['empty'],unique_train_cases=len(rows))]
    ds=DS(rows,mode,SEED+1000); gen=torch.Generator().manual_seed(SEED); sampler=RandomSampler(ds,replacement=False,generator=gen)
    dl=DataLoader(ds,batch_size=1,sampler=sampler,collate_fn=collate,num_workers=0); outdir.mkdir(parents=True,exist_ok=True)
    for e in range(1,EPOCHS+1):
        t=time.time(); tr=train_epoch(m,dl,opt,scaler); v=evaluate(m,val_rows); mins=(time.time()-t)/60
        z=dict(epoch=e,train_loss=tr[0],train_dice=tr[1],train_pred_fg=tr[2],val_loss=v['loss'],val_dice=v['dice'],val_pred_fg=v['pred_fg'],val_empty=v['empty'],unique_train_cases=len(rows),time_min=mins); hist.append(z)
        print(f"Epoch {e}: train_loss={tr[0]:.6f}, train_Dice={tr[1]:.6f}, val_loss={v['loss']:.6f}, val_Dice={v['dice']:.6f}, val_PredFG={v['pred_fg']:.1f}, empty={v['empty']}/{len(val_rows)}, time={mins:.2f} min")
        if v['empty']==len(val_rows): print("⚠ VALIDATION: all predictions are background-only.")
    jwrite(outdir/"history.json",hist); return hist


def main():
    print("="*82+"\nPART 42 PATH VALIDATION\n"+"="*82)
    ensure(CKPT.exists(),f"Missing initialization checkpoint: {CKPT}"); ensure(TRAIN_CSV.exists(),"Missing train cohort"); ensure(VAL_CSV.exists(),"Missing validation cohort")
    print("Project root                              : FOUND\nPart 11                                   : FOUND\nPart 15 initialization checkpoint        : FOUND\nPart 15 train cohort                      : FOUND\nPart 15 validation cohort                 : FOUND")
    print("\n"+"="*82+"\nPART 42 — CONTROLLED FOREGROUND-CENTERED SPATIAL SAMPLING\n"+"="*82)
    print(f"PyTorch                                   : {torch.__version__}\nDevice                                    : {DEVICE}\nFull preprocessed volume                  : {FULL}\nTraining crop size                        : {CROP}\nTrain subset                              : {TRAIN_N}\nValidation subset                         : {VAL_N}\nEpochs                                    : {EPOCHS}\nPart 15 overwritten                       : NO\nSPIDER used                               : NO\nTest set used                             : NO\nInitialization SHA256                     : {sha(CKPT)}")
    with open(TRAIN_CSV,newline="",encoding="utf-8") as f: alltr=list(csv.DictReader(f))
    with open(VAL_CSV,newline="",encoding="utf-8") as f: allva=list(csv.DictReader(f))
    ensure(len(alltr)==500,f"Expected 500 train rows, got {len(alltr)}"); ensure(len(allva)==100,f"Expected 100 val rows, got {len(allva)}")
    tr,va=alltr[:100],allva[:50]
    raw=torch.load(CKPT,map_location="cpu",weights_only=False); state=raw.get("model_state_dict",raw.get("state_dict",raw)) if isinstance(raw,dict) else raw; ensure(isinstance(state,dict),"Bad checkpoint state"); state={k:v.detach().cpu().clone() for k,v in state.items()}
    print(f"Exact Part 15 train cohort rows           : 500\nExact Part 15 validation cohort rows      : 100\nControlled train subset                   : 100\nControlled validation subset              : 50")
    print("\n"+"="*82+"\nPART 42 CROP COVERAGE AUDIT\n"+"="*82); cov=coverage(tr)
    print("\n"+"="*82+"\nPART 42 SHAPE SMOKE TEST\n"+"="*82)
    for i in range(3):
        im,ma=load_case(tr[i]); ri,rm,_=crop(im,ma,"random",random.Random(SEED+i)); fi,fm,_=crop(im,ma,"foreground_centered",random.Random(SEED+i)); print(f"Case {i+1}: full={tuple(im.shape)} random={tuple(ri.shape)} fg={int((rm>0).sum())} centered={tuple(fi.shape)} fg={int((fm>0).sum())}"); ensure(tuple(ri.shape)==CROP and tuple(fi.shape)==CROP,"Crop shape failure")
    print("✓ Shape smoke test PASSED.")
    bh=run("RANDOM SPATIAL CROPPING","random",tr,va,state,BASE_OUT)
    fh=run("FOREGROUND-CENTERED SPATIAL CROPPING","foreground_centered",tr,va,state,FG_OUT)
    b,f=bh[-1],fh[-1]
    cmp={"baseline_final_val_loss":b['val_loss'],"foreground_centered_final_val_loss":f['val_loss'],"baseline_final_val_dice":b['val_dice'],"foreground_centered_final_val_dice":f['val_dice'],"dice_delta_centered_minus_baseline":f['val_dice']-b['val_dice'],"baseline_final_predicted_fg_voxels":b['val_pred_fg'],"foreground_centered_final_predicted_fg_voxels":f['val_pred_fg'],"predicted_fg_delta_centered_minus_baseline":f['val_pred_fg']-b['val_pred_fg'],"baseline_final_empty_cases":b['val_empty'],"foreground_centered_final_empty_cases":f['val_empty'],"same_initialization_sha256":True,"initialization_sha256":sha(CKPT),"crop_shape":CROP,"mean_random_crop_fg":cov['mean_random_crop_fg'],"mean_foreground_centered_crop_fg":cov['mean_foreground_centered_crop_fg'],"random_zero_fg_cases":cov['random_zero_fg_cases'],"centered_zero_fg_cases":cov['centered_zero_fg_cases'],"interpretation":"FOREGROUND_CENTERED_SPATIAL_SAMPLING_CHANGED_BEHAVIOR" if (f['val_pred_fg']!=b['val_pred_fg'] or f['val_empty']!=b['val_empty'] or abs(f['val_dice']-b['val_dice'])>1e-4) else "NO_MEANINGFUL_BEHAVIORAL_DIFFERENCE_IN_THIS_PILOT","caution":"Pilot evidence only; this tests spatial crop presentation and does not establish clinical performance."}
    jwrite(OUT/"part42_controlled_comparison.json",cmp); print("\n"+"="*82+"\nPART 42 CONTROLLED COMPARISON\n"+"="*82); print(json.dumps(cmp,indent=2)); print("\n"+"="*82+"\nPART 42 COMPLETE\n"+"="*82); print(f"Output directory                           : {OUT}\nRandom spatial crop                        : {BASE_OUT}\nForeground-centered spatial crop           : {FG_OUT}\n✓ Part 15 was not modified. ✓ Same initialization/cohort/loss/optimizer/architecture. ✓ Only training spatial crop selection changed.")

if __name__=="__main__": main()
