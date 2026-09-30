
from __future__ import annotations
import hashlib, importlib.util, json, random, sys, time, traceback
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler

# PART 41: CONTROLLED FOREGROUND-AWARE CASE SAMPLING
# Only sampling changes between the two conditions.
PROJECT_ROOT = Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC_DIR = PROJECT_ROOT / "src"
OUT = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part41_controlled_foreground_sampling"
BASE = OUT / "baseline_random_sampling"
AWARE = OUT / "foreground_aware_weighted_sampling"
PART11 = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training.py"

P15 = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"

CKPT_DIR = P15 / "checkpoints"
CKPT = CKPT_DIR / "best_model.pth"

TRAIN_CSV = P15 / "part15_train_cohort.csv"
VAL_CSV = P15 / "part15_validation_cohort.csv"

TRAIN_N, VAL_N, EPOCHS, SEED = 100, 50, 3, 41041
PATCH = (64, 96, 96)
CLASSES, FEATURES = 6, 12
LR, WD = 1e-4, 1e-5
POWER, REPLACEMENT = 0.5, True
VAL_ROWS = None

def banner(s): print("\n" + "="*82 + "\n" + s + "\n" + "="*82)
def line(k,v): print(f"{k:<42}: {v}")
def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def import_p11():
    spec=importlib.util.spec_from_file_location("part11_p41",str(PART11))
    if spec is None or spec.loader is None: raise ImportError(PART11)
    m=importlib.util.module_from_spec(spec); sys.modules["part11_p41"]=m
    spec.loader.exec_module(m)
    for x in ["load_part9_module","load_tensor_case","dice_from_prediction","create_model","DiceCELoss"]:
        if not hasattr(m,x): raise AttributeError("Part 11 missing: "+x)
    return m

def case(p11,p9,row):
    im,ma,info=p11.load_tensor_case(row,p9)
    im=torch.as_tensor(im); ma=torch.as_tensor(ma)
    if im.ndim==4 and im.shape[0]==1: im=im[0]
    if ma.ndim==4 and ma.shape[0]==1: ma=ma[0]
    if im.ndim!=3 or ma.ndim!=3 or tuple(im.shape)!=PATCH or tuple(ma.shape)!=PATCH:
        raise ValueError(f"bad shapes {tuple(im.shape)} {tuple(ma.shape)}")
    return im.float().contiguous(),ma.long().contiguous(),info

def state_dict(ck):
    for k in ("model_state_dict","state_dict","model","weights"):
        if isinstance(ck,dict) and isinstance(ck.get(k),dict): return ck[k]
    if isinstance(ck,dict) and all(isinstance(v,torch.Tensor) for v in ck.values()): return ck
    raise RuntimeError("No model state_dict in checkpoint")

def metrics(logits,target,p11):
    with torch.no_grad():
        d,_=p11.dice_from_prediction(logits,target)
        fg=int((torch.argmax(logits,1)!=0).sum())
    return float(d),fg

def evaluate(model,rows,p11,p9,loss_fn,device):
    model.eval(); L=[]; D=[]; P=[]; T=[]; empty=0
    with torch.no_grad():
        for i,(_,r) in enumerate(rows.iterrows(),1):
            im,ma,_=case(p11,p9,r)
            x=im.unsqueeze(0).unsqueeze(0).to(device)
            y=ma.unsqueeze(0).to(device)
            with autocast(enabled=device.type=="cuda"):
                z=model(x); loss=loss_fn(z,y.unsqueeze(1))
            d,fg=metrics(z,y,p11); tf=int((y!=0).sum())
            L.append(float(loss.cpu())); D.append(d); P.append(fg); T.append(tf)
            if fg==0: empty+=1
            if i==1 or i%25==0 or i==len(rows):
                print(f"  VAL {i:03d}/{len(rows)} Loss={L[-1]:.5f} Dice={d:.6f} PredFG={fg} TargetFG={tf}")
    return dict(loss=float(np.mean(L)),dice=float(np.mean(D)),pred_fg=float(np.mean(P)),
                target_fg=float(np.mean(T)),empty_cases=empty)

def weights_for(rows,p11,p9):
    banner("BUILDING FOREGROUND-AWARE CASE WEIGHTS")
    fg=[]
    for i,(_,r) in enumerate(rows.iterrows(),1):
        _,ma,_=case(p11,p9,r); n=int((ma!=0).sum()); fg.append(n)
        if i==1 or i%25==0 or i==len(rows): print(f"  WEIGHT {i:03d}/{len(rows)} FG={n}")
    a=np.asarray(fg,dtype=np.float64); raw=np.power(a+1,POWER); w=raw/raw.sum()
    pd.DataFrame({"subset_index":np.arange(len(fg)),"foreground_voxels":fg,
                  "sampling_weight_raw":raw,"sampling_probability":w}).to_csv(
                      OUT/"foreground_sampling_weights.csv",index=False)
    with open(OUT/"foreground_sampling_weights.json","w",encoding="utf-8") as f:
        json.dump({"power":POWER,"replacement":REPLACEMENT,"foreground_voxels":fg,
                   "probabilities":w.tolist()},f,indent=2)
    line("Minimum foreground voxels",int(a.min())); line("Maximum foreground voxels",int(a.max()))
    line("Mean foreground voxels",float(a.mean()))
    line("Foreground fraction",float(a.sum()/(len(a)*np.prod(PATCH))))
    return w

def run(name,out,rows,p11,p9,device,init,mode,w=None):
    banner(name); out.mkdir(parents=True,exist_ok=True)
    model=p11.create_model(device); model.load_state_dict(init,strict=True); model.to(device)
    loss_fn=p11.DiceCELoss(to_onehot_y=True,softmax=True,include_background=False)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    scaler=GradScaler(enabled=device.type=="cuda")
    hist=[]; best=-1
    initial=evaluate(model,VAL_ROWS,p11,p9,loss_fn,device)
    print(f"Initialization: loss={initial['loss']:.6f}, Dice={initial['dice']:.6f}, PredFG={initial['pred_fg']:.1f}, empty={initial['empty_cases']}/{len(VAL_ROWS)}")
    for ep in range(1,EPOCHS+1):
        banner(f"{name} — EPOCH {ep}/{EPOCHS}"); model.train()
        g=torch.Generator(); g.manual_seed(SEED+ep+(500000 if mode=="aware" else 0))
        if mode=="baseline":
            sampler=torch.utils.data.RandomSampler(range(len(rows)),replacement=False,generator=g)
        else:
            sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(rows),replacement=REPLACEMENT,generator=g)
        inds=list(iter(sampler)); L=[];D=[];P=[];T=[];start=time.time()
        for step,idx in enumerate(inds,1):
            im,ma,_=case(p11,p9,rows.iloc[int(idx)])
            x=im.unsqueeze(0).unsqueeze(0).to(device); y=ma.unsqueeze(0).to(device)
            tf=int((ma!=0).sum()); opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type=="cuda"):
                z=model(x); loss=loss_fn(z,y.unsqueeze(1))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            d,fg=metrics(z.detach(),y,p11); lv=float(loss.detach().cpu())
            L.append(lv);D.append(d);P.append(fg);T.append(tf)
            if step==1 or step%25==0 or step==len(inds):
                print(f"  TRAIN {step:03d}/{len(inds)} Loss={lv:.5f} Dice={d:.6f} PredFG={fg} TargetFG={tf}")
        v=evaluate(model,VAL_ROWS,p11,p9,loss_fn,device)
        rec={"epoch":ep,"train_loss":float(np.mean(L)),"train_dice":float(np.mean(D)),
             "train_pred_fg":float(np.mean(P)),"sampled_mean_target_fg":float(np.mean(T)),
             "sampled_min_target_fg":int(np.min(T)),"sampled_max_target_fg":int(np.max(T)),
             "unique_cases_sampled":int(len(set(inds))),"val_loss":v["loss"],"val_dice":v["dice"],
             "val_pred_fg":v["pred_fg"],"val_target_fg":v["target_fg"],
             "val_empty_cases":v["empty_cases"],"minutes":(time.time()-start)/60}
        hist.append(rec)
        print(f"Epoch {ep}: train_loss={rec['train_loss']:.6f}, train_Dice={rec['train_dice']:.6f}, val_loss={v['loss']:.6f}, val_Dice={v['dice']:.6f}, val_PredFG={v['pred_fg']:.1f}, unique_train_cases={rec['unique_cases_sampled']}, time={rec['minutes']:.2f} min")
        ck={"epoch":ep,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),
            "best_val_dice":max(best,v["dice"]),"part41_condition":name,"sampler_mode":mode,
            "sampling_power":POWER,"replacement":REPLACEMENT,"seed":SEED}
        torch.save(ck,out/f"epoch_{ep:02d}.pth")
        if v["dice"]>best:
            best=v["dice"]; torch.save(ck,out/"best_model.pth"); print("✓ New best validation Dice checkpoint saved.")
        print("✓ VALIDATION: foreground predictions remain non-empty." if v["empty_cases"]<len(VAL_ROWS) else "⚠ VALIDATION: all predictions are background-only.")
        if device.type=="cuda": torch.cuda.empty_cache()
    pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
    with open(out/"summary.json","w",encoding="utf-8") as f: json.dump({"condition":name,"history":hist,"best_val_dice":best},f,indent=2)
    return hist

def main():
    global VAL_ROWS
    seed(SEED); banner("PART 41 PATH VALIDATION")
    for n,p in {"Project root":PROJECT_ROOT,"Part 11":PART11,"Part 15 checkpoint":CKPT,
                "Part 15 train cohort":TRAIN_CSV,"Part 15 validation cohort":VAL_CSV}.items():
        line(n,"FOUND" if p.exists() else f"MISSING: {p}")
        if not p.exists(): raise FileNotFoundError(p)
    OUT.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    banner("PART 41 — CONTROLLED FOREGROUND-AWARE CASE SAMPLING")
    line("PyTorch",torch.__version__); line("Device",device)
    if device.type=="cuda": line("GPU",torch.cuda.get_device_name(0))
    line("Patch size",PATCH); line("Feature size",FEATURES); line("Classes",CLASSES)
    line("Train subset",TRAIN_N); line("Validation subset",VAL_N); line("Epochs",EPOCHS)
    line("Sampling power",POWER); line("Replacement",REPLACEMENT)
    line("Part 15 overwritten","NO"); line("SPIDER used","NO"); line("Test set used","NO")
    p11=import_p11(); p9=p11.load_part9_module()
    tr=pd.read_csv(TRAIN_CSV); va=pd.read_csv(VAL_CSV)
    if len(tr)!=500 or len(va)!=100: raise ValueError(f"Unexpected cohorts: {len(tr)}, {len(va)}")
    tr=tr.iloc[:TRAIN_N].copy().reset_index(drop=True); VAL_ROWS=va.iloc[:VAL_N].copy().reset_index(drop=True)
    line("Exact Part 15 train cohort rows",500); line("Exact Part 15 validation cohort rows",100)
    line("Controlled train subset",len(tr)); line("Controlled validation subset",len(VAL_ROWS))
    ck=torch.load(CKPT,map_location="cpu",weights_only=False); init=state_dict(ck)
    init={k:v.detach().cpu().clone() for k,v in init.items()}
    init_sha=sha(CKPT); line("Initialization SHA256",init_sha)
    w=weights_for(tr,p11,p9)
    banner("PART 41 SHAPE SMOKE TEST")
    for j in range(3):
        im,ma,_=case(p11,p9,tr.iloc[j]); print(f"  Case {j+1}: image={tuple(im.shape)} input={(1,1,*im.shape)} target={(1,*ma.shape)} fg={int((ma!=0).sum())}")
    print("✓ Shape smoke test PASSED.")
    seed(SEED)
    bh=run("BASELINE RANDOM SAMPLING",BASE,tr,p11,p9,device,init,"baseline")
    if device.type=="cuda": torch.cuda.empty_cache()
    seed(SEED)
    ah=run("FOREGROUND-AWARE WEIGHTED SAMPLING",AWARE,tr,p11,p9,device,init,"aware",w)
    b,a=bh[-1],ah[-1]
    comp={"baseline_final_val_loss":b["val_loss"],"foreground_aware_final_val_loss":a["val_loss"],
          "baseline_final_val_dice":b["val_dice"],"foreground_aware_final_val_dice":a["val_dice"],
          "dice_delta_aware_minus_baseline":a["val_dice"]-b["val_dice"],
          "baseline_final_predicted_fg_voxels":b["val_pred_fg"],"foreground_aware_final_predicted_fg_voxels":a["val_pred_fg"],
          "predicted_fg_delta_aware_minus_baseline":a["val_pred_fg"]-b["val_pred_fg"],
          "baseline_final_empty_cases":b["val_empty_cases"],"foreground_aware_final_empty_cases":a["val_empty_cases"],
          "baseline_final_unique_cases_sampled":b["unique_cases_sampled"],"foreground_aware_final_unique_cases_sampled":a["unique_cases_sampled"],
          "same_initialization_sha256":True,"initialization_sha256":init_sha,"sampling_power":POWER,
          "replacement":REPLACEMENT,
          "interpretation":"FOREGROUND_AWARE_SAMPLING_CHANGED_TRAINING_BEHAVIOR" if abs(a["val_dice"]-b["val_dice"])>1e-6 or abs(a["val_pred_fg"]-b["val_pred_fg"])>1e-6 else "NO_MEANINGFUL_BEHAVIORAL_DIFFERENCE_IN_THIS_PILOT",
          "caution":"Pilot evidence only; case-level sampling does not test spatial foreground-centered cropping or establish clinical performance."}
    banner("PART 41 CONTROLLED COMPARISON"); print(json.dumps(comp,indent=2))
    with open(OUT/"comparison.json","w",encoding="utf-8") as f: json.dump(comp,f,indent=2)
    pd.DataFrame([comp]).to_csv(OUT/"comparison.csv",index=False)
    banner("PART 41 COMPLETE"); line("Output directory",OUT); line("Baseline",BASE); line("Foreground-aware",AWARE)
    print("✓ Part 15 was not modified. ✓ Same initialization/cohort/loss/optimizer/architecture. ✓ Only case sampling changed.")
if __name__=="__main__":
    try: main()
    except Exception as e: banner("PART 41 ERROR"); print(type(e).__name__,e); traceback.print_exc(); raise
