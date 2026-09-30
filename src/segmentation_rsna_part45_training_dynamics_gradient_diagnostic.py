
from pathlib import Path
import sys, csv, json, hashlib, random, importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss

ROOT=Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC=ROOT/"src"
PART11_PATH=SRC/"segmentation_rsna_part11_controlled_pilot_training.py"
PART9_PATH=SRC/"segmentation_rsna_part9_3d_dataset_loader.py"
P15=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"
INIT=P15/"checkpoints"/"part15_initialization_from_part11.pth"
TRCSV=P15/"part15_train_cohort.csv"
VACSV=P15/"part15_validation_cohort.csv"
OUT=ROOT/"outputs"/"segmentation"/"rsna_part45_training_dynamics_gradient_diagnostic"
REPORT=OUT/"reports"; CKPT=OUT/"checkpoints"
for p in (OUT,REPORT,CKPT): p.mkdir(parents=True,exist_ok=True)

TRAIN_N=50; VAL_N=20; EPOCHS=3
FULL_SHAPE=(64,96,96); CROP_SHAPE=(32,64,64)
LR=1e-4; WD=1e-5; SEED=145; DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def centered_crop(image,mask,shape,seed):
    D,H,W=image.shape; cd,ch,cw=shape; rng=np.random.default_rng(seed)
    fg=torch.nonzero(mask>0,as_tuple=False)
    if len(fg):
        p=fg[int(rng.integers(0,len(fg)))]
        c=[int(p[0])+int(rng.integers(-cd//8,cd//8+1)),
           int(p[1])+int(rng.integers(-ch//8,ch//8+1)),
           int(p[2])+int(rng.integers(-cw//8,cw//8+1))]
    else: c=[D//2,H//2,W//2]
    s=[max(0,min(D-cd,c[0]-cd//2)),max(0,min(H-ch,c[1]-ch//2)),max(0,min(W-cw,c[2]-cw//2))]
    d,h,w=s
    return image[d:d+cd,h:h+ch,w:w+cw],mask[d:d+cd,h:h+ch,w:w+cw]

def load_case(row,part11,part9,seed):
    z=part11.load_tensor_case(row,part9)
    image=torch.as_tensor(z[0]); mask=torch.as_tensor(z[1])
    if image.ndim==4 and image.shape[0]==1: image=image.squeeze(0)
    if mask.ndim==4 and mask.shape[0]==1: mask=mask.squeeze(0)
    image=image.float(); mask=mask.long()
    if tuple(image.shape)!=FULL_SHAPE or tuple(mask.shape)!=FULL_SHAPE:
        raise RuntimeError(f"Unexpected shapes image={tuple(image.shape)} mask={tuple(mask.shape)}")
    return centered_crop(image,mask,CROP_SHAPE,seed)

def preload(rows,part11,part9,label,offset):
    cases=[]
    for i,row in enumerate(rows):
        im,ma=load_case(row,part11,part9,SEED+offset+i); cases.append((im,ma))
        if i==0 or (i+1)%10==0 or i+1==len(rows):
            print(f"{label.upper()} {i+1:03d}/{len(rows)} cropFG={(ma>0).sum().item()}")
    print(f"Mean {label} crop foreground voxels : {np.mean([(m>0).sum().item() for _,m in cases]):.2f}")
    print(f"{label.capitalize()} zero-FG crops : {sum(int((m>0).sum()==0) for _,m in cases)}/{len(cases)}")
    return cases

dice_fn=DiceCELoss(to_onehot_y=True,softmax=True,lambda_dice=1.0,lambda_ce=0.0)
ce_fn=torch.nn.CrossEntropyLoss()

def components(logits,target):
    y=target.unsqueeze(1); d=dice_fn(logits,y); c=ce_fn(logits,target); return d+c,d,c

def metrics(logits,target):
    with torch.no_grad():
        p=torch.softmax(logits,1); pred=p.argmax(1); pf=pred>0; tf=target>0
        tp=(pf&tf).sum().item(); pc=pf.sum().item(); tc=tf.sum().item()
        fd=2*tp/(pc+tc) if pc+tc else 1.0
        fgprob=p[:,1:].sum(1).mean().item()
        fgmask=tf; bgmask=~tf
        fgl=p[:,1:].mean(1)[fgmask].mean().item() if fgmask.any() else float("nan")
        bgl=p[:,1:].mean(1)[bgmask].mean().item() if bgmask.any() else float("nan")
        return dict(fg_dice=float(fd),pred_fg=float(pc),target_fg=float(tc),fg_probability=float(fgprob),
                    mean_fg_logit=float(logits[:,1:].mean(1)[fgmask].mean().item()) if fgmask.any() else float("nan"),
                    mean_bg_logit=float(logits[:,1:].mean(1)[bgmask].mean().item()) if bgmask.any() else float("nan"),
                    fg_bg_logit_gap=float(fgl-bgl),empty=float(pc==0))

def grad_probe(model,cases):
    model.eval(); rec=[]
    for im,ma in cases:
        x=im[None,None].to(DEVICE); y=ma[None].to(DEVICE); model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=DEVICE.type=="cuda"): lo=model(x)
        lo.retain_grad(); total,d,c=components(lo,y); total.backward()
        g=lo.grad.detach().abs(); fg=g[:,1:].mean().item(); bg=g[:,0].mean().item()
        mm=metrics(lo.detach(),y)
        rec.append(dict(output_grad_mean_abs=g.mean().item(),foreground_channel_grad_mean_abs=fg,
                        background_channel_grad_mean_abs=bg,foreground_background_grad_ratio=fg/(bg+1e-12),
                        **mm))
        model.zero_grad(set_to_none=True)
    return {k:float(np.nanmean([r[k] for r in rec])) for k in rec[0]}

@torch.no_grad()
def evaluate(model,cases):
    model.eval(); rows=[]
    for im,ma in cases:
        x=im[None,None].to(DEVICE); y=ma[None].to(DEVICE)
        with torch.amp.autocast("cuda",enabled=DEVICE.type=="cuda"): lo=model(x); t,d,c=components(lo,y)
        rows.append(dict(total_loss=t.item(),dice_loss=d.item(),ce_loss=c.item(),**metrics(lo,y)))
    return {k:float(np.nanmean([r[k] for r in rows])) for k in rows[0]}

def make_model(part11):
    m=part11.create_model(DEVICE); s=torch.load(INIT,map_location=DEVICE)
    if isinstance(s,dict) and "model_state_dict" in s: s=s["model_state_dict"]
    elif isinstance(s,dict) and "state_dict" in s: s=s["state_dict"]
    m.load_state_dict(s,strict=True); return m

def train_epoch(model,cases,opt):
    model.train(); vals=[]
    for i,(im,ma) in enumerate(cases):
        x=im[None,None].to(DEVICE); y=ma[None].to(DEVICE); opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=DEVICE.type=="cuda"): lo=model(x); t,d,c=components(lo,y)
        lo.retain_grad(); t.backward(retain_graph=True)
        g=lo.grad.detach().abs(); fg=g[:,1:].mean().item(); bg=g[:,0].mean().item()
        opt.zero_grad(set_to_none=True); t.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        vals.append(dict(train_total_loss=t.item(),train_dice_loss=d.item(),train_ce_loss=c.item(),
                         train_grad_ratio=fg/(bg+1e-12),train_fg_grad=fg,train_bg_grad=bg,**metrics(lo.detach(),y)))
        if i==0 or i+1==len(cases): print(f"  batch {i+1:03d}/{len(cases)} loss={t.item():.6f} FGDice={vals[-1]['fg_dice']:.6f} PredFG={vals[-1]['pred_fg']:.1f} GradRatio={vals[-1]['train_grad_ratio']:.6f}")
    return {k:float(np.nanmean([r[k] for r in vals])) for k in vals[0]}

def main():
    print("="*82); print("PART 45 PATH VALIDATION"); print("="*82)
    for label,p in [("Project root",ROOT),("Part 11",PART11_PATH),("Part 9",PART9_PATH),("Part 15 initialization checkpoint",INIT),("Part 15 train cohort",TRCSV),("Part 15 validation cohort",VACSV)]:
        print(f"{label:<42}: {'FOUND' if p.exists() else 'MISSING'}")
    required=[ROOT,PART11_PATH,PART9_PATH,INIT,TRCSV,VACSV]
    if not all(p.exists() for p in required): raise FileNotFoundError("Required Part 45 input is missing.")
    print("\n"+"="*82); print("PART 45 — TRAINING-DYNAMICS / GRADIENT-DIRECTION DIAGNOSTIC"); print("="*82)
    print(f"PyTorch : {torch.__version__}\nDevice : {DEVICE}\nFull volume : {FULL_SHAPE}\nTraining crop : {CROP_SHAPE}\nTrain subset : {TRAIN_N}\nValidation subset : {VAL_N}\nEpochs : {EPOCHS}\nLearning rate : {LR}\nWeight decay : {WD}\nCrop policy : foreground-centered\nLoss : original Part 15 DiceCE\nPart 15 overwritten : NO\nSPIDER used : NO\nTest set used : NO\nInitialization SHA256 : {sha256_file(INIT)}")
    part11=load_module(PART11_PATH,"part11_p45"); part9=load_module(PART9_PATH,"part9_p45")
    trdf=pd.read_csv(TRCSV); vadf=pd.read_csv(VACSV)
    train=preload([trdf.iloc[i] for i in range(min(TRAIN_N,len(trdf)))],part11,part9,"train",1000)
    val=preload([vadf.iloc[i] for i in range(min(VAL_N,len(vadf)))],part11,part9,"validation",5000)
    print("\nPART 45 SHAPE SMOKE TEST")
    for i in range(min(3,len(train))): print(f"Case {i+1}: image={tuple(train[i][0].shape)} mask={tuple(train[i][1].shape)} cropFG={(train[i][1]>0).sum().item()}")
    print("✓ Shape smoke test PASSED.")
    m=make_model(part11); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
    hist=[]
    pr=grad_probe(m,val); ev=evaluate(m,val); hist.append(dict(epoch=0,**ev,**{f"grad_{k}":v for k,v in pr.items()}))
    print(f"\nInitialization: loss={ev['total_loss']:.6f}, FGDice={ev['fg_dice']:.6f}, PredFG={ev['pred_fg']:.1f}, FGProb={ev['fg_probability']:.6f}, GradRatio={pr['foreground_background_grad_ratio']:.6f}, FGGrad={pr['foreground_channel_grad_mean_abs']:.3e}, BGGrad={pr['background_channel_grad_mean_abs']:.3e}")
    for ep in range(1,EPOCHS+1):
        ti=train_epoch(m,train,opt); ev=evaluate(m,val); pr=grad_probe(m,val)
        row=dict(epoch=ep,**ti,**{f"val_{k}":v for k,v in ev.items()},**{f"val_grad_{k}":v for k,v in pr.items()}); hist.append(row)
        torch.save({"epoch":ep,"model_state_dict":m.state_dict(),"optimizer_state_dict":opt.state_dict()},CKPT/f"epoch_{ep:02d}.pth")
        print(f"\nEpoch {ep:02d}/{EPOCHS}: train_loss={ti['train_total_loss']:.6f}, train_FGDice={ti['fg_dice']:.6f}, train_GradRatio={ti['train_grad_ratio']:.6f}, val_loss={ev['total_loss']:.6f}, val_FGDice={ev['fg_dice']:.6f}, PredFG={ev['pred_fg']:.1f}, FGProb={ev['fg_probability']:.6f}, GradRatio={pr['foreground_background_grad_ratio']:.6f}, FGGrad={pr['foreground_channel_grad_mean_abs']:.3e}, BGGrad={pr['background_channel_grad_mean_abs']:.3e}, empty={int(ev['empty']*VAL_N)}/{VAL_N}")
    fields=sorted({k for r in hist for k in r})
    with open(REPORT/"part45_history.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(hist)
    final=hist[-1]; collapse=next((r["epoch"] for r in hist[1:] if r.get("val_pred_fg",1)==0),None)
    init_ratio=hist[0]["grad_foreground_background_grad_ratio"]; final_ratio=final["val_grad_foreground_background_grad_ratio"]
    interp="FOREGROUND_GRADIENT_SIGNAL_WEAKENED_DURING_COLLAPSE" if collapse is not None and final_ratio<init_ratio else ("BACKGROUND_COLLAPSE_CONFIRMED_BUT_GRADIENT_RATIO_NOT_DECISIVE" if collapse is not None else "NO_COMPLETE_COLLAPSE_WITHIN_DIAGNOSTIC_WINDOW")
    summary=dict(part=45,device=str(DEVICE),train_subset=TRAIN_N,validation_subset=VAL_N,epochs=EPOCHS,crop_shape=CROP_SHAPE,initialization_sha256=sha256_file(INIT),initial_gradient_ratio=init_ratio,final_gradient_ratio=final_ratio,first_zero_pred_fg_epoch=collapse,final_val_fg_dice=final["val_fg_dice"],final_val_pred_fg=final["val_pred_fg"],final_val_fg_probability=final["val_fg_probability"],interpretation=interp)
    with open(REPORT/"part45_summary.json","w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
    print("\n"+"="*82); print("PART 45 DIAGNOSTIC SUMMARY"); print("="*82)
    print(f"Initial foreground/background grad ratio : {init_ratio:.6f}")
    print(f"Final foreground/background grad ratio   : {final_ratio:.6f}")
    print(f"First zero-PredFG epoch                  : {collapse}")
    print(f"Final validation FG Dice                 : {final['val_fg_dice']:.6f}")
    print(f"Final validation PredFG                  : {final['val_pred_fg']:.2f}")
    print(f"Final validation FG probability          : {final['val_fg_probability']:.6f}")
    print(f"Interpretation                           : {interp}")
    print(f"Summary saved to: {REPORT/'part45_summary.json'}")
    print("\nPART 45 COMPLETE")

if __name__=="__main__": main()
