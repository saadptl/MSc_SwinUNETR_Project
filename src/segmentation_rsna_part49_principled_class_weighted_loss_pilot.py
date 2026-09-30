
from pathlib import Path
import sys, csv, json, hashlib, random, importlib.util
import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss

ROOT=Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC=ROOT/"src"
PART11_PATH=SRC/"segmentation_rsna_part11_controlled_pilot_training.py"
PART9_PATH=SRC/"segmentation_rsna_part9_3d_dataset_loader.py"
P15=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"
INIT=P15/"checkpoints"/"part15_initialization_from_part11.pth"
TRCSV=P15/"part15_train_cohort.csv"
VACSV=P15/"part15_validation_cohort.csv"
OUT=ROOT/"outputs"/"segmentation"/"rsna_part49_principled_class_weighted_loss_pilot"
BASE=OUT/"baseline_dicece"
WEIGHTED=OUT/"empirical_weighted_ce"
REPORT=OUT/"reports"
for p in (OUT,BASE,WEIGHTED,REPORT): p.mkdir(parents=True,exist_ok=True)

TRAIN_N=100; VAL_N=50; EPOCHS=3
FULL_SHAPE=(64,96,96); CROP_SHAPE=(32,64,64)
LR=1e-4; WD=1e-5; SEED=149
NUM_CLASSES=6
FOREGROUND_CLASSES=[1,2,3,4,5]
CLASS_NAMES={0:"Background",1:"Spinal_Canal_Stenosis",2:"Left_Neural_Foraminal_Narrowing",3:"Right_Neural_Foraminal_Narrowing",4:"Left_Subarticular_Stenosis",5:"Right_Subarticular_Stenosis"}
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def centered_crop(image,mask,shape,seed):
    D,H,W=image.shape; cd,ch,cw=shape; rng=np.random.default_rng(seed)
    fg=torch.nonzero(mask>0,as_tuple=False)
    if len(fg):
        p=fg[int(rng.integers(0,len(fg)))]
        c=[int(p[0])+int(rng.integers(-cd//8,cd//8+1)),int(p[1])+int(rng.integers(-ch//8,ch//8+1)),int(p[2])+int(rng.integers(-cw//8,cw//8+1))]
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

def preload(df,n,part11,part9,label,offset):
    cases=[]
    n=min(n,len(df))
    for i in range(n):
        cases.append(load_case(df.iloc[i],part11,part9,SEED+offset+i))
        if i==0 or (i+1)%25==0 or i+1==n:
            print(f"{label.upper()} {i+1:03d}/{n} FG={(cases[-1][1]>0).sum().item()}")
    fgs=[(m>0).sum().item() for _,m in cases]
    print(f"Mean {label} crop foreground voxels : {np.mean(fgs):.2f}")
    print(f"{label.capitalize()} zero-FG crops : {sum(x==0 for x in fgs)}/{len(fgs)}")
    return cases

def compute_class_weights(cases):
    counts=np.zeros(NUM_CLASSES,dtype=np.int64)
    for _,mask in cases:
        vals,freq=torch.unique(mask,return_counts=True)
        for v,n in zip(vals.tolist(),freq.tolist()):
            if 0<=int(v)<NUM_CLASSES: counts[int(v)]+=int(n)
    frequencies=counts/max(1,counts.sum())
    raw=1.0/np.sqrt(np.maximum(frequencies,1e-12))
    fg_mean=np.mean(raw[FOREGROUND_CLASSES])
    weights=raw/fg_mean
    weights[0]=min(weights[0],0.25)
    weights[FOREGROUND_CLASSES]=np.clip(weights[FOREGROUND_CLASSES],0.5,3.0)
    return counts,frequencies,weights

def make_model(part11):
    m=part11.create_model(DEVICE)
    s=torch.load(INIT,map_location=DEVICE)
    if isinstance(s,dict) and "model_state_dict" in s: s=s["model_state_dict"]
    elif isinstance(s,dict) and "state_dict" in s: s=s["state_dict"]
    m.load_state_dict(s,strict=True)
    return m

def build_losses(weights=None):
    dice=DiceCELoss(to_onehot_y=True,softmax=True,lambda_dice=1.0,lambda_ce=0.0)
    ce=torch.nn.CrossEntropyLoss() if weights is None else torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=DEVICE))
    return dice,ce

def metrics(logits,target):
    with torch.no_grad():
        p=torch.softmax(logits,1); pred=p.argmax(1); pf=pred>0; tf=target>0
        tp=(pf&tf).sum().item(); pc=pf.sum().item(); tc=tf.sum().item()
        r={"fg_dice":float(2*tp/(pc+tc) if pc+tc else 1.0),"pred_fg":float(pc),"target_fg":float(tc),"fg_probability":float(p[:,1:].sum(1).mean().item()),"background_probability":float(p[:,0].mean().item()),"empty":float(pc==0)}
        for c in range(NUM_CLASSES):
            tc_mask=target==c; pc_mask=pred==c; t=tc_mask.sum().item(); q=pc_mask.sum().item(); tp_c=(tc_mask&pc_mask).sum().item()
            r[f"class_{c}_target"]=float(t); r[f"class_{c}_pred"]=float(q); r[f"class_{c}_dice"]=float(2*tp_c/(t+q) if t+q else 1.0); r[f"class_{c}_prob"]=float(p[:,c].mean().item())
        return r

def evaluate(model,cases,dice,ce):
    model.eval(); rec=[]
    for im,ma in cases:
        x=im[None,None].to(DEVICE); y=ma[None].to(DEVICE)
        with torch.amp.autocast("cuda",enabled=DEVICE.type=="cuda"):
            lo=model(x); dl=dice(lo,y.unsqueeze(1)); cl=ce(lo,y); total=dl+cl
        rec.append({"loss":total.item(),"dice_loss":dl.item(),"ce_loss":cl.item(),**metrics(lo,y)})
    return {k:float(np.nanmean([r[k] for r in rec])) for k in rec[0]}

def train_epoch(model,cases,opt,dice,ce):
    model.train(); rec=[]
    for i,(im,ma) in enumerate(cases):
        x=im[None,None].to(DEVICE); y=ma[None].to(DEVICE); opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=DEVICE.type=="cuda"):
            lo=model(x); dl=dice(lo,y.unsqueeze(1)); cl=ce(lo,y); total=dl+cl
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        rec.append({"loss":total.item(),"dice_loss":dl.item(),"ce_loss":cl.item(),**metrics(lo.detach(),y)})
        if i==0 or i+1==len(cases): print(f"  batch {i+1:03d}/{len(cases)} loss={total.item():.6f} FGDice={rec[-1]['fg_dice']:.6f} PredFG={rec[-1]['pred_fg']:.1f}")
    return {k:float(np.nanmean([r[k] for r in rec])) for k in rec[0]}

def run(label,outdir,part11,train_cases,val_cases,weights):
    print("\n"+"="*82); print(label); print("="*82)
    model=make_model(part11); dice,ce=build_losses(weights); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    hist=[]; initial=evaluate(model,val_cases,dice,ce)
    print(f"Initialization: loss={initial['loss']:.6f} FGDice={initial['fg_dice']:.6f} PredFG={initial['pred_fg']:.1f} empty={int(initial['empty'])}/{len(val_cases)}")
    hist.append({"epoch":0,"phase":"initialization",**{f"val_{k}":v for k,v in initial.items()}})
    for ep in range(1,EPOCHS+1):
        tr=train_epoch(model,train_cases,opt,dice,ce); va=evaluate(model,val_cases,dice,ce)
        hist.append({"epoch":ep,"phase":"epoch",**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()}})
        torch.save({"epoch":ep,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"class_weights":weights},outdir/f"epoch_{ep:02d}.pth")
        print(f"\nEpoch {ep:02d}/{EPOCHS}: train_loss={tr['loss']:.6f}, train_FGDice={tr['fg_dice']:.6f}, val_loss={va['loss']:.6f}, val_FGDice={va['fg_dice']:.6f}, PredFG={va['pred_fg']:.1f}, FGProb={va['fg_probability']:.6f}, empty={int(va['empty'])}/{len(val_cases)}")
    with (outdir/"history.csv").open("w",newline="",encoding="utf-8") as f:
        fields=sorted({k for r in hist for k in r}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(hist)
    return hist

def main():
    print("="*82); print("PART 49 PATH VALIDATION"); print("="*82)
    for label,p in [("Project root",ROOT),("Part 11",PART11_PATH),("Part 9",PART9_PATH),("Part 15 initialization",INIT),("Part 15 train cohort",TRCSV),("Part 15 validation cohort",VACSV)]:
        print(f"{label:<42}: {'FOUND' if p.exists() else 'MISSING'}")
    if not all(p.exists() for p in [ROOT,PART11_PATH,PART9_PATH,INIT,TRCSV,VACSV]): raise FileNotFoundError("Required Part 49 input is missing.")
    print("\n"+"="*82); print("PART 49 — PRINCIPLED CLASS-WEIGHTED LOSS PILOT"); print("="*82)
    print(f"PyTorch : {torch.__version__}\nDevice : {DEVICE}\nTrain subset : {TRAIN_N}\nValidation subset : {VAL_N}\nEpochs : {EPOCHS}\nFull volume : {FULL_SHAPE}\nCrop : {CROP_SHAPE}\nLearning rate : {LR}\nWeight decay : {WD}\nCrop policy : foreground-centered\nCondition A : original Dice + unweighted CE\nCondition B : Dice + empirical weighted CE\nWeight method : inverse square-root class frequency\nPart 15 overwritten : NO\nSPIDER used : NO\nTest set used : NO\nInitialization SHA256 : {sha256_file(INIT)}")
    part11=load_module(PART11_PATH,"part11_part49"); part9=load_module(PART9_PATH,"part9_part49")
    tr=pd.read_csv(TRCSV); va=pd.read_csv(VACSV)
    print("\nPART 49 TRAIN DATA PRELOAD"); print("-"*82)
    train=preload(tr,TRAIN_N,part11,part9,"train",1000)
    print("\nPART 49 VALIDATION DATA PRELOAD"); print("-"*82)
    val=preload(va,VAL_N,part11,part9,"validation",5000)
    print("\nPART 49 SHAPE SMOKE TEST")
    for i in range(min(3,len(train))):
        print(f"Case {i+1}: image={tuple(train[i][0].shape)} mask={tuple(train[i][1].shape)} FG={(train[i][1]>0).sum().item()}")
    print("✓ Shape smoke test PASSED.")
    print("\n"+"="*82); print("PART 49 EMPIRICAL CLASS FREQUENCY CALCULATION"); print("="*82)
    counts,freq,weights=compute_class_weights(train)
    for c in range(NUM_CLASSES): print(f"Class {c} {CLASS_NAMES[c]:<42} voxels={counts[c]:>10d} freq={freq[c]:.10f} weight={weights[c]:.6f}")
    with (REPORT/"part49_class_weights.json").open("w",encoding="utf-8") as f: json.dump({"method":"inverse_square_root_frequency","normalization":"mean_foreground_weight=1","foreground_clip":[0.5,3.0],"background_cap":0.25,"class_counts":{str(i):int(counts[i]) for i in range(NUM_CLASSES)},"class_frequencies":{str(i):float(freq[i]) for i in range(NUM_CLASSES)},"class_weights":{str(i):float(weights[i]) for i in range(NUM_CLASSES)}},f,indent=2)
    a=make_model(part11); b=make_model(part11)
    same=all(torch.equal(pa,pb) for pa,pb in zip(a.parameters(),b.parameters()))
    print("\n"+"="*82); print("PART 49 CONTROL CHECK"); print("="*82); print(f"Identical model initialization : {same}")
    if not same: raise RuntimeError("Initialization control failed.")
    del a,b
    ha=run("CONDITION A — BASELINE ORIGINAL DICECE",BASE,part11,train,val,None)
    hb=run("CONDITION B — EMPIRICAL CLASS-WEIGHTED CE",WEIGHTED,part11,train,val,weights.tolist())
    a,b=ha[-1],hb[-1]
    comp={"baseline_final_val_loss":a["val_loss"],"weighted_final_val_loss":b["val_loss"],"baseline_final_fg_dice":a["val_fg_dice"],"weighted_final_fg_dice":b["val_fg_dice"],"fg_dice_delta_weighted_minus_baseline":b["val_fg_dice"]-a["val_fg_dice"],"baseline_final_pred_fg":a["val_pred_fg"],"weighted_final_pred_fg":b["val_pred_fg"],"baseline_final_fg_probability":a["val_fg_probability"],"weighted_final_fg_probability":b["val_fg_probability"],"baseline_empty_validation_cases":int(a["val_empty"]),"weighted_empty_validation_cases":int(b["val_empty"]),"interpretation":"WEIGHTED_LOSS_IMPROVED_FOREGROUND_BEHAVIOR" if b["val_fg_dice"]>a["val_fg_dice"]+0.01 else "WEIGHTED_LOSS_DID_NOT_PRODUCE_MEANINGFUL_FOREGROUND_DICE_GAIN"}
    cp=REPORT/"part49_comparison.json"
    with cp.open("w",encoding="utf-8") as f: json.dump(comp,f,indent=2)
    print("\n"+"="*82); print("PART 49 COMPARISON"); print("="*82)
    print(f"Baseline final val loss : {comp['baseline_final_val_loss']:.6f}\nWeighted final val loss : {comp['weighted_final_val_loss']:.6f}\nBaseline final FG Dice : {comp['baseline_final_fg_dice']:.6f}\nWeighted final FG Dice : {comp['weighted_final_fg_dice']:.6f}\nFG Dice delta : {comp['fg_dice_delta_weighted_minus_baseline']:+.6f}\nBaseline PredFG : {comp['baseline_final_pred_fg']:.1f}\nWeighted PredFG : {comp['weighted_final_pred_fg']:.1f}\nBaseline FG probability : {comp['baseline_final_fg_probability']:.6f}\nWeighted FG probability : {comp['weighted_final_fg_probability']:.6f}\nBaseline empty cases : {comp['baseline_empty_validation_cases']}/{VAL_N}\nWeighted empty cases : {comp['weighted_empty_validation_cases']}/{VAL_N}\nInterpretation : {comp['interpretation']}")
    print("\n"+"="*82); print("PART 49 COMPLETE"); print("="*82)
    print(f"Class weights : {REPORT/'part49_class_weights.json'}"); print(f"Comparison : {cp}"); print(f"Baseline history : {BASE/'history.csv'}"); print(f"Weighted history : {WEIGHTED/'history.csv'}")

if __name__=="__main__": main()
