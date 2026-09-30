# PART 44 — CONTROLLED LOSS-COMPONENT / CLASS-IMBALANCE DIAGNOSTIC
# This script is a diagnostic pilot. It preserves Part 15 and uses its exact
# initialization checkpoint, first 200 train rows, first 100 validation rows,
# foreground-centered crops, 5 epochs, AdamW 1e-4 / 1e-5.
#
# It compares original Dice+CE against Dice+foreground-aware weighted CE and
# records separate Dice loss, CE loss, foreground Dice, PredFG, foreground
# probability, and first-batch output-gradient magnitudes.

from pathlib import Path
import sys, json, hashlib, importlib.util, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.losses import DiceCELoss
from torch.cuda.amp import autocast, GradScaler

ROOT=Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC=ROOT/"src"
P11=SRC/"segmentation_rsna_part11_controlled_pilot_training.py"
P9=SRC/"segmentation_rsna_part9_3d_dataset_loader.py"
INIT=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"/"checkpoints"/"part15_initialization_from_part11.pth"
TRCSV=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"/"part15_train_cohort.csv"
VACSV=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"/"part15_validation_cohort.csv"

OUT=ROOT/"outputs"/"segmentation"/"rsna_part44_loss_component_class_imbalance_diagnostic"
BASE=OUT/"original_dicece"
AWARE=OUT/"foreground_aware_ce"
REPORT=OUT/"reports"

for x in (BASE,AWARE,REPORT):
    x.mkdir(parents=True,exist_ok=True)

SEED=4400; FULL=(64,96,96); CROP=(32,64,64); NTR=200; NVA=100; EPOCHS=5; LR=1e-4; WD=1e-5
WEIGHTS=[0.25,1.,1.,1.,1.,1.]
SHA="0900c0e6490fdaddf455763b465d4a607acfb310349c9f9ec82a61117016aec6"
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p11=None; p9=None

def imp(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s)
    sys.modules[name]=m; s.loader.exec_module(m); return m
def sh(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def rows(path,n):
    d=pd.read_csv(path); return [pd.Series(r) for _,r in d.iloc[:n].iterrows()]
def load(row):
    z=p11.load_tensor_case(row,p9); im,ma=z[0],z[1]
    im=torch.as_tensor(im,dtype=torch.float32); ma=torch.as_tensor(ma,dtype=torch.long)
    if im.ndim==4 and im.shape[0]==1: im=im.squeeze(0)
    if ma.ndim==4 and ma.shape[0]==1: ma=ma.squeeze(0)
    return im,ma
def center(im,ma):
    d,h,w=im.shape; cd,ch,cw=CROP; q=torch.nonzero(ma>0,as_tuple=False)
    if q.numel()==0: return random_crop(im,ma)
    z,y,x=q[random.randrange(q.shape[0])].tolist()
    a=max(0,min(int(z-cd//2),d-cd)); b=max(0,min(int(y-ch//2),h-ch)); c=max(0,min(int(x-cw//2),w-cw))
    return im[a:a+cd,b:b+ch,c:c+cw],ma[a:a+cd,b:b+ch,c:c+cw]
def random_crop(im,ma):
    d,h,w=im.shape; cd,ch,cw=CROP
    a=random.randint(0,d-cd); b=random.randint(0,h-ch); c=random.randint(0,w-cw)
    return im[a:a+cd,b:b+ch,c:c+cw],ma[a:a+cd,b:b+ch,c:c+cw]
class DS(Dataset):
    def __init__(self,rs,seed): self.rs=rs; self.seed=seed
    def __len__(self): return len(self.rs)
    def __getitem__(self,i):
        random.seed(self.seed+i*1009); im,ma=load(self.rs[i]); im,ma=center(im,ma)
        assert tuple(im.shape)==CROP and tuple(ma.shape)==CROP
        return {"image":im,"mask":ma}
def coll(b): return {"image":torch.stack([x["image"] for x in b]),"mask":torch.stack([x["mask"] for x in b])}
def model(): return p11.create_model(DEVICE)
class Losses(nn.Module):
    def __init__(self,aware):
        super().__init__(); self.d=DiceCELoss(to_onehot_y=True,softmax=True,lambda_dice=1.,lambda_ce=0.)
        self.w=torch.tensor(WEIGHTS,dtype=torch.float32) if aware else None
    def forward(self,lo,y):
        dl=self.d(lo,y)
        ce=F.cross_entropy(lo,y.squeeze(1),weight=self.w.to(lo.device) if self.w is not None else None)
        return dl+ce,dl,ce
@torch.no_grad()
def metrics(lo,y):
    pr=lo.argmax(1); fg=pr>0; tg=y>0; pf=int(fg.sum()); tf=int(tg.sum()); it=int((fg&tg).sum())
    fd=2*it/(pf+tf) if pf+tf else 1.; fp=float(torch.softmax(lo,1)[:,1:].sum(1).mean())
    return {"fg_dice":float(fd),"pred_fg":pf,"target_fg":tf,"fg_probability":fp}
@torch.no_grad()
def evaluate(m,rs,loss):
    m.eval(); vals=[]; empty=0
    dl=DataLoader(DS(rs,SEED+9000),1,False,collate_fn=coll,num_workers=0)
    for b in dl:
        x=b["image"].to(DEVICE).unsqueeze(1); y=b["mask"].to(DEVICE).unsqueeze(1)
        with autocast(enabled=DEVICE.type=="cuda"): lo=m(x); tot,di,ce=loss(lo,y)
        z=metrics(lo,y); empty+=z["pred_fg"]==0
        vals.append((tot.item(),di.item(),ce.item(),z))
    return {"total_loss":float(np.mean([x[0] for x in vals])),"dice_loss":float(np.mean([x[1] for x in vals])),
            "ce_loss":float(np.mean([x[2] for x in vals])),"fg_dice":float(np.mean([x[3]["fg_dice"] for x in vals])),
            "pred_fg":float(np.mean([x[3]["pred_fg"] for x in vals])),"target_fg":float(np.mean([x[3]["target_fg"] for x in vals])),
            "fg_probability":float(np.mean([x[3]["fg_probability"] for x in vals])),"empty":int(empty)}
def train(m,rs,loss,opt,scaler,epoch):
    m.train(); sums=[]; grad=None
    dl=DataLoader(DS(rs,SEED+epoch*100000),1,False,collate_fn=coll,num_workers=0)
    for i,b in enumerate(dl):
        x=b["image"].to(DEVICE).unsqueeze(1); y=b["mask"].to(DEVICE).unsqueeze(1); opt.zero_grad(set_to_none=True)
        with autocast(enabled=DEVICE.type=="cuda"): lo=m(x); tot,di,ce=loss(lo,y)
        if i==0:
            lo.retain_grad()
            if DEVICE.type=="cuda": scaler.scale(tot).backward(retain_graph=True); scaler.unscale_(opt)
            else: tot.backward(retain_graph=True)
            g=lo.grad.detach(); bg=float(g[:,0].abs().mean()); fg=float(g[:,1:].abs().mean())
            grad={"background_logit_grad":bg,"foreground_logit_grad":fg,"foreground_background_grad_ratio":fg/bg if bg else float("inf")}
            opt.zero_grad(set_to_none=True)
        if DEVICE.type=="cuda": scaler.scale(tot).backward(); scaler.step(opt); scaler.update()
        else: tot.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        z=metrics(lo.detach(),y); sums.append((tot.item(),di.item(),ce.item(),z))
    r={"total_loss":float(np.mean([x[0] for x in sums])),"dice_loss":float(np.mean([x[1] for x in sums])),
       "ce_loss":float(np.mean([x[2] for x in sums])),"fg_dice":float(np.mean([x[3]["fg_dice"] for x in sums])),
       "pred_fg":float(np.mean([x[3]["pred_fg"] for x in sums])),"target_fg":float(np.mean([x[3]["target_fg"] for x in sums])),
       "fg_probability":float(np.mean([x[3]["fg_probability"] for x in sums]))}
    r.update({"first_batch_"+k:v for k,v in grad.items()}); return r
def run(name,aware,tr,va,state,out):
    print("\n"+"="*82+"\n"+name+"\n"+"="*82)
    m=model(); m.load_state_dict(state,strict=True); loss=Losses(aware).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD); scaler=GradScaler(enabled=DEVICE.type=="cuda")
    ini=evaluate(m,va,loss); print(f"Initialization: total={ini['total_loss']:.6f}, DiceLoss={ini['dice_loss']:.6f}, CELoss={ini['ce_loss']:.6f}, FGDice={ini['fg_dice']:.6f}, PredFG={ini['pred_fg']:.1f}, FGProb={ini['fg_probability']:.6f}, empty={ini['empty']}/{len(va)}")
    hist=[{"epoch":0,"stage":"initialization","validation":ini}]
    for e in range(1,EPOCHS+1):
        t=time.time(); trr=train(m,tr,loss,opt,scaler,e); vaa=evaluate(m,va,loss); sec=time.time()-t
        hist.append({"epoch":e,"train":trr,"validation":vaa,"time_seconds":sec})
        print(f"Epoch {e:02d}/{EPOCHS}: train_total={trr['total_loss']:.6f}, train_DiceLoss={trr['dice_loss']:.6f}, train_CELoss={trr['ce_loss']:.6f}, train_FGDice={trr['fg_dice']:.6f}, val_total={vaa['total_loss']:.6f}, val_DiceLoss={vaa['dice_loss']:.6f}, val_CELoss={vaa['ce_loss']:.6f}, val_FGDice={vaa['fg_dice']:.6f}, PredFG={vaa['pred_fg']:.1f}, FGProb={vaa['fg_probability']:.6f}, empty={vaa['empty']}/{len(va)}, time={sec:.1f}s")
        torch.save({"model_state_dict":m.state_dict(),"epoch":e,"mode":name,"validation":vaa},out/f"epoch_{e:02d}.pth")
    r={"mode":name,"initialization":ini,"history":hist}; (out/"history.json").write_text(json.dumps(r,indent=2),encoding="utf-8"); return r
def main():
    global p11,p9
    print("="*82+"\nPART 44 PATH VALIDATION\n"+"="*82)
    for n,x in [("Project root",ROOT),("Part 11",P11),("Part 9",P9),("Part 15 initialization checkpoint",INIT),("Part 15 train cohort",TRCSV),("Part 15 validation cohort",VACSV)]:
        print(f"{n:<42}: {'FOUND' if x.exists() else 'MISSING'}")
    if any(not x.exists() for _,x in [("r",ROOT),("p11",P11),("p9",P9),("i",INIT),("tr",TRCSV),("va",VACSV)]): raise FileNotFoundError("Required Part 44 input missing.")
    actual=sh(INIT); assert actual==SHA,(actual,SHA)
    p11=imp("segmentation_rsna_part11_controlled_pilot_training",P11); p9=imp("segmentation_rsna_part9_3d_dataset_loader",P9)
    tr=rows(TRCSV,500); va=rows(VACSV,100); tr=tr[:NTR]; va=va[:NVA]
    print("\nPART 44 — CONTROLLED LOSS-COMPONENT / CLASS-IMBALANCE DIAGNOSTIC")
    print(f"PyTorch : {torch.__version__}\nDevice : {DEVICE}\nFull volume : {FULL}\nTraining crop : {CROP}\nTrain subset : {len(tr)}\nValidation subset : {len(va)}\nEpochs : {EPOCHS}\nLearning rate : {LR}\nWeight decay : {WD}\nCrop policy : foreground-centered\nPart 15 overwritten : NO\nSPIDER used : NO\nTest set used : NO\nInitialization SHA256 : {actual}\nForeground-aware CE weights : {WEIGHTS}")
    fgs=[]; cgs=[]
    for i,r in enumerate(tr):
        im,ma=load(r); _,cm=center(im,ma); fgs.append(int((ma>0).sum())); cgs.append(int((cm>0).sum()))
        if i in (0,49,99,149,199): print(f"COVERAGE {i+1:03d}/{len(tr)} sourceFG={fgs[-1]} centeredFG={cgs[-1]}")
    print(f"Mean source foreground voxels : {np.mean(fgs):.2f}\nMean centered crop FG : {np.mean(cgs):.2f}\nCentered zero-FG crops : {sum(x==0 for x in cgs)}/{len(cgs)}")
    print("\nPART 44 SHAPE SMOKE TEST")
    for i,r in enumerate(tr[:3]):
        im,ma=load(r); ci,cm=center(im,ma); print(f"Case {i+1}: full={tuple(im.shape)} crop={tuple(ci.shape)} cropFG={int((cm>0).sum())}"); assert tuple(ci.shape)==CROP
    print("✓ Shape smoke test PASSED.")
    st=torch.load(INIT,map_location="cpu"); state=st["model_state_dict"] if isinstance(st,dict) and "model_state_dict" in st else st
    b=run("ORIGINAL DICECE LOSS",False,tr,va,state,BASE)
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    a=run("FOREGROUND-AWARE CE WEIGHTING",True,tr,va,state,AWARE)
    bf=b["history"][-1]["validation"]; af=a["history"][-1]["validation"]
    dd=af["fg_dice"]-bf["fg_dice"]
    interp="LOSS_WEIGHTING_CHANGED_FOREGROUND_BEHAVIOR" if abs(dd)>=.03 or abs(af["pred_fg"]-bf["pred_fg"])>=100 else "NO_LARGE_FOREGROUND_BEHAVIOR_DIFFERENCE"
    print("\n"+"="*82+"\nPART 44 CONTROLLED COMPARISON\n"+"="*82)
    for k in ["total_loss","dice_loss","ce_loss","fg_dice","pred_fg","fg_probability","empty"]: print(f"{k:40}: original={bf[k]:.7f} aware={af[k]:.7f}")
    print(f"FG-aware minus original FG Dice       : {dd:+.7f}\nInterpretation                         : {interp}")
    summary={"part":44,"initialization_sha256":actual,"train_rows":len(tr),"validation_rows":len(va),"epochs":EPOCHS,"crop_shape":CROP,"ce_weights":WEIGHTS,"coverage":{"mean_source_fg":float(np.mean(fgs)),"mean_centered_crop_fg":float(np.mean(cgs))},"original":b,"foreground_aware":a,"interpretation":interp}
    path=REPORT/"part44_summary.json"; path.write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(f"Summary saved to: {path}\n\nPART 44 COMPLETE")
if __name__=="__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); main()
