from __future__ import annotations
import gc, hashlib, importlib.util, json, random, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss

ROOT=Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
PART11=ROOT/"src/segmentation_rsna_part11_controlled_pilot_training.py"
PART9=ROOT/"src/segmentation_rsna_part9_3d_dataset_loader.py"
P15=ROOT/"outputs/segmentation/rsna_part15_extended_controlled_training"
INIT=P15/"checkpoints/part15_initialization_from_part11.pth"
TRCSV=P15/"part15_train_cohort.csv"; VACSV=P15/"part15_validation_cohort.csv"
RSNA=ROOT/"dataset/rsna-2024-lumbar-spine-degenerative-classification"
COORD=RSNA/"train_label_coordinates.csv"
OUT=ROOT/"outputs/segmentation/rsna_part64_point_supervised_auxiliary_localization_pilot"
BASE=OUT/"baseline_dicece"; AUX=OUT/"point_auxiliary"; REPORT=OUT/"reports"
TRAIN_N=100; VAL_N=50; EPOCHS=3; FULL=(64,96,96); CROP=(32,64,64); RADIUS=2
LR=1e-4; WD=1e-5; SEED=6401; POINT_LAMBDA=.20; NEG_PER_POINT=2
CLASSES={1:"Spinal_Canal_Stenosis",2:"Left_Neural_Foraminal_Narrowing",3:"Right_Neural_Foraminal_Narrowing",4:"Left_Subarticular_Stenosis",5:"Right_Subarticular_Stenosis"}
NAME_TO_ID={v:k for k,v in CLASSES.items()}
NAME_TO_ID.update({"Spinal Canal Stenosis":1,"Left Neural Foraminal Narrowing":2,"Right Neural Foraminal Narrowing":3,"Left Subarticular Stenosis":4,"Right Subarticular Stenosis":5})

def banner(s): print("\n"+"="*82+"\n"+s+"\n"+"="*82)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def mod(p,n):
    sp=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def dil6(a,n):
    a=a.astype(bool).copy()
    for _ in range(n):
        b=a.copy(); b[1:]|=a[:-1]; b[:-1]|=a[1:]; b[:,1:]|=a[:,:-1]; b[:,:-1]|=a[:,1:]; b[:,:,1:]|=a[:,:,:-1]; b[:,:,:-1]|=a[:,:,1:]; a=b
    return a
def dilate(m,n):
    if n==0:return m.astype(np.int64).copy()
    parts=[]
    for c in range(1,6):
        q=m==c
        if q.any(): parts.append((int(q.sum()),c,dil6(q,n)))
    parts.sort(key=lambda x:(-x[0],x[1])); o=np.zeros_like(m,dtype=np.int64); used=np.zeros_like(m,bool)
    for _,c,q in parts: q=q&~used; o[q]=c; used|=q
    return o
def crop(im,m):
    q=np.argwhere(m>0); ctr=np.round(q.mean(0)).astype(int) if q.size else np.array([32,48,48])
    st=[max(0,min(int(ctr[i])-CROP[i]//2,FULL[i]-CROP[i])) for i in range(3)]
    z,y,x=st; dz,dy,dx=CROP
    return im[z:z+dz,y:y+dy,x:x+dx].astype(np.float32),m[z:z+dz,y:y+dy,x:x+dx].astype(np.int64),tuple(st)
def rzcoord(x,y,z,native):
    nz,nh,nw=map(float,native); tz,th,tw=map(float,FULL)
    return ((z+.5)*tz/nz-.5,(y+.5)*th/nh-.5,(x+.5)*tw/nw-.5)
def load_case(p11,p9,row):
    im,m,_=p11.load_case_robust(row,p9); im=np.asarray(im,np.float32); m=np.asarray(m,np.int64)
    im=p11.resize_3d(im,FULL,is_mask=False); m=p11.resize_3d(m,FULL,is_mask=True); m=dilate(m,RADIUS)
    im,m,st=crop(im,m)
    return {"image":im,"mask":m,"crop_start":st,"native_shape":tuple(np.asarray(im).shape) if False else tuple(map(int,p11.load_case_robust(row,p9)[0].shape)),"study_id":str(row["study_id"]),"series_id":str(row["series_id"])}
def load_one(p11,p9,row):
    im0,m0,_=p11.load_case_robust(row,p9); im0=np.asarray(im0,np.float32); m0=np.asarray(m0,np.int64)
    native=tuple(map(int,im0.shape)); im=p11.resize_3d(im0,FULL,is_mask=False); m=p11.resize_3d(m0,FULL,is_mask=True); m=dilate(m,RADIUS); im,m,st=crop(im,m)
    return {"image":im,"mask":m,"crop_start":st,"native_shape":native,"study_id":str(row["study_id"]),"series_id":str(row["series_id"])}
def points_for(p11,p9,row,case,coord):
    sid=case["study_id"]; ser=case["series_id"]; a=coord[(coord.study_id.astype(str)==sid)&(coord.series_id.astype(str)==ser)]
    if a.empty:return []
    _,ds,_=p11.read_dicom_series_robust(p11.resolve_series_dir(row))
    iz={int(d.get("InstanceNumber",0)):i for i,d in enumerate(ds)}; out=[]
    for _,r in a.iterrows():
        try: x=float(r.x); y=float(r.y); inst=int(float(r.instance_number))
        except: continue
        cid=NAME_TO_ID.get(str(r.condition).strip())
        if cid is None or inst not in iz: continue
        zf,yf,xf=rzcoord(x,y,iz[inst],case["native_shape"]); cz,cy,cx=case["crop_start"]; z,y,x=round(zf-cz),round(yf-cy),round(xf-cx)
        if 0<=z<CROP[0] and 0<=y<CROP[1] and 0<=x<CROP[2]: out.append({"class_id":cid,"z":int(z),"y":int(y),"x":int(x),"level":str(r.get("level",""))})
    return out
def preload(p11,p9,df,coord,label):
    out=[]
    for i,(_,row) in enumerate(df.iterrows(),1):
        c=load_one(p11,p9,row); c["index"]=i; c["points"]=points_for(p11,p9,row,c,coord); out.append(c)
        if i in (1,25,50,75,100): print(f"{label} {i:03d}/{len(df)} FG={(c['mask']>0).sum()} points={len(c['points'])}")
    return out
def point_loss(logits,pts,negs):
    if not pts:return logits.sum()*0
    lp=F.log_softmax(logits,1)[0]; terms=[-lp[int(p["class_id"]),int(p["z"]),int(p["y"]),int(p["x"])] for p in pts]
    terms += [-lp[0,z,y,x] for z,y,x in negs]
    return torch.stack(terms).mean()
def negatives(pts,mask,rng):
    used={(p["z"],p["y"],p["x"]) for p in pts}; out=[]; target=NEG_PER_POINT*len(pts)
    for _ in range(target*40+1):
        if len(out)>=target:break
        z=int(rng.integers(CROP[0])); y=int(rng.integers(CROP[1])); x=int(rng.integers(CROP[2]))
        if mask[z,y,x]>0 or (z,y,x) in used:continue
        used.add((z,y,x)); out.append((z,y,x))
    return out
@torch.no_grad()
def evaluate(m,cases,dev):
    m.eval(); lf=fd=pf=tf=fp=0.; empty=0; fn=DiceCELoss(to_onehot_y=True,softmax=True)
    for c in cases:
        x=torch.from_numpy(c["image"])[None,None].to(dev); y=torch.from_numpy(c["mask"])[None,None].long().to(dev); z=m(x); loss=fn(z,y); pr=torch.softmax(z,1); pred=pr.argmax(1)[0].cpu().numpy(); t=c["mask"]>0; p=pred>0; den=t.sum()+p.sum(); d=2*(t&p).sum()/den if den else 1.; lf+=loss.item(); fd+=d; pf+=p.sum(); tf+=t.sum(); fp+=pr[:,1:].sum(1).mean().item(); empty+=int(p.sum()==0)
        del x,y,z,pr
    n=len(cases); return {"loss":lf/n,"fg_dice":fd/n,"pred_fg":pf/n,"target_fg":tf/n,"fg_probability":fp/n,"empty":empty,"total":n}
def train(cond,out,train,val,p11,dev,use_aux):
    seed(SEED); m=p11.create_model(dev); m.to(dev); state=torch.load(INIT,map_location=dev,weights_only=False); state=state.get("model_state_dict",state.get("state_dict",state)); state={k[7:] if k.startswith("module.") else k:v for k,v in state.items()}; m.load_state_dict(state,strict=False)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD); dicece=DiceCELoss(to_onehot_y=True,softmax=True); rng=np.random.default_rng(SEED); hist=[]; v=evaluate(m,val,dev)
    print(f"INIT | val_loss={v['loss']:.6f} val_FGDice={v['fg_dice']:.6f} PredFG={v['pred_fg']:.1f} FGProb={v['fg_probability']:.6f} Empty={v['empty']}/{v['total']}")
    hist.append({"condition":cond,"epoch":0,"lr":LR,"train_total_loss":None,"train_dicece_loss":None,"train_point_loss":None,"train_fg_dice":None,**{f"val_{k}":v[k] for k in ("loss","fg_dice","pred_fg","target_fg","fg_probability","empty")}})
    for ep in range(1,EPOCHS+1):
        t=time.time(); m.train(); rt=rd=rp=rg=0.; order=np.arange(len(train)); rng.shuffle(order)
        for j in order:
            c=train[int(j)]; x=torch.from_numpy(c["image"])[None,None].to(dev); y=torch.from_numpy(c["mask"])[None,None].long().to(dev); opt.zero_grad(set_to_none=True); z=m(x); dl=dicece(z,y); pl=point_loss(z,c["points"],negatives(c["points"],c["mask"],rng)) if use_aux else z.sum()*0.; total=dl+POINT_LAMBDA*pl; total.backward(); opt.step()
            with torch.no_grad():
                pr=z.argmax(1)[0].cpu().numpy(); a=c["mask"]>0; b=pr>0; den=a.sum()+b.sum(); d=2*(a&b).sum()/den if den else 1.
            rt+=total.item(); rd+=dl.item(); rp+=pl.item(); rg+=d; del x,y,z,dl,pl,total
        v=evaluate(m,val,dev); sec=time.time()-t
        print(f"Epoch {ep:02d} | train_total={rt/len(train):.6f} train_DiceCE={rd/len(train):.6f} train_Point={rp/len(train):.6f} train_FGDice={rg/len(train):.6f} | val_loss={v['loss']:.6f} val_FGDice={v['fg_dice']:.6f} | PredFG={v['pred_fg']:.1f} TargetFG={v['target_fg']:.1f} FGProb={v['fg_probability']:.6f} Empty={v['empty']}/{v['total']} | {sec:.1f}s")
        out.mkdir(parents=True,exist_ok=True); torch.save({"epoch":ep,"model_state_dict":m.state_dict(),"optimizer_state_dict":opt.state_dict(),"condition":cond,"point_lambda":POINT_LAMBDA},out/f"epoch_{ep:02d}.pth")
        hist.append({"condition":cond,"epoch":ep,"lr":LR,"train_total_loss":rt/len(train),"train_dicece_loss":rd/len(train),"train_point_loss":rp/len(train),"train_fg_dice":rg/len(train),**{f"val_{k}":v[k] for k in ("loss","fg_dice","pred_fg","target_fg","fg_probability","empty")},"elapsed_seconds":sec})
    pd.DataFrame(hist).to_csv(out/"history.csv",index=False); del m,opt; gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return hist
def main():
    banner("PART 64 PATH VALIDATION")
    for n,q in [("Project root",ROOT),("Part 11",PART11),("Part 9",PART9),("Part 15 initialization",INIT),("Part 15 train cohort",TRCSV),("Part 15 validation cohort",VACSV),("RSNA coordinate CSV",COORD)]:
        print(f"{n:<40}: {'FOUND' if q.exists() else 'MISSING'}")
        if not q.exists():raise FileNotFoundError(q)
    banner("PART 64 — POINT-SUPERVISED AUXILIARY LOCALIZATION PILOT")
    print(f"Train subset : {TRAIN_N}\nValidation subset : {VAL_N}\nEpochs : {EPOCHS}\nFull volume : {FULL}\nCrop : {CROP}\nPseudo-mask radius : R{RADIUS}\nCrop strategy : foreground-centered\nA : original Part 15 DiceCELoss\nB : DiceCELoss + point auxiliary loss\nPoint auxiliary lambda : {POINT_LAMBDA}\nNegative points / positive : {NEG_PER_POINT}\nLR : {LR}\nWD : {WD}\nInitialization SHA256 : {sha(INIT)}\nSPIDER : NO\nTest set : NO\nPart 15 overwritten : NO")
    p11=mod(PART11,"p11_64"); p9=mod(PART9,"p9_64"); trdf=pd.read_csv(TRCSV).head(TRAIN_N); vadf=pd.read_csv(VACSV).head(VAL_N); coord=pd.read_csv(COORD)
    tr=preload(p11,p9,trdf,coord,"TRAIN"); va=preload(p11,p9,vadf,coord,"VALIDATION")
    banner("PART 64 SHAPE / POINT SMOKE TEST"); print(f"Image : {tr[0]['image'].shape}\nMask : {tr[0]['mask'].shape}\nLabels : {np.unique(tr[0]['mask']).tolist()}\nMapped points : {len(tr[0]['points'])}")
    assert tr[0]["image"].shape==CROP and tr[0]["mask"].shape==CROP and sum(len(c["points"]) for c in tr)>0 and sum(len(c["points"]) for c in va)>0
    print("✓ Shape / point smoke test PASSED.")
    banner("PART 64 POINT COVERAGE BY CLASS")
    for cid,name in CLASSES.items():
        a=sum(1 for c in tr for p in c["points"] if p["class_id"]==cid); b=sum(1 for c in va for p in c["points"] if p["class_id"]==cid); print(f"C{cid} {name:<36} train={a} val={b}")
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(f"\nPyTorch : {torch.__version__}\nDevice : {dev}")
    banner("PART 64 CONDITION A: ORIGINAL DICECE"); ha=train("A_original_dicece",BASE,tr,va,p11,dev,False)
    banner("PART 64 CONDITION B: DICECE + POINT AUXILIARY"); hb=train("B_dicece_point_auxiliary",AUX,tr,va,p11,dev,True)
    aa=ha[-1]; bb=hb[-1]; ba=max(ha[1:],key=lambda x:x["val_fg_dice"]); bc=max(hb[1:],key=lambda x:x["val_fg_dice"]); delta=bb["val_fg_dice"]-aa["val_fg_dice"]; bestdelta=bc["val_fg_dice"]-ba["val_fg_dice"]
    diagnosis="POINT_AUXILIARY_SUPERVISION_PRODUCED_MEANINGFUL_FOREGROUND_DICE_GAIN" if bestdelta>=.02 else ("POINT_AUXILIARY_SUPERVISION_SHOWED_SMALL_FOREGROUND_GAIN" if bestdelta>=.005 else "POINT_AUXILIARY_SUPERVISION_DID_NOT_MEANINGFULLY_IMPROVE_FOREGROUND_DICE")
    banner("PART 64 FINAL COMPARISON")
    print(f"A_original_dicece        | Final val_loss={aa['val_loss']:.6f} FGDice={aa['val_fg_dice']:.6f} PredFG={aa['val_pred_fg']:.1f} FGProb={aa['val_fg_probability']:.6f} Empty={aa['val_empty']}/{VAL_N}")
    print(f"B_dicece_point_auxiliary | Final val_loss={bb['val_loss']:.6f} FGDice={bb['val_fg_dice']:.6f} PredFG={bb['val_pred_fg']:.1f} FGProb={bb['val_fg_probability']:.6f} Empty={bb['val_empty']}/{VAL_N}")
    print(f"\nFinal Dice delta (B-A) : {delta:+.6f}\nBest A epoch : {ba['epoch']}\nBest A FGDice : {ba['val_fg_dice']:.6f}\nBest B epoch : {bc['epoch']}\nBest B FGDice : {bc['val_fg_dice']:.6f}\nBest Dice delta (B-A) : {bestdelta:+.6f}\nDiagnosis : {diagnosis}")
    REPORT.mkdir(parents=True,exist_ok=True); pd.DataFrame(ha+hb).to_csv(REPORT/"part64_training_trajectory.csv",index=False)
    pd.DataFrame([{"class_id":cid,"class_name":name,"train_points":sum(1 for c in tr for p in c["points"] if p["class_id"]==cid),"val_points":sum(1 for c in va for p in c["points"] if p["class_id"]==cid)} for cid,name in CLASSES.items()]).to_csv(REPORT/"part64_point_coverage_by_class.csv",index=False)
    with open(REPORT/"part64_comparison.json","w",encoding="utf-8") as f: json.dump({"part":64,"train_subset":TRAIN_N,"validation_subset":VAL_N,"epochs":EPOCHS,"radius":RADIUS,"point_lambda":POINT_LAMBDA,"negative_points_per_positive":NEG_PER_POINT,"initialization_sha256":sha(INIT),"final_A":aa,"final_B":bb,"best_A":ba,"best_B":bc,"final_dice_delta_B_minus_A":delta,"best_dice_delta_B_minus_A":bestdelta,"diagnosis":diagnosis,"scientific_note":"RSNA coordinates are point annotations, not manual segmentation masks; this pilot does not establish medical ground-truth segmentation accuracy."},f,indent=2)
    banner("PART 64 COMPLETE"); print(f"Output directory : {OUT}\nComparison JSON : {REPORT/'part64_comparison.json'}\nTrajectory CSV : {REPORT/'part64_training_trajectory.csv'}\nPoint coverage CSV : {REPORT/'part64_point_coverage_by_class.csv'}")
if __name__=="__main__":main()
