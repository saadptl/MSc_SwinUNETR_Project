from __future__ import annotations
import csv, gc, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.ndimage import distance_transform_edt

ROOT=Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
PART11=ROOT/"src/segmentation_rsna_part11_controlled_pilot_training.py"
PART9=ROOT/"src/segmentation_rsna_part9_3d_dataset_loader.py"
P15=ROOT/"outputs/segmentation/rsna_part15_extended_controlled_training"
INIT=P15/"checkpoints/part15_initialization_from_part11.pth"
VACSV=P15/"part15_validation_cohort.csv"
P58=ROOT/"outputs/segmentation/rsna_part58_r2_lr_stability_confirmation"
OUT=ROOT/"outputs/segmentation/rsna_part62_per_class_spatial_localization_diagnostic"
REPORT=OUT/"reports"
N=100; FULL=(64,96,96); CROP=(32,64,64); NC=6; RADIUS=2
THRESHOLDS=[.200,.225,.250,.275]; TOLS=[1,2,3]; EPOCHS=[1,2,3,4,5]
CLASSES={1:"Spinal_Canal_Stenosis",2:"Left_Neural_Foraminal_Narrowing",3:"Right_Neural_Foraminal_Narrowing",4:"Left_Subarticular_Stenosis",5:"Right_Subarticular_Stenosis"}
CONDS={"A_constant_5e5":P58/"A_constant_5e5","B_step_1e4_to_5e5":P58/"B_step_1e4_to_5e5"}

def banner(s): print("\n"+"="*82+"\n"+s+"\n"+"="*82)
def mod(path,name):
    sp=importlib.util.spec_from_file_location(name,str(path))
    m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def dil6(a,n):
    a=a.astype(bool).copy()
    for _ in range(n):
        b=a.copy(); b[1:]|=a[:-1]; b[:-1]|=a[1:]; b[:,1:]|=a[:,:-1]; b[:,:-1]|=a[:,1:]; b[:,:,1:]|=a[:,:,:-1]; b[:,:,:-1]|=a[:,:,1:]; a=b
    return a
def dilate(mask,n):
    if n==0:return mask.copy()
    rs=[]
    for c in range(1,NC):
        q=mask==c
        if q.any(): rs.append((int(q.sum()),c,dil6(q,n)))
    rs.sort(key=lambda x:(-x[0],x[1])); out=np.zeros_like(mask,dtype=np.int64); occ=np.zeros_like(mask,bool)
    for _,c,q in rs:
        q=q&~occ; out[q]=c; occ|=q
    return out
def crop(im,mask):
    q=np.argwhere(mask>0); ctr=np.round(q.mean(0)).astype(int) if q.size else np.array([x//2 for x in FULL])
    st=[max(0,min(int(ctr[i])-CROP[i]//2,FULL[i]-CROP[i])) for i in range(3)]
    z,y,x=st; dz,dy,dx=CROP
    return im[z:z+dz,y:y+dy,x:x+dx].astype(np.float32),mask[z:z+dz,y:y+dy,x:x+dx].astype(np.int64)
def load_case(p11,p9,row):
    z=p11.load_tensor_case(row,p9); im,ma=z[0],z[1]
    if torch.is_tensor(im): im=im.detach().cpu().numpy()
    if torch.is_tensor(ma): ma=ma.detach().cpu().numpy()
    im=np.asarray(im); ma=np.asarray(ma)
    if im.ndim==4 and im.shape[0]==1: im=im[0]
    if ma.ndim==4 and ma.shape[0]==1: ma=ma[0]
    if tuple(im.shape)!=FULL or tuple(ma.shape)!=FULL: raise RuntimeError(f"Unexpected shape {im.shape},{ma.shape}")
    return crop(im,dilate(ma.astype(np.int64),RADIUS))
def preload(p11,p9,df):
    out=[]
    for i,(_,r) in enumerate(df.iterrows(),1):
        im,ma=load_case(p11,p9,r); out.append({"image":im,"mask":ma,"index":i})
        if i in (1,25,50,75,100): print(f"VALIDATION {i:03d}/{len(df)} FG={int((ma>0).sum())}")
    return out
def ckpt(d,e):
    for q in (d/f"epoch_{e:02d}.pth",d/"checkpoints"/f"epoch_{e:02d}.pth"):
        if q.exists(): return q
    raise FileNotFoundError(d/e)
def load_model(m,q,dev):
    z=torch.load(q,map_location=dev,weights_only=False); s=z
    if isinstance(z,dict):
        for k in ("model_state_dict","state_dict","model"):
            if isinstance(z.get(k),dict): s=z[k]; break
    s={k[7:] if k.startswith("module.") else k:v for k,v in s.items()}
    miss,unexp=m.load_state_dict(s,strict=False)
    if miss or unexp: raise RuntimeError(f"Checkpoint mismatch: {miss} {unexp}")
    m.eval()
def probs(m,cases,dev):
    out=[]
    for c in cases:
        x=torch.from_numpy(c["image"]).unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad(): z=torch.softmax(m(x),1)[0].cpu().numpy()
        out.append(z); del x
    return out
def metrics(t,p):
    t=np.asarray(t,bool); p=np.asarray(p,bool); tp=int((t&p).sum()); fp=int((~t&p).sum()); fn=int((t&~p).sum()); tn=int((~t&~p).sum())
    tv=int(t.sum()); pv=int(p.sum()); d=2*tp/(tv+pv) if tv+pv else 1.; pr=tp/(tp+fp) if tp+fp else 0.; re=tp/(tp+fn) if tp+fn else 0.
    r={"dice":float(d),"precision":float(pr),"recall":float(re),"tp":tp,"fp":fp,"fn":fn,"target_voxels":tv,"pred_voxels":pv}
    td=distance_transform_edt(~t) if p.any() else None; pd=distance_transform_edt(~p) if t.any() else None
    for tol in TOLS:
        r[f"tol{tol}_precision"]=float(np.mean(td[p]<=tol)) if td is not None else 0.
        r[f"tol{tol}_recall"]=float(np.mean(pd[t]<=tol)) if pd is not None else 0.
    a=np.argwhere(t); b=np.argwhere(p); r["centroid_distance"]=float(np.linalg.norm(a.mean(0)-b.mean(0))) if a.size and b.size else None
    return r
def f1(p,r): return 2*p*r/(p+r) if p+r else 0.
def main():
    banner("PART 62 PATH VALIDATION")
    for name,q in [("Project root",ROOT),("Part 11",PART11),("Part 9",PART9),("Part 15 initialization",INIT),("Part 15 validation cohort",VACSV),("Part 58 output",P58)]:
        print(f"{name:<40}: {'FOUND' if q.exists() else 'MISSING'}")
        if not q.exists(): raise FileNotFoundError(q)
    for n,d in CONDS.items():
        print(f"{n+' directory':<40}: {'FOUND' if d.exists() else 'MISSING'}")
        for e in EPOCHS: ckpt(d,e)
    banner("PART 62 — PER-CLASS SPATIAL LOCALIZATION DIAGNOSTIC")
    print(f"Validation cohort : {N}\nPseudo-mask radius : R{RADIUS}\nFull volume : {FULL}\nCrop : {CROP}\nThresholds : {THRESHOLDS}\nSpatial tolerances : {TOLS} voxels\nCheckpoints : epochs {EPOCHS}\nClass-specific thresholding : YES\nTraining performed : NO\nOptimizer used : NO\nBackward pass : NO\nSPIDER : NO\nTest set : NO\nPart 15 modified : NO\nInitialization SHA256 : {sha(INIT)}")
    p11=mod(PART11,"p11_62"); p9=mod(PART9,"p9_62"); df=pd.read_csv(VACSV).head(N); cases=preload(p11,p9,df)
    banner("PART 62 SHAPE / LABEL SMOKE TEST"); print(f"Image : {cases[0]['image'].shape}\nMask : {cases[0]['mask'].shape}\nLabels : {np.unique(cases[0]['mask']).tolist()}")
    assert tuple(cases[0]["image"].shape)==CROP and tuple(cases[0]["mask"].shape)==CROP
    print("✓ Shape / label smoke test PASSED.")
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(f"\nPyTorch : {torch.__version__}\nDevice : {dev}")
    rows=[]; aggs=[]; best=[]
    for cond,d in CONDS.items():
        banner(f"PART 62 CONDITION: {cond}"); m=p11.create_model(dev); m.to(dev)
        for e in EPOCHS:
            q=ckpt(d,e); load_model(m,q,dev); print(f"\nCHECKPOINT EPOCH {e:02d} | SHA256={sha(q)}"); pp=probs(m,cases,dev)
            for t in THRESHOLDS:
                for cid,cname in CLASSES.items():
                    rs=[]
                    for c,pr in zip(cases,pp):
                        r=metrics(c["mask"]==cid,pr[cid]>=t); r.update(case_index=c["index"],condition=cond,epoch=e,threshold=t,class_id=cid,class_name=cname); rows.append(r); rs.append(r)
                    a={"condition":cond,"epoch":e,"threshold":t,"class_id":cid,"class_name":cname}
                    for k in ("dice","precision","recall","target_voxels","pred_voxels"): a[k]=float(np.mean([r[k] for r in rs]))
                    for k in ("tol1_precision","tol1_recall","tol2_precision","tol2_recall","tol3_precision","tol3_recall"): a[k]=float(np.mean([r[k] for r in rs]))
                    a["tol3_f1"]=f1(a["tol3_precision"],a["tol3_recall"]); a["empty_prediction_cases"]=sum(r["pred_voxels"]==0 for r in rs); a["target_present_cases"]=sum(r["target_voxels"]>0 for r in rs); a["total_cases"]=len(rs)
                    a["centroid_distance"]=float(np.mean([r["centroid_distance"] for r in rs if r["centroid_distance"] is not None])) if any(r["centroid_distance"] is not None for r in rs) else None
                    aggs.append(a)
                    print(f"  C{cid} {cname:<34} T={t:.3f} | Dice={a['dice']:.6f} | P={a['precision']:.6f} | R={a['recall']:.6f} | T3-P={a['tol3_precision']:.4f} T3-R={a['tol3_recall']:.4f} | T3-F1={a['tol3_f1']:.6f} | Pred={a['pred_voxels']:.1f}")
        del m; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    for cond in CONDS:
        for cid,cname in CLASSES.items():
            z=max((r for r in aggs if r["condition"]==cond and r["class_id"]==cid),key=lambda r:r["tol3_f1"]); best.append(z)
    banner("PART 62 BEST PER-CLASS SPATIAL RESULTS")
    for r in best: print(f"{r['condition']:<24} C{r['class_id']} {r['class_name']:<34} | E{r['epoch']} T={r['threshold']:.3f} | Dice={r['dice']:.6f} | T3-F1={r['tol3_f1']:.6f} | P3={r['tol3_precision']:.4f} R3={r['tol3_recall']:.4f}")
    summary=[]
    for cid,cname in CLASSES.items():
        z=[r for r in best if r["class_id"]==cid]
        summary.append({"class_id":cid,"class_name":cname,"best_t3_f1_mean":float(np.mean([r["tol3_f1"] for r in z])),"best_dice_mean":float(np.mean([r["dice"] for r in z])),"best_t3_precision_mean":float(np.mean([r["tol3_precision"] for r in z])),"best_t3_recall_mean":float(np.mean([r["tol3_recall"] for r in z]))})
    summary.sort(key=lambda r:r["best_t3_f1_mean"],reverse=True)
    banner("PART 62 CLASS RANKING")
    for i,r in enumerate(summary,1): print(f"{i}. C{r['class_id']} {r['class_name']:<34} | mean best T3-F1={r['best_t3_f1_mean']:.6f} | mean Dice={r['best_dice_mean']:.6f} | P3={r['best_t3_precision_mean']:.4f} | R3={r['best_t3_recall_mean']:.4f}")
    strong,weak=summary[0],summary[-1]
    if strong["best_t3_f1_mean"]>=.15: diagnosis="CLASS_SPECIFIC_SPATIAL_LOCALIZATION_IS_EVIDENT" if strong["best_t3_f1_mean"]>=1.5*weak["best_t3_f1_mean"] else "SPATIAL_SIGNAL_PRESENT_BUT_CLASS_SEPARATION_IS_WEAK"
    elif strong["best_t3_f1_mean"]>=.08: diagnosis="WEAK_PER_CLASS_SPATIAL_LOCALIZATION"
    else: diagnosis="NO_STRONG_PER_CLASS_SPATIAL_LOCALIZATION"
    banner("PART 62 DIAGNOSTIC INTERPRETATION"); print(f"Strongest class : C{strong['class_id']} {strong['class_name']}\nStrongest mean best T3-F1 : {strong['best_t3_f1_mean']:.6f}\nWeakest class : C{weak['class_id']} {weak['class_name']}\nWeakest mean best T3-F1 : {weak['best_t3_f1_mean']:.6f}\nDiagnosis : {diagnosis}")
    REPORT.mkdir(parents=True,exist_ok=True)
    def write(name,data):
        with open(REPORT/name,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    write("part62_per_class_case_metrics.csv",rows); write("part62_per_class_threshold_aggregate.csv",aggs); write("part62_best_per_class_spatial_checkpoint.csv",best); write("part62_class_ranking.csv",summary)
    js=REPORT/"part62_summary.json"
    with open(js,"w",encoding="utf-8") as f: json.dump({"part":62,"purpose":"Determine whether low-threshold foreground probability contains class-specific spatial localization.","validation_cases":N,"pseudo_mask_radius":RADIUS,"thresholds":THRESHOLDS,"tolerances_voxels":TOLS,"epochs":EPOCHS,"conditions":list(CONDS),"class_names":CLASSES,"training_performed":False,"optimizer_used":False,"backward_pass":False,"spider":False,"test_set":False,"part15_modified":False,"initialization_sha256":sha(INIT),"class_ranking":summary,"diagnosis":diagnosis,"scientific_note":"Targets are RSNA-derived pseudo-masks; spatial agreement does not establish medical ground-truth segmentation accuracy."},f,indent=2)
    banner("PART 62 COMPLETE"); print(f"Output directory : {OUT}\nPer-class case CSV : {REPORT/'part62_per_class_case_metrics.csv'}\nAggregate CSV : {REPORT/'part62_per_class_threshold_aggregate.csv'}\nBest-per-class CSV : {REPORT/'part62_best_per_class_spatial_checkpoint.csv'}\nClass ranking CSV : {REPORT/'part62_class_ranking.csv'}\nSummary JSON : {js}")
if __name__=="__main__": main()
