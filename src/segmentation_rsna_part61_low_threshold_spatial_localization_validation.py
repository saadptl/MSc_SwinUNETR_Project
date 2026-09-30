"""PART 61 — LOW-THRESHOLD SPATIAL LOCALIZATION VALIDATION

Diagnostic-only experiment using existing Part 58 R2 checkpoints.
Tests whether low-threshold foreground signal is spatially localized near
R2 pseudo-mask targets. No training/optimizer/backward/SPIDER/test set.
Targets are pseudo-masks; this does not establish medical ground-truth accuracy.
"""
from __future__ import annotations
import csv, gc, hashlib, importlib.util, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt

ROOT=Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
PART11=ROOT/"src"/"segmentation_rsna_part11_controlled_pilot_training.py"
PART9=ROOT/"src"/"segmentation_rsna_part9_3d_dataset_loader.py"
P15=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"
INIT=P15/"checkpoints"/"part15_initialization_from_part11.pth"
VACSV=P15/"part15_validation_cohort.csv"
P58=ROOT/"outputs"/"segmentation"/"rsna_part58_r2_lr_stability_confirmation"
OUT=ROOT/"outputs"/"segmentation"/"rsna_part61_low_threshold_spatial_localization_validation"
REPORT=OUT/"reports"
VAL_N=100; FULL_SHAPE=(64,96,96); CROP_SHAPE=(32,64,64); NUM_CLASSES=6; RADIUS=2
THRESHOLDS=[.150,.175,.200,.225,.250,.275]; TOLERANCES=[1,2,3]; EPOCHS=[1,2,3,4,5]
CONDITIONS={"A_constant_5e5":P58/"A_constant_5e5","B_step_1e4_to_5e5":P58/"B_step_1e4_to_5e5"}

def banner(s): print("\n"+"="*82+f"\n{s}\n"+"="*82)
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise RuntimeError(f"Unable to load module: {path}")
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def dilate_binary_6(a,n):
    r=a.astype(bool).copy()
    for _ in range(n):
        e=r.copy(); e[1:,:,:]|=r[:-1,:,:]; e[:-1,:,:]|=r[1:,:,:]; e[:,1:,:]|=r[:,:-1,:]; e[:,:-1,:]|=r[:,1:,:]; e[:,:,1:]|=r[:,:,:-1]; e[:,:,:-1]|=r[:,:,1:]; r=e
    return r
def dilate_multiclass(mask,radius):
    if radius==0:return mask.copy()
    regions=[]
    for cid in range(1,NUM_CLASSES):
        reg=mask==cid
        if reg.any(): regions.append((int(reg.sum()),cid,dilate_binary_6(reg,radius)))
    regions.sort(key=lambda x:(-x[0],x[1])); out=np.zeros_like(mask,dtype=np.int64); occ=np.zeros_like(mask,dtype=bool)
    for _,cid,reg in regions:
        a=reg&~occ; out[a]=cid; occ|=a
    return out
def foreground_center_crop(image,mask):
    c=np.round(np.argwhere(mask>0).mean(axis=0)).astype(int) if np.argwhere(mask>0).size else np.array([s//2 for s in image.shape])
    starts=[max(0,min(int(c[i])-CROP_SHAPE[i]//2,FULL_SHAPE[i]-CROP_SHAPE[i])) for i in range(3)]
    z,y,x=starts; dz,dy,dx=CROP_SHAPE
    return image[z:z+dz,y:y+dy,x:x+dx].astype(np.float32),mask[z:z+dz,y:y+dy,x:x+dx].astype(np.int64)
def load_case(part11,part9,row):
    loaded=part11.load_tensor_case(row,part9); image,mask=loaded[0],loaded[1]
    if torch.is_tensor(image): image=image.detach().cpu().numpy()
    if torch.is_tensor(mask): mask=mask.detach().cpu().numpy()
    image=np.asarray(image); mask=np.asarray(mask)
    if image.ndim==4 and image.shape[0]==1:image=image[0]
    if mask.ndim==4 and mask.shape[0]==1:mask=mask[0]
    if tuple(image.shape)!=FULL_SHAPE or tuple(mask.shape)!=FULL_SHAPE: raise RuntimeError(f"Unexpected shapes: {image.shape}, {mask.shape}")
    return foreground_center_crop(image,dilate_multiclass(mask.astype(np.int64),RADIUS))
def preload(part11,part9,df):
    banner("PART 61 VALIDATION PRELOAD"); cases=[]
    for i,(_,row) in enumerate(df.iterrows(),1):
        image,mask=load_case(part11,part9,row); cases.append({"image":image,"mask":mask,"index":i})
        if i in (1,25,50,75,100): print(f"VALIDATION {i:03d}/{len(df)} FG={int((mask>0).sum())}")
    return cases
def threshold_prediction(probs,t):
    fg=probs[1:]; cls=np.argmax(fg,axis=0).astype(np.int64)+1; cls[np.max(fg,axis=0)<t]=0; return cls
def basic_metrics(target,pred):
    a=target>0;b=pred>0;tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());pc=int(b.sum());tc=int(a.sum());d=pc+tc
    return {"dice":2*tp/d if d else 1.0,"precision":tp/(tp+fp) if tp+fp else 0.0,"recall":tp/(tp+fn) if tp+fn else 0.0,"tp":tp,"fp":fp,"fn":fn,"pred_fg":pc,"target_fg":tc}
def distance_stats(source,dest):
    if not source.any() or not dest.any(): return {"mean":None,"median":None,"p95":None,"max":None}
    d=distance_transform_edt(~dest)[source]
    return {"mean":float(np.mean(d)),"median":float(np.median(d)),"p95":float(np.percentile(d,95)),"max":float(np.max(d))}
def tol_metrics(target,pred,t):
    a=target>0;b=pred>0
    if not a.any() or not b.any():return {"precision":0.0,"recall":0.0}
    return {"precision":float(np.mean(distance_transform_edt(~a)[b]<=t)),"recall":float(np.mean(distance_transform_edt(~b)[a]<=t))}
def case_metrics(target,pred):
    r=basic_metrics(target,pred);p=distance_stats(pred>0,target>0);q=distance_stats(target>0,pred>0); r.update({"pred_to_target_mean_distance":p["mean"],"pred_to_target_median_distance":p["median"],"pred_to_target_p95_distance":p["p95"],"pred_to_target_max_distance":p["max"],"target_to_pred_mean_distance":q["mean"],"target_to_pred_median_distance":q["median"],"target_to_pred_p95_distance":q["p95"],"target_to_pred_max_distance":q["max"]})
    a=np.argwhere(target>0);b=np.argwhere(pred>0);r["centroid_distance"]=float(np.linalg.norm(a.mean(0)-b.mean(0))) if a.size and b.size else None
    for t in TOLERANCES:
        z=tol_metrics(target,pred,t);r[f"tol{t}_precision"]=z["precision"];r[f"tol{t}_recall"]=z["recall"]
    return r
def aggregate(records):
    md=lambda k:float(np.mean([r[k] for r in records if r[k] is not None])) if any(r[k] is not None for r in records) else None
    return {"foreground_dice":float(np.mean([r["dice"] for r in records])),"precision":float(np.mean([r["precision"] for r in records])),"recall":float(np.mean([r["recall"] for r in records])),"pred_fg_mean":float(np.mean([r["pred_fg"] for r in records])),"target_fg_mean":float(np.mean([r["target_fg"] for r in records])),"empty_cases":int(sum(r["pred_fg"]==0 for r in records)),"total_cases":len(records),"pred_to_target_mean_distance":md("pred_to_target_mean_distance"),"pred_to_target_median_distance":md("pred_to_target_median_distance"),"pred_to_target_p95_distance":md("pred_to_target_p95_distance"),"target_to_pred_mean_distance":md("target_to_pred_mean_distance"),"target_to_pred_median_distance":md("target_to_pred_median_distance"),"target_to_pred_p95_distance":md("target_to_pred_p95_distance"),"centroid_distance_mean":md("centroid_distance"),**{f"tol{t}_{x}":float(np.mean([r[f"tol{t}_{x}"] for r in records])) for t in TOLERANCES for x in ("precision","recall")}}
def checkpoint_path(directory,epoch):
    for p in (directory/f"epoch_{epoch:02d}.pth",directory/"checkpoints"/f"epoch_{epoch:02d}.pth"):
        if p.exists():return p
    raise FileNotFoundError(f"Epoch {epoch} checkpoint not found in {directory}")
def load_checkpoint(model,path,device):
    ck=torch.load(path,map_location=device,weights_only=False); state=ck
    if isinstance(ck,dict):
        for k in ("model_state_dict","state_dict","model"):
            if isinstance(ck.get(k),dict):state=ck[k];break
    state={k[7:] if k.startswith("module.") else k:v for k,v in state.items()};missing,unexpected=model.load_state_dict(state,strict=False)
    if missing or unexpected:raise RuntimeError(f"Checkpoint mismatch: {path}\nMissing={missing}\nUnexpected={unexpected}")
    model.eval()
def main():
    random.seed(161);np.random.seed(161);torch.manual_seed(161);banner("PART 61 PATH VALIDATION")
    for name,p in [("Project root",ROOT),("Part 11",PART11),("Part 9",PART9),("Part 15 initialization",INIT),("Part 15 validation cohort",VACSV),("Part 58 output",P58)]:
        print(f"{name:<40}: {'FOUND' if p.exists() else 'MISSING'}");
        if not p.exists():raise FileNotFoundError(p)
    for name,d in CONDITIONS.items():
        print(f"{name+' directory':<40}: {'FOUND' if d.exists() else 'MISSING'}");
        if not d.exists():raise FileNotFoundError(d)
        for e in EPOCHS:checkpoint_path(d,e)
    banner("PART 61 — LOW-THRESHOLD SPATIAL LOCALIZATION VALIDATION");print(f"Validation cohort : {VAL_N}\nPseudo-mask radius : R{RADIUS}\nFull volume : {FULL_SHAPE}\nCrop : {CROP_SHAPE}\nThresholds : {THRESHOLDS}\nSpatial tolerances : {TOLERANCES} voxels\nCheckpoints : epochs {EPOCHS}\nTraining performed : NO\nOptimizer used : NO\nBackward pass : NO\nSPIDER : NO\nTest set : NO\nPart 15 modified : NO\nInitialization SHA256 : {sha256(INIT)}")
    p11=load_module(PART11,"part11_part61");p9=load_module(PART9,"part9_part61");df=pd.read_csv(VACSV).head(VAL_N);cases=preload(p11,p9,df)
    banner("PART 61 SHAPE / LABEL SMOKE TEST");print(f"Image : {cases[0]['image'].shape}\nMask : {cases[0]['mask'].shape}\nLabels : {np.unique(cases[0]['mask']).tolist()}");assert tuple(cases[0]["image"].shape)==CROP_SHAPE and tuple(cases[0]["mask"].shape)==CROP_SHAPE;print("✓ Shape / label smoke test PASSED.")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");print(f"\nPyTorch : {torch.__version__}\nDevice : {device}");all_rows=[];aggs=[];model=None
    for condition,directory in CONDITIONS.items():
        banner(f"PART 61 CONDITION: {condition}");model=p11.create_model(device);model.to(device);model.eval()
        for epoch in EPOCHS:
            ck=checkpoint_path(directory,epoch);load_checkpoint(model,ck,device);print(f"\nCHECKPOINT EPOCH {epoch:02d} | SHA256={sha256(ck)}")
            probs_cache=[]
            for case in cases:
                x=torch.from_numpy(case["image"]).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():logits=model(x);probs=torch.softmax(logits,dim=1)[0].cpu().numpy()
                probs_cache.append(probs);del x,logits
            for t in THRESHOLDS:
                recs=[]
                for case,probs in zip(cases,probs_cache):
                    pred=threshold_prediction(probs,t);m=case_metrics(case["mask"],pred);m["case_index"]=case["index"];recs.append(m);all_rows.append({"condition":condition,"epoch":epoch,"threshold":t,**m})
                a=aggregate(recs);a.update({"condition":condition,"epoch":epoch,"threshold":t});aggs.append(a)
                print(f"  T={t:.3f} | Dice={a['foreground_dice']:.6f} | P={a['precision']:.6f} | R={a['recall']:.6f} | PredFG={a['pred_fg_mean']:.1f} | P@1={a['tol1_precision']:.4f} R@1={a['tol1_recall']:.4f} | P@3={a['tol3_precision']:.4f} R@3={a['tol3_recall']:.4f} | P→T mean={a['pred_to_target_mean_distance']}")
            del probs_cache
        del model;model=None;gc.collect();
        if torch.cuda.is_available():torch.cuda.empty_cache()
    banner("PART 61 CROSS-CHECKPOINT SPATIAL SUMMARY")
    def f1(p,r):return 2*p*r/(p+r) if p+r else 0.0
    best=[]
    for condition in CONDITIONS:
        for epoch in EPOCHS:
            rows=[r for r in aggs if r["condition"]==condition and r["epoch"]==epoch];r=max(rows,key=lambda z:f1(z["tol3_precision"],z["tol3_recall"]));best.append({"condition":condition,"epoch":epoch,"best_spatial_threshold":r["threshold"],"tol3_f1":f1(r["tol3_precision"],r["tol3_recall"]),"tol3_precision":r["tol3_precision"],"tol3_recall":r["tol3_recall"],"tol2_precision":r["tol2_precision"],"tol2_recall":r["tol2_recall"],"tol1_precision":r["tol1_precision"],"tol1_recall":r["tol1_recall"],"dice":r["foreground_dice"],"pred_fg_mean":r["pred_fg_mean"],"pred_to_target_mean_distance":r["pred_to_target_mean_distance"],"target_to_pred_mean_distance":r["target_to_pred_mean_distance"],"centroid_distance_mean":r["centroid_distance_mean"]})
            print(f"{condition:<24} E{epoch} | T={r['threshold']:.3f} | T3-F1={best[-1]['tol3_f1']:.6f} | Dice={r['foreground_dice']:.6f} | P3={r['tol3_precision']:.4f} R3={r['tol3_recall']:.4f} | PredFG={r['pred_fg_mean']:.1f}")
    global_best=max(best,key=lambda r:r["tol3_f1"]);common=[]
    for t in THRESHOLDS:
        rows=[r for r in aggs if r["threshold"]==t];fs=[f1(r["tol3_precision"],r["tol3_recall"]) for r in rows];common.append({"threshold":t,"mean_dice":float(np.mean([r["foreground_dice"] for r in rows])),"mean_tol3_precision":float(np.mean([r["tol3_precision"] for r in rows])),"mean_tol3_recall":float(np.mean([r["tol3_recall"] for r in rows])),"mean_tol3_f1":float(np.mean(fs)),"mean_pred_fg":float(np.mean([r["pred_fg_mean"] for r in rows]))})
    common_best=max(common,key=lambda r:r["mean_tol3_f1"]);print(f"\nBest common threshold by mean T3-F1 : {common_best['threshold']:.3f}\nMean T3-F1 : {common_best['mean_tol3_f1']:.6f}\nMean T3 precision : {common_best['mean_tol3_precision']:.6f}\nMean T3 recall : {common_best['mean_tol3_recall']:.6f}")
    if common_best["mean_tol3_recall"]>=.20 and common_best["mean_tol3_precision"]>=.01:diagnosis="LOW_THRESHOLD_SIGNAL_SHOWS_SPATIAL_LOCALIZATION"
    elif common_best["mean_tol3_recall"]>=.10:diagnosis="LOW_THRESHOLD_SIGNAL_SHOWS_WEAK_SPATIAL_LOCALIZATION"
    else:diagnosis="LOW_THRESHOLD_SIGNAL_IS_NOT_SPATIALLY_LOCALIZED"
    banner("PART 61 DIAGNOSTIC INTERPRETATION");print(f"Global best spatial checkpoint : {global_best['condition']} E{global_best['epoch']}\nBest spatial threshold : {global_best['best_spatial_threshold']:.3f}\nBest T3-F1 : {global_best['tol3_f1']:.6f}\nBest T3 precision : {global_best['tol3_precision']:.6f}\nBest T3 recall : {global_best['tol3_recall']:.6f}\nBest common threshold : {common_best['threshold']:.3f}\nCommon mean T3-F1 : {common_best['mean_tol3_f1']:.6f}\nDiagnosis : {diagnosis}")
    REPORT.mkdir(parents=True,exist_ok=True)
    def write_csv(path,rows):
        with open(path,"w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    case_csv=REPORT/"part61_case_spatial_metrics.csv";write_csv(case_csv,all_rows)
    agg_csv=REPORT/"part61_checkpoint_threshold_spatial_aggregate.csv";write_csv(agg_csv,aggs)
    best_csv=REPORT/"part61_best_spatial_threshold_by_checkpoint.csv";write_csv(best_csv,best)
    common_csv=REPORT/"part61_cross_checkpoint_threshold_spatial_aggregate.csv";write_csv(common_csv,common)
    summary=REPORT/"part61_summary.json"
    summary.write_text(json.dumps({"part":61,"purpose":"Validate whether low-threshold foreground probability is spatially localized near the R2 pseudo-mask target.","validation_cases":VAL_N,"pseudo_mask_radius":RADIUS,"full_shape":FULL_SHAPE,"crop_shape":CROP_SHAPE,"thresholds":THRESHOLDS,"tolerances_voxels":TOLERANCES,"epochs":EPOCHS,"conditions":list(CONDITIONS),"training_performed":False,"optimizer_used":False,"backward_pass":False,"spider":False,"test_set":False,"part15_modified":False,"initialization_sha256":sha256(INIT),"global_best_spatial":global_best,"best_common_threshold":common_best,"diagnosis":diagnosis,"scientific_note":"Targets are pseudo-masks. Spatial agreement with pseudo-masks does not establish medical ground-truth segmentation accuracy."},indent=2),encoding="utf-8")
    banner("PART 61 COMPLETE");print(f"Output directory : {OUT}\nCase metrics CSV : {case_csv}\nAggregate CSV : {agg_csv}\nBest-spatial CSV : {best_csv}\nThreshold aggregate CSV : {common_csv}\nSummary JSON : {summary}")
if __name__=="__main__":main()
