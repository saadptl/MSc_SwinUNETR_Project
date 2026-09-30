"""PART127: inference-only probability calibration of protected Part104."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
OUT=ROOT/"outputs"/"segmentation"/"rsna_part127_part104_probability_calibration"
VAL=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"/"part15_validation_cohort.csv"
P104=ROOT/"outputs"/"segmentation"/"rsna_part104_final_checkpoint_selection"/"checkpoints"/"final_segmentation_model.pth"
P9=SRC/"segmentation_rsna_part9_3d_dataset_loader.py"
P11 = SRC / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
SHA="fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
FULL=(64,96,96); ROI=(32,64,64); THRESHOLDS=[.30,.40,.50,.60,.70,.80,.90]
COMPONENTS=[0,5,20,50,100]; DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imp(path,name):
    s=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(s)
    sys.modules[name]=m; s.loader.exec_module(m); return m

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def dice(p,t):
    p=p.astype(bool); t=t.astype(bool); a=p.sum(); b=t.sum()
    if a==0 and b==0:return None
    return float(2*np.logical_and(p,t).sum()/(a+b)) if a+b else None

def metrics(p,t):
    pf=p>0; tf=t>0
    tp=np.logical_and(pf,tf).sum(); fp=np.logical_and(pf,~tf).sum()
    fn=np.logical_and(~pf,tf).sum()
    cd=[dice(p==c,t==c) for c in range(1,6)]
    usable=[x for x in cd if x is not None]
    return dict(
        foreground_dice=dice(pf,tf) or 0,
        macro_foreground_dice=float(np.mean(usable)) if usable else 0,
        precision=float(tp/(tp+fp)) if tp+fp else 0,
        recall=float(tp/(tp+fn)) if tp+fn else 0,
        pred_foreground_voxels=int(pf.sum()),target_foreground_voxels=int(tf.sum()),
        pred_target_volume_ratio=float(pf.sum()/tf.sum()) if tf.sum() else 0,
        class_dice=cd)

def filter_cc(mask,n):
    if n<=0:return mask
    from scipy import ndimage
    fg=mask>0
    if not fg.any():return mask
    lab,k=ndimage.label(fg,ndimage.generate_binary_structure(3,2))
    sizes=np.bincount(lab.ravel()); keep=sizes>=n; keep[0]=False
    z=mask.copy(); z[~keep[lab]]=0; return z

@torch.inference_mode()
def infer(model,img):
    x=img.unsqueeze(0).to(DEVICE)
    return sliding_window_inference(x,ROI,1,model,overlap=.50,mode="gaussian",sigma_scale=.125).cpu()

def main():
    OUT.mkdir(parents=True,exist_ok=True); (ROOT/"reports").mkdir(exist_ok=True)
    print("="*96); print("PART 127 — PART104 PROBABILITY CALIBRATION"); print("="*96)
    print("Project root :",ROOT); print("Output       :",OUT)
    print("PyTorch      :",torch.__version__); print("CUDA         :",torch.cuda.is_available()); print("Device       :",DEVICE)
    print("\nCHECKING REQUIRED ARTIFACTS")
    for n,x in [("Part9 source",P9),("Corrected Part11 source",P11),("Validation cohort",VAL),("Part104 checkpoint",P104)]:
        print(f"{n:<32}", "PASS" if x.exists() else "FAIL")
        if not x.exists(): raise FileNotFoundError(x)
    actual=sha(P104); print("\nPart104 SHA256 :",actual)
    if actual!=SHA: raise RuntimeError("Protected Part104 SHA mismatch.")
    print("Part104 SHA verification : PASS")
    p9=imp(P9,"p127_part9"); p11=imp(P11,"p127_part11")
    print("Part9 import : PASS"); print("Corrected Part11 import : PASS")
    df=pd.read_csv(VAL); print("Validation cases :",len(df))
    model=p11.create_model(DEVICE)
    params=sum(x.numel() for x in model.parameters() if x.requires_grad)
    print("Canonical parameters :",params)
    if params!=4078116: raise RuntimeError("Unexpected model parameter count.")
    ck=torch.load(P104,map_location=DEVICE,weights_only=False)
    state=ck.get("state_dict",ck.get("model_state_dict",ck.get("model",ck))) if isinstance(ck,dict) else ck
    if all(str(k).startswith("module.") for k in state): state={str(k)[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True); model.eval(); print("Part104 initialization : PASS")
    print("\nFULL-VOLUME SLIDING-WINDOW INFERENCE")
    cache=[]; failures=[]
    for i,row in df.iterrows():
        try:
            img, mask, metadata = p11.load_tensor_case(row, p9)
            if img.ndim==3: img=img.unsqueeze(0)
            if tuple(img.shape[-3:])!=FULL: img=F.interpolate(img.unsqueeze(0).float(),size=FULL,mode="trilinear",align_corners=False).squeeze(0)
            if tuple(mask.shape[-3:])!=FULL: mask=F.interpolate(mask[None,None].float(),size=FULL,mode="nearest").squeeze().long()
            logits=infer(model,img.float())
            prob=torch.softmax(logits,1); fg=prob[:,1:].sum(1)[0].numpy()
            cls=(torch.argmax(logits[:,1:],1)+1)[0].numpy()
            cache.append((int(i),str(row.get("study_id","")),str(row.get("series_id","")),mask.numpy(),fg,cls))
            if len(cache)%10==0: print(f"  Inference: {len(cache)}/{len(df)}")
        except Exception as e: failures.append({"index":int(i),"error":repr(e)}); print("  WARNING:",i,e)
    print("Completed:",len(cache),"/",len(df))
    rows=[]; case_rows=[]; comp=[]
    for th in THRESHOLDS:
        vals=[]
        for i,study,series,t,prob,cls in cache:
            pred=np.where(prob>=th,cls,0).astype(np.int16); m=metrics(pred,t); vals.append(m)
            r={"threshold":th,"filter_min_voxels":0,"index":i,"study_id":study,"series_id":series,**{k:v for k,v in m.items() if k!="class_dice"}}
            for c,v in enumerate(m["class_dice"],1): r[f"dice_class_{c}"]="" if v is None else v
            case_rows.append(r)
        rows.append({"threshold":th,"mean_foreground_dice":np.mean([x["foreground_dice"] for x in vals]),
                     "mean_macro_foreground_dice":np.mean([x["macro_foreground_dice"] for x in vals]),
                     "mean_precision":np.mean([x["precision"] for x in vals]),
                     "mean_recall":np.mean([x["recall"] for x in vals]),
                     "mean_pred_target_volume_ratio":np.mean([x["pred_target_volume_ratio"] for x in vals]),
                     "pred_foreground_cases":sum(x["pred_foreground_voxels"]>0 for x in vals)})
        for n in COMPONENTS:
            vals=[]
            for _,_,_,t,prob,cls in cache:
                pred=np.where(prob>=th,cls,0).astype(np.int16); pred=filter_cc(pred,n); vals.append(metrics(pred,t))
            comp.append({"threshold":th,"min_component_voxels":n,
                "mean_foreground_dice":np.mean([x["foreground_dice"] for x in vals]),
                "mean_macro_foreground_dice":np.mean([x["macro_foreground_dice"] for x in vals]),
                "mean_precision":np.mean([x["precision"] for x in vals]),
                "mean_recall":np.mean([x["recall"] for x in vals]),
                "mean_pred_target_volume_ratio":np.mean([x["pred_target_volume_ratio"] for x in vals]),
                "pred_foreground_cases":sum(x["pred_foreground_voxels"]>0 for x in vals)})
    td=pd.DataFrame(rows); cd=pd.DataFrame(comp)
    td.to_csv(OUT/"part127_threshold_results.csv",index=False); pd.DataFrame(case_rows).to_csv(OUT/"part127_case_results.csv",index=False); cd.to_csv(OUT/"part127_component_filter_results.csv",index=False)
    best=cd.sort_values(["mean_macro_foreground_dice","mean_precision"],ascending=False).iloc[0].to_dict()
    raw=td.sort_values("mean_macro_foreground_dice",ascending=False).iloc[0].to_dict()
    result={"part":127,"clinical_validation":False,"part104_sha256":actual,"validation_cases":len(df),"completed":len(cache),"failures":failures,"best_raw_threshold":raw,"best_component":best,"full_shape":FULL,"roi":ROI}
    (OUT/"part127_final_metrics.json").write_text(json.dumps(result,indent=2,default=float),encoding="utf-8")
    report=ROOT/"reports"/"part127_part104_probability_calibration_report.txt"
    report.write_text(json.dumps(result,indent=2,default=float)+"\n\nClinical validation: NO\n",encoding="utf-8")
    print("\n"+"="*96); print("PART127 COMPLETED"); print("="*96)
    print("Best raw threshold :",raw["threshold"]," macro Dice:",raw["mean_macro_foreground_dice"])
    print("Best component     :",best["threshold"],"min voxels:",int(best["min_component_voxels"])," macro Dice:",best["mean_macro_foreground_dice"])
    print("Threshold CSV :",OUT/"part127_threshold_results.csv"); print("Case CSV      :",OUT/"part127_case_results.csv")
    print("Metrics JSON  :",OUT/"part127_final_metrics.json"); print("Report        :",report)
    print("Part104 protected : YES"); print("Clinical validation : NO")

if __name__=="__main__": main()
