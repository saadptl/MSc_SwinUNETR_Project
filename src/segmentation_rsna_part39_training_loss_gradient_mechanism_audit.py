from __future__ import annotations
"""PHASE 4 - PART 39
RSNA-ONLY TRAINING-LOSS / GRADIENT / OPTIMIZATION-MECHANISM AUDIT

Audit only: no optimizer, no optimizer.step(), no weight modification,
no SPIDER, no RSNA test set. Uses the exact Part 15 500-case cohort and
initialization/epoch1/epoch2 checkpoints to investigate the rapid foreground
collapse established by Parts 28, 29 and 38.
"""
import gc, hashlib, importlib.util, json, math, random, sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
RSNA=ROOT/"dataset"/"rsna-2024-lumbar-spine-degenerative-classification"
P11=SRC/"segmentation_rsna_part11_controlled_pilot_training_corrected.py"
P15=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"
COHORT=P15/"part15_train_cohort.csv"
MANIFEST=ROOT/"outputs"/"segmentation"/"rsna_part8_dataset_construction"/"manifests"/"rsna_part8_train_manifest.csv"
CKPT=P15/"checkpoints"
OUT=ROOT/"outputs"/"segmentation"/"rsna_part39_training_loss_gradient_mechanism_audit"
REPORT=OUT/"reports"
CHECKPOINTS={"initialization":CKPT/"part15_initialization_from_part11.pth","epoch_01":CKPT/"epoch_01.pth","epoch_02":CKPT/"epoch_02.pth"}
SEED=42; NC=6; AUDIT_CASES=12; TRAIN_CASES=500
CLASSES={0:"Background",1:"Spinal Canal Stenosis",2:"Left Neural Foraminal Narrowing",3:"Right Neural Foraminal Narrowing",4:"Left Subarticular Stenosis",5:"Right Subarticular Stenosis"}

def banner(s): print("\n"+"="*78+"\n"+s+"\n"+"="*78)
def line(k,v): print(f"{k:<42}: {v}")
def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    try: torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    except Exception: pass
def clean():
    gc.collect()
    if torch.cuda.is_available():
        try: torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        except Exception: pass
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def import_p11():
    spec=importlib.util.spec_from_file_location("part11_p39",P11)
    if not spec or not spec.loader: raise ImportError(P11)
    m=importlib.util.module_from_spec(spec); sys.modules["part11_p39"]=m; spec.loader.exec_module(m)
    for x in ("load_part9_module","load_tensor_case","create_model"):
        if not hasattr(m,x): raise AttributeError("Part 11 missing "+x)
    return m

def safe_case(part11, part9, row):
    image, mask, info = part11.load_tensor_case(row, part9)

    # Part 11 returns image as either [D,H,W] or [1,D,H,W].
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    # Mask should normally already be [D,H,W].
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]

    if image.ndim != 3:
        raise ValueError(
            f"Expected image [D,H,W], got {tuple(image.shape)}"
        )

    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask [D,H,W], got {tuple(mask.shape)}"
        )

    return (
        image.float().contiguous(),
        mask.long().contiguous(),
        info,
    )

def state_dict(ck):
    if isinstance(ck,dict):
        for k in ("model_state_dict","state_dict","model","weights"):
            if isinstance(ck.get(k),dict): return ck[k]
        if all(torch.is_tensor(v) for v in ck.values()): return ck
    raise RuntimeError("No model state dictionary found")

def load_model(p11,path,device):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    m=p11.create_model(torch.device("cpu"))
    sd={(k[7:] if k.startswith("module.") else k):v for k,v in state_dict(ck).items()}
    r=m.load_state_dict(sd,strict=False)
    if r.missing_keys or r.unexpected_keys: raise RuntimeError(f"checkpoint mismatch {r}")
    return m.to(device).train(),ck

def target_stats(mask):
    flat=mask.reshape(-1); c=torch.bincount(flat.clamp(0,NC-1),minlength=NC); n=int(flat.numel()); fg=int(c[1:].sum())
    d={"total_voxels":n,"background_voxels":int(c[0]),"foreground_voxels":fg,"foreground_fraction":fg/n if n else 0.0}
    for i in range(NC): d[f"class_{i}_voxels"]=int(c[i]); d[f"class_{i}_fraction"]=int(c[i])/n if n else 0.0
    return d

def head(model):
    xs=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Conv3d) and m.out_channels==NC]
    return xs[-1] if xs else (None,None)

def head_snapshot(model):
    n,h=head(model)
    if h is None:return {"found":False}
    w=h.weight.detach().float().cpu(); d={"found":True,"name":n,"weight_shape":list(w.shape)}
    d["weight_l2"]=float(torch.linalg.vector_norm(w)); d["weight_abs_mean"]=float(w.abs().mean())
    d["weight_class_l2"]={str(c):float(torch.linalg.vector_norm(w[c])) for c in range(NC)}
    if h.bias is not None:
        b=h.bias.detach().float().cpu(); d["bias_by_class"]={str(c):float(b[c]) for c in range(NC)}
    return d

def optimizer_head_state(model,ck):
    opt=ck.get("optimizer_state_dict") if isinstance(ck,dict) else None
    if not isinstance(opt,dict): return {"available":False,"reason":"no optimizer_state_dict"}
    ids=[]
    for g in opt.get("param_groups",[]): ids += g.get("params",[])
    params=list(model.parameters()); n,h=head(model)
    if len(ids)!=len(params): return {"available":False,"reason":"optimizer/model parameter count mismatch"}
    mapping={id(p):ids[i] for i,p in enumerate(params)}
    out={"available":True,"head_name":n}
    for lab,p in (("weight",h.weight),("bias",h.bias)):
        if p is None: continue
        pid=mapping.get(id(p)); st=opt.get("state",{}).get(pid)
        if st is None: continue
        q={}
        for k in ("step","exp_avg","exp_avg_sq"):
            if k not in st: continue
            v=st[k]
            if torch.is_tensor(v):
                v=v.float(); q[k]={"mean":float(v.mean()),"abs_mean":float(v.abs().mean()),"l2":float(torch.linalg.vector_norm(v.reshape(-1))),"max_abs":float(v.abs().max())}
            else: q[k]=str(v)
        out[lab]=q
    return out

def actual_loss(logits,target):
    return DiceCELoss(to_onehot_y=True,softmax=True,include_background=False)(logits,target.unsqueeze(1))

def ce_diagnostics(logits,target):
    ce=F.cross_entropy(logits,target,reduction="none")
    d={"ce_loss":float(ce.mean().detach().cpu())}
    for c in range(NC):
        s=target==c; d[f"target_class_{c}_count"]=int(s.sum())
        d[f"target_class_{c}_mean_ce"]=float(ce[s].mean().detach().cpu()) if s.any() else 0.0
    d["background_target_mean_ce"]=float(ce[target==0].mean().detach().cpu())
    d["foreground_target_mean_ce"]=float(ce[target>0].mean().detach().cpu()) if (target>0).any() else 0.0
    return d,ce

def gradient_diagnostics(model,logits,target):
    # Actual Part 15 loss; gradients are computed but never assigned to model .grad.
    loss=actual_loss(logits,target)
    params=[p for p in model.parameters() if p.requires_grad]
    grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    sq=sum(float((g.detach().float()**2).sum().cpu()) for g in grads if g is not None)
    logg=torch.autograd.grad(loss,logits,retain_graph=True)[0].detach().float()
    bg=logg[:,0]; fg=logg[:,1:]
    ce=F.cross_entropy(logits,target,reduction="mean")
    ceg=torch.autograd.grad(ce,logits,retain_graph=False)[0].detach().float()
    cb=ceg[:,0]; cf=ceg[:,1:]
    ba=float(bg.abs().mean().cpu()); fa=float(fg.abs().mean().cpu()); cba=float(cb.abs().mean().cpu()); cfa=float(cf.abs().mean().cpu())
    return {"actual_dicece_loss":float(loss.detach().cpu()),"total_parameter_gradient_l2":math.sqrt(sq),"gradient_parameter_tensors_nonzero":sum(1 for g in grads if g is not None and torch.any(g!=0)),"background_logit_gradient_mean_abs":ba,"foreground_logit_gradient_mean_abs":fa,"foreground_background_gradient_ratio":fa/max(ba,1e-12),"ce_background_logit_gradient_mean_abs":cba,"ce_foreground_logit_gradient_mean_abs":cfa,"ce_foreground_background_gradient_ratio":cfa/max(cba,1e-12)}

def forward_diag(logits,target):
    p=torch.softmax(logits.detach().float(),1); pred=torch.argmax(logits.detach().float(),1)[0]; t=target[0]; fg=pred>0; tf=t>0; tp=int((fg&tf).sum()); fp=int((fg&~tf).sum()); fn=int((~fg&tf).sum()); dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0
    return {"pred_foreground_voxels":int(fg.sum()),"target_foreground_voxels":int(tf.sum()),"background_prediction_fraction":float((pred==0).float().mean()),"mean_background_probability":float(p[:,0].mean().cpu()),"mean_best_foreground_probability":float(p[:,1:].max(1).values.mean().cpu()),"foreground_dice":dice,"tp":tp,"fp":fp,"fn":fn}

def audit_case(model,row,p11,p9,device):
    image,mask,info=safe_case(p11,p9,row); x=image.unsqueeze(0).to(device); y=mask.unsqueeze(0).to(device); logits=model(x)
    d={"study_id":str(row.study_id),"series_id":str(row.series_id),"loader":info.get("part11_loader","")}; d.update(target_stats(mask.cpu())); d.update(forward_diag(logits,y)); ce,_=ce_diagnostics(logits,y); d.update(ce); d.update(gradient_diagnostics(model,logits,y)); return d

def main():
    seed(); banner("PHASE 4 - PART 39")
    print("RSNA-ONLY TRAINING-LOSS / GRADIENT / OPTIMIZATION-MECHANISM AUDIT")
    print("Evaluation only. NO optimizer. NO optimizer.step(). NO weight modification. NO SPIDER. NO RSNA test set.")
    required=[RSNA,MANIFEST,P11,COHORT,*CHECKPOINTS.values()]; missing=[str(x) for x in required if not x.exists()]
    if missing: raise FileNotFoundError("Missing required input(s):\n"+"\n".join(missing))
    OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); line("Device",device); line("PyTorch",torch.__version__)
    if device.type=="cuda": line("GPU",torch.cuda.get_device_name(0)); line("GPU memory",f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    line("Audit cases",AUDIT_CASES); line("Patch size","(64, 96, 96)"); line("Classes",NC)
    banner("IMPORTING VALIDATED PART 11"); p11=import_p11(); p9=p11.load_part9_module(); line("Part 11","IMPORTED"); line("Part 9","IMPORTED THROUGH PART 11")
    banner("EXACT PART 15 TRAIN COHORT"); rows=pd.read_csv(COHORT).reset_index(drop=True)
    if len(rows)!=TRAIN_CASES: raise ValueError(f"Expected {TRAIN_CASES} rows, got {len(rows)}")
    audit=rows.head(AUDIT_CASES).copy(); line("Cohort rows",len(rows)); line("Audit rows",len(audit))
    banner("TARGET IMBALANCE AUDIT"); targets=[]
    for i,(_,r) in enumerate(audit.iterrows(),1):
        try:
            _,m,info=safe_case(p11,p9,r); s=target_stats(m.cpu()); targets.append({"case_index":i,"study_id":str(r.study_id),"series_id":str(r.series_id),**s}); print(f"[{i:02d}] fg={s['foreground_voxels']} fraction={s['foreground_fraction']:.8f}")
        except Exception as e: print(f"[{i:02d}] ERROR {e}")
    tdf=pd.DataFrame(targets); tdf.to_csv(OUT/"part39_target_imbalance_metrics.csv",index=False)
    banner("CHECKPOINT OUTPUT-HEAD / SAVED ADAMW STATE AUDIT"); snaps={}
    for label,path in CHECKPOINTS.items():
        clean(); m=ck=None
        try:
            line("Checkpoint",label); line("SHA256",sha(path)); m,ck=load_model(p11,path,device); meta={k:ck.get(k) for k in ("epoch","best_val_dice","seed") if isinstance(ck,dict)}; snaps[label]={"metadata":meta,"head":head_snapshot(m),"optimizer":optimizer_head_state(m,ck)}; line("Epoch",meta.get("epoch")); line("Recorded best Dice",meta.get("best_val_dice")); line("Output head",head(m)[0])
        finally:
            if m is not None: del m
            if ck is not None: del ck
            clean()
    (OUT/"part39_checkpoint_head_optimizer_snapshot.json").write_text(json.dumps(snaps,indent=2,default=str),encoding="utf-8")
    banner("REAL TRAINING-SAMPLE LOSS / GRADIENT AUDIT"); records=[]
    for label,path in CHECKPOINTS.items():
        clean(); m=ck=None; print(f"\nCHECKPOINT: {label}")
        try:
            m,ck=load_model(p11,path,device)
            for i,(_,r) in enumerate(audit.iterrows(),1):
                try:
                    d=audit_case(m,r,p11,p9,device); d.update({"checkpoint":label,"case_index":i}); records.append(d); print(f"[{i:02d}] fg={d['target_foreground_voxels']} pred_fg={d['pred_foreground_voxels']} loss={d['actual_dicece_loss']:.6f} bg_prob={d['mean_background_probability']:.6f} FG/BG_grad={d['foreground_background_gradient_ratio']:.6f}")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower(): print(f"[{i:02d}] CUDA OOM; skipped"); clean()
                    else: print(f"[{i:02d}] ERROR {e}")
                except Exception as e: print(f"[{i:02d}] ERROR {e}")
        finally:
            if m is not None: del m
            if ck is not None: del ck
            clean()
    df=pd.DataFrame(records)
    if df.empty: raise RuntimeError("No Part 39 case results")
    df.to_csv(OUT/"part39_case_loss_gradient_metrics.csv",index=False)
    agg=df.groupby("checkpoint",sort=False).agg(cases=("case_index","count"),mean_target_fg=("target_foreground_voxels","mean"),mean_target_fg_fraction=("foreground_fraction","mean"),mean_pred_fg=("pred_foreground_voxels","mean"),mean_bg_probability=("mean_background_probability","mean"),mean_best_fg_probability=("mean_best_foreground_probability","mean"),mean_dice=("foreground_dice","mean"),mean_actual_dicece_loss=("actual_dicece_loss","mean"),mean_ce_loss=("ce_loss","mean"),mean_param_grad_l2=("total_parameter_gradient_l2","mean"),mean_bg_logit_grad=("background_logit_gradient_mean_abs","mean"),mean_fg_logit_grad=("foreground_logit_gradient_mean_abs","mean"),mean_fg_bg_grad_ratio=("foreground_background_gradient_ratio","mean"),mean_ce_fg_bg_ratio=("ce_foreground_background_gradient_ratio","mean")).reset_index()
    agg.to_csv(OUT/"part39_checkpoint_loss_gradient_summary.csv",index=False)
    banner("MECHANISM DECISION")
    mf=float(tdf.foreground_fraction.mean()) if not tdf.empty else None
    parts=[]
    if mf is not None and mf<0.01: parts.append("EXTREME_FOREGROUND_VOXEL_IMBALANCE")
    ai=agg.set_index("checkpoint")
    if "initialization" in ai.index and "epoch_01" in ai.index and float(ai.loc["epoch_01","mean_pred_fg"]) < max(float(ai.loc["initialization","mean_pred_fg"])*0.1,1): parts.append("RAPID_FOREGROUND_OUTPUT_SUPPRESSION_BY_EPOCH_1")
    if "epoch_01" in ai.index and "epoch_02" in ai.index and float(ai.loc["epoch_02","mean_pred_fg"]) <= max(float(ai.loc["epoch_01","mean_pred_fg"])*0.1,0): parts.append("FURTHER_FOREGROUND_COLLAPSE_BY_EPOCH_2")
    parts += ["BACKGROUND_FAVORED_OPTIMIZATION_UNDER_EXTREME_CLASS_IMBALANCE","CAUSALITY_NOT_EXCLUSIVELY_PROVEN_BY_AUDIT"] if parts else ["NO_SINGLE_MECHANISM_CONFIRMED"]
    diagnosis=";".join(parts); line("Mean foreground fraction",f"{mf:.8f}" if mf is not None else "N/A"); line("Diagnosis",diagnosis)
    summary={"audit":"Part 39","training_performed":False,"optimizer_created":False,"optimizer_step":False,"weights_modified":False,"spider_used":False,"test_set_used":False,"audit_cases":AUDIT_CASES,"successful_measurements":len(df),"mean_target_foreground_fraction":mf,"diagnosis":diagnosis}
    (OUT/"phase4_part39_training_loss_gradient_mechanism_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    rep=REPORT/"phase4_part39_training_loss_gradient_mechanism_report.txt"
    with rep.open("w",encoding="utf-8") as f:
        f.write("PHASE 4 - PART 39\nRSNA-ONLY TRAINING-LOSS / GRADIENT / OPTIMIZATION-MECHANISM AUDIT\n\n")
        f.write("Evaluation only; no optimizer, no optimizer step, no weight modification, no SPIDER, no test set.\n\n")
        f.write(f"Exact Part 15 cohort: {len(rows)} cases\nAudit cases: {AUDIT_CASES}\n")
        f.write(f"Mean target foreground fraction: {mf:.8f}\n" if mf is not None else "Mean target foreground fraction: N/A\n")
        f.write("\nCHECKPOINT SUMMARY\n")
        for _,r in agg.iterrows(): f.write(f"{r.checkpoint}: pred_fg={r.mean_pred_fg:.3f}, bg_prob={r.mean_bg_probability:.6f}, DiceCELoss={r.mean_actual_dicece_loss:.6f}, FG/BG_grad={r.mean_fg_bg_grad_ratio:.6f}\n")
        f.write(f"\nDIAGNOSIS\n{diagnosis}\n\n")
        f.write("Interpretation: Part 39 separates actual DiceCELoss behaviour, transparent CE diagnostics, output-head gradients and saved AdamW state. The leading mechanism is reported conservatively; this audit does not by itself prove that class imbalance is the sole causal factor. Metrics are against RSNA point-derived pseudo-masks, not manual clinical segmentation ground truth.\n")
    banner("PART 39 FINAL SUMMARY"); line("Successful case measurements",len(df)); line("Mean target foreground fraction",f"{mf:.8f}" if mf is not None else "N/A")
    for _,r in agg.iterrows(): print(f"{r.checkpoint}: pred_fg={r.mean_pred_fg:.3f}, bg_prob={r.mean_bg_probability:.6f}, loss={r.mean_actual_dicece_loss:.6f}, FG/BG_grad={r.mean_fg_bg_grad_ratio:.6f}")
    line("Diagnosis",diagnosis); print("\nOUTPUT DIRECTORY\n"+str(OUT)); banner("PHASE 4 - PART 39 COMPLETE")
if __name__=="__main__": main()
