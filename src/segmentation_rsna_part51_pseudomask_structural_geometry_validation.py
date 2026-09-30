from __future__ import annotations
import csv, hashlib, importlib.util, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
P11=ROOT/"src"/"segmentation_rsna_part11_controlled_pilot_training.py"
P9=ROOT/"src"/"segmentation_rsna_part9_3d_dataset_loader.py"
P15=ROOT/"outputs"/"segmentation"/"rsna_part15_extended_controlled_training"
TR=P15/"part15_train_cohort.csv"; VA=P15/"part15_validation_cohort.csv"
OUT=ROOT/"outputs"/"segmentation"/"rsna_part51_pseudomask_structural_geometry_validation"
REP=OUT/"reports"; CSV=OUT/"csv"
NTR,NVA=100,50; FULL=(64,96,96); CROP=(32,64,64); NC=6; SMALL=5
NAMES=["Background","Spinal_Canal_Stenosis","Left_Neural_Foraminal_Narrowing","Right_Neural_Foraminal_Narrowing","Left_Subarticular_Stenosis","Right_Subarticular_Stenosis"]

def loadmod(path,name):
    s=importlib.util.spec_from_file_location(name,str(path))
    if s is None or s.loader is None: raise RuntimeError(path)
    m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m

def validate():
    print("="*82); print("PART 51 PATH VALIDATION"); print("="*82)
    for n,x in [("Project root",ROOT),("Part 11",P11),("Part 9",P9),("Part 15 train cohort",TR),("Part 15 validation cohort",VA)]:
        print(f"{n:<40}: {'FOUND' if x.exists() else 'MISSING'}")
        if not x.exists(): raise FileNotFoundError(x)

def crop(img,mask):
    fg=np.argwhere(mask>0)
    if fg.size:
        c=np.round(fg.mean(0)).astype(int)
        starts=[max(0,min(int(c[i])-CROP[i]//2,FULL[i]-CROP[i])) for i in range(3)]
    else: starts=[(FULL[i]-CROP[i])//2 for i in range(3)]
    z,y,x=starts; dz,dy,dx=CROP
    return img[z:z+dz,y:y+dy,x:x+dx].astype(np.float32),mask[z:z+dz,y:y+dy,x:x+dx].astype(np.int64)

def loadcases(p11,p9,path,n,label):
    df=pd.read_csv(path).head(n)
    if len(df)!=n: raise RuntimeError(f"{label}: expected {n}, found {len(df)}")
    out=[]; print("\n"+"="*82); print(f"PART 51 {label.upper()} DATA PRELOAD"); print("-"*82)
    for i,(_,row) in enumerate(df.iterrows(),1):
        a=p11.load_tensor_case(row,p9); img,mask=a[0],a[1]
        if torch.is_tensor(img): img=img.detach().cpu().numpy()
        if torch.is_tensor(mask): mask=mask.detach().cpu().numpy()
        img=np.asarray(img); mask=np.asarray(mask)
        if img.ndim==4 and img.shape[0]==1: img=img[0]
        if mask.ndim==4 and mask.shape[0]==1: mask=mask[0]
        if tuple(img.shape)!=FULL or tuple(mask.shape)!=FULL: raise RuntimeError(f"{label} {i}: {img.shape}/{mask.shape}")
        ci,cm=crop(img.astype(np.float32),mask.astype(np.int64)); out.append(cm)
        if i==1 or i==n or i%25==0: print(f"{label.upper()} {i:03d}/{n} FG={(cm>0).sum()}")
    return out

def comps(a):
    a=np.asarray(a,bool)
    if not a.any(): return []
    seen=np.zeros(a.shape,bool); D,H,W=a.shape; sizes=[]
    for z in range(D):
      for y in range(H):
       for x in range(W):
        if not a[z,y,x] or seen[z,y,x]: continue
        st=[(z,y,x)]; seen[z,y,x]=1; sz=0
        while st:
         cz,cy,cx=st.pop(); sz+=1
         for nz,ny,nx in ((cz-1,cy,cx),(cz+1,cy,cx),(cz,cy-1,cx),(cz,cy+1,cx),(cz,cy,cx-1),(cz,cy,cx+1)):
          if 0<=nz<D and 0<=ny<H and 0<=nx<W and a[nz,ny,nx] and not seen[nz,ny,nx]:
           seen[nz,ny,nx]=1; st.append((nz,ny,nx))
        sizes.append(sz)
    return sorted(sizes,reverse=True)

def geom(mask,cid):
    b=mask==cid; n=int(b.sum())
    if not n: return dict(voxels=0,components=0,largest=0,median=0,small_fraction=0,bbox_z=0,bbox_y=0,bbox_x=0,active_z=0)
    q=np.argwhere(b); lo=q.min(0); hi=q.max(0); sz=comps(b)
    return dict(voxels=n,components=len(sz),largest=int(sz[0]),median=float(np.median(sz)),small_fraction=sum(x for x in sz if x<SMALL)/n,bbox_z=int(hi[0]-lo[0]+1),bbox_y=int(hi[1]-lo[1]+1),bbox_x=int(hi[2]-lo[2]+1),active_z=int(b.any((1,2)).sum()))

def analyze(mask,split,i):
    fg=mask>0; fc=comps(fg); q={"split":split,"case":i,"fg_voxels":int(fg.sum()),"fg_fraction":float(fg.mean()),"fg_components":len(fc),"fg_largest":int(fc[0]) if fc else 0,"active_z":int(fg.any((1,2)).sum()),"z_fraction":float(fg.any((1,2)).mean()),"z_transitions":int(np.abs(np.diff(fg.any((1,2)).astype(np.int8))).sum()),"invalid":int(((mask<0)|(mask>=NC)).sum())}
    for c in range(NC):
        for k,v in geom(mask,c).items(): q[f"c{c}_{k}"]=v
    return q

def summary(rows,split):
    r=[x for x in rows if x["split"]==split]
    s={"split":split,"n":len(r)}
    for k in ["fg_voxels","fg_fraction","fg_components","fg_largest","z_fraction","z_transitions"]:
        s["mean_"+k]=float(np.mean([x[k] for x in r])); s["median_"+k]=float(np.median([x[k] for x in r]))
    s["zero_fg"]=sum(x["fg_voxels"]==0 for x in r); s["invalid_cases"]=sum(x["invalid"]>0 for x in r); s["classes"]={}
    for c in range(NC):
        vals=[x[f"c{c}_voxels"] for x in r]; present=[v>0 for v in vals]
        d={"name":NAMES[c],"total_voxels":int(sum(vals)),"mean_voxels":float(np.mean(vals)),"median_voxels":float(np.median(vals)),"cases_present":int(sum(present))}
        for k in ["components","largest","median","small_fraction","bbox_z","bbox_y","bbox_x","active_z"]:
            a=[x[f"c{c}_{k}"] for x in r if x[f"c{c}_voxels"]>0]; d["mean_"+k]=float(np.mean(a)) if a else 0.0
        s["classes"][str(c)]=d
    return s

def main():
    validate(); print("\n"+"="*82); print("PART 51 — PSEUDO-MASK STRUCTURAL / GEOMETRY VALIDATION"); print("="*82)
    print(f"Train subset : {NTR}\nValidation subset : {NVA}\nFull volume : {FULL}\nCrop : {CROP}\nSmall component threshold : < {SMALL} voxels\nTraining performed : NO\nOptimizer used : NO\nSPIDER used : NO\nTest set used : NO\nPart 15 overwritten : NO")
    p11=loadmod(P11,"p11_part51"); p9=loadmod(P9,"p9_part51")
    train=loadcases(p11,p9,TR,NTR,"train"); val=loadcases(p11,p9,VA,NVA,"validation")
    print("\n"+"="*82); print("PART 51 SHAPE / LABEL SMOKE TEST"); print("-"*82)
    for i,m in enumerate(train[:3],1):
        print(f"Case {i}: mask={m.shape} FG={(m>0).sum()} labels={sorted(np.unique(m).tolist())}")
        assert tuple(m.shape)==CROP and m.min()>=0 and m.max()<NC
    print("✓ Shape / label smoke test PASSED.")
    rows=[]
    for split,cases in [("train",train),("validation",val)]:
        print("\n"+"="*82); print(f"PART 51 {split.upper()} STRUCTURAL ANALYSIS"); print("="*82)
        for i,m in enumerate(cases,1):
            q=analyze(m,split,i); rows.append(q)
            if i==1 or i==len(cases) or i%25==0: print(f"{split.upper()} {i:03d}/{len(cases)} FG={q['fg_voxels']} components={q['fg_components']} largest={q['fg_largest']} activeZ={q['z_fraction']:.3f}")
    CSV.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
    fields=[] 
    for q in rows:
        for k in q:
            if k not in fields: fields.append(k)
    with open(CSV/"part51_case_geometry.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    ts,vs=summary(rows,"train"),summary(rows,"validation")
    sparse=ts["mean_fg_fraction"]<.01 and vs["mean_fg_fraction"]<.01
    fragmented=ts["mean_fg_components"]>=10 and vs["mean_fg_components"]>=10
    tiny=ts["mean_fg_largest"]<10 and vs["mean_fg_largest"]<10
    diagnosis=("HIGHLY_SPARSE_AND_FRAGMENTED_PSEUDOMASK_STRUCTURE" if sparse and fragmented else "HIGHLY_SPARSE_PSEUDOMASK_STRUCTURE" if sparse else "FRAGMENTED_PSEUDOMASK_STRUCTURE" if fragmented else "VERY_SMALL_FOREGROUND_COMPONENTS" if tiny else "NO_EXTREME_STRUCTURAL_PATHOLOGY_DETECTED")
    data={"part":51,"configuration":{"train_subset":NTR,"validation_subset":NVA,"full_shape":FULL,"crop_shape":CROP,"crop_policy":"foreground-centered","small_component_threshold":SMALL,"class_names":NAMES},"method":{"training":False,"optimizer":False,"spider":False,"test_set":False,"part15_overwritten":False},"train_summary":ts,"validation_summary":vs,"diagnosis":diagnosis,"note":"Structural pseudo-mask diagnostic only; not medical segmentation ground truth."}
    with open(REP/"part51_summary.json","w",encoding="utf-8") as f: json.dump(data,f,indent=2)
    print("\n"+"="*82); print("PART 51 SUMMARY"); print("="*82)
    for s in [ts,vs]:
        print(f"\n{s['split'].upper()}: mean FG={s['mean_fg_voxels']:.2f}, median FG={s['median_fg_voxels']:.2f}, mean FG fraction={s['mean_fg_fraction']:.6f}, mean components={s['mean_fg_components']:.2f}, mean largest={s['mean_fg_largest']:.2f}, active-Z={s['mean_z_fraction']:.4f}, zero-FG={s['zero_fg']}/{s['n']}")
    print("\nPER-CLASS")
    for c in range(NC):
        a=ts["classes"][str(c)]; b=vs["classes"][str(c)]
        print(f"Class {c} {NAMES[c]}: train present {a['cases_present']}/{NTR}, mean voxels {a['mean_voxels']:.2f}, components {a['mean_components']:.2f}, largest {a['mean_largest']:.2f}; val present {b['cases_present']}/{NVA}, mean voxels {b['mean_voxels']:.2f}, components {b['mean_components']:.2f}, largest {b['mean_largest']:.2f}")
    print(f"\nDiagnosis : {diagnosis}")
    print(f"Case geometry CSV : {CSV/'part51_case_geometry.csv'}")
    print(f"Summary JSON : {REP/'part51_summary.json'}")
    print("\n"+"="*82); print("PART 51 COMPLETE"); print("="*82)
if __name__=="__main__": main()
