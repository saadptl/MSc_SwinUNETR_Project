from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
P11=ROOT/"src/segmentation_rsna_part11_controlled_pilot_training.py"
P9=ROOT/"src/segmentation_rsna_part9_3d_dataset_loader.py"
P15=ROOT/"outputs/segmentation/rsna_part15_extended_controlled_training"
VACSV=P15/"part15_validation_cohort.csv"
RSNA=ROOT/"dataset/rsna-2024-lumbar-spine-degenerative-classification"
COORD=RSNA/"train_label_coordinates.csv"
OUT=ROOT/"outputs/segmentation/rsna_part67_point_pseudomask_label_consistency_forensics"
REPORT=OUT/"reports"

N=100; FULL=(64,96,96); CROP=(32,64,64); R=2
CLASSES={1:"Spinal_Canal_Stenosis",2:"Left_Neural_Foraminal_Narrowing",3:"Right_Neural_Foraminal_Narrowing",4:"Left_Subarticular_Stenosis",5:"Right_Subarticular_Stenosis"}
NAME={v:k for k,v in CLASSES.items()}
NAME.update({"Spinal Canal Stenosis":1,"Left Neural Foraminal Narrowing":2,"Right Neural Foraminal Narrowing":3,"Left Subarticular Stenosis":4,"Right Subarticular Stenosis":5})

def banner(s): print("\n"+"="*82+"\n"+s+"\n"+"="*82)
def loadmod(path,name):
    sp=importlib.util.spec_from_file_location(name,str(path))
    if sp is None or sp.loader is None: raise ImportError(path)
    m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m

def dil6(a,n):
    a=a.astype(bool).copy()
    for _ in range(n):
        b=a.copy()
        b[1:]|=a[:-1]; b[:-1]|=a[1:]
        b[:,1:]|=a[:,:-1]; b[:,:-1]|=a[:,1:]
        b[:,:,1:]|=a[:,:,:-1]; b[:,:,:-1]|=a[:,:,1:]
        a=b
    return a

def dilate_labels(m,n):
    if n==0:return m.astype(np.int64).copy()
    parts=[]
    for c in range(1,6):
        q=m==c
        if q.any():parts.append((int(q.sum()),c,dil6(q,n)))
    parts.sort(key=lambda x:(-x[0],x[1]))
    o=np.zeros_like(m,dtype=np.int64); used=np.zeros_like(m,bool)
    for _,c,q in parts:q=q&~used;o[q]=c;used|=q
    return o

def map_coord(x,y,z,native):
    nz,nh,nw=map(float,native)
    return ((z+.5)*FULL[0]/nz-.5,(y+.5)*FULL[1]/nh-.5,(x+.5)*FULL[2]/nw-.5)

def crop_mask(m,center):
    st=[max(0,min(int(round(center[i]))-CROP[i]//2,FULL[i]-CROP[i])) for i in range(3)]
    z,y,x=st; dz,dy,dx=CROP
    return m[z:z+dz,y:y+dy,x:x+dx],tuple(st)

def offsets(r):
    return [(a,b,c) for a in range(-r,r+1) for b in range(-r,r+1) for c in range(-r,r+1) if abs(a)+abs(b)+abs(c)<=r]

def local_label_fractions(m,z,y,x,c,r):
    vals=[]; correct=fg=0
    for a,b,d in offsets(r):
        zz,yy,xx=z+a,y+b,x+d
        if 0<=zz<m.shape[0] and 0<=yy<m.shape[1] and 0<=xx<m.shape[2]:
            v=int(m[zz,yy,xx]); vals.append(v); fg+=int(v>0); correct+=int(v==c)
    n=len(vals)
    return (correct/n if n else 0.,fg/n if n else 0.)

def nearest_dist(m,z,y,x,c,maxd=20):
    target=(m==c)
    if 0<=z<m.shape[0] and 0<=y<m.shape[1] and 0<=x<m.shape[2] and target[z,y,x]:return 0.
    for d in range(1,maxd+1):
        for a,b,cc in offsets(d):
            if abs(a)+abs(b)+abs(cc)!=d:continue
            zz,yy,xx=z+a,y+b,x+cc
            if 0<=zz<m.shape[0] and 0<=yy<m.shape[1] and 0<=xx<m.shape[2] and target[zz,yy,xx]:return float(d)
    return float("inf")

def load_case(p11,p9,row,coord):
    im,m,_=p11.load_case_robust(row,p9)
    im=np.asarray(im,np.float32); m=np.asarray(m,np.int64); native=m.shape
    im=p11.resize_3d(im,FULL,is_mask=False)
    m=dilate_labels(p11.resize_3d(m,FULL,is_mask=True),R)
    sid,ser=str(row["study_id"]),str(row["series_id"])
    ann=coord[(coord.study_id.astype(str)==sid)&(coord.series_id.astype(str)==ser)]
    _,ds,_=p11.read_dicom_series_robust(p11.resolve_series_dir(row))
    inst_to_z={int(d.get("InstanceNumber",0)):i for i,d in enumerate(ds)}
    pts=[]
    for _,a in ann.iterrows():
        cid=NAME.get(str(a["condition"]).strip())
        try:x,y,inst=float(a["x"]),float(a["y"]),int(float(a["instance_number"]))
        except:continue
        if cid is None or inst not in inst_to_z:continue
        z=inst_to_z[inst]
        if not(0<=x<native[2] and 0<=y<native[1] and 0<=z<native[0]):continue
        zz,yy,xx=map_coord(x,y,z,native)
        pts.append({"class_id":cid,"class_name":CLASSES[cid],"z":zz,"y":yy,"x":xx,"level":str(a.get("level",""))})
    return im,m,pts

def main():
    banner("PART 67 PATH VALIDATION")
    for n,q in [("Project root",ROOT),("Part 11",P11),("Part 9",P9),("Part 15 validation cohort",VACSV),("RSNA coordinate CSV",COORD)]:
        print(f"{n:<40}: {'FOUND' if q.exists() else 'MISSING'}")
        if not q.exists():raise FileNotFoundError(q)
    banner("PART 67 — POINT-TO-PSEUDOMASK LABEL CONSISTENCY FORENSICS")
    print(f"Validation subset : {N}\nFull volume : {FULL}\nCrop : {CROP}\nPseudo-mask radius : R{R}\nLocal neighborhood radii : [1,2,3]\nTraining performed : NO\nOptimizer used : NO\nBackward pass : NO\nSPIDER : NO\nTest set : NO\nPart 15 modified : NO")
    p11,p9=loadmod(P11,"p11_67"),loadmod(P9,"p9_67")
    df=pd.read_csv(VACSV).head(N); coord=pd.read_csv(COORD)
    cases=[]
    banner("PART 67 VALIDATION PRELOAD")
    for i,(_,row) in enumerate(df.iterrows(),1):
        im,m,pts=load_case(p11,p9,row,coord); cases.append((im,m,pts))
        if i in (1,10,20,25,40,50,75,100):print(f"VALIDATION {i:03d}/{N} FG={(m>0).sum()} points={len(pts)}")
    banner("PART 67 SHAPE / LABEL SMOKE TEST")
    print(f"Image : {cases[0][0].shape}\nMask : {cases[0][1].shape}\nLabels : {np.unique(cases[0][1]).tolist()}\nMapped points : {len(cases[0][2])}")
    print("✓ Shape / label smoke test PASSED.")
    rows=[]; classrows=[]
    for i,(im,m,pts) in enumerate(cases,1):
        q=np.argwhere(m>0); center=q.mean(0) if len(q) else np.array([31.5,47.5,47.5])
        cm,st=crop_mask(m,center)
        for p in pts:
            inside=(st[0]<=p["z"]<st[0]+CROP[0] and st[1]<=p["y"]<st[1]+CROP[1] and st[2]<=p["x"]<st[2]+CROP[2])
            rec={"case":i,"class_id":p["class_id"],"class_name":p["class_name"],"level":p["level"],"inside_crop":int(inside)}
            if not inside:
                rec.update({"point_label":"OUTSIDE_CROP","local_r1_same":np.nan,"local_r2_same":np.nan,"local_r3_same":np.nan,"nearest_same_class":np.nan,"nearest_any_fg":np.nan})
            else:
                z,y,x=[int(round(p[k]-st[j])) for j,k in enumerate(("z","y","x"))]
                if not all(0<=v<CROP[j] for j,v in enumerate((z,y,x))):
                    rec["inside_crop"]=0; rec.update({"point_label":"ROUNDING_OUTSIDE","local_r1_same":np.nan,"local_r2_same":np.nan,"local_r3_same":np.nan,"nearest_same_class":np.nan,"nearest_any_fg":np.nan})
                else:
                    lab=int(cm[z,y,x])
                    r1,_=local_label_fractions(cm,z,y,x,p["class_id"],1)
                    r2,_=local_label_fractions(cm,z,y,x,p["class_id"],2)
                    r3,_=local_label_fractions(cm,z,y,x,p["class_id"],3)
                    rec.update({"point_label":("CORRECT_CLASS" if lab==p["class_id"] else ("OTHER_FOREGROUND" if lab>0 else "BACKGROUND")),"local_r1_same":r1,"local_r2_same":r2,"local_r3_same":r3,"nearest_same_class":nearest_dist(cm,z,y,x,p["class_id"],20),"nearest_any_fg":nearest_dist(cm,z,y,x,None,20)})
            rows.append(rec)
    pdf=pd.DataFrame(rows)
    banner("PART 67 OVERALL POINT OUTCOME")
    total=len(pdf); inside=pdf.inside_crop.eq(1).sum()
    for lab in ["CORRECT_CLASS","OTHER_FOREGROUND","BACKGROUND","OUTSIDE_CROP","ROUNDING_OUTSIDE"]:
        n=int((pdf.point_label==lab).sum()); print(f"{lab:<20}: {n:4d} / {total} = {n/total:.4f}")
    print(f"Inside crop : {inside}/{total} = {inside/total:.4f}")
    banner("PART 67 LOCAL RECOVERY")
    for r in (1,2,3):
        col=f"local_r{r}_same"; g=pdf[pdf.inside_crop==1][col].dropna()
        print(f"R{r} mean same-class neighborhood fraction : {g.mean():.4f}")
    banner("PART 67 PER-CLASS MISMATCH SOURCE")
    for cid,name in CLASSES.items():
        g=pdf[pdf.class_id==cid]; ins=g[g.inside_crop==1]
        correct=(ins.point_label=="CORRECT_CLASS").mean() if len(ins) else np.nan
        bg=(ins.point_label=="BACKGROUND").mean() if len(ins) else np.nan
        other=(ins.point_label=="OTHER_FOREGROUND").mean() if len(ins) else np.nan
        outside=(g.inside_crop==0).mean()
        print(f"C{cid} {name:<38} n={len(g):3d} outside={outside:.4f} correct={correct:.4f} bg={bg:.4f} otherFG={other:.4f}")
        classrows.append({"class_id":cid,"class_name":name,"n":len(g),"outside_crop_rate":outside,"correct_class_rate_inside":correct,"background_rate_inside":bg,"other_foreground_rate_inside":other,"r1_mean":ins.local_r1_same.mean(),"r2_mean":ins.local_r2_same.mean(),"r3_mean":ins.local_r3_same.mean()})
    ins=pdf[pdf.inside_crop==1]
    if len(ins):
        bg=float((ins.point_label=="BACKGROUND").mean()); other=float((ins.point_label=="OTHER_FOREGROUND").mean()); corr=float((ins.point_label=="CORRECT_CLASS").mean())
        if bg>other and bg>corr*.5: diag="PSEUDOMASK_MISSES_MANY_RETAINED_POINTS"
        elif other>bg and other>0.20: diag="PSEUDOMASK_CLASS_ID_MISMATCH_IS_SUBSTANTIAL"
        else: diag="POINT_PSEUDOMASK_MISMATCH_HAS_MIXED_SOURCES"
    else: diag="INSUFFICIENT_INSIDE_POINTS"
    banner("PART 67 DIAGNOSTIC INTERPRETATION")
    print(f"Inside-crop points : {len(ins)}/{total} = {len(ins)/total:.4f}")
    print(f"Correct-class at point : {(ins.point_label=='CORRECT_CLASS').mean():.4f}")
    print(f"Background at point : {(ins.point_label=='BACKGROUND').mean():.4f}")
    print(f"Other foreground class : {(ins.point_label=='OTHER_FOREGROUND').mean():.4f}")
    print(f"Diagnosis : {diag}")
    print("\nScientific limitation: RSNA coordinates are point annotations, not manual segmentation masks; this is a target-consistency diagnostic, not medical segmentation accuracy.")
    REPORT.mkdir(parents=True,exist_ok=True)
    pdf.to_csv(REPORT/"part67_point_level_consistency.csv",index=False)
    pd.DataFrame(classrows).to_csv(REPORT/"part67_per_class_mismatch_summary.csv",index=False)
    result={"part":67,"validation_subset":N,"radius":R,"total_annotations":total,"inside_crop":int(inside),"diagnosis":diag,"training_performed":False,"optimizer_used":False,"part15_modified":False}
    json.dump(result,open(REPORT/"part67_summary.json","w",encoding="utf-8"),indent=2)
    banner("PART 67 COMPLETE")
    print(f"Point-level CSV : {REPORT/'part67_point_level_consistency.csv'}\nPer-class CSV : {REPORT/'part67_per_class_mismatch_summary.csv'}\nSummary JSON : {REPORT/'part67_summary.json'}")
if __name__=="__main__": main()
