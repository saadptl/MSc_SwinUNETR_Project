from __future__ import annotations
import gc, importlib.util, json, math, sys
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
OUT=ROOT/"outputs/segmentation/rsna_part65_point_to_pseudomask_alignment_forensics"
REPORT=OUT/"reports"
VAL_N=100; FULL=(64,96,96); CROP=(32,64,64); RADIUS=2; RADII=(1,2,3)
CLASSES={1:"Spinal_Canal_Stenosis",2:"Left_Neural_Foraminal_Narrowing",3:"Right_Neural_Foraminal_Narrowing",4:"Left_Subarticular_Stenosis",5:"Right_Subarticular_Stenosis"}
NAME_TO_ID={v:k for k,v in CLASSES.items()}
NAME_TO_ID.update({"Spinal Canal Stenosis":1,"Left Neural Foraminal Narrowing":2,"Right Neural Foraminal Narrowing":3,"Left Subarticular Stenosis":4,"Right Subarticular Stenosis":5})

def banner(s): print("\n"+"="*82+"\n"+s+"\n"+"="*82)
def imp(path,name):
    sp=importlib.util.spec_from_file_location(name,str(path))
    if sp is None or sp.loader is None: raise ImportError(path)
    m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m

def offsets(r):
    return [(a,b,c) for a in range(-r,r+1) for b in range(-r,r+1) for c in range(-r,r+1) if abs(a)+abs(b)+abs(c)<=r]

def local_frac(m,z,y,x,c,r):
    q=offsets(r); n=hit=0
    for a,b,d in q:
        zz,yy,xx=z+a,y+b,x+d
        if 0<=zz<m.shape[0] and 0<=yy<m.shape[1] and 0<=xx<m.shape[2]:
            n+=1; hit+=int((m[zz,yy,xx]>0) if c is None else (m[zz,yy,xx]==c))
    return hit/n if n else 0.

def nearest(m,z,y,x,c,maxd=20):
    t=(m>0) if c is None else (m==c)
    if t[z,y,x]: return 0.
    for d in range(1,maxd+1):
        for a in range(-d,d+1):
            rem=d-abs(a)
            for b in range(-rem,rem+1):
                dx=rem-abs(b)
                for cc in ({0} if dx==0 else {-dx,dx}):
                    zz,yy,xx=z+a,y+b,x+cc
                    if 0<=zz<t.shape[0] and 0<=yy<t.shape[1] and 0<=xx<t.shape[2] and t[zz,yy,xx]: return float(d)
    return float("inf")

def map_native(x,y,z,shape):
    nz,nh,nw=map(float,shape); tz,th,tw=map(float,FULL)
    return ((z+.5)*tz/nz-.5,(y+.5)*th/nh-.5,(x+.5)*tw/nw-.5)

def dilate_labels_6connected(mask, radius):
    out = np.asarray(mask, dtype=np.int64).copy()
    if radius <= 0: return out
    for _ in range(int(radius)):
        prev=out.copy(); expanded=prev.copy()
        for cid in range(1,6):
            src=(prev==cid)
            if not src.any(): continue
            grown=src.copy()
            grown[:-1,:,:] |= src[1:,:,:]; grown[1:,:,:] |= src[:-1,:,:]
            grown[:,:-1,:] |= src[:,1:,:]; grown[:,1:,:] |= src[:,:-1,:]
            grown[:,:,:-1] |= src[:,:,1:]; grown[:,:,1:] |= src[:,:,:-1]
            expanded[(expanded==0)&grown]=cid
        out=expanded
    return out


def centered_crop_local(image, mask, crop_shape):
    # Local foreground-centered crop; Part 11 does not expose centered_crop().
    import numpy as np

    image = np.asarray(image)
    mask = np.asarray(mask)

    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {image.shape} vs {mask.shape}"
        )

    dz, dy, dx = map(int, crop_shape)
    sz, sy, sx = mask.shape

    if dz > sz or dy > sy or dx > sx:
        raise ValueError(
            f"Crop {crop_shape} is larger than volume {mask.shape}"
        )

    fg = np.argwhere(mask > 0)

    if len(fg):
        cz, cy, cx = np.mean(fg, axis=0)
    else:
        cz = (sz - 1) / 2.0
        cy = (sy - 1) / 2.0
        cx = (sx - 1) / 2.0

    def start(center, crop, size):
        s = int(round(float(center) - float(crop) / 2.0))
        return max(0, min(s, size - crop))

    z0 = start(cz, dz, sz)
    y0 = start(cy, dy, sy)
    x0 = start(cx, dx, sx)

    return (
        image[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx],
        mask[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx],
        (z0, y0, x0),
    )


def main():
    banner("PART 65 PATH VALIDATION")
    for n,q in [("Project root",ROOT),("Part 11",P11),("Part 9",P9),("Part 15 validation cohort",VACSV),("RSNA coordinate CSV",COORD)]:
        print(f"{n:<40}: {'FOUND' if q.exists() else 'MISSING'}")
        if not q.exists(): raise FileNotFoundError(q)
    banner("PART 65 — RSNA POINT-TO-PSEUDOMASK ALIGNMENT FORENSICS")
    print(f"Validation subset : {VAL_N}\nFull volume : {FULL}\nCrop : {CROP}\nPseudo-mask radius : R{RADIUS}\nLocal radii : {list(RADII)}\nTraining performed : NO\nOptimizer used : NO\nBackward pass : NO\nSPIDER : NO\nTest set : NO\nPart 15 modified : NO")
    p11=imp(P11,"p11_part65"); p9=imp(P9,"p9_part65")
    val=pd.read_csv(VACSV).head(VAL_N); coord=pd.read_csv(COORD)
    need={"study_id","series_id","condition","level","instance_number","x","y"}
    miss=need-set(coord.columns)
    if miss: raise RuntimeError(f"Missing coordinate columns: {sorted(miss)}")
    rows=[]; cases=[]
    banner("PART 65 VALIDATION PRELOAD")
    for no,(_,r) in enumerate(val.iterrows(),1):
        im,mask,info=p11.load_case_robust(r,p9); im=np.asarray(im,np.float32); mask=np.asarray(mask,np.int64); native=tuple(map(int,im.shape))
        fullmask=p11.resize_3d(mask,FULL,is_mask=True); fullmask=dilate_labels_6connected(fullmask,RADIUS)
        _,_,st=centered_crop_local(p11.resize_3d(im,FULL,is_mask=False),fullmask,CROP)
        z0,y0,x0=st; crop=fullmask[z0:z0+CROP[0],y0:y0+CROP[1],x0:x0+CROP[2]]
        cases.append((r,native,crop,st))
        ann=coord[(coord.study_id.astype(str)==str(r["study_id"]))&(coord.series_id.astype(str)==str(r["series_id"]))]
        try: _,ds,_=p11.read_dicom_series_robust(p11.resolve_series_dir(r)); instz={int(d.get("InstanceNumber",0)):i for i,d in enumerate(ds)}
        except Exception: instz={}
        for ai,(_,a) in enumerate(ann.iterrows(),1):
            cid=NAME_TO_ID.get(str(a["condition"]).strip())
            try: xx=float(a["x"]); yy=float(a["y"]); inst=int(float(a["instance_number"]))
            except: continue
            if cid is None or inst not in instz: continue
            z=instz[inst]
            if not(0<=xx<native[2] and 0<=yy<native[1] and 0<=z<native[0]): continue
            zf,yf,xf=map_native(xx,yy,z,native); cz,cy,cx=st; zz,zy,zx=zf-cz,yf-cy,xf-cx
            zi,yi,xi=round(zz),round(zy),round(zx)
            base={"case_no":no,"study_id":str(r["study_id"]),"series_id":str(r["series_id"]),"annotation_no":ai,"condition":str(a["condition"]),"class_id":cid,"class_name":CLASSES[cid],"level":str(a.get("level","")),"instance_number":inst,"native_x":xx,"native_y":yy,"native_z":z,"full_x":xf,"full_y":yf,"full_z":zf,"crop_x":zx,"crop_y":zy,"crop_z":zz,"voxel_x":xi,"voxel_y":yi,"voxel_z":zi,"inside_final_crop":bool(0<=zi<CROP[0] and 0<=yi<CROP[1] and 0<=xi<CROP[2])}
            if base["inside_final_crop"]:
                lab=int(crop[zi,yi,xi]); base.update({"point_label":lab,"point_label_name":"Background" if lab==0 else CLASSES.get(lab,f"Class_{lab}"),"point_any_foreground":bool(lab>0),"point_same_class":bool(lab==cid),"nearest_any_foreground_distance":nearest(crop,zi,yi,xi,None),"nearest_same_class_distance":nearest(crop,zi,yi,xi,cid)})
                for rr in RADII:
                    base[f"any_fg_fraction_r{rr}"]=local_frac(crop,zi,yi,xi,None,rr); base[f"same_class_fraction_r{rr}"]=local_frac(crop,zi,yi,xi,cid,rr)
            else:
                base.update({"point_label":np.nan,"point_label_name":"OUTSIDE_CROP","point_any_foreground":np.nan,"point_same_class":np.nan,"nearest_any_foreground_distance":np.inf,"nearest_same_class_distance":np.inf})
                for rr in RADII: base[f"any_fg_fraction_r{rr}"]=np.nan; base[f"same_class_fraction_r{rr}"]=np.nan
            rows.append(base)
        if no in (1,10,20,25,40,50,75,100): print(f"VALIDATION {no:03d}/{len(val)} annotations={len(ann)}")
    df=pd.DataFrame(rows); inside=df[df.inside_final_crop]
    banner("PART 65 SHAPE / ALIGNMENT SUMMARY")
    print(f"Annotation rows : {len(df)}\nInside final crop : {len(inside)}\nOutside final crop : {len(df)-len(inside)}\nCrop survival : {len(inside)/len(df):.4f}")
    if inside.empty: raise RuntimeError("No annotations survived the final crop.")
    print(f"Point on any foreground : {inside.point_any_foreground.mean():.4f}\nPoint on SAME class : {inside.point_same_class.mean():.4f}\nMean nearest any-FG distance : {inside.nearest_any_foreground_distance.replace(np.inf,np.nan).mean():.4f}\nMedian nearest any-FG distance : {inside.nearest_any_foreground_distance.replace(np.inf,np.nan).median():.4f}\nMean nearest SAME-class distance : {inside.nearest_same_class_distance.replace(np.inf,np.nan).mean():.4f}\nMedian nearest SAME-class distance : {inside.nearest_same_class_distance.replace(np.inf,np.nan).median():.4f}")
    for rr in RADII: print(f"R{rr} same-class local hit rate : {(inside[f'same_class_fraction_r{rr}']>0).mean():.4f}")
    banner("PART 65 PER-CLASS ALIGNMENT")
    for cid,name in CLASSES.items():
        g=inside[inside.class_id==cid]
        print(f"C{cid} {name:<36} n={len(g)} same-class={g.point_same_class.mean() if len(g) else float('nan'):.4f} nearest_same={g.nearest_same_class_distance.replace(np.inf,np.nan).mean() if len(g) else float('nan'):.3f} localR2={(g.same_class_fraction_r2>0).mean() if len(g) else float('nan'):.4f}")
    same=float(inside.point_same_class.mean()); hit=float((inside.same_class_fraction_r2>0).mean())
    med=float(inside.nearest_same_class_distance.replace(np.inf,np.nan).median())
    if same>=.8 and hit>=.9 and med<=1: diag="STRONG_POINT_TO_PSEUDOMASK_ALIGNMENT"
    elif same>=.5 and hit>=.75: diag="MODERATE_POINT_TO_PSEUDOMASK_ALIGNMENT"
    elif hit>=.5: diag="WEAK_POINT_TO_PSEUDOMASK_ALIGNMENT"
    else: diag="POOR_POINT_TO_PSEUDOMASK_ALIGNMENT"
    banner("PART 65 DIAGNOSTIC INTERPRETATION")
    print(f"Diagnosis : {diag}\nPoint same-class rate : {same:.4f}\nR2 same-class local hit rate : {hit:.4f}\nMedian nearest same-class distance : {med:.4f}")
    print("\nScientific limitation: RSNA coordinates are point annotations, not manual segmentation masks. Alignment is a target-consistency diagnostic, not medical segmentation accuracy.")
    REPORT.mkdir(parents=True,exist_ok=True)
    df.to_csv(REPORT/"part65_annotation_alignment.csv",index=False)
    summ=[]
    for label,g in [("ALL",inside)]+[(f"C{c}_{n}",inside[inside.class_id==c]) for c,n in CLASSES.items()]:
        if g.empty: continue
        x={"group":label,"annotations":len(g),"same_class_rate":float(g.point_same_class.mean()),"any_foreground_rate":float(g.point_any_foreground.mean()),"mean_nearest_same_class":float(g.nearest_same_class_distance.replace(np.inf,np.nan).mean()),"median_nearest_same_class":float(g.nearest_same_class_distance.replace(np.inf,np.nan).median())}
        for rr in RADII:x[f"same_class_hit_r{rr}"]=float((g[f"same_class_fraction_r{rr}"]>0).mean())
        summ.append(x)
    pd.DataFrame(summ).to_csv(REPORT/"part65_alignment_summary.csv",index=False)
    case=pd.DataFrame([{"case_no":i,"study_id":c[0]["study_id"],"series_id":c[0]["series_id"],"crop_start_z":c[3][0],"crop_start_y":c[3][1],"crop_start_x":c[3][2]} for i,c in enumerate(cases,1)])
    case.to_csv(REPORT/"part65_case_crop_geometry.csv",index=False)
    out={"part":65,"validation_subset":len(val),"annotation_rows":len(df),"inside_crop":len(inside),"outside_crop":len(df)-len(inside),"crop_survival_rate":len(inside)/len(df),"point_any_foreground_rate":float(inside.point_any_foreground.mean()),"point_same_class_rate":same,"median_nearest_same_class_distance":med,"same_class_local_hit_rates_r1_r2_r3":{str(rr):float((inside[f"same_class_fraction_r{rr}"]>0).mean()) for rr in RADII},"diagnosis":diag,"training_performed":False,"optimizer_used":False,"backward_pass":False,"spider_used":False,"test_set_used":False,"part15_modified":False}
    with open(REPORT/"part65_summary.json","w",encoding="utf-8") as f: json.dump(out,f,indent=2)
    banner("PART 65 COMPLETE"); print(f"Annotation alignment CSV : {REPORT/'part65_annotation_alignment.csv'}\nAlignment summary CSV : {REPORT/'part65_alignment_summary.csv'}\nCase crop geometry CSV : {REPORT/'part65_case_crop_geometry.csv'}\nSummary JSON : {REPORT/'part65_summary.json'}")
    gc.collect()

if __name__=="__main__": main()
