from pathlib import Path
import importlib.util, sys, json, numpy as np, pandas as pd

ROOT=Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
P11=ROOT/"src/segmentation_rsna_part11_controlled_pilot_training.py"
P9=ROOT/"src/segmentation_rsna_part9_3d_dataset_loader.py"
P15=ROOT/"outputs/segmentation/rsna_part15_extended_controlled_training"
VACSV=P15/"part15_validation_cohort.csv"
COORD=ROOT/"dataset/rsna-2024-lumbar-spine-degenerative-classification"/"train_label_coordinates.csv"
OUT=ROOT/"outputs/segmentation/rsna_part66_point_aware_crop_coverage_diagnostic"
REPORT=OUT/"reports"
N=100; FULL=(64,96,96); CROP=(32,64,64); R=2
CLASSES={1:"Spinal_Canal_Stenosis",2:"Left_Neural_Foraminal_Narrowing",3:"Right_Neural_Foraminal_Narrowing",4:"Left_Subarticular_Stenosis",5:"Right_Subarticular_Stenosis"}
NAME={v:k for k,v in CLASSES.items()}
NAME.update({"Spinal Canal Stenosis":1,"Left Neural Foraminal Narrowing":2,"Right Neural Foraminal Narrowing":3,"Left Subarticular Stenosis":4,"Right Subarticular Stenosis":5})

def mod(p,n):
    s=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
def banner(s): print("\n"+"="*82+"\n"+s+"\n"+"="*82)
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
def rz(x,y,z,native):
    nz,nh,nw=map(float,native); tz,th,tw=map(float,FULL)
    return ((z+.5)*tz/nz-.5,(y+.5)*th/nh-.5,(x+.5)*tw/nw-.5)
def crop(im,m,center):
    st=[max(0,min(int(round(center[i]))-CROP[i]//2,FULL[i]-CROP[i])) for i in range(3)]
    z,y,x=st; dz,dy,dx=CROP
    return im[z:z+dz,y:y+dy,x:x+dx],m[z:z+dz,y:y+dy,x:x+dx],tuple(st)
def point_center(pts):
    if pts:return np.mean([[p["z"],p["y"],p["x"]] for p in pts],axis=0)
    return np.array([31.5,47.5,47.5])
def pcoverage(pts,st):
    if not pts:return 0,0,0,0
    z,y,x=st; q=[p for p in pts if z<=p["z"]<z+CROP[0] and y<=p["y"]<y+CROP[1] and x<=p["x"]<x+CROP[2]]
    return len(q)/len(pts),len(q),len(set(p["class_id"] for p in q)),len(set(p["level"] for p in q if p["level"]))
def load(p11,p9,row,coord):
    im,m,_=p11.load_case_robust(row,p9); im=np.asarray(im,np.float32); m=np.asarray(m,np.int64); native=m.shape
    im=p11.resize_3d(im,FULL,is_mask=False); m=dilate(p11.resize_3d(m,FULL,is_mask=True),R)
    sid,ser=str(row["study_id"]),str(row["series_id"]); a=coord[(coord.study_id.astype(str)==sid)&(coord.series_id.astype(str)==ser)]
    _,ds,_=p11.read_dicom_series_robust(p11.resolve_series_dir(row)); iz={int(d.get("InstanceNumber",0)):i for i,d in enumerate(ds)}
    pts=[]
    for _,r in a.iterrows():
        cid=NAME.get(str(r["condition"]).strip())
        try:x,y,inst=float(r["x"]),float(r["y"]),int(float(r["instance_number"]))
        except:continue
        if cid is None or inst not in iz:continue
        z=iz[inst]
        if 0<=z<native[0] and 0<=y<native[1] and 0<=x<native[2]:
            zz,yy,xx=rz(x,y,z,native); pts.append({"class_id":cid,"class_name":CLASSES[cid],"z":zz,"y":yy,"x":xx,"level":str(r.get("level",""))})
    return im,m,pts
def main():
    banner("PART 66 PATH VALIDATION")
    for n,q in [("Project root",ROOT),("Part 11",P11),("Part 9",P9),("Part 15 validation cohort",VACSV),("RSNA coordinate CSV",COORD)]:
        print(f"{n:<40}: {'FOUND' if q.exists() else 'MISSING'}")
        if not q.exists():raise FileNotFoundError(q)
    banner("PART 66 — POINT-AWARE CROP COVERAGE DIAGNOSTIC")
    print(f"Validation subset : {N}\nFull volume : {FULL}\nCrop : {CROP}\nPseudo-mask radius : R{R}\nStrategies : pseudo-mask-centered / point-centered / combined-center\nTraining performed : NO\nOptimizer used : NO\nBackward pass : NO\nSPIDER : NO\nTest set : NO\nPart 15 modified : NO")
    p11,p9=mod(P11,"p11_66"),mod(P9,"p9_66"); df=pd.read_csv(VACSV).head(N); coord=pd.read_csv(COORD)
    cases=[]; rows=[]; points_rows=[]
    banner("PART 66 VALIDATION PRELOAD")
    for i,(_,row) in enumerate(df.iterrows(),1):
        im,m,pts=load(p11,p9,row,coord); cases.append((im,m,pts))
        if i in (1,10,20,25,40,50,75,100):print(f"VALIDATION {i:03d}/{N} FG={(m>0).sum()} points={len(pts)}")
    banner("PART 66 SHAPE / POINT SMOKE TEST")
    print(f"Full image : {cases[0][0].shape}\nFull mask : {cases[0][1].shape}\nLabels : {np.unique(cases[0][1]).tolist()}\nMapped points : {len(cases[0][2])}\nCrop : {CROP}")
    for i,(im,m,pts) in enumerate(cases,1):
        pc=np.argwhere(m>0).mean(0) if (m>0).any() else np.array([31.5,47.5,47.5])
        tc=point_center(pts); cc=(pc+tc)/2
        for name,center in [("pseudo_mask_centered",pc),("point_centered",tc),("combined_center",cc)]:
            ci,cm,st=crop(im,m,center); cov,inside,classes,levels=pcoverage(pts,st)
            rows.append({"case":i,"strategy":name,"target_fg":int((cm>0).sum()),"occupancy":float((cm>0).mean()),"annotations":len(pts),"inside_points":inside,"coverage":cov,"classes_inside":classes,"levels_inside":levels,"crop_z":st[0],"crop_y":st[1],"crop_x":st[2]})
            for p in pts:
                inside_pt=any([False]) if False else (st[0]<=p["z"]<st[0]+CROP[0] and st[1]<=p["y"]<st[1]+CROP[1] and st[2]<=p["x"]<st[2]+CROP[2])
                if inside_pt:
                    q=(round(p["z"]-st[0]),round(p["y"]-st[1]),round(p["x"]-st[2]))
                    # A point exactly on the crop boundary can round to CROP,
                    # which is outside the valid zero-based index range.
                    if not all(0 <= q[j] < CROP[j] for j in range(3)):
                        continue
                    lab=int(cm[q])
                    points_rows.append({"case":i,"strategy":name,"class_id":p["class_id"],"class_name":p["class_name"],"level":p["level"],"label_at_point":lab,"same_class":int(lab==p["class_id"])})
    sdf=pd.DataFrame(rows); pdf=pd.DataFrame(points_rows); summary=[]
    banner("PART 66 OVERALL COMPARISON")
    for s in ["pseudo_mask_centered","point_centered","combined_center"]:
        g=sdf[sdf.strategy==s]; summary.append({"strategy":s,"mean_fg":g.target_fg.mean(),"mean_occupancy":g.occupancy.mean(),"mean_annotation_coverage":g.coverage.mean(),"median_annotation_coverage":g.coverage.median(),"full_coverage_cases":int((g.coverage>=.999999).sum()),"mean_classes_inside":g.classes_inside.mean(),"mean_levels_inside":g.levels_inside.mean()})
        print(f"{s:<25} | FG={g.target_fg.mean():.2f} occ={g.occupancy.mean():.6f} | coverage={g.coverage.mean():.4f} median={g.coverage.median():.4f} | full={int((g.coverage>=.999999).sum())}/{N} | classes={g.classes_inside.mean():.2f} | levels={g.levels_inside.mean():.2f}")
    banner("PART 66 POINT-TO-PSEUDOMASK CONSISTENCY BY STRATEGY")
    for s in ["pseudo_mask_centered","point_centered","combined_center"]:
        g=pdf[pdf.strategy==s]; print(f"{s:<25} | inside points={len(g)} | same-class={g.same_class.mean() if len(g) else float('nan'):.4f}")
    banner("PART 66 PER-CLASS CROP COVERAGE")
    classrows=[]
    for cid,name in CLASSES.items():
        total=sum(1 for _,_,pts in cases for p in pts if p["class_id"]==cid); print(f"\nC{cid} — {name}")
        for s in ["pseudo_mask_centered","point_centered","combined_center"]:
            g=pdf[(pdf.strategy==s)&(pdf.class_id==cid)]; inside=len(g); cov=inside/total if total else np.nan; same=g.same_class.mean() if len(g) else np.nan
            print(f"  {s:<23} coverage={cov:.4f} same-class={same:.4f} inside={inside}/{total}")
            classrows.append({"class_id":cid,"class_name":name,"strategy":s,"total_points":total,"inside_points":inside,"coverage":cov,"same_class":same})
    summarydf=pd.DataFrame(summary); base=summarydf[summarydf.strategy=="pseudo_mask_centered"].iloc[0]; best=summarydf.sort_values(["mean_annotation_coverage","mean_classes_inside"],ascending=False).iloc[0]; gain=float(best.mean_annotation_coverage-base.mean_annotation_coverage)
    diag="POINT_AWARE_CROP_SUBSTANTIALLY_IMPROVES_ANNOTATION_COVERAGE" if best.strategy!="pseudo_mask_centered" and gain>=.10 else ("POINT_AWARE_CROP_MODERATELY_IMPROVES_ANNOTATION_COVERAGE" if best.strategy!="pseudo_mask_centered" and gain>=.05 else "NO_LARGE_POINT_AWARE_CROP_COVERAGE_ADVANTAGE")
    banner("PART 66 DIAGNOSTIC INTERPRETATION"); print(f"Best strategy by mean annotation coverage : {best.strategy}\nMean annotation coverage : {best.mean_annotation_coverage:.4f}\nCoverage gain vs pseudo-mask-centered : {gain:+.4f}\nDiagnosis : {diag}\n\nScientific limitation: RSNA coordinates are point annotations, not manual segmentation masks.")
    REPORT.mkdir(parents=True,exist_ok=True); sdf.to_csv(REPORT/"part66_crop_strategy_case_metrics.csv",index=False); pdf.to_csv(REPORT/"part66_point_crop_alignment.csv",index=False); summarydf.to_csv(REPORT/"part66_crop_strategy_summary.csv",index=False); pd.DataFrame(classrows).to_csv(REPORT/"part66_per_class_crop_coverage.csv",index=False)
    json.dump({"part":66,"validation_subset":N,"strategies":[x for x in summary],"best_strategy":str(best.strategy),"coverage_gain_vs_pseudo_mask_centered":gain,"diagnosis":diag,"training_performed":False,"part15_modified":False},open(REPORT/"part66_summary.json","w",encoding="utf-8"),indent=2)
    banner("PART 66 COMPLETE"); print(f"Case metrics CSV : {REPORT/'part66_crop_strategy_case_metrics.csv'}\nPoint alignment CSV : {REPORT/'part66_point_crop_alignment.csv'}\nStrategy summary CSV : {REPORT/'part66_crop_strategy_summary.csv'}\nPer-class CSV : {REPORT/'part66_per_class_crop_coverage.csv'}\nSummary JSON : {REPORT/'part66_summary.json'}")
if __name__=="__main__":main()
