"""PHASE 4 - PART 9
RSNA 3D SEGMENTATION DATASET LOADER + SANITY VALIDATION

RSNA ONLY. No training. No weight modification. No SPIDER.
"""
from pathlib import Path
from typing import Any, Optional
import json, traceback, sys
import numpy as np
import pandas as pd
import pydicom
import torch

ROOT=Path(__file__).resolve().parents[1]
RSNA=ROOT/'dataset'/'rsna-2024-lumbar-spine-degenerative-classification'
DICOM_ROOT=RSNA/'train_images'
P8=ROOT/'outputs'/'segmentation'/'rsna_part8_dataset_construction'
MAN=P8/'manifests'
OUT=ROOT/'outputs'/'segmentation'/'rsna_part9_3d_dataset_loader'
SAMPLES=OUT/'sample_arrays'
REPORTS=OUT/'reports'
TRAIN=MAN/'rsna_part8_train_manifest.csv'
VAL=MAN/'rsna_part8_validation_manifest.csv'
TEST=MAN/'rsna_part8_test_manifest.csv'
CLASSES={0:'Background',1:'Spinal Canal Stenosis',2:'Left Neural Foraminal Narrowing',3:'Right Neural Foraminal Narrowing',4:'Left Subarticular Stenosis',5:'Right Subarticular Stenosis'}


def h(s): print('\n'+'='*78+'\n'+s+'\n'+'='*78)
def jint(x):
    try: return int(float(x))
    except: return None

def load_dicoms(series_dir):
    files=sorted(series_dir.glob('*.dcm'))
    if not files: raise RuntimeError(f'No DICOM files: {series_dir}')
    items=[]
    for p in files:
        try:
            ds=pydicom.dcmread(str(p),force=True)
            if hasattr(ds,'PixelData') and hasattr(ds,'Rows') and hasattr(ds,'Columns'): items.append(ds)
        except Exception: pass
    if len(items)<2: raise RuntimeError(f'Only {len(items)} readable DICOM slices: {series_dir}')
    def key(ds):
        ipp=getattr(ds,'ImagePositionPatient',None)
        try:
            if ipp is not None and len(ipp)>=3: return (0,float(ipp[2]))
        except: pass
        try: return (1,float(ds.SliceLocation))
        except: pass
        try: return (2,int(ds.InstanceNumber))
        except: return (3,0)
    items.sort(key=key)
    arr=[]
    for ds in items:
        a=ds.pixel_array.astype(np.float32)
        a*=float(getattr(ds,'RescaleSlope',1.0)); a+=float(getattr(ds,'RescaleIntercept',0.0)); arr.append(a)
    shapes={x.shape for x in arr}
    if len(shapes)!=1: raise RuntimeError(f'Inconsistent DICOM shapes: {shapes}')
    vol=np.stack(arr).astype(np.float32)
    if not np.isfinite(vol).all(): raise RuntimeError('DICOM volume contains non-finite values')
    return vol,items

def mask_key(keys):
    for k in ('mask','pseudo_mask','segmentation','labels','label'):
        if k in keys:return k
    for k in keys:
        if any(x in k.lower() for x in ('mask','segment','label')):return k
    return None

def load_npz(path):
    with np.load(str(path),allow_pickle=True) as z:
        keys=list(z.keys()); k=mask_key(keys)
        if k is None: raise RuntimeError(f'No mask array in {path}; keys={keys}')
        mask=np.asarray(z[k]); meta={}
        for x in keys:
            if x==k: continue
            try:
                v=np.asarray(z[x]); v=v.item() if v.ndim==0 else v
                meta[x]=v.tolist() if isinstance(v,np.ndarray) else v
            except: pass
    return mask,meta

def meta_int(meta,names):
    low={str(k).lower().replace('-','_'):v for k,v in meta.items()}
    for n in names:
        if n in low:
            v=jint(low[n])
            if v is not None:return v
    return None

def align_mask(mask,shape,meta,dsets):
    d,h,w=shape; m=np.squeeze(mask)
    if m.ndim==3:
        if tuple(m.shape)==shape:return m.astype(np.int16),'native_3d_exact',None
        if m.shape[1:]==(h,w) and m.shape[0]<=d:
            out=np.zeros(shape,np.int16);out[:m.shape[0]]=m;return out,'3d_depth_padded',None
        if m.shape[:2]==(h,w) and m.shape[2]<=d:
            t=np.transpose(m,(2,0,1));out=np.zeros(shape,np.int16);out[:t.shape[0]]=t;return out,'3d_axis_permuted',None
        raise RuntimeError(f'Cannot align 3D mask {m.shape} to {shape}')
    if m.ndim!=2: raise RuntimeError(f'Unsupported mask shape {m.shape}')
    if tuple(m.shape)!=(h,w): raise RuntimeError(f'2D mask {m.shape} != DICOM slice {(h,w)}')
    idx=meta_int(meta,['slice_index','z_index','mask_slice_index','image_slice_index'])
    if idx is not None and 0<=idx<d: pass
    else:
        inst=meta_int(meta,['instance_number','slice_number','dicom_instance','slice'])
        idx=None
        if inst is not None:
            for i,ds in enumerate(dsets):
                if jint(getattr(ds,'InstanceNumber',None))==inst: idx=i;break
            if idx is None and 1<=inst<=d: idx=inst-1
    if idx is None: raise RuntimeError('2D pseudo-mask has no reliable slice/instance metadata; refusing to guess')
    out=np.zeros(shape,np.int16);out[idx]=m
    return out,'2d_to_3d_single_slice',idx

def normalize(v):
    x=v[np.isfinite(v)]
    if x.size==0: raise RuntimeError('No finite image voxels')
    lo,hi=np.percentile(x,[1,99])
    if hi<=lo: lo,hi=float(x.min()),float(x.max())
    if hi<=lo:return np.zeros_like(v,np.float32)
    return (np.clip(v,lo,hi)-lo)/(hi-lo)

def load_case(row):
    sid=str(row['study_id']); ser=str(row['series_id'])
    sp=DICOM_ROOT/sid/ser
    mp=Path(str(row['pseudo_mask_path']))
    if not sp.exists():raise FileNotFoundError(sp)
    if not mp.exists():raise FileNotFoundError(mp)
    vol,ds=load_dicoms(sp); raw,meta=load_npz(mp)
    mask,mode,idx=align_mask(raw,vol.shape,meta,ds)
    img=normalize(vol)
    return img,mask,{'study_id':sid,'series_id':ser,'series_description':str(row.get('series_description','')),'raw_mask_shape':str(raw.shape),'alignment_mode':mode,'aligned_slice_index':idx,'volume_shape':str(vol.shape)}

def main():
    h('PHASE 4 - PART 9\nRSNA 3D SEGMENTATION DATASET LOADER + SANITY VALIDATION')
    print('PROJECT ROOT\n',ROOT);print('\nRSNA DATASET\n',RSNA);print('\nPART 8 MANIFESTS\n',MAN);print('\nOUTPUT DIRECTORY\n',OUT)
    OUT.mkdir(parents=True,exist_ok=True);SAMPLES.mkdir(exist_ok=True);REPORTS.mkdir(exist_ok=True)
    h('DATASET PATH VALIDATION')
    required={'RSNA root':RSNA,'train_images':DICOM_ROOT,'train manifest':TRAIN,'validation manifest':VAL,'test manifest':TEST}
    for n,p in required.items():print(f'{n:<22}: {"FOUND" if p.exists() else "MISSING"}')
    if not all(p.exists() for p in required.values()):raise RuntimeError('Required RSNA/Part 8 path missing')
    h('PYTORCH ENVIRONMENT');print('PyTorch:',torch.__version__);print('CUDA available:',torch.cuda.is_available());print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    h('LOADING PART 8 MANIFESTS')
    manifests={k:pd.read_csv(p) for k,p in [('train',TRAIN),('validation',VAL),('test',TEST)]}
    for k,df in manifests.items():print(f'{k:<12}: {len(df)} series')
    rows=[];MAX=5
    h('REPRESENTATIVE 3D LOADER VALIDATION')
    for split,df in manifests.items():
        idxs=np.linspace(0,len(df)-1,min(MAX,len(df)),dtype=int) if len(df) else []
        for n,i in enumerate(idxs,1):
            row=df.iloc[int(i)];print(f'[{n}/{len(idxs)}] {split} index={i}')
            try:
                img,mask,info=load_case(row)
                checks={'image_3d':img.ndim==3,'mask_3d':mask.ndim==3,'shape_match':img.shape==mask.shape,'finite':bool(np.isfinite(img).all()),'range_0_1':bool(img.min()>=-1e-6 and img.max()<=1.000001),'labels_valid':set(np.unique(mask).astype(int)).issubset(CLASSES),'foreground_present':bool((mask>0).any())}
                passed=all(checks.values());labels=sorted(np.unique(mask).astype(int).tolist())
                print('  Study:',info['study_id']);print('  Series:',info['series_id']);print('  Type:',info['series_description']);print('  Image:',img.shape);print('  Mask:',mask.shape,'raw:',info['raw_mask_shape']);print('  Labels:',labels);print('  Foreground:',int((mask>0).sum()));print('  Alignment:',info['alignment_mode']);print('  RESULT:', 'PASS' if passed else 'FAIL')
                if passed:
                    prefix=f'{split}_{info["study_id"]}_{info["series_id"]}';np.save(SAMPLES/(prefix+'_image.npy'),img.astype(np.float32));np.save(SAMPLES/(prefix+'_mask.npy'),mask.astype(np.int16))
                rec={'split':split,'index':int(i),'study_id':info['study_id'],'series_id':info['series_id'],'series_description':info['series_description'],'volume_shape':str(img.shape),'mask_shape':str(mask.shape),'raw_mask_shape':info['raw_mask_shape'],'labels':str(labels),'foreground_voxels':int((mask>0).sum()),'alignment_mode':info['alignment_mode'],'aligned_slice_index':info['aligned_slice_index'],'passed':passed,'error':''};rec.update({f'check_{k}':v for k,v in checks.items()})
            except Exception as e:
                print('  ERROR:',type(e).__name__,e);rec={'split':split,'index':int(i),'study_id':str(row.get('study_id','')),'series_id':str(row.get('series_id','')),'passed':False,'error':f'{type(e).__name__}: {e}'}
            rows.append(rec)
    result=pd.DataFrame(rows);result.to_csv(OUT/'rsna_part9_loader_case_results.csv',index=False)
    passed=int(result['passed'].sum()) if len(result) else 0;failed=len(result)-passed
    align=result[result['passed']==True]['alignment_mode'].value_counts().to_dict() if len(result) else {}
    h('PART 9 FINAL SUMMARY');print('Cases checked      :',len(result));print('Cases passed       :',passed);print('Cases failed       :',failed);print('Alignment modes    :');[print(f'  {k}: {v}') for k,v in align.items()]
    summary={'phase':'Phase 4 - Part 9','rsna_only':True,'spider_used':False,'training_performed':False,'model_weights_modified':False,'train_series':len(manifests['train']),'validation_series':len(manifests['validation']),'test_series':len(manifests['test']),'cases_checked':len(result),'cases_passed':passed,'cases_failed':failed,'alignment_modes':{str(k):int(v) for k,v in align.items()},'classes':CLASSES,'final_decision':'PASS - RSNA 3D loader and tensor sanity validation passed' if failed==0 and len(result)>0 else 'FAIL - review loader case results'}
    (OUT/'phase4_part9_loader_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (REPORTS/'phase4_part9_loader_validation_report.txt').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nSPIDER used              : NO');print('Training performed       : NO');print('Model weights modified   : NO');print('\nFINAL DECISION');print(summary['final_decision']);print('\nOUTPUT DIRECTORY\n',OUT)
    if failed: raise RuntimeError('Part 9 failed. See rsna_part9_loader_case_results.csv')
    h('PHASE 4 - PART 9 COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception as e:print('\n'+'='*78+'\nPART 9 ERROR\n'+'='*78);traceback.print_exc();sys.exit(1)
