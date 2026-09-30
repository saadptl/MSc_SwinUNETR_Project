from pathlib import Path
import numpy as np
import pydicom

SERIES = Path(r"dataset\rsna-2024-lumbar-spine-degenerative-classification\train_images\11943292\1212326388")


def vec(v):
    a=np.asarray([float(x) for x in v],dtype=float)
    return np.round(a,6).tolist()

files=sorted(SERIES.glob('*.dcm'))
print('SERIES:', SERIES)
print('DICOM files:', len(files))
if not files:
    raise SystemExit('ERROR: no .dcm files found')

dss=[]
for p in files:
    try:
        ds=pydicom.dcmread(str(p), force=True, stop_before_pixels=False)
        if hasattr(ds,'PixelData'):
            dss.append(ds)
    except Exception as e:
        print('READ ERROR:',p.name,e)

print('Readable pixel slices:',len(dss))
if not dss:
    raise SystemExit('ERROR: no readable pixel-bearing DICOMs')

ds=dss[0]
iop=getattr(ds,'ImageOrientationPatient',None)
ipp=getattr(ds,'ImagePositionPatient',None)
ps=getattr(ds,'PixelSpacing',None)
print('\n--- FIRST SLICE ---')
print('SOPInstanceUID:',getattr(ds,'SOPInstanceUID',''))
print('InstanceNumber:',getattr(ds,'InstanceNumber',''))
print('Rows x Columns:',getattr(ds,'Rows','?'),'x',getattr(ds,'Columns','?'))
print('PixelSpacing:',ps)
print('SliceThickness:',getattr(ds,'SliceThickness','NA'))
print('SpacingBetweenSlices:',getattr(ds,'SpacingBetweenSlices','NA'))
print('ImageOrientationPatient:',iop)
print('ImagePositionPatient:',ipp)
print('PatientPosition:',getattr(ds,'PatientPosition','NA'))

if iop is not None and len(iop)>=6:
    row=np.array([float(x) for x in iop[:3]])
    col=np.array([float(x) for x in iop[3:6]])
    row/=np.linalg.norm(row); col/=np.linalg.norm(col)
    normal=np.cross(row,col); normal/=np.linalg.norm(normal)
    print('\n--- ORIENTATION ---')
    print('Row direction:',vec(row))
    print('Column direction:',vec(col))
    print('Slice normal:',vec(normal))
    ax=np.argmax(np.abs(normal))
    label=['Sagittal (normal≈X)','Coronal (normal≈Y)','Axial (normal≈Z)'][ax]
    print('Source plane:',label)
else:
    normal=np.array([0.,0.,1.])
    print('\nWARNING: missing/invalid ImageOrientationPatient')

# Sort by physical slice position along normal.
records=[]
for ds in dss:
    p=getattr(ds,'ImagePositionPatient',None)
    if p is not None and len(p)>=3:
        pos=np.array([float(x) for x in p[:3]])
        coord=float(np.dot(pos,normal))
    else:
        coord=float(getattr(ds,'InstanceNumber',0) or 0)
    records.append((coord,int(getattr(ds,'InstanceNumber',0) or 0),ds))
records.sort(key=lambda x:(x[0],x[1]))
coords=np.array([r[0] for r in records])
print('\n--- SLICE GEOMETRY ---')
print('First 10 physical slice coordinates:',np.round(coords[:10],6).tolist())
if len(coords)>1:
    diffs=np.abs(np.diff(coords))
    print('First 10 absolute differences:',np.round(diffs[:10],6).tolist())
    nz=diffs[diffs>1e-4]
    if len(nz):
        print('Median non-zero slice spacing:',float(np.median(nz)),'mm')
        print('Min/Max non-zero spacing:',float(np.min(nz)),float(np.max(nz)),'mm')
    else:
        print('WARNING: no non-zero physical slice separation detected')

print('\n--- ALL UNIQUE SHAPES ---')
shapes=[]
for ds in dss:
    try: shapes.append(tuple(ds.pixel_array.shape))
    except Exception: pass
print(sorted(set(shapes)))

print('\n--- RECOMMENDED MPR AXES ---')
print('Canonical array should be [Z, Y, X].')
print('Axial    = volume[Z, :, :]')
print('Coronal  = volume[:, Y, :]')
print('Sagittal = volume[:, :, X]')
print('\nCopy the COMPLETE output into ChatGPT before changing the MPR code.')
