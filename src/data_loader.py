
import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

label_mapping={"Normal/Mild":0,"Moderate":1,"Severe":2}

condition_prefix={
"Spinal Canal Stenosis":"spinal_canal_stenosis",
"Left Neural Foraminal Narrowing":"left_neural_foraminal_narrowing",
"Right Neural Foraminal Narrowing":"right_neural_foraminal_narrowing",
"Left Subarticular Stenosis":"left_subarticular_stenosis",
"Right Subarticular Stenosis":"right_subarticular_stenosis",
}

def load_three_slices(row,train_images):
    study_id=int(row["study_id"]);series_id=int(row["series_id"]);instance=int(row["instance_number"])
    folder=os.path.join(train_images,str(study_id),str(series_id))
    files=sorted([f for f in os.listdir(folder) if f.endswith(".dcm")],key=lambda x:int(x[:-4]))
    nums=[int(f[:-4]) for f in files]
    cur=nums.index(instance)
    idxs=[max(cur-1,0),cur,min(cur+1,len(nums)-1)]
    imgs=[]
    import pydicom
    for i in idxs:
        imgs.append(pydicom.dcmread(os.path.join(folder,files[i])).pixel_array)
    return imgs

def preprocess_image(images,size=224):
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    out=[]
    for img in images:
        img=img.astype(np.float32)
        img=(img-img.min())/(img.max()-img.min()+1e-8)
        img=(img*255).astype(np.uint8)
        img=clahe.apply(img)
        img=cv2.resize(img,(size,size))
        out.append(img)
    return np.stack(out,-1)

def get_target_label(row,train_df):
    prefix=condition_prefix.get(row["condition"])
    if prefix is None:return None
    col=f"{prefix}_{row['level'].lower().replace('/','_')}"
    if col not in train_df.columns:return None
    p=train_df[train_df.study_id==row["study_id"]]
    if p.empty:return None
    sev=p.iloc[0][col]
    if pd.isna(sev):return None
    return label_mapping.get(sev)

class LumbarDataset(Dataset):
    def __init__(self,dataframe,train_df,train_images):
        self.train_df=train_df
        self.train_images=train_images
        rows=[]
        for _,r in dataframe.iterrows():
            try:
                if get_target_label(r,train_df) is None: continue
                folder=os.path.join(train_images,str(int(r["study_id"])),str(int(r["series_id"])))
                if os.path.isdir(folder):
                    rows.append(r)
            except Exception:
                pass
        self.df=pd.DataFrame(rows).reset_index(drop=True)
        print("Valid samples:",len(self.df))
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        row=self.df.iloc[index]
        imgs=load_three_slices(row,self.train_images)
        img=preprocess_image(imgs)
        img=torch.tensor(img,dtype=torch.float32).permute(2,0,1)/255.0
        label=get_target_label(row,self.train_df)
        return {"image":img,
                "label":torch.tensor(label,dtype=torch.long),
                "study_id":int(row["study_id"]),
                "series_id":int(row["series_id"]),
                "instance_number":int(row["instance_number"]),
                "condition":row["condition"],
                "level":row["level"],
                "x":float(row["x"]),
                "y":float(row["y"])}
