import os
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

dataset_path = r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project\dataset\rsna-2024-lumbar-spine-degenerative-classification"

train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))

series_df = pd.read_csv(os.path.join(dataset_path, "train_series_descriptions.csv"))

coord_df = pd.read_csv(os.path.join(dataset_path, "train_label_coordinates.csv"))

print(train_df.head())
print(series_df.head())
print(coord_df.head())

image_root = os.path.join(dataset_path, "train_images")

patients = sorted(os.listdir(image_root))

print("Total Patients:", len(patients))

patient = patients[0]

patient_folder = os.path.join(image_root, patient)

print(os.listdir(patient_folder))

series = os.listdir(patient_folder)[2]

series_folder = os.path.join(patient_folder, series)

print(os.listdir(series_folder)[:10])

dicom_path = os.path.join(series_folder, os.listdir(series_folder)[2])

ds = pydicom.dcmread(dicom_path)

image = ds.pixel_array

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.title("MRI Slice")
plt.show()
