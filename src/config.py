import os

PROJECT_ROOT = r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

for folder in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    os.makedirs(folder, exist_ok=True)


    import os

PROJECT_ROOT = r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "rsna-2024-lumbar-spine-degenerative-classification")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")