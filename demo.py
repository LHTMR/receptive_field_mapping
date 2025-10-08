import os
from glob import glob
from src.train_predict import dlc_utils

project_path = "/Users/sarmc72/Library/CloudStorage/OneDrive-Linköpingsuniversitet/projects - in progress/RF mapping with DLC/td_res_3-conv_vid-2025-03-18/"

# Strip surrounding quotes if they exist
project_path = project_path.strip('"').strip("'")

config_candidate = os.path.join(project_path, "config.yaml")
if os.path.exists(config_candidate):
    config_path = config_candidate
    videos_dir = os.path.join(project_path, "videos")

    # Auto-detect the training folder
    train_folders = glob(
        os.path.join(project_path, "dlc-models-pytorch",
                    "iteration-0", "*", "train")
    )
    train_folder = train_folders[0] if train_folders else None
    print("Project Loaded!")
else:
    print("❌ 'config.yaml' not found in the provided folder.")

# Initialize project
dlc_utils.init_project(config_path=config_path, project_path=project_path)
dlc_utils.clean_snapshots(train_folder=train_folder)
os.makedirs(videos_dir, exist_ok=True)
print("project_initialized")
