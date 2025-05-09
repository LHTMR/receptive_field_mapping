
import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import deeplabcut
from glob import glob
import subprocess
import tempfile
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
import pathlib
import deeplabcut
import streamlit as st
import subprocess
import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from src.post_processing.datadlc import DataDLC
from src.post_processing.plotting_plotly import PlottingPlotly
from pathlib import Path
from PIL import Image

# Init session state flags
if "project_initialized" not in st.session_state:
    st.session_state["project_initialized"] = False

# Setup paths
#project_path = r"C:\Users\sweer\Desktop\td_res_3-conv_vid-2025-03-18"
project_path = r"C:\Python Programming\LIU\projects\td_res_3-conv_vid-2025-03-18"
config_path = os.path.join(project_path, "config.yaml")
videos_dir = os.path.join(project_path, "videos")
### Needs to change with new model!
training_folder = os.path.join(
            project_path,
            "dlc-models-pytorch",
            "iteration-0",
            "td_res_3Mar18-trainset90shuffle1",
            "train"
        )

# Save to session state
st.session_state["project_path"] = project_path
st.session_state["config_path"] = config_path
st.session_state["videos_dir"] = videos_dir
st.session_state["training_folder"] = training_folder

# Create Tabs
tab1, tab2 = st.tabs(["Create Labeled Video", "Labeling / Retraining"])

def init_project(config_path, project_path):
    try:
        yaml = YAML()
        yaml.preserve_quotes = True

        # 1. Clear video_sets in config.yaml
        with open(config_path, 'r') as f:
            cfg = yaml.load(f)
        cfg['video_sets'] = {}
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f)

        # 2. Clean out 'videos' and 'labeled-data' folders
        folders_to_clean = ['videos', 'labeled-data']
        for folder in folders_to_clean:
            full_path = os.path.join(project_path, folder)
            if os.path.exists(full_path):
                for file in os.listdir(full_path):
                    file_path = os.path.join(full_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

        st.success("✅ Project initialized: config cleared, videos/labeled-data cleaned.")
    
    except Exception as e:
        st.error(f"❌ Failed to initialize project: {e}")

def clean_snapshots(training_folder):
    """
    Deletes all snapshot files in the training folder except for the
    ones explicitly allowed to remain.
    """
    keep_files = {"snapshot-075.pt", "snapshot-detector-200.pt"}

    if os.path.exists(training_folder):
        for file in os.listdir(training_folder):
            if file.startswith("snapshot-") and file.endswith(".pt") and file not in keep_files:
                os.remove(os.path.join(training_folder, file))
        st.success("🧹 Snapshots cleaned: Only selected files kept.")
    else:
        st.warning("⚠️ Training folder not found.")

def show_detector_training_loss(train_folder):
    detector_stats_path = os.path.join(train_folder, "learning_stats_detector.csv")
    if not os.path.exists(detector_stats_path):
        return None

    df = pd.read_csv(detector_stats_path)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["step"], df["losses/train.total_loss"], label="Detector Train Loss", color='green')
    ax.set_title("Detector Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig

def show_pose_training_loss(train_folder):
    stats_path = os.path.join(train_folder, "learning_stats.csv")
    if not os.path.exists(stats_path):
        return None

    df = pd.read_csv(stats_path)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["step"], df["losses/train.total_loss"], label="Pose Train Loss", color='blue')
    ax.set_title("Pose Model Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig

def show_training_plots(train_folder):
    col1, col2 = st.columns(2)

    with col1:
        fig1 = show_pose_training_loss(train_folder)
        if fig1:
            st.pyplot(fig1)
        else:
            st.warning("No pose training loss data to display.")

    with col2:
        fig2 = show_detector_training_loss(train_folder)
        if fig2:
            st.pyplot(fig2)
        else:
            st.warning("No detector training loss data to display.")

def clear_training_datasets(project_path):
    training_datasets_path = os.path.join(project_path, "training-datasets")
    if os.path.exists(training_datasets_path):
        shutil.rmtree(training_datasets_path)
        print("✅ Cleared existing training-datasets folder.")
    else:
        print("ℹ️ No training-datasets folder found, nothing to clear.")

def update_numframes2pick(config_path, selected_value):
    try:
        yaml = YAML()
        yaml.preserve_quotes = True

        with open(config_path, 'r') as f:
            cfg = yaml.load(f)

        current_value = cfg.get('numframes2pick', 10)
        if current_value != selected_value:
            cfg['numframes2pick'] = selected_value
            with open(config_path, 'w') as f:
                yaml.dump(cfg, f)
            st.success(f"✅ Updated 'numframes2pick' from {current_value} to {selected_value}")
        else:
            st.info("ℹ️ 'numframes2pick' already set to the selected value. No update needed.")
    except Exception as e:
        st.error(f"❌ Failed to check/update config.yaml: {e}")
        raise

def save_h5_to_session(videos_dir: str) -> str | None:
    """Find the first .h5 file in the videos directory and store it in session state."""
    try:
        h5_files = glob(os.path.join(videos_dir, "*.h5"))
        if not h5_files:
            return None
        h5_path = h5_files[0]
        st.session_state["h5_path"] = h5_path
        st.success("Labels Saved")
        return h5_path
    except Exception as e:
        st.error(f"Error accessing .h5 files: {e}")
        return None

def delete_prev_pred(project_path):
    predictions_removed = False

    # DeepLabCut stores results in:
    results_dir = os.path.join(project_path, 'videos')

    if not os.path.isdir(results_dir):
        st.warning(f"Results folder not found at: {results_dir}")
        return

    # File extensions DeepLabCut generates
    extensions = ['*.h5', '*.pickle', '*_labeled.mp4']

    for ext in extensions:
        for file_path in glob(os.path.join(results_dir, ext)):
            try:
                os.remove(file_path)
                predictions_removed = True
            except Exception as e:
                st.warning(f"⚠️ Could not delete {file_path}: {e}")

    if not predictions_removed:
        st.info("No prediction-related files found to remove.")

def add_video_to_config(config_path, processed_video_path):
    try:
        yaml = YAML()
        yaml.preserve_quotes = True

        with open(config_path, 'r') as f:
            cfg = yaml.load(f)

        normalized_path = pathlib.Path(processed_video_path).as_posix()
        quoted_path = DoubleQuotedScalarString(normalized_path)

        if cfg.get('video_sets') is None:
            cfg['video_sets'] = {}

        if quoted_path not in cfg['video_sets']:
            cfg['video_sets'][quoted_path] = {'crop': '0,1274,0,720'}
            with open(config_path, 'w') as f:
                yaml.dump(cfg, f)
            st.success("✅ Video added to config.yaml.")
        else:
            st.info("ℹ️ Video already exists in config.yaml.")
    except Exception as e:
        st.error(f"❌ Error updating config.yaml: {e}")
        raise

def run_labeling(config_path, processed_video_path):
    try:
        add_video_to_config(config_path, processed_video_path)

        st.info("🔍 Extracting frames...")
        deeplabcut.extract_frames(
            config_path,
            mode='automatic',
            crop=False,
            userfeedback=False
        )
        st.success("🖼️ Frames extracted!")

        subprocess.Popen([sys.executable, "-m", "napari"])
        napari_instructions()
        st.warning("📝 Napari is running—label your frames, then close Napari when you’re done.")

    except Exception as e:
        st.error(f"❌ Frame extraction or labeling failed: {e}")

def napari_instructions():
    st.markdown("""
    ### 📝 Napari Instructions
    All these steps must be followed in order; otherwise, Napari-DeepLabCut may not display elements correctly.
    """)

    ASSETS_PATH = Path("assets")

    # Step 1
    st.markdown("""
    #### Step 1
    Click `Plugins -> Keypoint controls` in the top left, which will load a sidebar on the right side. Feel free to ignore the tutorial pop-up.
    """)
    st.image(ASSETS_PATH / "step 1.png", caption="")

    # Step 2
    st.markdown("""
    #### Step 2
    Click `File -> Open File(s)...`, navigate to the folder containing the project folder, then select open on the config file.
    """)
    st.image(ASSETS_PATH / "step 2.png", caption="")
    st.image(ASSETS_PATH / "step 3.png", caption="")

    # Step 3
    st.markdown("""
    #### Step 3
    Click `File -> Open Folder...`, navigate into the `labeled-data` folder. Within should now be a folder with a name similar to the uploaded video. 
    Select open on that folder. Finally, select `napari DeepLabCut`. Now label the images!
    """)
    st.image(ASSETS_PATH / "step 4.png", caption="")
    st.image(ASSETS_PATH / "step 5.png", caption="")

    # Step 4
    st.markdown("""
    #### Step 4
    Once labeling has been completed, click `File -> Save Selected Layer(s)` while `CollectedData_conv_vid` is selected. 
    Then you can now close Napari and move on!
    """)
    st.image(ASSETS_PATH / "step 6.png", caption="")

def run_retraining(config_path, processed_video_path,
                   num_epochs=25, num_detector_epochs=50):
    # Add video to config.yaml
    add_video_to_config(config_path, processed_video_path)
    # Remove undesired snapshots
    clean_snapshots(training_folder=training_folder)
    # Remove previous predictions
    try:
        st.info("🛠️ Removing previous predictions")
        delete_prev_pred(project_path=os.path.dirname(config_path))
        st.success("🗑️ Previous predictions removed!")
    except Exception as e:
        st.error(f"❌ Failed to remove previous predictions: {e}")
        st.stop()  # Stop execution immediately

    # Create training dataset
    try:
        clear_training_datasets(project_path)
        st.info("🛠️ Creating training dataset...")
        deeplabcut.create_training_dataset(config_path,
                                           num_shuffles=1,
                                           weight_init=None,
                                           net_type="top_down_resnet_50",
                                           userfeedback=False)
        st.success("📦 Training dataset created!")
    except Exception as e:
        st.error(f"❌ Failed to create training dataset: {e}")
        st.stop()  # Also use st.stop here for consistency

    # Train the model
    try:
        detector_path = os.path.join(
        training_folder,
        "snapshot-detector-200.pt"   
        )
        snapshot_path = os.path.join(
        training_folder,
        "snapshot-075.pt"   
        )
        st.info("🧠 Starting model training...")
        deeplabcut.train_network(config_path,
                                 detector_path=detector_path,
                                 snapshot_path=snapshot_path,
                                 epochs=num_epochs,
                                 save_epochs=num_epochs,
                                 detector_epochs=num_detector_epochs,
                                 detector_save_epochs=num_detector_epochs,
                                 shuffle=1,
                                 batch_size=2,
                                 detector_batch_size=2,
                                 autotune=False,
                                 keepdeconvweights=True
                                 )
        st.success("✅ Training complete!")
        st.markdown("### 📊 Training Loss Overview")
        show_training_plots(training_folder)
    except Exception as e:
        st.error(f"❌ Failed to train the model: {e}")
        return  # Exit early if training fails

def predict_and_show_labeled_video(config_path: str, video_path: str, videos_dir: str):
    try:
        st.info("📈 Analyzing video with DeepLabCut...")
        deeplabcut.analyze_videos(config_path, [video_path], shuffle=1)
        st.success("🎉 New predictions generated!")

        h5_path = save_h5_to_session(videos_dir=videos_dir)
        dlc_data = DataDLC(h5_file=h5_path)
        vid_bit = PlottingPlotly.generate_labeled_video(dlc_data, video_path)
        st.video(vid_bit)

    except Exception as e:
        st.error(f"❌ Could not complete prediction or labeling: {e}")


def preprocess_video(input_video_path, output_video_path):
    """Preprocesses the video: removes audio, scales it to fit 1274x720, centers it, and sets 30 FPS."""
    target_width, target_height = 1280, 720
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (target_width, target_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))

        # Create a black canvas and center the resized frame
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

        out.write(canvas)

    cap.release()
    out.release()
    return output_video_path

with tab1:
    st.title("DeepLabCut Video Prediction")

    # Initialize project
    if not st.session_state["project_initialized"]:
        init_project(config_path=config_path, project_path=project_path)
        os.makedirs(videos_dir, exist_ok=True)
        st.session_state["project_initialized"] = True

    # Upload and preprocess video
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None and "processed_video_path" not in st.session_state:
        original_name = Path(uploaded_video.name).stem
        temp_input_path = os.path.join(videos_dir, uploaded_video.name)

        with open(temp_input_path, "wb") as f:
            f.write(uploaded_video.read())

        processed_video_name = f"processed_{original_name}.mp4"
        processed_video_path = os.path.join(videos_dir, processed_video_name)

        st.write("Preprocessing video...")
        preprocess_video(temp_input_path, processed_video_path)
        os.remove(temp_input_path)
        st.success("✅ Video preprocessed and saved.")
        st.session_state["processed_video_path"] = processed_video_path

    # Prediction and make labeled video
    if "processed_video_path" in st.session_state:
        if st.button("Run Prediction and Create Labeled Video"):
            predict_and_show_labeled_video(config_path,
                                        st.session_state["processed_video_path"],
                                        videos_dir)
            st.markdown("### ✅ Happy with the result? Continue to **Post Processing** page")
            st.markdown("### If not, move to the **Labeling/Retraining** tab:")

with tab2:
    if "config_path" in st.session_state and "processed_video_path" in st.session_state:
        config_path = st.session_state["config_path"]
        processed_video_path = st.session_state["processed_video_path"]

        st.markdown("## 🖼️ Extract Frames for Labeling")

        num_frames = st.slider(
            "Number of frames to extract:",
            min_value=5,
            max_value=50,
            step=5,
            value=10
        )

        if st.button("1️⃣ Extract frames & launch labeling"):
            update_numframes2pick(config_path, num_frames)
            run_labeling(config_path, processed_video_path)

        st.markdown("---")

        col1, col2 = st.columns(2)
        num_epochs = col1.slider("Number of epochs to retrain the model:", 5, 50, step=5, value=25)
        num_detector_epochs = col2.slider("Number of epochs for the detector:", 5, 100, step=5, value=50)

        if st.button("2️⃣ Done labeling? Continue to retrain"):
            run_retraining(config_path, processed_video_path, num_epochs, num_detector_epochs)
            predict_and_show_labeled_video(config_path,
                                    st.session_state["processed_video_path"],
                                    videos_dir)
            st.markdown("### ✅ Happy with the result? Continue to **Post Processing** page")
            st.markdown("### If not, try label more frames or increase number of epochs :")
    else:
        st.warning("⚠️ Please upload and process a video in Tab 1 first.")

