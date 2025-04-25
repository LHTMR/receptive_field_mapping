
import streamlit as st
import os
import cv2
import numpy as np
import deeplabcut
import glob
import subprocess
import tempfile
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
import pathlib
import deeplabcut
import streamlit as st
import subprocess
import sys
import yaml
import shutil

## TODO!
### Clear video_sets in config.yaml before each run. DONE!
### Make sure labeled_data and videos folder are cleared before each run. DONE!
#### Create init function to remove above. DONE!
### Change path written to config depending on Windows/Linux?
### Create new tab for retraining, containing guide for Napari! (also make config and video paths global) DONE!
### Make the final h5-file into a global variable in ST. DONE!
### Change buttons workflow. DONE!
### Remove unnecessary methods. predict, create label etc. DONE!

tab1, tab2 = st.tabs(["Create Labeled Video", "Labeling / Retraining"])

def init_project(config_path, project_path):
    try:
        yaml = YAML()
        yaml.preserve_quotes = True

        # 1. Load and clear video_sets in config.yaml
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

        st.success("✅ Project initialized: config cleared, videos and labeled-data cleaned.")

    except Exception as e:
        st.error(f"❌ Failed to initialize project: {e}")

def update_numframes2pick(config_path, selected_value):
    try:
        yaml = YAML()
        yaml.preserve_quotes = True

        with open(config_path, 'r') as f:
            cfg = yaml.load(f)

        current_value = cfg.get('numframes2pick', 5)
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
            algo='uniform',
            crop=False,
            userfeedback=False
        )
        st.success("🖼️ Frames extracted!")

        st.info("🚀 Launching labeling workflow (Napari)...")
        # Try open Napari in Streamlit environment using subprocess
        subprocess.Popen([sys.executable, "-c", f"import deeplabcut; deeplabcut.label_frames(r'{config_path}')"])
        st.warning("📝 Label your frames, be sure to save layers and close Napari before clicking the next button!")

    except Exception as e:
        st.error(f"❌ Frame extraction or labeling failed: {e}")

def run_retraining(config_path, processed_video_path):
    try:
        st.info("🛠️ Creating training dataset...")
        deeplabcut.create_training_dataset(config_path)
        st.success("📦 Training dataset created!")

        st.info("🧠 Starting model training...")
        # Remove previous predictions
        remove_previous_predictions(video_path=processed_video_path)
        deeplabcut.train_network(config_path, autotune=False)
        st.success("✅ Training complete!")

        st.info("📈 Analyzing video again with updated model...")
        deeplabcut.analyze_videos(config_path, [processed_video_path])
        st.success("🎉 New predictions generated!")

    except Exception as e:
        st.error(f"❌ Retraining failed: {e}")

def convert_to_streamlit_friendly(input_video_path):
    """Convert video to H.264-encoded MP4 for Streamlit compatibility using ffmpeg."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_file.name

    command = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-i", input_video_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except subprocess.CalledProcessError:
        print("FFmpeg failed to convert video.")
        return None

def preprocess_video(input_video_path, output_video_path):
    """Preprocesses the video: removes audio, scales it to fit 1274x720, centers it, and sets 30 FPS."""
    target_width, target_height = 1274, 720
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

def remove_previous_predictions(video_path):
    base, _ = os.path.splitext(video_path)
    
    # Remove .h5 and .pickle files
    prediction_files = glob(base + "*.h5") + glob(base + "*.pickle")
    for file in prediction_files:
        try:
            os.remove(file)
        except Exception as e:
            st.warning(f"Could not delete {file}: {e}")
    
    # Remove the labeled video if it exists
    labeled_video_path = base + "_labeled.mp4"
    if os.path.exists(labeled_video_path):
        try:
            os.remove(labeled_video_path)
            st.success(f"Removed previous labeled video: {labeled_video_path}")
        except Exception as e:
            st.warning(f"Could not delete {labeled_video_path}: {e}")
    else:
        st.warning("No labeled video found to remove.")

def create_labeled_video(config_path, video_path):
    # Print the video path for debugging
    print(f"Creating labeled video for: {video_path}")
    
    # Create labeled video with DeepLabCut (this will automatically save it in the videos folder)
    try:
        deeplabcut.create_labeled_video(config_path, videos=video_path)
        print("DeepLabCut create_labeled_video function executed")
    except Exception as e:
        print(f"Error creating labeled video: {e}")
    
    # Get the path of the labeled video by searching for files in the same folder as the input video
    video_dir = os.path.dirname(video_path)
    labeled_video_files = [f for f in os.listdir(video_dir) if f.endswith('_labeled.mp4')]

    if labeled_video_files:
        # If labeled video is found, return its full path
        labeled_video_path = os.path.join(video_dir, labeled_video_files[0])
        print(f"Labeled video found at {labeled_video_path}")
        return labeled_video_path
    else:
        print(f"Labeled video not found in {video_dir}")
        return None


with tab1:
    st.title("DeepLabCut Video Prediction")

    # Init session state flags
    if "labels_saved" not in st.session_state:
        st.session_state["labels_saved"] = False
    if "project_initialized" not in st.session_state:
        st.session_state["project_initialized"] = False

    # Setup paths
    project_path = r"C:\Users\sweer\Desktop\td_res_3-conv_vid-2025-03-18"
    config_path = os.path.join(project_path, "config.yaml")
    videos_dir = os.path.join(project_path, "videos")

    # Save to session state
    st.session_state["project_path"] = project_path
    st.session_state["config_path"] = config_path
    st.session_state["videos_dir"] = videos_dir

    # Init project, clear directories and config file once starting anew
    if not st.session_state["project_initialized"]:
        init_project(config_path=config_path, project_path=project_path)
        os.makedirs(videos_dir, exist_ok=True)
        st.session_state["project_initialized"] = True

    # Upload and preprocess
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None and "processed_video_path" not in st.session_state:
        temp_input_path = os.path.join(videos_dir, uploaded_video.name)
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_video.read())

        processed_video_name = "processed_" + uploaded_video.name
        processed_video_path = os.path.join(videos_dir, processed_video_name)

        st.write("Preprocessing video...")
        preprocess_video(temp_input_path, processed_video_path)
        os.remove(temp_input_path)
        st.success("✅ Video preprocessed and saved.")
        st.session_state["processed_video_path"] = processed_video_path

    # Prediction button
    if "processed_video_path" in st.session_state and "labeled_video_path" not in st.session_state:
        if st.button("Run Prediction and Create Labeled Video"):
            st.write("Running DeepLabCut...")
            deeplabcut.analyze_videos(config_path, [st.session_state["processed_video_path"]])
            labeled_video_path = create_labeled_video(config_path, st.session_state["processed_video_path"])

            if labeled_video_path and os.path.exists(labeled_video_path):
                st.session_state["labeled_video_path"] = labeled_video_path
                st.success("✅ Labeled video created!")

    # Show labeled video and Save Labels button
    if "labeled_video_path" in st.session_state and os.path.exists(st.session_state["labeled_video_path"]):
        st.video(convert_to_streamlit_friendly(st.session_state["labeled_video_path"]))
        st.markdown("### ✅ Happy with the result? Save labels:")

        if st.button("💾 Save Labels") and not st.session_state["labels_saved"]:
            try:
                h5_files = glob.glob(os.path.join(videos_dir, "*.h5"))
                if not h5_files:
                    st.error("No .h5 file found in videos directory.")
                else:
                    st.session_state["data_dlc"] = h5_files[0]
                    st.session_state["labels_saved"] = True
                    st.success(f"✅ Labels saved: {st.session_state['data_dlc']}")
    
                    # Add redirect button
                    if st.button("➡️ Go to Post Processing"):
                        st.switch_page("pages/03_Post_Processing.py")
                    st.stop()
            except Exception as e:
                st.error(f"Error saving labels: {e}")

    # Show success message after saving
    if st.session_state["labels_saved"]:
        st.success("✅ Labels already saved. Move to the Post Processing page.")



with tab2:
    if "config_path" in st.session_state and "processed_video_path" in st.session_state:
        config_path = st.session_state["config_path"]
        processed_video_path = st.session_state["processed_video_path"]

        st.markdown("## 🖼️ Extract Frames for Labeling")

        # 👇 Frame count selector
        num_frames = st.slider(
            "Number of frames to extract:",
            min_value=5,
            max_value=50,
            step=5,
            value=5
        )

        # 👇 Button to trigger full pipeline
        if st.button("1️⃣ Extract frames & launch labeling"):
            update_numframes2pick(config_path, num_frames)
            run_labeling(config_path, processed_video_path)

        st.markdown("---")

        # 🧠 Continue to retraining
        if st.button("2️⃣ Done labeling? Continue to retrain"):
            run_retraining(config_path, processed_video_path)

    else:
        st.warning("⚠️ Please upload and process a video in Tab 1 first.")
