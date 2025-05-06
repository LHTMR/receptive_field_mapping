
import streamlit as st
import os
import cv2
import numpy as np
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
import yaml
import shutil
import time
from pathlib import Path

## TODO!
### Change path written to config depending on Windows/Linux?
### Make visual instruction for Napari when loading module
### Make some graphics/plots during re-training?


# Init session state flags
if "labels_saved" not in st.session_state:
    st.session_state["labels_saved"] = False
if "project_initialized" not in st.session_state:
    st.session_state["project_initialized"] = False

# Setup paths
project_path = r"C:\Users\sweer\Desktop\td_res_3-conv_vid-2025-03-18"
#project_path = r"C:\Python Programming\LIU\projects\td_res_3-conv_vid-2025-03-18"
config_path = os.path.join(project_path, "config.yaml")
videos_dir = os.path.join(project_path, "videos")

# Save to session state
st.session_state["project_path"] = project_path
st.session_state["config_path"] = config_path
st.session_state["videos_dir"] = videos_dir

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

        # 3. Clean snapshots except the ones we want to keep
        training_folder = os.path.join(
            project_path,
            "dlc-models-pytorch",
            "iteration-0",
            "td_res_3Mar18-trainset90shuffle1",
            "train"
        )
        if os.path.exists(training_folder):
            for file in os.listdir(training_folder):
                if (
                    file.startswith("snapshot-") and file.endswith(".pt")
                    and file not in ["snapshot-075.pt", "snapshot-detector-200.pt"]
                ):
                    os.remove(os.path.join(training_folder, file))

        st.success("✅ Project initialized: config cleared, videos/labeled-data cleaned, snapshots trimmed.")

    except Exception as e:
        st.error(f"❌ Failed to initialize project: {e}")


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

def delete_prev_pred(project_path):
    predictions_removed = False

    # DeepLabCut usually stores results in:
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
            algo='uniform',
            crop=False,
            userfeedback=False
        )
        st.success("🖼️ Frames extracted!")

        subprocess.Popen([sys.executable, "-m", "napari"])
        #! add instructions here explaining what to upload and in what order
        st.warning("📝 Napari is running—label your frames, then close Napari when you’re done.")

    except Exception as e:
        st.error(f"❌ Frame extraction or labeling failed: {e}")

def run_retraining(config_path, processed_video_path,
                   num_epochs=25, num_detector_epochs=50):
    add_video_to_config(config_path, processed_video_path)
    # Step 1: Remove previous predictions
    try:
        st.info("🛠️ Removing previous predictions")
        delete_prev_pred(project_path=os.path.dirname(config_path))
        st.success("🗑️ Previous predictions removed!")
    except Exception as e:
        st.error(f"❌ Failed to remove previous predictions: {e}")
        st.stop()  # Stop execution immediately
    # TODO Remove previous training-set
    # Step 2: Create training dataset
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

    # Step 3: Train the model
    try:
        detector_path = os.path.join(
        project_path,
        "snapshot-detector-200.pt"   
        )
        snapshot_path = os.path.join(
        project_path,
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
    except Exception as e:
        st.error(f"❌ Failed to train the model: {e}")
        return  # Exit early if training fails

    # Step 4: Analyze video with the updated model
    try:
        st.info("📈 Analyzing video again with updated model...")
        deeplabcut.analyze_videos(config_path, [processed_video_path], shuffle=1)
        st.success("🎉 New predictions generated!")
    except Exception as e:
        st.error(f"❌ Failed to analyze video: {e}")
        return  # Exit early if analysis fails

    # Step 5: Create labeled video
    try:
        st.info("Creating Labeled Video")
        deeplabcut.create_labeled_video(config=config_path,
                                        videos=processed_video_path,
                                        shuffle=1)
        st.success("🎬 Labeled video created!")
    except Exception as e:
        st.error(f"❌ Failed to create labeled video: {e}")


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

def create_labeled_video(config_path, video_path, shuffle=1):
    # Print the video path for debugging
    print(f"Creating labeled video for: {video_path}")
    
    # Create labeled video with DeepLabCut (this will automatically save it in the videos folder)
    try:
        deeplabcut.create_labeled_video(config_path,
                                        videos=video_path,
                                        shuffle=shuffle)
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

    
    # Init project, clear video directory, video_sets in config file
    # and old snapshots
    if not st.session_state["project_initialized"]:
        init_project(config_path=config_path, project_path=project_path)
        os.makedirs(videos_dir, exist_ok=True)
        st.session_state["project_initialized"] = True
    
    # Upload and preprocess
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None and "processed_video_path" not in st.session_state:
        original_name = Path(uploaded_video.name).stem  # No extension
        temp_input_path = os.path.join(videos_dir, uploaded_video.name)
        
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_video.read())

        processed_video_name = f"processed_{original_name}.mp4"  # Always .mp4 now
        processed_video_path = os.path.join(videos_dir, processed_video_name)

        st.write("Preprocessing video...")
        preprocess_video(temp_input_path, processed_video_path)
        os.remove(temp_input_path)
        st.success("✅ Video preprocessed and saved.")
        st.session_state["processed_video_path"] = processed_video_path

    # Prediction button
    if "processed_video_path" in st.session_state and "labeled_video_path" not in st.session_state:
        if st.button("Run Prediction and Create Labeled Video"):
            st.write("Running DeepLabCut Prediction...")
            deeplabcut.analyze_videos(config_path, [st.session_state["processed_video_path"]])
            labeled_video_path = create_labeled_video(config_path, st.session_state["processed_video_path"])

            if labeled_video_path and os.path.exists(labeled_video_path):
                st.session_state["labeled_video_path"] = labeled_video_path
                st.success("✅ Labeled video created!")

    # Show labeled video and Save Labels button
    if "labeled_video_path" in st.session_state and os.path.exists(st.session_state["labeled_video_path"]):
        st.video(convert_to_streamlit_friendly(st.session_state["labeled_video_path"]))
        st.markdown("### ✅ Happy with the result? Save labels:")

        if st.button("💾 Save Labels", key="save_labels_tab1") and not st.session_state["labels_saved"]:
            try:
                h5_files = glob(os.path.join(videos_dir, "*.h5"))
                if not h5_files:
                    st.error("No .h5 file found in videos directory.")
                else:
                    st.session_state["h5_path"] = h5_files[0]
                    st.session_state["labels_saved"] = True
                    st.success(f"✅ Labels saved: {st.session_state['h5_path']}")
                    st.info("🔜 You can now move to the *Post Processing* page.")
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
            value=10
        )

        # 👇 Button to trigger full pipeline
        if st.button("1️⃣ Extract frames & launch labeling"):
            update_numframes2pick(config_path, num_frames)
            run_labeling(config_path, processed_video_path)

        st.markdown("---")
        # Create two columns for sliders
        col1, col2 = st.columns(2)

        # Slider 1: Number of epochs to retrain the model
        num_epochs = col1.slider(
            "Number of epochs to retrain the model:",
            min_value=5,
            max_value=50,
            step=5,
            value=25
        )

        # Slider 2: Number of epochs for the detector
        num_detector_epochs = col2.slider(
            "Number of epochs for the detector:",
            min_value=5,
            max_value=100,
            step=5,
            value=50
        )
        # 🧠 Continue to retraining
        if st.button("2️⃣ Done labeling? Continue to retrain"):
            run_retraining(config_path, processed_video_path, num_epochs, num_detector_epochs)

            # Try to find the new labeled video
            labeled_videos = glob(os.path.join(videos_dir, "*_labeled.mp4"))
            if labeled_videos:
                st.session_state["labeled_video_path"] = labeled_videos[0]
                st.session_state["labels_saved"] = False  # Reset saved state

        if "labeled_video_path" in st.session_state and os.path.exists(st.session_state["labeled_video_path"]):
            st.video(convert_to_streamlit_friendly(st.session_state["labeled_video_path"]))
            st.markdown("### ✅ Happy with the result? Save labels:")

            if st.button("💾 Save Labels", key="save_labels_tab2") and not st.session_state.get("labels_saved", False):
                try:
                    h5_files = glob(os.path.join(videos_dir, "*.h5"))
                    if not h5_files:
                        st.error("No .h5 file found in videos directory.")
                    else:
                        st.session_state["h5_path"] = h5_files[0]
                        st.session_state["labels_saved"] = True
                except Exception as e:
                    st.error(f"Error saving labels: {e}")

        # ✅ Always show success message if labels have been saved
        if st.session_state.get("labels_saved", False):
            st.success(f"✅ Labels saved: {st.session_state['h5_path']}")
            st.info("🔜 You can now move to the *Post Processing* page.")

    else:
        st.warning("⚠️ Please upload and process a video in Tab 1 first.")
