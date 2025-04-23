
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

## TODO!
### Clear video_sets in config.yaml before each run
### Make sure labeled_data and videos folder are cleared before each run
#### Create init function to remove above
### Change path written to config depending on Windows/Linux?
### Create new tab for retraining, containing guide for Napari! (also make config and video paths global) DONE
### Make the final h5-file into a global variable in ST. DONE
### Change buttons workflow. DONE
### Remove unnecessary methods. predict, create label etc.

import deeplabcut
import subprocess
import pathlib
import sys
import streamlit as st
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

tab1, tab2 = st.tabs(["Main App", "Labeling / Retraining"])

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
        st.warning("📝 Label 5 frames and close Napari before clicking the next button!")

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


def run_prediction(video_path, config_path):
    """Runs DeepLabCut prediction on the video."""
    deeplabcut.analyze_videos(config_path, [video_path])

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
    def main():
        st.title("DeepLabCut Video Prediction")
        ### Change when moving
        project_path = r"C:\Python Programming\LIU\projects\td_res_3-conv_vid-2025-03-18"
        config_path = os.path.join(project_path, "config.yaml")
        videos_dir = os.path.join(project_path, "videos")
        # Save them globally
        st.session_state["project_path"] = project_path
        st.session_state["config_path"] = config_path
        st.session_state["videos_dir"] = videos_dir

        #os.makedirs(results_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)

        uploaded_video = st.file_uploader("Upload a video for prediction", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            temp_input_path = os.path.join(videos_dir, uploaded_video.name)

            # Save the uploaded file temporarily
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_video.read())

            processed_video_name = "processed_" + uploaded_video.name
            processed_video_path = os.path.join(videos_dir, processed_video_name)

            st.write("Preprocessing video...")
            preprocess_video(temp_input_path, processed_video_path)
            os.remove(temp_input_path)  # Clean up the uploaded original
            st.success("Video preprocessed and saved in videos/")
            st.session_state["processed_video_path"] = processed_video_path
            if st.button("Run Prediction and create labeled Video"):
                st.write("Running DeepLabCut Prediction...")
                run_prediction(processed_video_path, config_path)
                labeled_video_path = create_labeled_video(config_path, processed_video_path)

                if labeled_video_path and os.path.exists(labeled_video_path):
                    st.success("Labeled Video Created!")

                    # Convert for Streamlit compatibility
                    st.write("Converting video for display...")
                    display_path = convert_to_streamlit_friendly(labeled_video_path)

                    if display_path and os.path.exists(display_path):
                        st.video(display_path)
                    else:
                        st.error("Failed to convert video for display.")


                else:
                    st.error("Labeled video could not be found.")
                st.markdown("---")
                st.markdown("### ✅ If you are happy with the results, press the Save labels button and continue to the Post processing page:")
                st.markdown("If you are not satisfied with the video, scroll up and press the Labeling / Retrain tab")

                if st.button("💾 Save Labels"):
                    try:
                        # Find the only .h5 file in the videos directory
                        h5_files = glob(os.path.join(videos_dir, "*.h5"))

                        if not h5_files:
                            st.error("No .h5 file found in videos directory.")
                        else:
                            st.session_state["data_dlc"] = h5_files[0]
                            st.session_state["labeled_video_path"] = labeled_video_path
                            st.success(f"Labels saved and ready: {st.session_state['data_dlc']}")

                    except Exception as e:
                        st.error(f"Failed to set .h5 file in session state: {e}")

with tab2:
    if "config_path" in st.session_state and "processed_video_path" in st.session_state:
        config_path = st.session_state["config_path"]
        processed_video_path = st.session_state["processed_video_path"]
        
        # Print paths for debugging 
        st.write(f"Using config: {config_path}")
        st.write(f"Using video: {processed_video_path}")
        
        # Call your retraining or other functions here
    else:
        st.warning("Please upload a video first in Tab 1.")

    if st.button("1️⃣ Extract frames & launch labeling"):
        # Add input from user on number of frames to pick?
        run_labeling(config_path, processed_video_path)

    if st.button("2️⃣ Done labeling? Continue to retrain"):
        run_retraining(config_path, processed_video_path)

if __name__ == "__main__":
     main()