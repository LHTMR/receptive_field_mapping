
import streamlit as st
import os
import cv2
import deeplabcut
import glob
import subprocess
import tempfile
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
import pathlib
import deeplabcut
import streamlit as st

def retrain_pipeline(config_path, processed_video_path):
    # Step 1: Add video to config.yaml if not already added (safely with ruamel.yaml)

    try:
        yaml = YAML()
        yaml.preserve_quotes = True

        with open(config_path, 'r') as f:
            cfg = yaml.load(f)

        # Normalize to forward slashes
        normalized_path = pathlib.Path(processed_video_path).as_posix()
        quoted_path = DoubleQuotedScalarString(normalized_path)

        if cfg.get('video_sets') is None:
            cfg['video_sets'] = {}

        if quoted_path not in cfg['video_sets']:
            cfg['video_sets'][quoted_path] = {'crop': '0,1274,0,720'}
            with open(config_path, 'w') as f:
                yaml.dump(cfg, f)
            st.success("Video added to config successfully.")
        else:
            st.info("Video already in config.")

    except Exception as e:
        st.error(f"Error while adding video to config: {e}")
        return

    # Step 2: Extract and label frames
    try:
        st.info("Extracting frames and launching labeling workflow...")
        deeplabcut.extract_frames(config_path,
                                    mode='automatic',
                                    algo='uniform',
                                    crop=False,
                                    userfeedback=False
                                )

        deeplabcut.label_frames(config_path)
        st.success("Labeling launched! Please label 5 frames in Napari and close it when done.")
    except Exception as e:
        st.error(f"Something went wrong during frame extraction or labeling: {e}")
        return

    # Step 3: Create training dataset
    try:
        st.info("Creating training dataset...")
        deeplabcut.create_training_dataset(config_path)
        st.success("Training dataset created.")
    except Exception as e:
        st.error(f"Something went wrong while creating training dataset: {e}")
        return

    # Step 4: Retrain
    try:
        st.info("Starting training...")
        remove_previous_predictions(video_path=processed_video_path)
        deeplabcut.train_network(config_path, autotune=False)
        st.success("Training complete!")
    except Exception as e:
        st.error(f"Something went wrong during retraining: {e}")
        return

    # Step 5: Analyze the same video again
    try:
        st.info("Running new predictions on updated video...")
        deeplabcut.analyze_videos(config_path, [processed_video_path])
        st.success("New predictions completed!")
    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")


def extract_frames_with_dlc(config_path, video_path):
    deeplabcut.extract_frames(config_path, videos=[video_path]
    )

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


import cv2
import numpy as np

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
    prediction_files = glob.glob(base + "*.h5") + glob.glob(base + "*.pickle")
    for file in prediction_files:
        try:
            os.remove(file)
        except Exception as e:
            st.warning(f"Could not delete {file}: {e}")

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



def main():
    st.title("DeepLabCut Video Prediction")
    ### Cahnge when moving
    project_path = r"C:\Users\sweer\Desktop\td_res_3-conv_vid-2025-03-18"
    config_path = os.path.join(project_path, "config.yaml")
    videos_dir = os.path.join(project_path, "videos")

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

        if st.button("Run Prediction"):
            st.write("Running DeepLabCut Prediction...")
            run_prediction(processed_video_path, config_path)
            st.success("Prediction Complete!")

        if st.button("Create Labeled Video"):
            st.write("Creating Labeled Video...")
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
                
        if st.button("Extract and Retrain"):
            retrain_pipeline(config_path, processed_video_path)

if __name__ == "__main__":
    main()
