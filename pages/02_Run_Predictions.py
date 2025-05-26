
import streamlit as st
import os

from pathlib import Path
from glob import glob

from src.train_predict import dlc_utils

# Init session state flags
if "project_initialized" not in st.session_state:
    st.session_state["project_initialized"] = False

# Create Tabs
tab1, tab2 = st.tabs(["Create Labeled Video", "Labeling / Retraining"])

with tab1:
    st.title("DeepLabCut Video Prediction")
    # Get project path from user
    project_path = st.text_input(
        "📁 Enter the full path to your DeepLabCut project folder:\nExample: C:\\....\\td_res_3-conv_vid-2025-03-18"
    )


    if project_path:
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
            st.success("Project Loaded!")
        else:
            st.error("❌ 'config.yaml' not found in the provided folder.")


        # Save to session state
        st.session_state["config_path"] = config_path
        st.session_state["project_path"] = project_path
        st.session_state["videos_dir"] = videos_dir
        st.session_state["training_folder"] = train_folder

        # Initialize project
        if not st.session_state["project_initialized"]:
            dlc_utils.init_project(config_path=config_path, project_path=project_path)
            dlc_utils.clean_snapshots(train_folder=train_folder)
            os.makedirs(videos_dir, exist_ok=True)
            st.session_state["project_initialized"] = True
        else:
            st.warning("⬆️ Enter the project folder path to continue.")

        # Upload and preprocess video
        uploaded_video = st.file_uploader("Upload a video",
                                          type=["mp4", "avi", "mov"])
        if uploaded_video is not None and\
            "processed_video_path" not in st.session_state:
            original_name = Path(uploaded_video.name).stem
            temp_input_path = os.path.join(videos_dir, uploaded_video.name)

            with open(temp_input_path, "wb") as f:
                f.write(uploaded_video.read())

            processed_video_name = f"processed_{original_name}.mp4"
            processed_video_path = os.path.join(videos_dir, processed_video_name)

            st.write("Preprocessing video...")
            dlc_utils.preprocess_video(temp_input_path, processed_video_path)
            os.remove(temp_input_path)
            st.success("✅ Video preprocessed and saved.")
            st.session_state["processed_video_path"] = processed_video_path

        # Prediction and make labeled video
        if "processed_video_path" in st.session_state:
            if st.button("Run Prediction and Create Labeled Video"):
                dlc_utils.predict_and_show_labeled_video(config_path,
                                    st.session_state["processed_video_path"],
                                    videos_dir)
                st.markdown("### ✅ Happy with the result? Continue to **Post Processing** page on the left.")
                st.markdown("### If not, scroll up to the **Labeling/Retraining** tab at the top of this page.")

with tab2:
    if "config_path" in st.session_state and\
    "processed_video_path" in st.session_state:
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
            dlc_utils.update_num_frames2pick(config_path, num_frames)
            dlc_utils.run_labeling(config_path, processed_video_path)

        st.markdown("---")
        # Sliders for choosing epochs
        col1, col2 = st.columns(2)
        num_epochs = col1.slider("Number of epochs to retrain the model:",
                                         5,
                                         50,
                                         step=5,
                                         value=25)
        num_detector_epochs = col2.slider("Number of epochs for the detector:",
                                          5,
                                          100,
                                          step=5,
                                          value=50)

        if st.button("2️⃣ Done with step 1? Click here to retrain the Model"):
            if not dlc_utils.is_labeling_done(project_path):
                st.warning("⚠️ No labeled data found. Please label at least 5 frames and save before continuing.")
            else:
                # Add video to config.yaml
                dlc_utils.add_video_to_config(config_path, processed_video_path)
                st.info("🛠️ Removing previous predictions")
                # Remove other snapshots
                dlc_utils.clean_snapshots(train_folder=train_folder)
                # Remove previous predictions
                dlc_utils.delete_prev_pred(videos_dir)
                # Remove any previous training sets
                dlc_utils.clear_training_datasets(project_path)
                # Retrain the model
                dlc_utils.run_retraining(config_path,train_folder,
                            num_epochs,
                            num_detector_epochs)
                st.markdown("### 📊 Training Loss Overview")
                dlc_utils.show_training_plots(train_folder)
                # Make prediction and show labeled video
                dlc_utils.predict_and_show_labeled_video(config_path,
                                        st.session_state["processed_video_path"],
                                        videos_dir)
                st.markdown("### ✅ Happy with the result? Continue to **Post Processing** page on the left")
                st.markdown("### If not, try label more frames or increase number of epochs")
    else:
        st.warning("⚠️ Please upload and process a video in Tab 1 first.")

