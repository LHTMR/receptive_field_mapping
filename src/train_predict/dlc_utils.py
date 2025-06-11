import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import deeplabcut
import subprocess
import pathlib
import deeplabcut
import sys
import shutil
import matplotlib.pyplot as plt

from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from glob import glob

from src.post_processing.datadlc import DataDLC
from src.post_processing.plotting_plotly import PlottingPlotly

# Project Setup and Utilities

def init_project(config_path: str, project_path: str) -> None:
    """
    Initializes a DeepLabCut project by updating the config.yaml file and cleaning project folders.

    This function performs the following actions:
    1. Clears the 'video_sets' entry in the config.yaml file and updates the 'project_path'.
    2. Deletes all contents inside the 'videos' and 'labeled-data' directories within the project folder.

    Args:
        config_path (str): Full path to the config.yaml file.
        project_path (str): Path to the root of the DeepLabCut project directory.

    Returns:
        None

    Displays:
        Streamlit success or error messages indicating the outcome of the initialization.
    """
    try:
        yaml = YAML()
        yaml.preserve_quotes = True

        # 1. Clear video_sets and add project path in config.yaml
        with open(config_path, 'r') as f:
            cfg = yaml.load(f)
        cfg['video_sets'] = {}
        cfg['project_path'] = str(project_path)
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

        st.success("✅ Project initialized: config updated, videos/labeled-data cleaned.")
    
    except Exception as e:
        st.error(f"❌ Failed to initialize project: {e}")

def save_h5_to_session(videos_dir: str) -> str | None:
    """
    Finds the first .h5 file in the provided videos directory and stores it 
    in the session state.

    This function searches for .h5 files in the given videos directory. If it 
    finds any, it stores the path of the first .h5 file found in the session 
    state under the key "h5_path". If no .h5 files are found, it returns None.

    Args:
        videos_dir (str): The directory to search for .h5 files.

    Returns:
        str | The path to the first .h5 file found, or None if no .h5 
        files are found.

    Raises:
        Exception: If an error occurs while accessing or reading the files.
    """
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

def update_num_frames2pick(config_path: str, selected_value: int) -> None:
    """
    Updates the 'numframes2pick' value in the config.yaml file if the selected
    value differs from the current one.

    This function checks the current 'numframes2pick' value in the provided 
    config.yaml file and updates it to the selected value if it differs. 
    If the value is already set to the selected value, no changes are made.

    Args:
        config_path (str): The path to the config.yaml file to be updated.
        selected_value (int): The value to set for 'numframes2pick' in the config file.

    Returns:
        None

    Raises:
        Exception: If the config.yaml file cannot be read or written to.
    """
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
            st.success(f"✅ Updated 'numframes2pick'")
        else:
            st.info("ℹ️ 'numframes2pick' already set to the selected value. No update needed.")
    except Exception as e:
        st.error(f"❌ Failed to check/update config.yaml: {e}")
        raise

def add_video_to_config(config_path: str, processed_video_path: str) -> None:
    """
    Adds a video file path to the 'video_sets' section of the config.yaml file.

    This function normalizes the provided video path, checks if the 'video_sets' 
    section exists in the config file, and adds the video if it's not already listed.
    If the video is already in the config, it does not make changes. The path is stored 
    with a default crop value.

    Args:
        config_path (str): The path to the config.yaml file.
        processed_video_path (str): The path to the processed video to be added to the config.

    Returns:
        None: This function modifies the config file directly and provides feedback through Streamlit.

    Raises:
        Exception: If any errors occur during the file operations or YAML manipulation, 
                   they are caught and displayed in the Streamlit interface.
    """
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

def delete_prev_pred(videos_dir: str) -> None:
    """
    Deletes previous prediction-related files from the specified videos directory.

    This function removes files with extensions such as .h5, .pickle, and any 
    video files with '_labeled.mp4' from the given directory. It provides 
    feedback on whether any files were removed or if no relevant files were found.

    Args:
        videos_dir (str): The directory where the prediction files (e.g., .h5, 
                          .pickle, *_labeled.mp4) are located.

    Returns:
        None: This function doesn't return any values. It only updates the 
              Streamlit interface with success or warning messages.

    Raises:
        None: If any errors occur during file deletion, they are caught and 
              displayed as warnings in the Streamlit interface.
    """
    predictions_removed = False

    if not os.path.isdir(videos_dir):
        st.warning(f"Videos folder not found at: {videos_dir}")
        return

    # File extensions DeepLabCut generates and we want removed
    extensions = ['*.h5', '*.pickle', '*_labeled.mp4']

    for ext in extensions:
        for file_path in glob(os.path.join(videos_dir, ext)):
            try:
                os.remove(file_path)
                predictions_removed = True
            except Exception as e:
                st.warning(f"⚠️ Could not delete {file_path}: {e}")

    if predictions_removed:
        st.success("🧹 Previous predictions removed.")
    else:
        st.info("No prediction-related files found to remove.")

def clean_snapshots(train_folder: str) -> None:
    """
    Deletes unnecessary snapshot files from the training folder, keeping only selected ones.

    Specifically retains:
    - 'snapshot-100.pt'
    - 'snapshot-detector-200.pt'

    Args:
        train_folder (str): Path to the training folder containing snapshot files.

    Returns:
        None

    Displays:
        Streamlit success or warning messages depending on the result.
    """
    # Change this for new model!!
    keep_files = {"snapshot-100.pt", "snapshot-detector-200.pt"}

    if os.path.exists(train_folder):
        for file in os.listdir(train_folder):
            if file.startswith("snapshot-") and file.endswith(".pt")\
                  and file not in keep_files:
                os.remove(os.path.join(train_folder, file))
        st.success("🧹 Snapshots cleaned: Only selected files kept.")
    else:
        st.warning("⚠️ Training folder not found.")

def clear_training_datasets(project_path: str) -> None:
    """
    Clears the 'training-datasets' folder in the provided project directory.

    This function removes the 'training-datasets' folder and all its contents
    if it exists. If the folder does not exist, a message is printed indicating
    that there is nothing to clear.

    Args:
        project_path (str): The path to the project directory containing the
                             'training-datasets' folder.

    Returns:
        None
    """
    training_datasets_path = os.path.join(project_path, "training-datasets")
    if os.path.exists(training_datasets_path):
        shutil.rmtree(training_datasets_path)
        print("✅ Cleared existing training-datasets folder.")
    else:
        print("ℹ️ No training-datasets folder found, nothing to clear.")

def is_labeling_done(project_path: str) -> bool:
    """
    Checks whether any labeled data exists in the 'labeled-data' folder of a DeepLabCut project.

    This function looks for subdirectories inside 'labeled-data' and checks if they contain
    at least one label file (.h5 or .csv). These files indicate that at least one frame has
    been labeled and saved using the labeling GUI.

    Args:
        project_path (str): The path to the root of the DeepLabCut project.

    Returns:
        bool: True if labeled data is found, False otherwise.
    """
    labeled_data_path = os.path.join(project_path, 'labeled-data')
    if not os.path.exists(labeled_data_path):
        return False
    for subdir in os.listdir(labeled_data_path):
        subdir_path = os.path.join(labeled_data_path, subdir)
        if os.path.isdir(subdir_path):
            label_files = [f for f in os.listdir(subdir_path)\
                            if f.endswith('.h5') or f.endswith('.csv')]
            if label_files:  # Labeled data exists
                return True
    return False

def preprocess_video(input_video_path: str, output_video_path: str) -> str:
    """
    Preprocesses a video by removing audio, scaling it to a target resolution, 
    centering it, and setting the frame rate to 30 FPS.

    The video is resized to fit within a 1274x720 resolution while maintaining 
    its aspect ratio. If the resized video does not fill the target resolution, 
    it is centered on a black canvas. The output video is saved as an MP4 file 
    with 30 FPS.

    Args:
        input_video_path (str): The file path to the input video to be processed.
        output_video_path (str): The file path where the processed video will be saved.

    Returns:
        str: The file path to the processed video.

    Raises:
        None: Assumes OpenCV functions handle any internal exceptions related to file reading/writing.
    """
    target_width, target_height = 1274, 720
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path,
                          fourcc,
                          30.0,
                          (target_width, target_height))

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

# Visualization
def show_detector_training_loss(train_folder):
    """
    Loads detector training loss statistics and generates a plot.

    This function reads the 'learning_stats_detector.csv' file from the specified
    training folder, extracts the training loss values, and creates a matplotlib
    figure showing the loss over training steps.

    Args:
        train_folder (str): Path to the folder containing 'learning_stats_detector.csv'.

    Returns:
        matplotlib.figure.Figure or None: The generated training loss plot if the
        file exists, otherwise None.
    """
    detector_stats_path = os.path.join(train_folder, "learning_stats_detector.csv")
    if not os.path.exists(detector_stats_path):
        return None

    df = pd.read_csv(detector_stats_path)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["step"], df["losses/train.total_loss"],
                        label="Detector Train Loss",
                        color='green')
    ax.set_title("Detector Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig

def show_pose_training_loss(train_folder):
    """
    Loads pose model training loss statistics and generates a plot.

    This function reads the 'learning_stats.csv' file from the specified training
    folder, extracts the pose model's total training loss values, and creates a 
    matplotlib figure showing the loss over training steps.

    Args:
        train_folder (str): Path to the folder containing 'learning_stats.csv'.

    Returns:
        matplotlib.figure.Figure or None: The generated training loss plot if the
        file exists, otherwise None.
    """
    stats_path = os.path.join(train_folder, "learning_stats.csv")
    if not os.path.exists(stats_path):
        return None

    df = pd.read_csv(stats_path)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["step"], df["losses/train.total_loss"],
                        label="Pose Train Loss",
                        color='blue')
    ax.set_title("Pose Model Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig

def show_training_plots(train_folder: str) -> None:
    """
    Displays side-by-side training loss plots for pose and detector models in Streamlit.

    This function retrieves and visualizes training loss data using helper functions
    (`show_pose_training_loss` and `show_detector_training_loss`) and presents the
    plots in two Streamlit columns. If data is missing, warnings are shown.

    Args:
        train_folder (str): Path to the folder containing the training statistics CSV files.
    """
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

def napari_instructions():
    """
    Displays a set of instructions for using Napari with DeepLabCut, 
    guiding the user through the necessary steps for labeling frames 
    and saving the labeled data.

    This function provides a series of instructions, accompanied by images, 
    to ensure that users properly set up Napari for labeling frames extracted 
    from videos. The instructions include navigating through Napari's menu, 
    loading project files, and saving the labeled data.

    The steps are displayed in the following order:
        1. Open the keypoint controls in Napari.
        2. Open the config file from the project folder.
        3. Open the labeled data folder and select the appropriate folder for labeling.
        4. Save the labeled frames and close Napari.

    The images are loaded from the "assets" folder, which should contain the 
    necessary screenshots for the instructions.

    Args:
        None: This function does not take any input arguments.

    Returns:
        None: This function is responsible for rendering instructions in Streamlit.

    Raises:
        None: Assumes the assets folder contains the necessary images.
    """
    st.markdown("""
    ### 📝 Napari Instructions
    All these steps must be followed in order; otherwise,
    Napari-DeepLabCut may not display elements correctly.
    """)

    ASSETS_PATH = Path("assets")

    # Step 1
    st.markdown("""
    #### Step 1
    Click `Plugins -> Keypoint controls` in the top left, 
    which will load a sidebar on the right side.
    Feel free to ignore the tutorial pop-up.
    """)
    st.image(ASSETS_PATH / "step 1.png", caption="")

    # Step 2
    st.markdown("""
    #### Step 2
    Click `File -> Open File(s)...`, navigate to the folder containing
    the project folder, then select open on the config file.
    """)
    st.image(ASSETS_PATH / "step 2.png", caption="")
    st.image(ASSETS_PATH / "step 3.png", caption="")

    # Step 3
    st.markdown("""
    #### Step 3
    Click `File -> Open Folder...`, navigate into the `labeled-data` folder. 
    Within should now be a folder with a name similar to the uploaded video. 
    Select open on that folder. 
    Finally, select `napari DeepLabCut`. 
    Now label the images like shown in the picture!
    """)
    st.image(ASSETS_PATH / "step 4.png", caption="")
    st.image(ASSETS_PATH / "step 5.png", caption="")
    st.image(ASSETS_PATH / "labels_shown.png", caption="")
    # Step 4
    st.markdown("""
    #### Step 4
    Once labeling has been completed,
    click `File -> Save Selected Layer(s)` while `CollectedData_conv_vid` is selected. 
    Then you can now close Napari and move on!
    """)
    st.image(ASSETS_PATH / "step 6.png", caption="")

# Training / Labeling
def run_labeling(config_path, processed_video_path):
    """
    Extracts frames from a video and opens Napari for frame labeling.

    This function adds the processed video to the config.yaml, extracts frames using 
    DeepLabCut, and then opens Napari for labeling. It also provides instructions to 
    the user to label the frames and close Napari once done.

    Args:
        config_path (str): The path to the config.yaml file.
        processed_video_path (str): The path to the processed video to be labeled.

    Returns:
        None: The function directly modifies the config file and interacts with Napari.

    Raises:
        Exception: If any error occurs during frame extraction, Napari launch, or video processing, 
                   it is caught and displayed in the Streamlit interface.
    """
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
        st.info("Napari will now open in a separate window.\n Scroll down and follow the instructions.")
        subprocess.Popen([sys.executable, "-m", "napari"])
        napari_instructions()
        st.warning("📝 Napari is running—label your frames, then close Napari when you’re done.")

    except Exception as e:
        st.error(f"❌ Frame extraction or labeling failed: {e}")

def run_retraining(config_path, train_folder,
                   num_epochs=25, num_detector_epochs=50):
    """
    Runs the retraining process for the DeepLabCut model.

    This function creates a new training dataset and trains the model
    using the provided config and training folder.

    Args:
        config_path (str): The path to the `config.yaml` file containing the model's configuration.
        train_folder (str): The path to the folder containing the training snapshots and model files.
        num_epochs (int, optional): The number of epochs to train the pose model (default is 25).
        num_detector_epochs (int, optional): The number of epochs to train the detector model (default is 50).

    Returns:
        None: The function modifies the model and dataset and updates the Streamlit interface with 
              the training progress and results.

    Raises:
        Exception: If any error occurs during the retraining steps, it will be caught, and an error message 
                   will be displayed in the Streamlit interface.

    """
    # Create training dataset
    try:       
        st.info("🛠️ Creating training dataset...")
        deeplabcut.create_training_dataset(config_path,
                                    num_shuffles=1,
                                    weight_init=None,
                                    net_type="top_down_hrnet_w48",
                                    detector_type="fasterrcnn_resnet50_fpn_v2",
                                    userfeedback=False)
        st.success("📦 Training dataset created!")
    except Exception as e:
        st.error(f"❌ Failed to create training dataset: {e}")
        st.stop()

    try:
        detector_path = os.path.join(
        train_folder,
        "snapshot-detector-200.pt"   # Change this for new model!
        )
        snapshot_path = os.path.join(
        train_folder,
        "snapshot-100.pt"   # Change this for new model!
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
    
def predict_and_show_labeled_video(config_path: str,
                                   video_path: str,
                                   videos_dir: str):
    """
    Analyzes a video using DeepLabCut, generates predictions, and displays the labeled video.

    This function performs the following steps:
    1. Analyzes the provided video using DeepLabCut's `analyze_videos` function.
    2. Saves the generated `.h5` file to the session state.
    3. Loads the `.h5` file and generates a labeled video.
    4. Displays the labeled video in the Streamlit interface.

    Args:
        config_path (str): The path to the config.yaml file for DeepLabCut.
        video_path (str): The path to the video to be analyzed and labeled.
        videos_dir (str): The directory where the videos are stored.

    Returns:
        None: The function directly displays the labeled video in Streamlit.

    Raises:
        Exception: If any error occurs during the analysis, labeling, or video display, an error message is shown in Streamlit.
    """
    try:
        st.info("📈 Running predictions and generating the labeled video. Please wait...") 
        st.write("This video include imputed outlier computations with default value, this can be changed in the Post Processing page")
        deeplabcut.analyze_videos(config_path, [video_path], shuffle=1)
        st.success("🎉 New predictions generated!")

        h5_path = save_h5_to_session(videos_dir=videos_dir)
        dlc_data = DataDLC(h5_file=h5_path)
        vid_bit = PlottingPlotly.generate_labeled_video(dlc_data, video_path)
        st.video(vid_bit)

    except Exception as e:
        st.error(f"❌ Could not complete prediction or labeling: {e}")