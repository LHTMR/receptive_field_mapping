import unittest
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
from ruamel.yaml import YAML,YAMLError
from src.train_predict import dlc_utils

class TestDLCUtils(unittest.TestCase):
########################################################################
    @patch("src.train_predict.dlc_utils.st")
    def test_successful_init_project(self, mock_st):
        with tempfile.TemporaryDirectory() as project_dir:
            config_path = os.path.join(project_dir, "config.yaml")

            # Create dummy config
            cfg = {
                "video_sets": {"dummy": "video.mp4"},
                "project_path": "old/path"
            }
            yaml_obj = YAML()
            with open(config_path, "w") as f:
                yaml_obj.dump(cfg, f)

            # Create dummy files/folders
            for folder in ["videos", "labeled-data"]:
                folder_path = os.path.join(project_dir, folder)
                os.makedirs(folder_path)
                with open(os.path.join(folder_path, "dummy.txt"), "w") as f:
                    f.write("test")

            dlc_utils.init_project(config_path, project_dir)

            # Reload and check config
            with open(config_path, "r") as f:
                updated_cfg = yaml_obj.load(f)
            self.assertEqual(updated_cfg["video_sets"], {})
            self.assertEqual(updated_cfg["project_path"], project_dir)

            # Folders should be empty
            for folder in ["videos", "labeled-data"]:
                self.assertEqual(os.listdir(os.path.join(project_dir,
                                                         folder)), [])

            mock_st.success.assert_called_once()
#------------------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    def test_missing_folders_handled_gracefully(self, mock_st):
        with tempfile.TemporaryDirectory() as project_dir:
            config_path = os.path.join(project_dir, "config.yaml")

            cfg = {"video_sets": {"v": "a"}, "project_path": "old/path"}
            yaml_obj = YAML()
            with open(config_path, "w") as f:
                yaml_obj.dump(cfg, f)

            # Do not create 'videos' or 'labeled-data'
            dlc_utils.init_project(config_path, project_dir)

            with open(config_path, "r") as f:
                updated_cfg = yaml_obj.load(f)
            self.assertEqual(updated_cfg["video_sets"], {})
            self.assertEqual(updated_cfg["project_path"], project_dir)

            mock_st.success.assert_called_once()
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    @patch("src.train_predict.dlc_utils.YAML.load",
            side_effect=YAMLError("YAML parse error"))
    def test_config_load_failure_triggers_error(self, mock_yaml_load, mock_st):
        with tempfile.TemporaryDirectory() as project_dir:
            config_path = os.path.join(project_dir, "config.yaml")

            # Valid YAML
            with open(config_path, "w") as f:
                f.write("video_sets: {}")

            dlc_utils.init_project(config_path, project_dir)

            mock_st.error.assert_called_once()
########################################################################
    @patch("src.train_predict.dlc_utils.glob")
    @patch("src.train_predict.dlc_utils.st")
    def test_save_h5_to_session_found(self, mock_st, mock_glob):

        # Arrange
        mock_glob.return_value = ["/fake/videos/some_file.h5"]

        # Make session_state behave like a real dictionary
        mock_st.session_state = {}  # Reset session_state as a real dict
        
        # Act
        result = dlc_utils.save_h5_to_session("/fake/videos")

        # Assert
        self.assertEqual(result, "/fake/videos/some_file.h5")
        self.assertEqual(mock_st.session_state["h5_path"],
                         "/fake/videos/some_file.h5")
        mock_st.success.assert_called_once_with("Labels Saved")
#-----------------------------------------------------------------------  
    @patch("src.train_predict.dlc_utils.glob", return_value=[])
    @patch("src.train_predict.dlc_utils.st")
    def test_save_h5_to_session_not_found(self, mock_st, mock_glob):
        result = dlc_utils.save_h5_to_session("/fake/videos")
        self.assertIsNone(result)
        mock_st.success.assert_not_called()
        mock_st.error.assert_not_called()
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.glob",
           side_effect=Exception("Something went wrong"))
    @patch("src.train_predict.dlc_utils.st")
    def test_save_h5_to_session_error(self, mock_st, mock_glob):
        result = dlc_utils.save_h5_to_session("/fake/videos")
        self.assertIsNone(result)
        mock_st.error.assert_called_once()
########################################################################
    @patch("src.train_predict.dlc_utils.st")
    def test_update_numframes2pick_when_different(self, mock_st):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                         suffix=".yaml") as tmp:
            yaml = YAML()
            yaml.preserve_quotes = True
            cfg = {"numframes2pick": 10}
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        # Run with different value
        dlc_utils.update_num_frames2pick(tmp_path, 15)

        with open(tmp_path, "r") as f:
            updated_cfg = yaml.load(f)
        os.remove(tmp_path)

        self.assertEqual(updated_cfg["numframes2pick"], 15)
        mock_st.success.assert_called_once_with("✅ Updated 'numframes2pick'")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    def test_update_numframes2pick_when_same(self, mock_st):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                         suffix=".yaml") as tmp:
            yaml = YAML()
            yaml.preserve_quotes = True
            cfg = {"numframes2pick": 20}
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        # Run with same value
        dlc_utils.update_num_frames2pick(tmp_path, 20)

        with open(tmp_path, "r") as f:
            updated_cfg = yaml.load(f)
        os.remove(tmp_path)

        self.assertEqual(updated_cfg["numframes2pick"], 20)
        mock_st.info.assert_called_once_with("ℹ️ 'numframes2pick' already set to the selected value. No update needed.")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    def test_update_numframes2pick_invalid_yaml(self, mock_st):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                         suffix=".yaml") as tmp:
            tmp.write("::: not valid yaml :::")
            tmp_path = tmp.name

        with self.assertRaises(Exception):
            dlc_utils.update_num_frames2pick(tmp_path, 30)

        mock_st.error.assert_called_once()
        os.remove(tmp_path)
########################################################################
    @patch("src.train_predict.dlc_utils.st")
    def test_add_video_to_config_when_not_exists(self, mock_st):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                         suffix=".yaml") as tmp:
            yaml = YAML()
            yaml.preserve_quotes = True
            cfg = {"video_sets": {}}
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        # Provide video path
        video_path = "processed_video.mp4"
        dlc_utils.add_video_to_config(tmp_path, video_path)

        with open(tmp_path, "r") as f:
            updated_cfg = yaml.load(f)
        os.remove(tmp_path)

        # Check that the video path was added
        self.assertIn(Path(video_path).as_posix(), updated_cfg["video_sets"])
        mock_st.success.assert_called_once_with("✅ Video added to config.yaml.")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    def test_add_video_to_config_when_exists(self, mock_st):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                         suffix=".yaml") as tmp:
            yaml = YAML()
            yaml.preserve_quotes = True
            cfg = {"video_sets": {"/path/to/video": {"crop": "0,1274,0,720"}}}
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        # Provide the same video path
        video_path = "/path/to/video"
        dlc_utils.add_video_to_config(tmp_path, video_path)

        with open(tmp_path, "r") as f:
            updated_cfg = yaml.load(f)
        os.remove(tmp_path)

        # Check that the video path was not added again
        self.assertIn(video_path, updated_cfg["video_sets"])
        mock_st.info.assert_called_once_with("ℹ️ Video already exists in config.yaml.")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    def test_add_video_to_config_invalid_yaml(self, mock_st):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False,
                                         suffix=".yaml") as tmp:
            tmp.write("::: not valid yaml :::")
            tmp_path = tmp.name

        # Run with invalid yaml content
        with self.assertRaises(Exception):
            dlc_utils.add_video_to_config(tmp_path, "processed_video.mp4")

        mock_st.error.assert_called_once()
        os.remove(tmp_path)
########################################################################
    @patch("src.train_predict.dlc_utils.st")
    @patch("src.train_predict.dlc_utils.os.remove")
    def test_delete_prev_pred_when_files_exist(self, mock_remove, mock_st):

        # Create a temporary directory to simulate the videos directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files to be removed
            video_file = os.path.join(tmp_dir, "test_video_labeled.mp4")
            h5_file = os.path.join(tmp_dir, "test_data.h5")
            pickle_file = os.path.join(tmp_dir, "test_data.pickle")
            with open(video_file, 'w'), open(h5_file, 'w'), open(pickle_file,
            'w'):
                pass

            # Run delete_prev_pred
            dlc_utils.delete_prev_pred(tmp_dir)

            # Assert that remove was called for each file
            mock_remove.assert_any_call(video_file)
            mock_remove.assert_any_call(h5_file)
            mock_remove.assert_any_call(pickle_file)

            # Check that success message is shown
            mock_st.success.assert_called_once_with("🧹 Previous predictions removed.")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    def test_delete_prev_pred_when_no_files(self, mock_st):
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Run delete_prev_pred on an empty directory
            dlc_utils.delete_prev_pred(tmp_dir)

            # Check that the appropriate info message is shown
            mock_st.info.assert_called_once_with("No prediction-related files found to remove.")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    def test_delete_prev_pred_when_directory_not_found(self, mock_st):

        # Test with a non-existing directory
        non_existent_dir = "/non/existent/directory"
        dlc_utils.delete_prev_pred(non_existent_dir)

        # Check that the warning message is shown
        mock_st.warning.assert_called_once_with(f"Videos folder not found at: {non_existent_dir}")
########################################################################
    @patch("src.train_predict.dlc_utils.os.remove")
    @patch("src.train_predict.dlc_utils.st")
    def test_clean_snapshots_when_files_exist(self, mock_st, mock_remove):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create snapshot files
            snapshot_075 = os.path.join(tmp_dir, "snapshot-075.pt")
            snapshot_200 = os.path.join(tmp_dir, "snapshot-detector-200.pt")
            snapshot_extra = os.path.join(tmp_dir, "snapshot-extra.pt")

            for path in [snapshot_075, snapshot_200, snapshot_extra]:
                with open(path, 'w'):
                    pass

            # Call the function under test
            dlc_utils.clean_snapshots(tmp_dir)

            # Check that only the unwanted snapshot was removed
            mock_remove.assert_called_once_with(snapshot_extra)
            mock_st.success.assert_called_once_with("🧹 Snapshots cleaned: Only selected files kept.")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    def test_clean_snapshots_when_no_files_to_remove(self, mock_st):
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Create only the files that should be kept
            snapshot_075 = os.path.join(tmp_dir, "snapshot-075.pt")
            snapshot_200 = os.path.join(tmp_dir, "snapshot-detector-200.pt")
            with open(snapshot_075, 'w'), open(snapshot_200, 'w'):
                pass

            # Run clean_snapshots
            dlc_utils.clean_snapshots(tmp_dir)

            # Ensure that no removal actions were triggered and that the success message is shown
            mock_st.success.assert_called_once_with("🧹 Snapshots cleaned: Only selected files kept.")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.st")
    def test_clean_snapshots_when_folder_not_found(self, mock_st):

        # Test with a non-existing folder
        non_existent_dir = "/non/existent/folder"
        dlc_utils.clean_snapshots(non_existent_dir)
########################################################################
    @patch("src.train_predict.dlc_utils.shutil.rmtree")
    def test_clear_training_datasets_when_folder_exists(self, mock_rmtree):

        # Create a temporary directory to simulate the project folder
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_datasets_path = os.path.join(tmp_dir, "training-datasets")

            # Create the folder to be cleared
            os.mkdir(training_datasets_path)

            # Run clear_training_datasets
            dlc_utils.clear_training_datasets(tmp_dir)

            # Check if rmtree was called to remove the 'training-datasets' folder
            mock_rmtree.assert_called_once_with(training_datasets_path)

            # Check that the success message is shown
            print("✅ Cleared existing training-datasets folder.")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.shutil.rmtree")
    def test_clear_training_datasets_when_folder_does_not_exist(self,
                                                                mock_rmtree):

        # Create a temporary directory to simulate the project folder
        # without the 'training-datasets' folder
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Run clear_training_datasets
            dlc_utils.clear_training_datasets(tmp_dir)

            # Check that rmtree was not called since the folder doesn't exist
            mock_rmtree.assert_not_called()

            # Check that the informational message is printed
            print("ℹ️ No training-datasets folder found, nothing to clear.")
########################################################################
    @patch("src.train_predict.dlc_utils.cv2.VideoCapture")
    @patch("src.train_predict.dlc_utils.cv2.VideoWriter")
    def test_preprocess_video(self, MockVideoWriter, MockVideoCapture):
        # Create temporary files for input and output videos
        with tempfile.NamedTemporaryFile(delete=False) as temp_input_video:
            input_video_path = temp_input_video.name
        with tempfile.NamedTemporaryFile(delete=False) as temp_output_video:
            output_video_path = temp_output_video.name

        # Mocking cv2.VideoCapture and cv2.VideoWriter
        mock_capture = MagicMock()
        mock_writer = MagicMock()

        MockVideoCapture.return_value = mock_capture
        MockVideoWriter.return_value = mock_writer

        # Mock the behavior of video capture
        mock_capture.isOpened.return_value = True
        mock_capture.read.side_effect = [
            (True, np.zeros((100, 100, 3), dtype=np.uint8)),  # Simulate reading a frame
            (False, None)  # Simulate end of video
        ]

        # Run preprocess_video
        result = dlc_utils.preprocess_video(input_video_path,
                                            output_video_path)

        # Check if the output video path is returned
        self.assertEqual(result, output_video_path)

        # Verify that VideoCapture and VideoWriter were called correctly
        MockVideoCapture.assert_called_once_with(input_video_path)
        MockVideoWriter.assert_called_once_with(output_video_path, ANY,
                                                30.0, (1274, 720))

        # Verify that frames were written to the output
        mock_writer.write.assert_called()

        # Clean up the temporary files after the test
        os.remove(input_video_path)
        os.remove(output_video_path)
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.cv2.VideoCapture")
    @patch("src.train_predict.dlc_utils.cv2.VideoWriter")
    def test_preprocess_video_no_frames(self,
                                        MockVideoWriter,
                                        MockVideoCapture):
        # Create temporary files for input and output videos
        with tempfile.NamedTemporaryFile(delete=False) as temp_input_video:
            input_video_path = temp_input_video.name
        with tempfile.NamedTemporaryFile(delete=False) as temp_output_video:
            output_video_path = temp_output_video.name

        # Mocking cv2.VideoCapture and cv2.VideoWriter
        mock_capture = MagicMock()
        mock_writer = MagicMock()

        MockVideoCapture.return_value = mock_capture
        MockVideoWriter.return_value = mock_writer

        # Mock the behavior of video capture (no frames to read)
        mock_capture.isOpened.return_value = False
        mock_capture.read.side_effect = [(False, None)]

        # Run preprocess_video
        result = dlc_utils.preprocess_video(input_video_path,
                                            output_video_path)

        # Check if the output video path is returned
        self.assertEqual(result, output_video_path)

        # Verify that VideoWriter was not called since there are no frames
        mock_writer.write.assert_not_called()

        # Clean up the temporary files after the test
        os.remove(input_video_path)
        os.remove(output_video_path)
########################################################################
    @patch("src.train_predict.dlc_utils.pd.read_csv")
    @patch("src.train_predict.dlc_utils.plt.subplots")
    def test_show_detector_training_loss(self,
                                         MockSubplots,
                                         MockReadCsv):
        # Mocking the DataFrame returned by pd.read_csv
        mock_df = pd.DataFrame({
            "step": [1, 2, 3],
            "losses/train.total_loss": [0.1, 0.05, 0.03]
        })
        MockReadCsv.return_value = mock_df

        # Mocking matplotlib's subplots method to prevent actual plot creation
        mock_figure = MagicMock()
        mock_axes = MagicMock()
        MockSubplots.return_value = (mock_figure, mock_axes)

        # Create a temporary folder and file
        with tempfile.TemporaryDirectory() as temp_dir:
            stats_file_path = os.path.join(temp_dir, "learning_stats_detector.csv")
            mock_df.to_csv(stats_file_path, index=False)

            # Call the function
            fig = dlc_utils.show_detector_training_loss(temp_dir)

            # Check if the figure is returned
            self.assertIsNotNone(fig)

            # Verify that read_csv was called with the correct path
            MockReadCsv.assert_called_once_with(stats_file_path)

            # Check if subplots were called to create the plot
            MockSubplots.assert_called_once_with(figsize=(5, 3))

            # Verify that the plot was created and axes were used
            mock_axes.plot.assert_called_once_with(mock_df["step"],
                                mock_df["losses/train.total_loss"],
                                label="Detector Train Loss", color='green')
            mock_axes.set_title.assert_called_once_with("Detector Training Loss")
            mock_axes.set_xlabel.assert_called_once_with("Epoch")
            mock_axes.set_ylabel.assert_called_once_with("Loss")
            mock_axes.legend.assert_called_once()
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.pd.read_csv")
    @patch("src.train_predict.dlc_utils.plt.subplots")
    @patch("src.train_predict.dlc_utils.os.path.exists", return_value=False)
    def test_show_detector_training_loss_file_not_found(self,
                                                        MockExists,
                                                        MockSubplots,
                                                        MockReadCsv):
        # Mock read_csv to simulate file not found scenario
        MockReadCsv.side_effect = FileNotFoundError

        with tempfile.TemporaryDirectory() as temp_dir:
            # Call the function and assert that None is returned
            fig = dlc_utils.show_detector_training_loss(temp_dir)
            self.assertIsNone(fig)
            
            # Check that read_csv was not called
            MockReadCsv.assert_not_called()

########################################################################
    @patch("src.train_predict.dlc_utils.pd.read_csv")
    @patch("src.train_predict.dlc_utils.plt.subplots")
    def test_show_pose_training_loss(self,
                                     MockSubplots,
                                     MockReadCsv):
        # Mocking the DataFrame returned by pd.read_csv
        mock_df = pd.DataFrame({
            "step": [1, 2, 3],
            "losses/train.total_loss": [0.2, 0.1, 0.05]
        })
        MockReadCsv.return_value = mock_df

        # Mocking matplotlib's subplots method to prevent actual plot creation
        mock_figure = MagicMock()
        mock_axes = MagicMock()
        MockSubplots.return_value = (mock_figure, mock_axes)

        # Create a temporary folder and file
        with tempfile.TemporaryDirectory() as temp_dir:
            stats_file_path = os.path.join(temp_dir, "learning_stats.csv")
            mock_df.to_csv(stats_file_path, index=False)

            # Call the function
            fig = dlc_utils.show_pose_training_loss(temp_dir)

            # Check if the figure is returned
            self.assertIsNotNone(fig)

            # Verify that read_csv was called with the correct path
            MockReadCsv.assert_called_once_with(stats_file_path)

            # Check if subplots were called to create the plot
            MockSubplots.assert_called_once_with(figsize=(5, 3))

            # Verify that the plot was created and axes were used
            mock_axes.plot.assert_called_once_with(mock_df["step"],\
            mock_df["losses/train.total_loss"], label="Pose Train Loss",
                                                color='blue')
            mock_axes.set_title.assert_called_once_with("Pose Model Training Loss")
            mock_axes.set_xlabel.assert_called_once_with("Epoch")
            mock_axes.set_ylabel.assert_called_once_with("Loss")
            mock_axes.legend.assert_called_once()
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.pd.read_csv")
    @patch("src.train_predict.dlc_utils.plt.subplots")
    @patch("src.train_predict.dlc_utils.os.path.exists", return_value=False)
    def test_show_pose_training_loss_file_not_found(self,
                                                    MockExists,
                                                    MockSubplots,
                                                    MockReadCsv):
        with tempfile.TemporaryDirectory() as temp_dir:
            fig = dlc_utils.show_pose_training_loss(temp_dir)
            self.assertIsNone(fig)

            MockReadCsv.assert_not_called()
########################################################################
    @patch("src.train_predict.dlc_utils.show_pose_training_loss")
    @patch("src.train_predict.dlc_utils.show_detector_training_loss")
    @patch("src.train_predict.dlc_utils.st.pyplot")
    def test_show_training_plots(self,
                                 MockPyplot,
                                 MockShowDetectorLoss,
                                 MockShowPoseLoss):
        # Mocking the returned figures from the training loss functions
        mock_pose_fig = MagicMock(spec=plt.Figure)
        mock_detector_fig = MagicMock(spec=plt.Figure)

        # Mocking the functions to return mock figures
        MockShowPoseLoss.return_value = mock_pose_fig
        MockShowDetectorLoss.return_value = mock_detector_fig

        with tempfile.TemporaryDirectory() as temp_dir:
            # Call the function
            dlc_utils.show_training_plots(temp_dir)

            # Check if the functions were called with the correct folder
            MockShowPoseLoss.assert_called_once_with(temp_dir)
            MockShowDetectorLoss.assert_called_once_with(temp_dir)

            # Verify if the figures are passed to Streamlit's st.pyplot
            MockPyplot.assert_any_call(mock_pose_fig)
            MockPyplot.assert_any_call(mock_detector_fig)
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.show_pose_training_loss")
    @patch("src.train_predict.dlc_utils.show_detector_training_loss")
    @patch("src.train_predict.dlc_utils.st.warning")
    def test_show_training_plots_missing_data(self,
                                              MockWarning,
                                              MockShowDetectorLoss,
                                              MockShowPoseLoss):
        # Mocking the functions to return None (missing data scenario)
        MockShowPoseLoss.return_value = None
        MockShowDetectorLoss.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Call the function
            dlc_utils.show_training_plots(temp_dir)

            # Verify that the warning messages are shown
            MockWarning.assert_any_call("No pose training loss data to display.")
            MockWarning.assert_any_call("No detector training loss data to display.")
########################################################################
    @patch("src.train_predict.dlc_utils.add_video_to_config")
    @patch("src.train_predict.dlc_utils.deeplabcut.extract_frames")
    @patch("src.train_predict.dlc_utils.subprocess.Popen")
    @patch("src.train_predict.dlc_utils.st.warning")
    @patch("src.train_predict.dlc_utils.st.success")
    @patch("src.train_predict.dlc_utils.st.info")
    @patch("src.train_predict.dlc_utils.st.error")
    def test_run_labeling(self, MockError, MockInfo, MockSuccess, MockWarning,
                          MockPopen, MockExtractFrames, MockAddVideo):
        # Prepare mock behavior
        MockAddVideo.return_value = None  # Mock add_video_to_config as no-op
        MockExtractFrames.return_value = None  # Mock the DeepLabCut frame extraction
        MockPopen.return_value = None  # Mock subprocess.Popen as no-op

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            processed_video_path = os.path.join(temp_dir, "processed_video.mp4")

            # Call the function
            dlc_utils.run_labeling(config_path, processed_video_path)

            # Verify that add_video_to_config was called
            MockAddVideo.assert_called_once_with(config_path,
                                                 processed_video_path)

            # Verify that extract_frames was called
            MockExtractFrames.assert_called_once_with(
                config_path,
                mode='automatic',
                crop=False,
                userfeedback=False
            )

            # Verify that subprocess.Popen (to launch napari) was called
            MockPopen.assert_called_once_with([sys.executable, "-m", "napari"])

            # Check if streamlit's info, success, and warning functions were called
            MockInfo.assert_any_call("🔍 Extracting frames...")
            MockSuccess.assert_any_call("🖼️ Frames extracted!")
            MockWarning.assert_any_call("📝 Napari is running—label your frames, then close Napari when you’re done.")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.add_video_to_config")
    @patch("src.train_predict.dlc_utils.deeplabcut.extract_frames")
    @patch("src.train_predict.dlc_utils.subprocess.Popen")
    @patch("src.train_predict.dlc_utils.st.error")
    def test_run_labeling_failure(self, MockError, MockPopen,
                                  MockExtractFrames, MockAddVideo):
        # Simulate exception during extract_frames
        MockAddVideo.return_value = None
        MockExtractFrames.side_effect = Exception("Frame extraction error")
        MockPopen.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            processed_video_path = os.path.join(temp_dir, "processed_video.mp4")

            # Run the function
            dlc_utils.run_labeling(config_path, processed_video_path)

            # Verify error message was shown
            MockError.assert_any_call("❌ Frame extraction or labeling failed: Frame extraction error")

########################################################################
    @patch("deeplabcut.train_network")
    @patch("deeplabcut.create_training_dataset")
    @patch("src.train_predict.dlc_utils.st.success")
    @patch("src.train_predict.dlc_utils.st.info")
    def test_run_retraining(self, MockInfo, MockSuccess,
                            MockCreateDataset, MockTrainNetwork):
        MockCreateDataset.return_value = None
        MockTrainNetwork.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            train_folder = os.path.join(temp_dir, "train_folder")
            os.makedirs(train_folder, exist_ok=True)

            # Call the function that triggers the retraining process
            dlc_utils.run_retraining(config_path, train_folder)

            # Verify that create_training_dataset was called
            MockCreateDataset.assert_called_once_with(
                config_path,
                num_shuffles=1,
                weight_init=None,
                net_type="top_down_resnet_50",
                userfeedback=False
            )

            # Verify that train_network was called
            MockTrainNetwork.assert_called_once_with(
                config_path,
                detector_path=os.path.join(train_folder,
                                           "snapshot-detector-200.pt"),
                snapshot_path=os.path.join(train_folder,
                                           "snapshot-075.pt"),
                epochs=25,
                save_epochs=25,
                detector_epochs=50,
                detector_save_epochs=50,
                shuffle=1,
                batch_size=2,
                detector_batch_size=2,
                autotune=False,
                keepdeconvweights=True
            )

            # Verify the success and info messages are called
            MockInfo.assert_any_call("🛠️ Creating training dataset...")
            MockSuccess.assert_any_call("📦 Training dataset created!")
            MockInfo.assert_any_call("🧠 Starting model training...")
            MockSuccess.assert_any_call("✅ Training complete!")
#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.deeplabcut.create_training_dataset")
    @patch("src.train_predict.dlc_utils.deeplabcut.train_network")
    @patch("src.train_predict.dlc_utils.st.error")
    def test_run_retraining_failure(self, MockError,
                                    MockTrainNetwork, MockCreateDataset):
        # Simulate a failure in training dataset creation
        MockCreateDataset.side_effect = Exception("Dataset creation error")
        MockTrainNetwork.return_value = None  # Mock the model training

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            train_folder = os.path.join(temp_dir, "train_folder")

            os.makedirs(train_folder, exist_ok=True)

            # Call the function
            dlc_utils.run_retraining(config_path, train_folder)

            # Verify error message is shown for dataset creation failure
            MockError.assert_any_call("❌ Failed to create training dataset: Dataset creation error")

        # Simulate a failure in model training
        MockCreateDataset.side_effect = None  # Reset the side effect
        MockTrainNetwork.side_effect = Exception("Training error")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            train_folder = os.path.join(temp_dir, "train_folder")

            os.makedirs(train_folder, exist_ok=True)

            # Call the function
            dlc_utils.run_retraining(config_path, train_folder)

            # Verify error message is shown for model training failure
            MockError.assert_any_call("❌ Failed to train the model: Training error")
########################################################################
    @patch("src.train_predict.dlc_utils.st.video")
    @patch("src.train_predict.dlc_utils.st.error")
    @patch("src.train_predict.dlc_utils.st.success")
    @patch("src.train_predict.dlc_utils.st.info")
    @patch("src.train_predict.dlc_utils.PlottingPlotly.generate_labeled_video")
    @patch("src.train_predict.dlc_utils.DataDLC")
    @patch("src.train_predict.dlc_utils.save_h5_to_session")
    @patch("src.train_predict.dlc_utils.deeplabcut.analyze_videos")
    def test_predict_and_show_labeled_video_success(
        self,
        MockAnalyze,
        MockSaveH5,
        MockDataDLC,
        MockGenerateVideo,
        MockInfo,
        MockSuccess,
        MockError,
        MockVideo,
    ):
        # Prepare mocks
        MockAnalyze.return_value = None
        MockSaveH5.return_value = "mock_h5_path"
        MockDataDLC.return_value = MagicMock()
        MockGenerateVideo.return_value = "mock_video"
        MockVideo.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            video_path = os.path.join(temp_dir, "video.mp4")
            videos_dir = temp_dir

            # Call the function
            dlc_utils.predict_and_show_labeled_video(config_path,
                                                     video_path,
                                                     videos_dir)

            # Verify analyze_videos was called with correct arguments
            MockAnalyze.assert_called_once_with(config_path,
                                                [video_path],
                                                shuffle=1)

            # Verify save_h5_to_session was called
            MockSaveH5.assert_called_once_with(videos_dir=videos_dir)

            # Verify DataDLC object was created with the correct H5 path
            MockDataDLC.assert_called_once_with(h5_file="mock_h5_path")

            # Verify generate_labeled_video was called with correct arguments
            MockGenerateVideo.assert_called_once_with(MockDataDLC.return_value,
                                                      video_path)

            # Verify Streamlit's video function was called
            MockVideo.assert_called_once_with("mock_video")

            # Verify the appropriate Streamlit success messages were shown
            MockInfo.assert_any_call("📈 Analyzing video with DeepLabCut...")
            MockSuccess.assert_any_call("🎉 New predictions generated!")

#-----------------------------------------------------------------------
    @patch("src.train_predict.dlc_utils.deeplabcut.analyze_videos")
    @patch("src.train_predict.dlc_utils.save_h5_to_session")
    @patch("src.train_predict.dlc_utils.st.error")
    def test_predict_and_show_labeled_video_failure(self, MockError,
                                                    MockSaveH5, MockAnalyze):
        # Simulate a failure in the analysis step
        MockAnalyze.side_effect = Exception("Analysis failed")
        # Mock the return of save_h5_to_session
        MockSaveH5.return_value = "mock_h5_path"

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            video_path = os.path.join(temp_dir, "video.mp4")
            videos_dir = temp_dir

            # Call the function
            dlc_utils.predict_and_show_labeled_video(config_path,
                                                     video_path,
                                                     videos_dir)

            # Verify that the error message is displayed
            MockError.assert_any_call("❌ Could not complete prediction or labeling: Analysis failed")

        # Simulate a failure in the save_h5_to_session step
        # Reset side effect for analyze_videos
        MockAnalyze.side_effect = None  
        MockSaveH5.side_effect = Exception("Saving H5 failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            video_path = os.path.join(temp_dir, "video.mp4")
            videos_dir = temp_dir

            # Call the function
            dlc_utils.predict_and_show_labeled_video(config_path,
                                                     video_path,
                                                     videos_dir)

            # Verify that the error message is displayed for saving H5 failure
            MockError.assert_any_call("❌ Could not complete prediction or labeling: Saving H5 failed")
########################################################################
if __name__ == '__main__':
    unittest.main()
