import unittest
import streamlit as st

from unittest.mock import patch, MagicMock, Mock
from src.post_processing import processing_utils

class TestProcessingUtils(unittest.TestCase):

    def setUp(self):
        # Clear Streamlit session state before each test
        st.session_state.clear()

    def test_get_temp_video_path_existing_key(self):
        st.session_state["labeled_video_path"] = "temp/path.mp4"
        result = processing_utils.get_temp_video_path("dummy_file")
        self.assertEqual(result, "temp/path.mp4")

    def test_creates_temp_file_and_returns_path(self):
        # session_state is cleared in setUp, so no key present here

        # Mock video_file with a read() method returning bytes
        mock_video_file = MagicMock()
        mock_video_file.read.return_value = b"dummy video content"

        result_path = processing_utils.get_temp_video_path(mock_video_file)

        self.assertIsInstance(result_path, str)
        self.assertTrue(result_path.endswith(".mp4"))

        self.assertIn("labeled_video_path", st.session_state)
        self.assertEqual(st.session_state["labeled_video_path"], result_path)

        # Check file contents
        with open(result_path, "rb") as f:
            content = f.read()
            self.assertEqual(content, b"dummy video content")

    @patch("streamlit.file_uploader")
    @patch("streamlit.success")
    def test_assign_video_path_existing(self, mock_success, mock_uploader):
        st.session_state["labeled_video_path"] = "existing/path.mp4"
        processing_utils.assign_video_path()
        mock_success.assert_called_with("Labeled video path already in session state!")

    @patch("streamlit.file_uploader")
    @patch("streamlit.success")
    def test_assign_video_path_upload_new(self, mock_success, mock_uploader):
        # Simulate file upload with a mock object
        mock_file = MagicMock()
        mock_file.name = "new_video.mp4"
        mock_uploader.return_value = mock_file

        with patch("src.post_processing.processing_utils.get_temp_video_path",
                   return_value="temp/new_video.mp4"):
            processing_utils.assign_video_path()
            self.assertEqual(st.session_state["labeled_video_path"],
                             "temp/new_video.mp4")
            mock_success.assert_called_with("Labeled video path assigned successfully!")

    def test_get_all_matplotlib_cmaps(self):
        cmaps = processing_utils.get_all_matplotlib_cmaps()
        self.assertIsInstance(cmaps, dict)
        self.assertIn("viridis", cmaps)

    @patch("streamlit.columns")
    @patch("streamlit.text_input")
    @patch("streamlit.color_picker")
    def test_get_plot_inputs(self,
                             mock_color_picker,
                             mock_text_input,
                             mock_columns):
        # Setup mock return values
        mock_text_input.side_effect = ["My Title", "x", "y"]
        mock_color_picker.return_value = "#123456"
        mock_columns.return_value = [MagicMock()] * 4

        result = processing_utils.get_plot_inputs()
        self.assertEqual(result, ("My Title", "x", "y", "#123456"))

    @patch("streamlit.columns")
    @patch("streamlit.text_input")
    @patch("streamlit.color_picker")
    @patch("streamlit.checkbox")
    def test_get_dual_y_axis_plot_inputs(self,
                                         mock_checkbox,
                                         mock_color_picker,
                                         mock_text_input,
                                         mock_columns):
        mock_text_input.side_effect = ["Plot Title", "time",
                                       "amplitude", "firing"]
        mock_color_picker.side_effect = ["#abc123", "#def456"]
        mock_checkbox.return_value = True
        mock_columns.return_value = [MagicMock()] * 7

        result = processing_utils.get_dual_y_axis_plot_inputs()
        self.assertEqual(result, ("Plot Title", "time", "amplitude",
                                  "firing", "#abc123", "#def456", True))

    def test_get_all_plotly_cmaps(self):
        cmaps = processing_utils.get_all_plotly_cmaps()
        self.assertIsInstance(cmaps, dict)
        self.assertIn("Viridis", cmaps)  # typical cmap


if __name__ == '__main__':
    unittest.main()
