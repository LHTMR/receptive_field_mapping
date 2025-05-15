import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from src.post_processing.plotting_plotly import PlottingPlotly
from src.post_processing.datadlc import DataDLC
from parameterized import parameterized
import cv2
import tempfile
import os

class TestPlottingPlotly(unittest.TestCase):
    def setUp(self):
        self.valid_df = pd.DataFrame({
            'col1': np.random.rand(10),
            'col2': np.random.rand(10)
        })
        self.valid_columns = ['col1', 'col2']
        self.xlabel = "Time"
        self.ylabel_1 = "Signal 1"
        self.ylabel_2 = "Signal 2"
        self.title = "Test Plot"
        self.color_1 = "#1f77b4"
        self.color_2 = "#d62728"
        self.homography_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        # Set up dummy DLC and video for video tests
        self.dlc_data = self._make_dummy_dlc()
        self.video_path = self._make_dummy_video()

    def tearDown(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)

    def _make_dummy_dlc(self):
        num_frames = 5
        points = 2
        df = lambda: pd.DataFrame(
            np.random.randint(0, 100, size=(num_frames, points * 2)),
            columns=[f"x{i}" if j % 2 == 0 else f"y{i}" for i in range(points) for j in range(2)]
        )
        class DummyDLC:
            df_square = df()
            df_monofil = df()
        return DummyDLC()

    def _make_dummy_video(self):
        path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (100, 100))
        for _ in range(5):
            out.write(np.zeros((100, 100, 3), dtype=np.uint8))
        out.release()
        return path

    def test_get_lim(self):
        lim = PlottingPlotly._get_lim(self.homography_points)
        self.assertIsInstance(lim, tuple)
        self.assertEqual(len(lim), 2)
        self.assertTrue(all(isinstance(x, (int, float)) for x in lim))



    @parameterized.expand([
        ("no_invert", False),
        ("invert", True),
    ])
    def test_plot_dual_y_axis(self, name, invert_y_2):
        fig = PlottingPlotly.plot_dual_y_axis(
            df=self.valid_df,
            columns=self.valid_columns,
            xlabel=self.xlabel,
            ylabel_1=self.ylabel_1,
            ylabel_2=self.ylabel_2,
            title=self.title,
            color_1=self.color_1,
            color_2=self.color_2,
            invert_y_2=invert_y_2
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].name, self.ylabel_1)
        self.assertEqual(fig.data[1].name, self.ylabel_2)
        y2_data = fig.data[1].y
        expected = (
            -self.valid_df['col2'].reset_index(drop=True)
            if invert_y_2 else
            self.valid_df['col2'].reset_index(drop=True)
        )
        pd.testing.assert_series_equal(
            pd.Series(y2_data),
            expected,
            check_names=False
        )

    @parameterized.expand([
        ("columns_not_list", {"columns": "col1,col2"}, TypeError),
        ("columns_not_strings", {"columns": [1, 2]}, TypeError),
        ("columns_missing", {"columns": ['col1', 'not_in_df']}, ValueError),
        ("xlabel_not_string", {"xlabel": 123}, TypeError),
        ("ylabel_1_none", {"ylabel_1": None}, TypeError),
        ("ylabel_2_dict", {"ylabel_2": {}}, TypeError),
        ("title_empty_list", {"title": []}, TypeError),
        ("color_1_number", {"color_1": 999}, TypeError),
        ("color_2_none", {"color_2": None}, TypeError),
        ("df_is_none", {"df": None}, TypeError),
        ("df_wrong_columns", {"df": pd.DataFrame({'wrong': [1, 2] * 5})}, ValueError),
    ])
    def test_plot_dual_y_axis_invalid(self, name, override_args, expected_exception):
        kwargs = {
            "df": self.valid_df,
            "columns": self.valid_columns,
            "xlabel": self.xlabel,
            "ylabel_1": self.ylabel_1,
            "ylabel_2": self.ylabel_2,
            "title": self.title,
            "color_1": self.color_1,
            "color_2": self.color_2,
        }
        kwargs.update(override_args)
        with self.assertRaises(expected_exception):
            PlottingPlotly.plot_dual_y_axis(**kwargs)



    @parameterized.expand([
        ("default_colormaps", "Accent", "Blues"),
        ("custom_colormaps", "viridis", "plasma"),
    ])
    def test_generate_labeled_video_valid(self, name, square_cmap, filament_cmap):
        video_bytes = PlottingPlotly.generate_labeled_video(
            dlc_data=self.dlc_data,
            video_path=self.video_path,
            square_cmap=square_cmap,
            filament_cmap=filament_cmap
        )
        self.assertIsInstance(video_bytes, bytes)
        self.assertGreater(len(video_bytes), 0)

    @parameterized.expand([
        ("bad_dlc_type", "not_a_dlc", "valid_path", "Accent", "Blues", TypeError),
        ("bad_path_ext", "_valid_dlc", "video.txt", "Accent", "Blues", ValueError),
        ("nonexistent_file", "_valid_dlc", "/non/exist/path.mp4", "Accent", "Blues", FileNotFoundError),
        ("bad_square_cmap", "_valid_dlc", "valid_path", "notacolor", "Blues", ValueError),
        ("bad_filament_cmap", "_valid_dlc", "valid_path", "Accent", "notacolor", ValueError),
    ])
    def test_generate_labeled_video_invalid(self, name, dlc, video_path, square_cmap, filament_cmap, expected_exc):
        if dlc == "_valid_dlc":
            dlc = self.dlc_data
        if video_path == "valid_path":
            video_path = self.video_path
        with self.assertRaises(expected_exc):
            PlottingPlotly.generate_labeled_video(
                dlc_data=dlc,
                video_path=video_path,
                square_cmap=square_cmap,
                filament_cmap=filament_cmap
            )

if __name__ == "__main__":
    unittest.main()
