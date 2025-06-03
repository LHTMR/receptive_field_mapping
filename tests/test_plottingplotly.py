import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from src.post_processing.plotting_plotly import PlottingPlotly
from src.post_processing.datadlc import DataDLC
from src.post_processing.dataneuron import DataNeuron
from src.post_processing.mergeddata import MergedData
from parameterized import parameterized
import cv2
import tempfile
import os
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") # Prevent GUI windows in test runs


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

        # Set up mock DataDLC obj
        self.dlc_data = DataDLC("tests/mock_dlc_data.h5")
        self.dlc_data.get_bending_coefficients()
        self.dlc_data.apply_homography()
        # Set up mock DataNeuron obj
        self.neuron_data = DataNeuron("tests/mock_neuron_data.csv", original_freq=10)
        self.neuron_data.downsample(3)
        # Set up mock MergedData obj
        self.merged_data = MergedData(
            dlc=self.dlc_data,
            neuron=self.neuron_data,
            max_gap_fill=2,
            threshold=0.15
        )
        # Set up dummy video
        self.video_path = self._make_dummy_video()

        # 4 homography points
        self.homography_points = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
        # 5 frames, 4 points (so 8 columns: x0,y0,x1,y1,x2,y2,x3,y3)
        data = np.random.rand(5, 8)
        columns = [f"{axis}{i}" for i in range(4) for axis in ("x", "y")]
        self.df_transformed = pd.DataFrame(data, columns=columns)

    def tearDown(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)

    def _make_dummy_video(self):
        path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (100, 100))
        for _ in range(3):
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

    @parameterized.expand([
        ("default_args", 30, "Homography Animation", "x (mm)", "y (mm)", "#d62728", (12, 12)),
        ("custom_args", 15, "Custom Title", "X", "Y", "#0760A0", (8, 8)),
    ])
    def test_generate_homography_video_valid(self, name, fps, title, x_label, y_label, color, figsize):
        # Create dummy homography points and transformed monofil DataFrame
        homography_points = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
        # 5 frames, 4 points (so 8 columns: x0,y0,x1,y1,x2,y2,x3,y3)
        data = np.random.rand(5, 8)
        columns = [f"{axis}{i}" for i in range(4) for axis in ("x", "y")]
        df_transformed = pd.DataFrame(data, columns=columns)

        # Each row should be flattened to 8 values, then reshaped to (4,2) in the function
        video_bytes = PlottingPlotly.generate_homography_video(
            homography_points=homography_points,
            df_transformed_monofil=df_transformed,
            fps=fps,
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            figsize=figsize
        )
        self.assertIsInstance(video_bytes, bytes)
        self.assertGreater(len(video_bytes), 0)

    @parameterized.expand([
        ("bad_homography_shape",
        np.zeros((3, 2)), pd.DataFrame(np.random.rand(5, 8)), 30,
        "Test", "x", "y", "#d62728", (12, 12), ValueError),
        ("bad_df_type",
        np.zeros((4, 2)), "not_a_df", 30,
        "Test", "x", "y", "#d62728", (12, 12), TypeError),
        ("bad_fps",
        np.zeros((4, 2)), pd.DataFrame(np.random.rand(5, 8)), -1,
        "Test", "x", "y", "#d62728", (12, 12), ValueError),
        ("bad_figsize",
        np.zeros((4, 2)), pd.DataFrame(np.random.rand(5, 8)), 30,
        "Test", "x", "y", "#d62728", "not_a_tuple", TypeError),
    ])
    def test_generate_homography_video_invalid(self, name,
                                            homography_points, df_transformed, fps,
                                            title, x_label, y_label, color, figsize,
                                            expected_exc):
        with self.assertRaises(expected_exc):
            PlottingPlotly.generate_homography_video(
                homography_points=homography_points,
                df_transformed_monofil=df_transformed,
                fps=fps,
                title=title,
                x_label=x_label,
                y_label=y_label,
                color=color,
                figsize=figsize
            )

    @parameterized.expand([
        ("default_args", "Homography Animation", "royalblue", "x (mm)", "y (mm)"),
        ("custom_args", "Custom Title", "#d62728", "X", "Y"),
    ])
    def test_plot_homography_interactive_valid(self, name, title, color, x_label, y_label):
        fig = PlottingPlotly.plot_homography_interactive(
            homography_points=self.homography_points,
            df_transformed_monofil=self.df_transformed,
            title=title,
            color=color,
            x_label=x_label,
            y_label=y_label
        )
        self.assertIsInstance(fig, Figure)
        self.assertTrue(len(fig.frames) == len(self.df_transformed))

    @parameterized.expand([
        ("bad_homography_shape",
         np.zeros((3, 2)), pd.DataFrame(np.random.rand(5, 8)), "Test",
         "#d62728", "x (mm)", "y (mm)", ValueError),
        ("bad_df_type",
         np.zeros((4, 2)), "not_a_df", "Test",
         "#d62728", "x (mm)", "y (mm)", TypeError),
        ("bad_title_type",
         np.zeros((4, 2)), pd.DataFrame(np.random.rand(5, 8)), 123,
         "#d62728", "x (mm)", "y (mm)", TypeError),
        ("bad_color_type",
         np.zeros((4, 2)), pd.DataFrame(np.random.rand(5, 8)), "Test",
         123, "x (mm)", "y (mm)", TypeError),
        ("bad_x_label_type",
         np.zeros((4, 2)), pd.DataFrame(np.random.rand(5, 8)), "Test",
         "#d62728", 123, "y (mm)", TypeError),
        ("bad_y_label_type",
         np.zeros((4, 2)), pd.DataFrame(np.random.rand(5, 8)), "Test",
         "#d62728", "x (mm)", 123, TypeError)
    ])
    def test_plot_homography_interactive_invalid(self,
                                                 name,
                                                 homography_points,
                                                 df_transformed,
                                                 title,
                                                 color,
                                                 x_label, y_label,
                                                 expected_exc):
        with self.assertRaises(expected_exc):
            PlottingPlotly.plot_homography_interactive(
                homography_points=homography_points,
                df_transformed_monofil=df_transformed,
                title=title,
                color=color,
                x_label=x_label,
                y_label=y_label
            )

    @parameterized.expand([
        ("default_args", False, False, "x", "y", "size", "color", "RF Mapping Animation", 30, (12, 12), "viridis"),
        ("bending_true", True, False, "x", "y", "size", "color", "Bending On", 24, (8, 8), "plasma"),
        ("spikes_true", False, True, "x", "y", "size", "Spikes", "Spikes On", 15, (10, 10), "viridis"),
    ])
    def test_plot_rf_mapping_animated_valid(self, name, bending, spikes, x_col, y_col, size_col, color_col, title, fps, figsize, cmap):
        # Create a dummy MergedData with required columns
        num_frames = 5
        df = pd.DataFrame({
            x_col: np.random.rand(num_frames) * 20,
            y_col: np.random.rand(num_frames) * 20,
            size_col: np.random.rand(num_frames) * 10 + 1,
            color_col: np.random.rand(num_frames) * 5 if color_col != "Spikes" else [0, 1, 0, 1, 0],
            "Spikes": [0, 1, 0, 1, 0],  # Always present for spikes
        })
        merged_data = MagicMock(spec=MergedData)
        merged_data.threshold_data.return_value = df

        video_bytes = PlottingPlotly.plot_rf_mapping_animated(
            merged_data=merged_data,
            x_col=x_col,
            y_col=y_col,
            homography_points=self.homography_points,
            size_col=size_col,
            color_col=color_col,
            title=title,
            bending=bending,
            spikes=spikes,
            xlabel="x",
            ylabel="y",
            fps=fps,
            figsize=figsize,
            cmap=cmap
        )
        self.assertIsInstance(video_bytes, bytes)
        self.assertGreater(len(video_bytes), 0)

    @parameterized.expand([
        ("bad_merged_data",
        "not_merged", "x", "y", "size", "color", "Title",
        False, False, 30, (12, 12), "viridis", np.zeros((4, 2)),
        TypeError),
        ("bad_homography_shape",
        MagicMock(spec=MergedData), "x", "y", "size", "color", "Title",
        False, False, 30, (12, 12), "viridis", np.zeros((3, 2)),
        ValueError),
        ("bad_x_col_type",
         MagicMock(spec=MergedData), 123, "y", "size", "color", "Title",
         False, False, 30, (12, 12), "viridis", np.zeros((4, 2)),
         TypeError),
        ("bad_y_col_type",
         MagicMock(spec=MergedData), "x", None, "size", "color", "Title",
         False, False, 30, (12, 12), "viridis", np.zeros((4, 2)),
         TypeError),
        ("bad_bending_type",
         MagicMock(spec=MergedData), "x", "y", "size", "color", "Title",
         "not_bool", False, 30, (12, 12), "viridis", np.zeros((4, 2)),
         TypeError),
        ("bad_spikes_type",
         MagicMock(spec=MergedData), "x", "y", "size", "color", "Title",
         False, "not_bool", 30, (12, 12), "viridis", np.zeros((4, 2)),
         TypeError),
        ("bad_fps",
         MagicMock(spec=MergedData), "x", "y", "size", "color", "Title",
         False, False, -1, (12, 12), "viridis", np.zeros((4, 2)),
         ValueError),
        ("bad_figsize",
         MagicMock(spec=MergedData), "x", "y", "size", "color", "Title",
         False, False, 30, "not_a_tuple", "viridis", np.zeros((4, 2)),
         TypeError),
        ("bad_cmap",
         MagicMock(spec=MergedData), "x", "y", "size", "color", "Title",
         False, False, 30, (12, 12), "notacolor", np.zeros((4, 2)),
         ValueError),
    ])
    def test_plot_rf_mapping_animated_invalid(self, name, merged_data, x_col, y_col, size_col, color_col, title, bending, spikes, fps, figsize, cmap, homography_points, expected_exc):
        # For valid merged_data, mock threshold_data to return a valid DataFrame
        if isinstance(merged_data, MagicMock):
            num_frames = 5
            df = pd.DataFrame({
                x_col if isinstance(x_col, str) else "x": np.random.rand(num_frames) * 20,
                y_col if isinstance(y_col, str) else "y": np.random.rand(num_frames) * 20,
                size_col if isinstance(size_col, str) else "size": np.random.rand(num_frames) * 10 + 1,
                color_col if isinstance(color_col, str) else "color": np.random.rand(num_frames) * 5,
                "Spikes": [0, 1, 0, 1, 0],
            })
            merged_data.threshold_data.return_value = df

        with self.assertRaises(expected_exc):
            PlottingPlotly.plot_rf_mapping_animated(
                merged_data=merged_data,
                x_col=x_col,
                y_col=y_col,
                homography_points=homography_points,
                size_col=size_col,
                color_col=color_col,
                title=title,
                bending=bending,
                spikes=spikes,
                xlabel="x",
                ylabel="y",
                fps=fps,
                figsize=figsize,
                cmap=cmap
            )

    def test_compute_kde_valid(self):
        # Create a simple DataFrame with two columns
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100)
        })
        grid_limits = (-3, 3, -3, 3)
        xx, yy, zz = PlottingPlotly._compute_kde(df, "x", "y", grid_limits, bw_method=0.2)
        self.assertIsInstance(xx, np.ndarray)
        self.assertIsInstance(yy, np.ndarray)
        self.assertIsInstance(zz, np.ndarray)
        self.assertEqual(xx.shape, yy.shape)
        self.assertEqual(xx.shape, zz.shape)
        self.assertEqual(xx.shape, (200, 200))

    def test_compute_kde_invalid_column(self):
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100)
        })
        grid_limits = (-3, 3, -3, 3)
        # Should raise KeyError for missing column
        with self.assertRaises(KeyError):
            PlottingPlotly._compute_kde(df, "not_x", "y", grid_limits)

    def test_compute_kde_invalid_grid_limits(self):
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100)
        })
        # grid_limits should be a tuple of 4 numbers
        with self.assertRaises(ValueError):
            PlottingPlotly._compute_kde(df, "x", "y", (-3, 3, -3), bw_method=0.2)

    def test_compute_kde_invalid_bw_method(self):
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100)
        })
        grid_limits = (-3, 3, -3, 3)
        # Should raise an error for invalid bandwidth method
        with self.assertRaises(Exception):
            PlottingPlotly._compute_kde(df, "x", "y", grid_limits, bw_method="invalid")

    @parameterized.expand([
        ("bending_only", True, False, 0.2, 0.2, "KDE Plot", "x (mm)", "y (mm)", "Viridis", "Reds", 0.05),
        ("spikes_only", False, True, 0.2, 0.2, "KDE Plot", "x (mm)", "y (mm)", "Viridis", "Reds", 0.05),
        ("both", True, True, 0.2, 0.2, "Both KDE", "X", "Y", "Viridis", "Reds", 0.1),
    ])
    def test_plot_kde_density_interactive_valid(self, name, bending, spikes, bw_bending, bw_spikes, title, xlabel, ylabel, cmap_bending, cmap_spikes, threshold_percentage):
        # Create a dummy MergedData with required columns
        num_points = 100
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, num_points),
            "y": np.random.normal(0, 1, num_points),
            "Spikes": np.random.randint(0, 2, num_points),
            "Bending": np.random.rand(num_points),
        })
        merged_data = MagicMock(spec=MergedData)
        merged_data.threshold_data.side_effect = lambda bending_flag, spikes_flag: df

        fig = PlottingPlotly.plot_kde_density_interactive(
            merged_data=merged_data,
            x_col="x",
            y_col="y",
            homography_points=self.homography_points,
            bending=bending,
            spikes=spikes,
            bw_bending=bw_bending,
            bw_spikes=bw_spikes,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            cmap_bending=cmap_bending,
            cmap_spikes=cmap_spikes,
            threshold_percentage=threshold_percentage
        )
        self.assertIsInstance(fig, Figure)
        self.assertGreater(len(fig.data), 0)

    @parameterized.expand([
        ("bad_merged_data", "not_merged", "x", "y", np.zeros((4, 2)), True, False, 0.2, 0.2, "KDE Plot", "x", "y", "Viridis", "Reds", 0.05, TypeError),
        ("bad_homography_shape", MagicMock(spec=MergedData), "x", "y", np.zeros((3, 2)), True, False, 0.2, 0.2, "KDE Plot", "x", "y", "Viridis", "Reds", 0.05, ValueError),
        ("bad_x_col_type", MagicMock(spec=MergedData), 123, "y", np.zeros((4, 2)), True, False, 0.2, 0.2, "KDE Plot", "x", "y", "Viridis", "Reds", 0.05, TypeError),
        ("bad_y_col_type", MagicMock(spec=MergedData), "x", None, np.zeros((4, 2)), True, False, 0.2, 0.2, "KDE Plot", "x", "y", "Viridis", "Reds", 0.05, TypeError),
        ("bad_bending_type", MagicMock(spec=MergedData), "x", "y", np.zeros((4, 2)), "not_bool", False, 0.2, 0.2, "KDE Plot", "x", "y", "Viridis", "Reds", 0.05, TypeError),
        ("bad_spikes_type", MagicMock(spec=MergedData), "x", "y", np.zeros((4, 2)), False, "not_bool", 0.2, 0.2, "KDE Plot", "x", "y", "Viridis", "Reds", 0.05, TypeError),
        ("bad_cmap_bending", MagicMock(spec=MergedData), "x", "y", np.zeros((4, 2)), True, False, 0.2, 0.2, "KDE Plot", "x", "y", "notacolor", "Reds", 0.05, ValueError),
        ("bad_cmap_spikes", MagicMock(spec=MergedData), "x", "y", np.zeros((4, 2)), False, True, 0.2, 0.2, "KDE Plot", "x", "y", "Viridis", "notacolor", 0.05, ValueError),
    ])
    def test_plot_kde_density_interactive_invalid(self, name, merged_data, x_col, y_col, homography_points, bending, spikes, bw_bending, bw_spikes, title, xlabel, ylabel, cmap_bending, cmap_spikes, threshold_percentage, expected_exc):
        # For valid merged_data, mock threshold_data to return a valid DataFrame
        if isinstance(merged_data, MagicMock):
            num_points = 100
            df = pd.DataFrame({
                "x": np.random.normal(0, 1, num_points),
                "y": np.random.normal(0, 1, num_points),
                "Spikes": np.random.randint(0, 2, num_points),
                "Bending": np.random.rand(num_points),
            })
            merged_data.threshold_data.side_effect = lambda bending_flag, spikes_flag: df

        with self.assertRaises(expected_exc):
            PlottingPlotly.plot_kde_density_interactive(
                merged_data=merged_data,
                x_col=x_col,
                y_col=y_col,
                homography_points=homography_points,
                bending=bending,
                spikes=spikes,
                bw_bending=bw_bending,
                bw_spikes=bw_spikes,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                cmap_bending=cmap_bending,
                cmap_spikes=cmap_spikes,
                threshold_percentage=threshold_percentage
            )

    @parameterized.expand([
        ("default_args", "x", "y", "size", "color", False, False, "Scatter Plot", "x (mm)", "y (mm)", "Viridis"),
        ("spikes_col", "x", "y", "size", "Spikes", False, True, "Spikes Only", "X", "Y", "Viridis"),
    ])
    def test_plot_scatter_interactive_valid(self, name, x_col, y_col, size_col, color_col, bending, spikes, title, xlabel, ylabel, cmap):
        # Create a dummy MergedData with required columns
        num_points = 50
        df = pd.DataFrame({
            x_col: np.random.rand(num_points) * 20,
            y_col: np.random.rand(num_points) * 20,
            size_col: np.random.rand(num_points) * 10 + 1,
            "Spikes": np.random.randint(0, 2, num_points),
            "color": np.random.rand(num_points) * 5,
        })
        merged_data = MagicMock(spec=MergedData)
        merged_data.threshold_data.return_value = df

        fig = PlottingPlotly.plot_scatter_interactive(
            merged_data=merged_data,
            x_col=x_col,
            y_col=y_col,
            homography_points=self.homography_points,
            size_col=size_col,
            color_col=color_col,
            bending=bending,
            spikes=spikes,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap
        )
        self.assertIsInstance(fig, Figure)
        self.assertGreater(len(fig.data), 0)

    @parameterized.expand([
        ("bad_merged_data", "not_merged", "x", "y", "size", "color", False, False, "Scatter Plot", "x", "y", "Viridis", np.zeros((4, 2)), TypeError),
        ("bad_homography_shape", MagicMock(spec=MergedData), "x", "y", "size", "color", False, False, "Scatter Plot", "x", "y", "Viridis", np.zeros((3, 2)), ValueError),
        ("bad_x_col_type", MagicMock(spec=MergedData), 123, "y", "size", "color", False, False, "Scatter Plot", "x", "y", "Viridis", np.zeros((4, 2)), TypeError),
        ("bad_y_col_type", MagicMock(spec=MergedData), "x", None, "size", "color", False, False, "Scatter Plot", "x", "y", "Viridis", np.zeros((4, 2)), TypeError),
        ("bad_size_col_type", MagicMock(spec=MergedData), "x", "y", 123, "color", False, False, "Scatter Plot", "x", "y", "Viridis", np.zeros((4, 2)), TypeError),
        ("bad_color_col_type", MagicMock(spec=MergedData), "x", "y", "size", 123, False, False, "Scatter Plot", "x", "y", "Viridis", np.zeros((4, 2)), TypeError),
        ("bad_bending_type", MagicMock(spec=MergedData), "x", "y", "size", "color", "not_bool", False, "Scatter Plot", "x", "y", "Viridis", np.zeros((4, 2)), TypeError),
        ("bad_spikes_type", MagicMock(spec=MergedData), "x", "y", "size", "color", False, "not_bool", "Scatter Plot", "x", "y", "Viridis", np.zeros((4, 2)), TypeError),
        ("bad_cmap", MagicMock(spec=MergedData), "x", "y", "size", "color", False, False, "Scatter Plot", "x", "y", "notacolor", np.zeros((4, 2)), ValueError),
    ])
    def test_plot_scatter_interactive_invalid(self, name, merged_data, x_col, y_col, size_col, color_col, bending, spikes, title, xlabel, ylabel, cmap, homography_points, expected_exc):
        # For valid merged_data, mock threshold_data to return a valid DataFrame
        if isinstance(merged_data, MagicMock):
            num_points = 50
            df = pd.DataFrame({
                x_col if isinstance(x_col, str) else "x": np.random.rand(num_points) * 20,
                y_col if isinstance(y_col, str) else "y": np.random.rand(num_points) * 20,
                size_col if isinstance(size_col, str) else "size": np.random.rand(num_points) * 10 + 1,
                "Spikes": np.random.randint(0, 2, num_points),
                "color": np.random.rand(num_points) * 5,
            })
            merged_data.threshold_data.return_value = df

        with self.assertRaises(expected_exc):
            PlottingPlotly.plot_scatter_interactive(
                merged_data=merged_data,
                x_col=x_col,
                y_col=y_col,
                homography_points=homography_points,
                size_col=size_col,
                color_col=color_col,
                bending=bending,
                spikes=spikes,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                cmap=cmap
            )

    def test_background_framing_valid_no_video(self):
        # Valid call with no video_path/index (just draws homography lines)
        merged_data = MagicMock(spec=MergedData)
        fig, ax = plt.subplots()
        homography_points = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
        # Should not raise
        PlottingPlotly.background_framing(
            merged_data=merged_data,
            ax=ax,
            homography_points=homography_points,
            video_path=None,
            index=None
        )
        plt.close(fig)

    @patch("cv2.VideoCapture")
    def test_background_framing_valid_with_video(self, mock_cv2cap):
        # Setup mocks
        merged_data = MagicMock(spec=MergedData)
        merged_data.dlc = MagicMock()
        merged_data.dlc._get_homography_matrix.return_value = np.eye(3)
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.ones((100, 100, 3), dtype=np.uint8))
        mock_cap.isOpened.return_value = True
        mock_cv2cap.return_value = mock_cap

        fig, ax = plt.subplots()
        homography_points = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
        # Should not raise
        PlottingPlotly.background_framing(
            merged_data=merged_data,
            ax=ax,
            homography_points=homography_points,
            video_path="tests/fake.mp4",
            index=0
        )
        plt.close(fig)

    @parameterized.expand([
        ("bad_homography_shape", np.zeros((3, 2)), 0, ValueError),
        ("bad_index_type", np.zeros((4, 2)), "not_an_int", TypeError),
    ])
    def test_background_framing_invalid(self, name, homography_points, index, expected_exc):
        merged_data = MagicMock(spec=MergedData)
        fig, ax = plt.subplots()
        kwargs = {
            "merged_data": merged_data,
            "ax": ax,
            "homography_points": homography_points,
            "video_path": "tests/fake.mp4" if index is not None else None,
            "index": index if index is not None else None
        }
        with self.assertRaises(expected_exc):
            PlottingPlotly.background_framing(**kwargs)
        plt.close(fig)

    @parameterized.expand([
        ("default_args", "x", "y", False, False, "KDE Plot", "x (mm)", "y (mm)", (8, 8), "vlag"),
        ("bending_true", "x", "y", True, False, "Bending KDE", "X", "Y", (10, 10), "plasma"),
        ("spikes_true", "x", "y", False, True, "Spikes KDE", "X", "Y", (12, 12), "Reds"),
    ])
    def test_plot_kde_density_valid(self, name, x_col, y_col, bending, spikes, title, xlabel, ylabel, figsize, cmap):
        # Create a dummy MergedData with required columns
        num_points = 100
        df = pd.DataFrame({
            x_col: np.random.normal(0, 1, num_points),
            y_col: np.random.normal(0, 1, num_points),
            "Spikes": np.random.randint(0, 2, num_points),
            "Bending": np.random.rand(num_points),
        })
        merged_data = MagicMock(spec=MergedData)
        merged_data.threshold_data.side_effect = lambda bending_flag, spikes_flag: df

        fig, ax = PlottingPlotly.plot_kde_density(
            merged_data=merged_data,
            x_col=x_col,
            y_col=y_col,
            homography_points=self.homography_points,
            bending=bending,
            spikes=spikes,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            cmap=cmap
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    @parameterized.expand([
        ("bad_merged_data", "not_merged", "x", "y", np.zeros((4, 2)), False, False, "KDE Plot", "x", "y", (8, 8), "vlag", TypeError),
        ("bad_homography_shape", MagicMock(spec=MergedData), "x", "y", np.zeros((3, 2)), False, False, "KDE Plot", "x", "y", (8, 8), "vlag", ValueError),
        ("bad_x_col_type", MagicMock(spec=MergedData), 123, "y", np.zeros((4, 2)), False, False, "KDE Plot", "x", "y", (8, 8), "vlag", TypeError),
        ("bad_y_col_type", MagicMock(spec=MergedData), "x", None, np.zeros((4, 2)), False, False, "KDE Plot", "x", "y", (8, 8), "vlag", TypeError),
        ("bad_bending_type", MagicMock(spec=MergedData), "x", "y", np.zeros((4, 2)), "not_bool", False, "KDE Plot", "x", "y", (8, 8), "vlag", TypeError),
        ("bad_spikes_type", MagicMock(spec=MergedData), "x", "y", np.zeros((4, 2)), False, "not_bool", "KDE Plot", "x", "y", (8, 8), "vlag", TypeError),
        ("bad_cmap", MagicMock(spec=MergedData), "x", "y", np.zeros((4, 2)), False, False, "KDE Plot", "x", "y", (8, 8), "notacolor", ValueError),
    ])
    def test_plot_kde_density_invalid(self, name, merged_data, x_col, y_col, homography_points, bending, spikes, title, xlabel, ylabel, figsize, cmap, expected_exc):
        # For valid merged_data, mock threshold_data to return a valid DataFrame
        if isinstance(merged_data, MagicMock):
            num_points = 100
            df = pd.DataFrame({
                "x": np.random.normal(0, 1, num_points),
                "y": np.random.normal(0, 1, num_points),
                "Spikes": np.random.randint(0, 2, num_points),
                "Bending": np.random.rand(num_points),
            })
            merged_data.threshold_data.side_effect = lambda bending_flag, spikes_flag: df

        with self.assertRaises(expected_exc):
            PlottingPlotly.plot_kde_density(
                merged_data=merged_data,
                x_col=x_col,
                y_col=y_col,
                homography_points=homography_points,
                bending=bending,
                spikes=spikes,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                cmap=cmap
            )

    @patch("cv2.VideoCapture")
    @patch("imageio.get_writer")
    def test_generate_scroll_over_video_valid(self, mock_writer, mock_cv2cap):
        # Mock video capture
        mock_cap = MagicMock()
        # Simulate 3 frames
        mock_cap.isOpened.side_effect = [True, True, True, False]
        mock_cap.read.side_effect = [
            (True, np.ones((100, 100, 3), dtype=np.uint8)),
            (True, np.ones((100, 100, 3), dtype=np.uint8)),
            (True, np.ones((100, 100, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_cap.get.side_effect = lambda x: {5: 30, 3: 100, 4: 100}[x]  # FPS, WIDTH, HEIGHT
        mock_cv2cap.return_value = mock_cap

        # Mock imageio writer
        mock_writer_inst = MagicMock()
        mock_writer.return_value = mock_writer_inst

        # Mock merged_data
        merged_data = MagicMock(spec=MergedData)
        merged_data.df_merged = pd.DataFrame({
            "A": np.random.rand(150),
            "B": np.random.rand(150),
            "Bending_ZScore": np.random.rand(150)
        })
        merged_data.threshold = 0.5

        # Patch open to return dummy bytes
        with patch("builtins.open", mock_open(read_data=b"video_bytes")) as m:
            result = PlottingPlotly.generate_scroll_over_video(
                merged_data=merged_data,
                columns=["A", "Bending_ZScore"],
                video_path="fake.mp4",
                title="Test Scroll",
                color_1="#1f77b4",
                color_2="#d62728"
            )
            self.assertIsInstance(result, bytes)

    @patch("cv2.VideoCapture")
    def test_generate_scroll_over_video_invalid_video(self, mock_cv2cap):
        # Simulate video cannot be opened
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2cap.return_value = mock_cap

        merged_data = MagicMock(spec=MergedData)
        merged_data.df_merged = pd.DataFrame({"A": [1, 2, 3]})
        merged_data.threshold = 0.5

        with self.assertRaises(Exception):
            PlottingPlotly.generate_scroll_over_video(
                merged_data=merged_data,
                columns=["A"],
                video_path="fake.mp4"
            )

    @parameterized.expand([
        ("bad_merged_data",
         "not_merged", ["A"], "fake.mp4", "#1f77b4", "#d62728",
         TypeError),
        ("bad_columns_type",
         MagicMock(spec=MergedData), [123], "fake.mp4", "#1f77b4", "#d62728",
         TypeError),
        ("bad_video_path",
         MagicMock(spec=MergedData), ["A"], "not_a_video.txt", "#1f77b4", "#d62728",
         ValueError),
        ("bad_color_1",
         MagicMock(spec=MergedData), ["A"], "fake.mp4", 123, "#d62728",
         TypeError),
    ])
    def test_generate_scroll_over_video_invalid_args(self, name, merged_data, columns, video_path, color_1="#1f77b4", color_2="#d62728", expected_exc=TypeError):
        if isinstance(merged_data, MagicMock):
            merged_data.df_merged = pd.DataFrame({"A": [1, 2, 3]})
            merged_data.threshold = 0.5

        with self.assertRaises(expected_exc):
            PlottingPlotly.generate_scroll_over_video(
                merged_data=merged_data,
                columns=columns,
                video_path=video_path,
                color_1=color_1,
                color_2=color_2
            )

if __name__ == "__main__":
    unittest.main()
