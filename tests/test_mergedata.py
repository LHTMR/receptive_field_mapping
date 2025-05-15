import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.post_processing.mergeddata import MergedData
from src.post_processing.datadlc import DataDLC
from src.post_processing.dataneuron import DataNeuron
from parameterized import parameterized


class TestMergedData(unittest.TestCase):
    def setUp(self):
        # Mock DataDLC
        self.mock_dlc = MagicMock(spec=DataDLC)
        self.mock_dlc._merge_data.return_value = pd.DataFrame({
            'Bending_Coefficient': [0.1, 0.2, 0.3],
            'Other_Column': [1, 2, 3]
        })

        # Mock DataNeuron
        self.mock_neuron = MagicMock(spec=DataNeuron)
        self.mock_neuron.downsampled_df = pd.DataFrame({
            'Spikes': [0, 1, 0],
            'IFF': [0, 5, 0]
        })

        # Initialize MergedData
        self.merged_data = MergedData(
            dlc=self.mock_dlc,
            neuron=self.mock_neuron,
            max_gap_fill=2,
            threshold=0.15
        )

    def test_init(self):
        # Test initialization
        self.assertIsInstance(self.merged_data, MergedData)
        self.assertIsInstance(self.merged_data.dlc, DataDLC)
        self.assertIsInstance(self.merged_data.neuron, DataNeuron)
        self.assertEqual(self.merged_data.max_gap_fill, 2)
        self.assertEqual(self.merged_data.threshold, 0.15)
        self.assertIsNotNone(self.merged_data.df_merged)
        self.assertIsNotNone(self.merged_data.df_merged_cleaned)

    @parameterized.expand([
        ("invalid_dlc_type",
         "not_a_dlc_object", MagicMock(spec=DataNeuron), 10, 0.1, TypeError),
        ("invalid_neuron_type",
         MagicMock(spec=DataDLC), "not_a_neuron_object", 10, 0.1, TypeError),
        ("invalid_max_gap_fill_type",
         MagicMock(spec=DataDLC), MagicMock(spec=DataNeuron), "not_an_int", 0.1, TypeError),
        ("negative_max_gap_fill",
         MagicMock(spec=DataDLC), MagicMock(spec=DataNeuron), -5, 0.1, ValueError),
        ("zero_max_gap_fill",
         MagicMock(spec=DataDLC), MagicMock(spec=DataNeuron), 0, 0.1, ValueError),
        ("invalid_threshold_type",
         MagicMock(spec=DataDLC), MagicMock(spec=DataNeuron), 10, "not_a_float", TypeError),
        ("negative_threshold",
         MagicMock(spec=DataDLC), MagicMock(spec=DataNeuron), 10, -0.1, ValueError),
        ("above_one_threshold",
         MagicMock(spec=DataDLC), MagicMock(spec=DataNeuron), 10, 1.5, ValueError)
    ])
    def test_invalid_init(self, name, dlc, neuron, max_gap_fill, threshold, expected_exception):
        with self.assertRaises(expected_exception):
            MergedData(dlc=dlc, neuron=neuron,
                       max_gap_fill=max_gap_fill, threshold=threshold)

    def test_merge(self):
        # Test the _merge method
        merged_df = self.merged_data._merge()
        self.assertIsInstance(merged_df, pd.DataFrame)
        self.assertIn('Bending_Coefficient', merged_df.columns)
        self.assertIn('Spikes', merged_df.columns)
        self.assertIn('IFF', merged_df.columns)

    def test_clean(self):
        # Test the _clean method
        cleaned_df = self.merged_data._clean()
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertTrue(
            all((cleaned_df['Bending_Binary'] == 1) |
                (cleaned_df['Spikes'] != 0))
        )

    def test_threshold_data(self):
        # Test the threshold_data method
        filtered_df = self.merged_data.threshold_data(bending=True,
                                                      spikes=False)
        self.assertIsInstance(filtered_df, pd.DataFrame)
        self.assertTrue(all(filtered_df['Bending_Binary'] == 1))

        filtered_df = self.merged_data.threshold_data(bending=False,
                                                      spikes=True)
        self.assertTrue(all(filtered_df['Spikes'] != 0))

        unfiltered_df = self.merged_data.threshold_data(bending=False,
                                                        spikes=False)
        self.assertTrue(unfiltered_df.equals(self.merged_data.df_merged))

    @parameterized.expand([
        ("invalid_bending_type", "not_a_bool", True, TypeError),
        ("invalid_spikes_type", True, "not_a_bool", TypeError),
    ])
    def test_threshold_data_invalid(self, name, bending, spikes, expected_exception):
        # Test the threshold_data method with invalid inputs
        with self.assertRaises(expected_exception):
            self.merged_data.threshold_data(bending=bending, spikes=spikes)

    def test_plotting_split(self):
        # Test the plotting_split method
        high_bend_w_neuron, high_bend_wo_neuron, low_bend_w_neuron = \
            self.merged_data.plotting_split()

        self.assertIsInstance(high_bend_w_neuron, pd.DataFrame)
        self.assertIsInstance(high_bend_wo_neuron, pd.DataFrame)
        self.assertIsInstance(low_bend_w_neuron, pd.DataFrame)

        self.assertTrue(all(high_bend_w_neuron['Bending_Binary'] == 1))
        self.assertTrue(all(high_bend_w_neuron['Spikes'] == 1))
        self.assertTrue(all(high_bend_wo_neuron['Bending_Binary'] == 1))
        self.assertTrue(all(high_bend_wo_neuron['Spikes'] == 0))
        self.assertTrue(all(low_bend_w_neuron['Bending_Binary'] == 0))
        self.assertTrue(all(low_bend_w_neuron['Spikes'] == 1))

    @patch('src.post_processing.mergeddata.MergedData._save_data')
    def test_save_full_data_csv(self, mock_save_data):
        # Test the save_full_data method
        self.merged_data.save_full_data(
            path="mock_path.csv", file_format="csv")
        mock_save_data.assert_called_once_with(
            self.merged_data.df_merged, "mock_path.csv", "csv")

    @patch('src.post_processing.mergeddata.MergedData._save_data')
    def test_save_full_data_xlsx(self, mock_save_data):
        # Test the save_full_data method
        self.merged_data.save_full_data(
            path="mock_path.xlsx", file_format="xlsx")
        mock_save_data.assert_called_once_with(
            self.merged_data.df_merged, "mock_path.xlsx", "xlsx")

    @patch('src.post_processing.mergeddata.MergedData._save_data')
    def test_save_cleaned_data_csv(self, mock_save_data):
        # Test the save_cleaned_data method
        self.merged_data.save_cleaned_data(
            path="mock_path.csv", file_format="csv")
        mock_save_data.assert_called_once_with(
            self.merged_data.df_merged_cleaned, "mock_path.csv", "csv")

    @patch('src.post_processing.mergeddata.MergedData._save_data')
    def test_save_cleaned_data_xlsx(self, mock_save_data):
        # Test the save_cleaned_data method
        self.merged_data.save_cleaned_data(
            path="mock_path.xlsx", file_format="xlsx")
        mock_save_data.assert_called_once_with(
            self.merged_data.df_merged_cleaned, "mock_path.xlsx", "xlsx")

    @parameterized.expand([
        ("invalid_df_type", "not_a_dataframe", "mock_path.csv", "csv", TypeError),
        ("invalid_path_type", pd.DataFrame(), 123, "csv", TypeError),
        ("invalid_file_format_type", pd.DataFrame(),
         "mock_path.csv", 123, TypeError),
        ("unsupported_file_format", pd.DataFrame(),
         "mock_path.txt", "txt", ValueError),
    ])
    def test_save_data_invalid_inputs(self, name, df, path, file_format, expected_exception):
        with self.assertRaises(expected_exception):
            self.merged_data._save_data(df, path, file_format)


if __name__ == "__main__":
    unittest.main()
