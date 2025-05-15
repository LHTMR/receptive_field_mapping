import os
import unittest
import numpy as np
import pandas as pd
from src.post_processing.dataneuron import DataNeuron
from unittest.mock import patch
from parameterized import parameterized


class TestDataNeuron(unittest.TestCase):
    def setUp(self):
        # Create mock data for testing
        self.mock_xlsx_file = "mock_data.xlsx"
        self.mock_data = {
            "Time": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "Spikes": [0, 1, 0, 1, 0, 1],
            "IFF": [0, 0, 0, 5, 5, 5]
        }
        self.mock_df = pd.DataFrame(self.mock_data)
        self.mock_df.to_excel(self.mock_xlsx_file, index=False)

        # Initialize DataNeuron instance
        self.data_neuron = DataNeuron(self.mock_xlsx_file, original_freq=10)

    def tearDown(self):
        # Clean up mock file
        if os.path.exists(self.mock_xlsx_file):
            os.remove(self.mock_xlsx_file)

    def test_init(self):
        # Test initialization
        self.assertIsInstance(self.data_neuron, DataNeuron)
        self.assertEqual(self.data_neuron.original_freq, 10)
        self.assertIsInstance(self.data_neuron.df, pd.DataFrame)
        self.assertIsNone(self.data_neuron.downsampled_df)

    def test_validate_required_columns(self):
        # Test that required columns are validated during initialization
        with patch("src.components.validation.Validation.validate_dataframe") as mock_validate:
            DataNeuron(self.mock_xlsx_file, original_freq=10)
            mock_validate.assert_called()

    def test_invalid_file_path(self):
        # Test invalid file path
        with self.assertRaises(FileNotFoundError):
            DataNeuron("invalid_path.xlsx", original_freq=10)

    @parameterized.expand([
        ("negative_freq", -10, ValueError),
        ("zero_freq", 0, ValueError),
        ("float_freq", 2.5, TypeError),
    ])
    def test_invalid_original_freq(self, name, original_freq, expected_exception):
        with self.assertRaises(expected_exception):
            DataNeuron(self.mock_xlsx_file, original_freq=original_freq)

    def test_calculate_iff(self):
        # Remove the IFF column and recalculate it
        self.data_neuron.df.drop(columns=["IFF"], inplace=True)
        self.data_neuron.calculate_iff()

        # Expected IFF values
        expected_iff = [0, 0, 0, 5, 5, 5]
        np.testing.assert_array_almost_equal(self.data_neuron.df["IFF"].values, expected_iff)

    def test_get_frequency(self):
        # Test the frequency calculation
        freq = self.data_neuron._get_frequency()
        self.assertEqual(freq, 10)

    def test_fill_samples(self):
        # Modify the DataFrame to simulate missing samples
        self.data_neuron.df = self.data_neuron.df.iloc[[1, 3, 5]] # Keep spike rows
        self.data_neuron.fill_samples()

        # Check that the DataFrame has been filled correctly
        np.testing.assert_array_almost_equal(self.data_neuron.df["Time"].values,
                                             self.mock_data["Time"])
        np.testing.assert_array_almost_equal(self.data_neuron.df["Spikes"].values,
                                             self.mock_data["Spikes"])
        np.testing.assert_array_almost_equal(self.data_neuron.df["IFF"].values,
                                             self.mock_data["IFF"])

    @parameterized.expand([
        ("valid_target_freq", 5, 3),  # Valid target frequency
        ("same_freq", 10, 6),  # Same as original frequency
        ("edge_case_freq", 1, 1)  # Edge case frequency
    ])
    def test_downsample_valid_inputs(self, name, target_freq, expected_length):
        downsampled_df = self.data_neuron.downsample(target_freq=target_freq)

        # Check the downsampled DataFrame
        self.assertEqual(len(downsampled_df), expected_length)
        self.assertIn("IFF", downsampled_df.columns)
        self.assertIn("Spikes", downsampled_df.columns)

    @parameterized.expand([
        ("negative_target_freq", -5, ValueError),
        ("zero_target_freq", 0, ValueError),
        ("non_integer_target_freq", 2.5, TypeError),
        ("larger_than_original_freq", 20, ValueError),
    ])
    def test_downsample_invalid_inputs(self, name, target_freq, expected_exception):
        with self.assertRaises(expected_exception):
            self.data_neuron.downsample(target_freq=target_freq)

    def test_fill_downsample_length(self):
        # Downsample the data first
        self.data_neuron.downsample(target_freq=5)
        last_iff = self.data_neuron.df["IFF"].iloc[-1]
        len_df = len(self.data_neuron.downsampled_df)

        # Fill the downsampled data to a target length
        filled_df = self.data_neuron._fill_downsample_length(target_length=10)

        # Check the filled DataFrame
        self.assertEqual(len(filled_df), 10)
        # Check that the new rows are filled with the last IFF and Spikes as 0
        self.assertTrue(np.all(filled_df["IFF"].iloc[len_df:] == last_iff))
        self.assertTrue(np.all(filled_df["Spikes"].iloc[len_df:] == 0))


if __name__ == "__main__":
    unittest.main()
