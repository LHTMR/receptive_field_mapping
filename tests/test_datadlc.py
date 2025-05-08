import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
import pandas as pd
from src.post_processing.datadlc import DataDLC
from unittest.mock import patch, MagicMock
from parameterized import parameterized


class TestDataDLC(unittest.TestCase):
    def setUp(self):
        # Create mock data for testing
        self.mock_h5_data = {
            ('filename', 'Top_left', 'x'): [0, 10, 20],
            ('filename', 'Top_left', 'y'): [0, 10, 20],
            ('filename', 'Top_left', 'likelihood'): [0.9, 0.8, 0.85],
            ('filename', 'Top_right', 'x'): [10, 20, 30],
            ('filename', 'Top_right', 'y'): [0, 10, 20],
            ('filename', 'Top_right', 'likelihood'): [0.9, 0.8, 0.85],
            ('filename', 'Bottom_right', 'x'): [10, 20, 30],
            ('filename', 'Bottom_right', 'y'): [10, 20, 30],
            ('filename', 'Bottom_right', 'likelihood'): [0.9, 0.8, 0.85],
            ('filename', 'Bottom_left', 'x'): [0, 10, 20],
            ('filename', 'Bottom_left', 'y'): [10, 20, 30],
            ('filename', 'Bottom_left', 'likelihood'): [0.9, 0.8, 0.85],
            ('filename', 'FR1', 'x'): [1, 2, 3],
            ('filename', 'FR1', 'y'): [4, 5, 6],
            ('filename', 'FR1', 'likelihood'): [0.9, 0.8, 0.85],
            ('filename', 'FR2', 'x'): [7, 8, 9],
            ('filename', 'FR2', 'y'): [10, 11, 12],
            ('filename', 'FR2', 'likelihood'): [0.9, 0.8, 0.85],
            ('filename', 'FG1', 'x'): [13, 14, 15],
            ('filename', 'FG1', 'y'): [16, 17, 18],
            ('filename', 'FG1', 'likelihood'): [0.9, 0.8, 0.85],
            ('filename', 'FG2', 'x'): [19, 20, 21],
            ('filename', 'FG2', 'y'): [22, 23, 24],
            ('filename', 'FG2', 'likelihood'): [0.9, 0.8, 0.85],
            ('filename', 'FB1', 'x'): [25, 26, 27],
            ('filename', 'FB1', 'y'): [28, 29, 30],
            ('filename', 'FB1', 'likelihood'): [0.9, 0.8, 0.85],
            ('filename', 'FB2', 'x'): [31, 32, 33],
            ('filename', 'FB2', 'y'): [34, 35, 36],
            ('filename', 'FB2', 'likelihood'): [0.9, 0.8, 0.85],
        }
        self.mock_h5_file = "mock_data.h5"
        self.mock_df = pd.DataFrame(self.mock_h5_data)
        self.mock_df.columns = pd.MultiIndex.from_tuples(self.mock_df.columns)
        self.mock_df.to_hdf(self.mock_h5_file, key='df', mode='w')

        # Initialize DataDLC instance
        self.data_dlc = DataDLC(self.mock_h5_file)

    def tearDown(self):
        # Clean up mock file
        if os.path.exists(self.mock_h5_file):
            os.remove(self.mock_h5_file)

    def test_init(self):
        self.assertIsInstance(self.data_dlc, DataDLC)
        self.assertEqual(self.data_dlc.df_square.shape, (3, 8))
        self.assertEqual(self.data_dlc.df_monofil.shape, (3, 12))
        self.assertEqual(self.data_dlc.df_likelihoods.shape, (3, 10))
        self.assertIsNone(self.data_dlc.df_merged)
        self.assertIsNone(self.data_dlc.df_bending_coefficients)
        self.assertIsNone(self.data_dlc.df_transformed_monofil)
        
        # Internally in the init method, the homography points are set
        # to a default via the assign_homography_points method.
        self.assertIsNotNone(self.data_dlc.homography_points)
        self.assertEqual(self.data_dlc.homography_points.shape, (4, 2))
        self.assertTrue(np.array_equal(self.data_dlc.homography_points, np.array(
            [[0, 20], [20, 20], [20, 0], [0, 0]], dtype=np.float32
            )))

    def test_get_avg_likelihoods(self):
        # Call the method
        result = self.data_dlc.get_avg_likelihoods()

        # Calculate expected values using self.mock_df
        likelihood_columns = self.mock_df.columns[self.mock_df.columns.get_level_values(2) == 'likelihood']
        likelihood_values = self.mock_df[likelihood_columns]

        overall_average = likelihood_values.mean().mean()
        bodypart_averages = likelihood_values.mean()

        # Construct the expected string
        expected_result = f"Overall average likelihood: \n{overall_average}\n" + \
                          f"Bodypart average likelihoods: \n{bodypart_averages}"

        # Assert the result matches the expected string
        self.assertEqual(result.strip(), expected_result.strip())

    @parameterized.expand([
        ("valid_inputs", 5, 15),
        ("negative_integers", -5, -15),
        ("start_greater_than_end", 20, 10),
    ])
    def test_assign_homography_points(self, name, start, end):
        points = self.data_dlc.assign_homography_points(start=start, end=end)
        self.assertTrue(np.array_equal(points,
                                       np.array([[start, end],
                                                 [end, end],
                                                 [end, start],
                                                 [start, start]], dtype=np.float32)))

    @parameterized.expand([
        ("same_inputs", 5, 5, ValueError),
        ("float_inputs", 5.5, 15.5, TypeError),
        ("string_inputs", "5", "15", TypeError),
        ("tuple_inputs", (5,), (15,), TypeError),
        ("list_inputs", [5], [15], TypeError),
        ("none_inputs", None, None, TypeError),
    ])
    def test_assign_homography_points_invalid_inputs(self, name, start, end, expected_exception):
        with self.assertRaises(expected_exception):
            self.data_dlc.assign_homography_points(start=start, end=end)

    @parameterized.expand([
        ("valid_std_threshold_int", 2, None, True, False, None, True),
        ("valid_std_threshold_float", 2.5, None, True, False, None, True),
        ("invalid_std_threshold_str", "2", None, True, False, None, False),
        ("invalid_std_threshold_list", [2], None, True, False, None, False),
        ("valid_model_name", 2, "RFR", True, False, None, True),
        ("invalid_model_name_int", 2, 123, True, False, None, False),
        ("invalid_model_name_list", 2, ["model_v1"], True, False, None, False),
        ("square_and_filament_true", 2, None, True, True, None, False),  # Both square and filament cannot be True
        ("square_and_filament_false", 2, None, False, False, None, False),  # Both square and filament cannot be False
    ])
    def test_impute_outliers(self, name, std_threshold, model_name, square, filament, mock_return, should_pass):
        with patch('src.post_processing.outlierimputer.OutlierImputer.impute_outliers', return_value=mock_return) as mock_imputer:
            if should_pass:
                # Call the method with valid inputs
                self.data_dlc.impute_outliers(std_threshold=std_threshold, square=square, filament=filament, model_name=model_name)
                if square:
                    mock_imputer.assert_called_once_with(self.data_dlc.df_square, std_threshold, model_name)
                elif filament:
                    mock_imputer.assert_called_once_with(self.data_dlc.df_monofil, std_threshold, model_name)
            else:
                # Expect an exception for invalid inputs
                with self.assertRaises((TypeError, ValueError)):
                    self.data_dlc.impute_outliers(std_threshold=std_threshold, square=square, filament=filament, model_name=model_name)

    def test_get_homography_matrix_valid(self):
        dst_points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        h_matrix = self.data_dlc._get_homography_matrix(index=0, dst_points=dst_points)
        self.assertEqual(h_matrix.shape, (3, 3))
        self.assertTrue(np.allclose(h_matrix[2], [0, 0, 1]))

    def test_get_homography_matrix_invalid_index(self):
        with self.assertRaises(IndexError):
            self.data_dlc._get_homography_matrix(index=100)

    def test_get_homography_matrix_invalid_dst_points(self):
        invalid_dst_points = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.data_dlc._get_homography_matrix(index=0, dst_points=invalid_dst_points)

    def test_apply_homography(self):
        with patch.object(self.data_dlc, '_get_homography_matrix', return_value=np.eye(3)) as mock_homography:
            transformed_points = self.data_dlc.apply_homography()
            self.assertEqual(transformed_points.shape, (3, 12))  # 3 rows, 12 columns (6 points * 2 coordinates)
            mock_homography.assert_called()

    def test_get_bending_coefficients(self):
        bending_coefficients = self.data_dlc.get_bending_coefficients()
        self.assertEqual(len(bending_coefficients), 3)  # 3 rows of data
        self.assertTrue(all(isinstance(coeff, float) for coeff in bending_coefficients))


if __name__ == "__main__":
    unittest.main()