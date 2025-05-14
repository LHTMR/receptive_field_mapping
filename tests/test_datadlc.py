import os
import unittest
import numpy as np
import pandas as pd
from src.post_processing.datadlc import DataDLC
from unittest.mock import patch
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

        # Expected result
        overall_average = self.data_dlc.df_likelihoods.mean().mean()
        bodypart_average = self.data_dlc.df_likelihoods.mean()
        bodypart_average_str = bodypart_average.to_string(index=True, header=False)
        expected_result = f"Overall average likelihood: \n{overall_average}\n" + \
                        f"Bodypart average likelihoods: \n{bodypart_average_str}"

        # Assert the result matches the expected string
        self.assertEqual(result.strip(), expected_result.strip())

    def test_assign_homography_points_defaults(self):
        # Call the method without passing start and end
        points = self.data_dlc.assign_homography_points()
        
        # Verify that the default homography points are used
        expected_points = np.array([[0, 20],
                                    [20, 20],
                                    [20, 0],
                                    [0, 0]], dtype=np.float32)
        self.assertTrue(np.array_equal(points, expected_points))

    @parameterized.expand([
        ("valid_inputs", 5, 15),
        ("zero_start", 0, 10),
        ("start_greater_than_end", 20, 10)
    ])
    def test_assign_homography_points(self,
                                      name,
                                      start, end):
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
        ("negative_integers", -5, -15, ValueError),
    ])
    def test_assign_homography_points_invalid_inputs(self,
                                                     name,
                                                     start, end,
                                                     expected_exception):
        with self.assertRaises(expected_exception):
            self.data_dlc.assign_homography_points(start=start, end=end)

    # I realize now I should've split the method being tested into two
    # separate methods, one for square and one for filament.
    @parameterized.expand([
        # square
        ("valid_std_threshold_int_square", 2, None, True, False, None),
        ("valid_std_threshold_float_square", 2.5, None, True, False, None),
        ("valid_model_name_square", 2, "RFR", True, False, None),
        # filament
        ("valid_std_threshold_int_filament", 2, None, False, True, None),
        ("valid_std_threshold_float_filament", 2.5, None, False, True, None),
        ("valid_model_name_filament", 2, "RFR", False, True, None),
    ])
    def test_impute_outliers_valid_inputs(self,
                                          name,
                                          std_threshold,
                                          model_name,
                                          square,
                                          filament,
                                          mock_return):
        with patch('src.post_processing.outlierimputer.OutlierImputer.impute_outliers',
                   return_value=mock_return) as mock_imputer:
            # Fetch df's before the imputation (since it removes outliers)
            # and we want to compare the original df's
            df_square = self.data_dlc.df_square.copy()
            df_monofil = self.data_dlc.df_monofil.copy()

            # Call the method with valid inputs
            self.data_dlc.impute_outliers(std_threshold=std_threshold,
                                          square=square,
                                          filament=filament,
                                          model_name=model_name)
            
            # Verify the mock was called with the correct arguments
            called_args, _ = mock_imputer.call_args

            if square:
                pd.testing.assert_frame_equal(called_args[0], df_square)
                self.assertEqual(called_args[1], std_threshold)
                self.assertEqual(called_args[2], model_name)
            elif filament:
                pd.testing.assert_frame_equal(called_args[0], df_monofil)
                self.assertEqual(called_args[1], std_threshold)
                self.assertEqual(called_args[2], model_name)

    @parameterized.expand([
        ("invalid_std_threshold_str", "2", None, True, False, None, TypeError),
        ("invalid_std_threshold_list", [2], None, True, False, None, TypeError),
        ("invalid_model_name_int", 2, 123, True, False, None, TypeError),
        ("invalid_model_name_list", 2, ["model_v1"], True, False, None, TypeError),
        # Both square and filament cannot be True
        ("square_and_filament_true", 2, None, True, True, None, ValueError),
        # Both square and filament cannot be False
        ("square_and_filament_false", 2, None, False, False, None, ValueError),
    ])
    def test_impute_outliers_invalid_inputs(self,
                                            name,
                                            std_threshold,
                                            model_name,
                                            square,
                                            filament,
                                            mock_return,
                                            expected_exception):
        with patch('src.post_processing.outlierimputer.OutlierImputer.impute_outliers',
                   return_value=mock_return):
            # Expect an exception for invalid inputs
            with self.assertRaises(expected_exception):
                self.data_dlc.impute_outliers(std_threshold=std_threshold,
                                              square=square,
                                              filament=filament,
                                              model_name=model_name)

    def test_get_bending_coefficients(self):
        bending_coefficients = self.data_dlc.get_bending_coefficients()
        self.assertIsInstance(bending_coefficients, pd.Series)
        self.assertEqual(len(bending_coefficients), 3)  # 3 rows of data
        self.assertTrue(all(isinstance(coeff, float) for coeff in bending_coefficients))
        self.assertEqual(self.data_dlc.df_bending_coefficients.shape, (3,))

    @parameterized.expand([
        ("default_dst_points", 0, None, np.eye(3)),  # Default case with no dst_points
        ("custom_dst_points", 0, np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32), np.eye(3)),  # Custom dst_points
    ])
    def test_get_homography_matrix_valid_cases(self, name, index, dst_points, expected_matrix):
        h_matrix = self.data_dlc._get_homography_matrix(index=index, dst_points=dst_points)
        self.assertEqual(h_matrix.shape, (3, 3))
        self.assertTrue(np.allclose(h_matrix[2], [0, 0, 1]))

    @parameterized.expand([
        # Invalid index cases
        ("out_of_bounds_index", 100, None, IndexError),
        ("float_index", 1.5, None, TypeError),
        ("none_index", None, None, TypeError),
        ("string_index", "invalid", None, TypeError),
        # Invalid dst_points cases
        ("too_few_points", 0, np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32), ValueError),
        ("too_many_points", 0, np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], dtype=np.float32), ValueError),
        ("not_2d_array", 0, np.array([0, 10, 20, 30], dtype=np.float32), ValueError),
        ("empty_array", 0, np.array([], dtype=np.float32).reshape(0, 2), ValueError),
    ])
    def test_get_homography_matrix_invalid_inputs(self, name, index, dst_points, expected_exception):
        with self.assertRaises(expected_exception):
            self.data_dlc._get_homography_matrix(index=index, dst_points=dst_points)

    def test_apply_homography(self):
        with patch.object(self.data_dlc, '_get_homography_matrix', return_value=np.eye(3)) as mock_homography:
            transformed_points = self.data_dlc.apply_homography()
            self.assertEqual(transformed_points.shape, (3, 12))  # 3 rows, 12 columns (6 points * 2 coordinates)
            mock_homography.assert_called()
            self.assertEqual(mock_homography.call_count, 3)
            self.assertEqual(self.data_dlc.df_transformed_monofil.shape, (3, 12))

    def test_merge_data(self):
        # Create mock DataFrames
        self.data_dlc.df_square = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        self.data_dlc.df_monofil = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
        self.data_dlc.df_transformed_monofil = pd.DataFrame({'E': [9, 10], 'F': [11, 12]})
        self.data_dlc.df_bending_coefficients = pd.DataFrame({'G': [13, 14]})
        merged_df = self.data_dlc._merge_data()

        # Expected result
        expected_df = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4],
            'C': [5, 6],
            'D': [7, 8],
            'E': [9, 10],
            'F': [11, 12],
            'G': [13, 14],
        })

        # Assert the merged DataFrame matches the expected result
        pd.testing.assert_frame_equal(merged_df, expected_df)
        self.assertEqual(self.data_dlc.df_merged.shape, (2, 7))

if __name__ == "__main__":
    unittest.main()