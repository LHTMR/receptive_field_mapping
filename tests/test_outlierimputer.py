import unittest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock
from src.post_processing.outlierimputer import OutlierImputer
from sklearn.ensemble import RandomForestRegressor
from parameterized import parameterized


class TestOutlierImputer(unittest.TestCase):
    def setUp(self):
        # Create a mock DataFrame for testing
        self.mock_df = pd.DataFrame({
            "x1": [1, 2, 3, 4, 100],
            "y1": [2, 4, 6, 8, 200],
            "x2": [1, 2, 3, 4, 5],
            "y2": [5, 10, 15, 20, 25]
        })

        # Initialize OutlierImputer
        self.imputer = OutlierImputer(log_file="test_log.json")

    def tearDown(self):
        # Clean up the log file
        try:
            import os
            os.remove("test_log.json")
        except FileNotFoundError:
            pass

    def test_init(self):
        # Test default initialization
        self.assertEqual(self.imputer.log_file, "test_log.json")
        self.assertEqual(self.imputer.best_models, {})

    @parameterized.expand([
        ("invalid_log_file_type", 123, TypeError),  # Non-string log_file
        ("invalid_log_file_extension", "test_log.txt", ValueError),  # Non-JSON file extension
        ("empty_log_file", "", ValueError),  # Empty string for log_file
        ("none_log_file", None, TypeError),  # None as log_file
    ])
    def test_init_invalid(self, name, log_file, expected_exception):
        with self.assertRaises(expected_exception):
            OutlierImputer(log_file=log_file)

    def test_transform_to_derivative(self):
        # Test the transform_to_derivative method
        derivative_df = self.imputer.transform_to_derivative(self.mock_df)
        expected_df = self.mock_df.diff().abs()
        expected_df.iloc[0, :] = 0
        pd.testing.assert_frame_equal(derivative_df, expected_df)

    @parameterized.expand([
        ("non_dataframe_input", [1, 2, 3], TypeError),  # Non-DataFrame input
        ("empty_dataframe", pd.DataFrame(), ValueError),  # Empty DataFrame
        ("odd_columns_dataframe", pd.DataFrame({"x1": [1, 2, 3]}), ValueError),  # Odd number of columns
        ("non_numeric_dataframe", pd.DataFrame({"x1": ["a", "b", "c"], "y1": ["d", "e", "f"]}), ValueError),  # Non-numeric DataFrame
    ])
    def test_transform_to_derivative_invalid_input(self, name, input_data, expected_exception):
        # Test invalid input for transform_to_derivative
        with self.assertRaises(expected_exception):
            self.imputer.transform_to_derivative(input_data)

    @parameterized.expand([
        ("float_threshold", 0.5),
        ("full_float_threshold", 1.0),
        ("integer_threshold", 1),
    ])
    def test_detect_outliers_velocity(self, name, threshold):
        # Test the detect_outliers_velocity method
        outlier_df = self.imputer.detect_outliers_velocity(self.mock_df, threshold=threshold)
        self.assertTrue(np.isnan(outlier_df.iloc[4, 0]))  # Outlier in x1
        self.assertTrue(np.isnan(outlier_df.iloc[4, 1]))  # Outlier in y1

    @parameterized.expand([
        ("negative_threshold", -1.0, ValueError),  # Negative threshold
        ("zero_threshold", 0.0, ValueError),      # Zero threshold
        ("non_numeric_threshold", "invalid", TypeError),  # Non-numeric threshold
    ])
    def test_detect_outliers_velocity_invalid_thresholds(self, name, threshold, expected_exception):
        # Test the detect_outliers_velocity method with invalid thresholds
        with self.assertRaises(expected_exception):
            self.imputer.detect_outliers_velocity(self.mock_df, threshold=threshold)

    @patch("src.post_processing.outlierimputer.GridSearchCV")
    def test__grid_search_models_per_col_default(self, mock_grid_search):
        # Mock the GridSearchCV instance
        mock_grid_instance = MagicMock()
        mock_grid_instance.fit.return_value = None
        mock_grid_instance.best_score_ = -10
        mock_model = MagicMock()
        mock_grid_instance.best_estimator_ = mock_model
        mock_grid_search.return_value = mock_grid_instance

        # Remove outliers (optional if you want to keep the flow consistent)
        df_clean = self.imputer.detect_outliers_velocity(self.mock_df, threshold=2.0)
        self.imputer._grid_search_models_per_col(df_clean)

        self.assertTrue(mock_grid_search.called)
        # Check that each column got a best model set
        for col in ["x1", "y1", "x2", "y2"]:
            self.assertIn(col, self.imputer.best_models)
            # NaN present, so model should be mocked one
            self.assertEqual(self.imputer.best_models[col], mock_model)

    @patch("src.post_processing.outlierimputer.GridSearchCV")
    def test__grid_search_models_per_col_valid_model_names(self, mock_grid_search):
        mock_model = MagicMock()
        mock_grid_search.return_value = mock_model
        mock_model.fit.return_value = None
        mock_model.best_score_ = -5
        mock_model.best_estimator_ = "MockedBestModel"

        # Valid model names
        valid_models = [
            "RFR", "HGBR", "KNR", "XGB", "SVR", "BR", "Poly"
        ]

        # Fetch cleaned DataFrame
        cleaned_df = self.imputer.detect_outliers_velocity(self.mock_df, threshold=2.0)

        for model_name in valid_models:
            with self.subTest(model=model_name):
                self.imputer._grid_search_models_per_col(cleaned_df, model_name=model_name)
                self.assertTrue(all(v == "MockedBestModel"
                                    for v in self.imputer.best_models.values()))

    def test__grid_search_models_per_col_invalid_model_name_raises(self):
        # Fetch cleaned DataFrame
        cleaned_df = self.imputer.detect_outliers_velocity(self.mock_df, threshold=2.0)
        with self.assertRaises(ValueError) as context:
            self.imputer._grid_search_models_per_col(self.mock_df, model_name="NotARealModel")

        self.assertIn("Invalid model name 'NotARealModel'", str(context.exception))
        self.assertIn("Available models:", str(context.exception))

    @patch("src.post_processing.outlierimputer.IterativeImputer")
    def test_iterative_imputation_valid(self, mock_iter_imputer):
        # Setup mock return value
        mock_instance = MagicMock()
        mock_instance.fit_transform.return_value = np.ones((5, 4))
        mock_iter_imputer.return_value = mock_instance

        result_df = self.imputer.iterative_imputation(self.mock_df)

        # Assertions
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(result_df.shape, self.mock_df.shape)
        np.testing.assert_array_equal(result_df.columns, self.mock_df.columns)
        mock_iter_imputer.assert_called_once()
        mock_instance.fit_transform.assert_called_once()

    @parameterized.expand([
        ("non_dataframe_input", [1, 2, 3], TypeError),
        ("empty_dataframe", pd.DataFrame(), ValueError),
        ("non_numeric_dataframe", pd.DataFrame({"x1": ["a", "b", "c"], "y1": ["d", "e", "f"]}), ValueError),
        ("negative_max_iter", pd.DataFrame({"x1": [1, 2], "x2": [3, 4]}), ValueError, -10),
        ("non_integer_max_iter", pd.DataFrame({"x1": [1, 2], "x2": [3, 4]}), TypeError, "ten"),
    ])
    def test_iterative_imputation_invalid_inputs(self, name, df_input, expected_exception, max_iter=10):
        with self.assertRaises(expected_exception):
            self.imputer.iterative_imputation(df_input, max_iter=max_iter)

    @patch("src.post_processing.outlierimputer.OutlierImputer._grid_search_models_per_col")
    @patch("src.post_processing.outlierimputer.OutlierImputer.iterative_imputation")
    def test_impute_outliers(self, mock_iterative_imputation, mock_grid_search):
        # Mock methods
        mock_iterative_imputation.return_value = self.mock_df.copy()
        mock_grid_search.return_value = None
        # Create some info in imputer.best_models to get written in the log
        self.imputer.best_models = {
            "x1": RandomForestRegressor(),
            "y1": RandomForestRegressor(),
            "x2": RandomForestRegressor(),
            "y2": RandomForestRegressor()
        }

        # Run method
        imputed_df = self.imputer.impute_outliers(self.mock_df.copy(), std_threshold=2.0)

        # Assertions
        self.assertIsInstance(imputed_df, pd.DataFrame)
        self.assertEqual(imputed_df.shape, self.mock_df.shape)
        mock_grid_search.assert_called_once()
        mock_iterative_imputation.assert_called_once()

        # Check JSON output
        with open("test_log.json", "r") as f:
            log_data = json.load(f)
        self.assertIn("x1", log_data)
        self.assertIn("x2", log_data)

    @parameterized.expand([
        ("non_dataframe_input", [1, 2, 3], 2.0, TypeError),
        ("negative_threshold", pd.DataFrame({"x1": [1, 2], "x2": [3, 4]}), -1, ValueError),
        ("zero_threshold", pd.DataFrame({"x1": [1, 2], "x2": [3, 4]}), 0, ValueError),
        ("non_numeric_threshold", pd.DataFrame({"x1": [1, 2], "x2": [3, 4]}), "high", TypeError),
    ])
    def test_impute_outliers_invalid_inputs(self, name, df_input, std_threshold, expected_exception):
        with self.assertRaises(expected_exception):
            self.imputer.impute_outliers(df_input, std_threshold=std_threshold)


if __name__ == "__main__":
    unittest.main()