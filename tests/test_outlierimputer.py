import unittest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock
from src.post_processing.outlierimputer import OutlierImputer
from sklearn.ensemble import RandomForestRegressor


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
        # Test initialization
        self.assertEqual(self.imputer.log_file, "test_log.json")
        self.assertEqual(self.imputer.best_models, {})

    def test_init_invalid(self):
        # Test initialization
        self.assertEqual(self.imputer.log_file, "test_log.json")
        self.assertEqual(self.imputer.best_models, {})

    def test_transform_to_derivative(self):
        # Test the transform_to_derivative method
        derivative_df = self.imputer.transform_to_derivative(self.mock_df)
        expected_df = self.mock_df.diff().abs()
        expected_df.iloc[0, :] = 0
        pd.testing.assert_frame_equal(derivative_df, expected_df)

    def test_transform_to_derivative_invalid_input(self):
        # Test invalid input for transform_to_derivative
        with self.assertRaises(ValueError):
            self.imputer.transform_to_derivative(pd.DataFrame({"x1": [1, 2, 3]}))  # Odd number of columns

    def test_detect_outliers_velocity(self):
        # Test the detect_outliers_velocity method
        outlier_df = self.imputer.detect_outliers_velocity(self.mock_df, threshold=2.0)
        self.assertTrue(np.isnan(outlier_df.iloc[4, 0]))  # Outlier in x1
        self.assertTrue(np.isnan(outlier_df.iloc[4, 1]))  # Outlier in y1

    def test_detect_outliers_velocity_invalid_input(self):
        # Test invalid input for detect_outliers_velocity
        with self.assertRaises(ValueError):
            self.imputer.detect_outliers_velocity(self.mock_df, threshold=-1)  # Negative threshold

    @patch("src.post_processing.outlierimputer.GridSearchCV")
    def test_grid_search_models_per_col(self, mock_grid_search):
        # Mock GridSearchCV
        mock_grid_search.return_value.fit.return_value = None
        mock_grid_search.return_value.best_score_ = -10
        mock_grid_search.return_value.best_estimator_ = RandomForestRegressor()

        # Test the _grid_search_models_per_col method
        self.imputer._grid_search_models_per_col(self.mock_df)
        self.assertIn("x1", self.imputer.best_models)
        self.assertIsInstance(self.imputer.best_models["x1"], RandomForestRegressor)

    def test_iterative_imputation(self):
        # Test the iterative_imputation method
        self.imputer.best_models = {"x1": RandomForestRegressor()}
        imputed_df = self.imputer.iterative_imputation(self.mock_df)
        self.assertEqual(imputed_df.shape, self.mock_df.shape)

    def test_iterative_imputation_invalid_input(self):
        # Test invalid input for iterative_imputation
        with self.assertRaises(ValueError):
            self.imputer.iterative_imputation(self.mock_df, max_iter=-1)  # Negative max_iter

    @patch("src.post_processing.outlierimputer.OutlierImputer._grid_search_models_per_col")
    @patch("src.post_processing.outlierimputer.OutlierImputer.iterative_imputation")
    def test_impute_outliers(self, mock_iterative_imputation, mock_grid_search):
        # Mock methods
        mock_iterative_imputation.return_value = self.mock_df
        mock_grid_search.return_value = None

        # Test the impute_outliers method
        imputed_df = self.imputer.impute_outliers(self.mock_df, std_threshold=2.0)
        self.assertEqual(imputed_df.shape, self.mock_df.shape)

        # Check if the log file is created
        with open("test_log.json", "r") as f:
            log_data = json.load(f)
        self.assertIn("x1", log_data)

    def test_impute_outliers_invalid_input(self):
        # Test invalid input for impute_outliers
        with self.assertRaises(ValueError):
            self.imputer.impute_outliers(self.mock_df, std_threshold=-1)  # Negative threshold


if __name__ == "__main__":
    unittest.main()