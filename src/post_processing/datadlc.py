import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import cv2
from outlierimputer import OutlierImputer
from validation import Validation as Val

class DataDLC:
    homography_points = np.array([[0, 20],
                                  [20, 20],
                                  [20, 0],
                                  [0, 0]], dtype=np.float32)

    # This is the initial class, reading the h5 file for DLC data, then
    # getting bending coefficient, homography, plotting, and saving the homography video
    def __init__(self, h5_file) -> None:
        df = pd.read_hdf(h5_file)
        # Making slots for future dataframes/attributes
        self.df_square = None
        self.df_monofil = None
        self.df_likelihoods = None
        self.df_merged = None
        self.df_bending_coefficients = None
        self.df_transformed_monofil = None

        try:  # Extract desired parts from the h5 file
            df.columns = [
                f"{bodypart}_{coord}" for bodypart, coord in zip(
                    df.columns.get_level_values(1),
                    df.columns.get_level_values(2)
                )
            ]

            self.df_monofil = df.loc[:, df.columns.str.startswith(('FR', 'FG', 'FB')) &
                                     ~df.columns.str.endswith('likelihood')]

            self.df_square = df.loc[:, df.columns.str.startswith(
                ('Top_left', 'Top_right', 'Bottom_left', 'Bottom_right')) &
                ~df.columns.str.endswith('likelihood')]

            self.df_likelihoods = df.loc[:, df.columns.str.endswith('likelihood')]
        except AttributeError as e:
            raise AttributeError(
                f"Invalid h5 file. Please check the file format.\n{e}"
            )

    def get_likelihoods(self) -> str:

        overall_average = self.df_likelihoods.mean().mean()
        bodypart_average = self.df_likelihoods.mean()
        return f"Overall average likelihood: \n{overall_average}\n" + \
               f"Bodypart average likelihoods: \n{bodypart_average}"

    def impute_outliers(self,
                        std_threshold: int|float = 2,
                        square: bool = True,
                        filament: bool = False,) -> None:
        Val.validate_type(std_threshold, (int, float), "STD Threshold")
        Val.validate_positive(std_threshold, "STD Threshold", zero_allowed=True)
        Val.validate_type(square, bool, "Square")
        Val.validate_type(filament, bool, "Filament")

        outlier_imputer = OutlierImputer()

        # Impute outliers for the square and monofilament points
        if square:
            self.df_square = outlier_imputer.impute_outliers(self.df_square,
                                                             std_threshold)
        if filament:
            self.df_monofil = outlier_imputer.impute_outliers(self.df_monofil,
                                                              std_threshold)

    def get_bending_coefficients(self) -> pd.Series:

        # Initialize a list to store bending coefficients
        bending_coefficients = []

        # Process each row of df_monofil
        for index, row in self.df_monofil.iterrows():
            # Step 1: Extract x and y coordinates for the current row
            x_coords = row.filter(like="_x").values
            y_coords = row.filter(like="_y").values

            # Step 2: Center coordinates around their mean
            x_centered = x_coords - np.mean(x_coords)
            y_centered = y_coords - np.mean(y_coords)

            # Step 3: Fit a polynomial (degree 2)
            degree = 2
            coefficients = np.polyfit(x_centered, y_centered, degree)
            bending_coeff = coefficients[0]  # Coefficient of the quadratic term

            # Store the bending coefficient for this frame
            bending_coefficients.append(abs(bending_coeff))

        # Step 4: Add the bending coefficients as a new series attribute
        self.df_bending_coefficients = pd.Series(bending_coefficients,
                                                 name='Bending_Coefficient')
        return self.df_bending_coefficients

    def apply_homography(self) -> pd.DataFrame:

        transformed_monofil_points = []

        for i in range(len(self.df_square)):
            # Find the homography matrix
            h_matrix = self._get_homography_matrix(i)

            monofil_points = np.array([
                [self.df_monofil.iloc[i]['FR1_x'], self.df_monofil.iloc[i]['FR1_y']],
                [self.df_monofil.iloc[i]['FR2_x'], self.df_monofil.iloc[i]['FR2_y']],
                [self.df_monofil.iloc[i]['FG1_x'], self.df_monofil.iloc[i]['FG1_y']],
                [self.df_monofil.iloc[i]['FG2_x'], self.df_monofil.iloc[i]['FG2_y']],
                [self.df_monofil.iloc[i]['FB1_x'], self.df_monofil.iloc[i]['FB1_y']],
                [self.df_monofil.iloc[i]['FB2_x'], self.df_monofil.iloc[i]['FB2_y']]
            ], dtype=np.float32)

            # Apply homography to the monofilament points
            monofil_points_transformed = cv2.perspectiveTransform(
                monofil_points.reshape(-1, 1, 2), h_matrix
                ).reshape(-1, 2)
            # Store the transformed points
            transformed_monofil_points.append(
                monofil_points_transformed.flatten()
                )

        columns = ['tf_FR1_x', 'tf_FR1_y', 'tf_FR2_x', 'tf_FR2_y',
                   'tf_FG1_x', 'tf_FG1_y', 'tf_FG2_x', 'tf_FG2_y',
                   'tf_FB1_x', 'tf_FB1_y', 'tf_FB2_x', 'tf_FB2_y']
        self.df_transformed_monofil = pd.DataFrame(transformed_monofil_points,
                                                   columns=columns)

        return self.df_transformed_monofil

    def _get_homography_matrix(self,
                               index: int,
                               dst_points: np.ndarray=homography_points) -> np.ndarray:

        Val.validate_type(index, int, "Index")
        Val.validate_positive(index, "Index", zero_allowed=True)
        Val.validate_array_int_float(dst_points,
                                     shape=(4, 2),
                                     name="Destination Points")

        src_points = np.array([
            [self.df_square.iloc[index]['Top_left_x'],
             self.df_square.iloc[index]['Top_left_y']],
            [self.df_square.iloc[index]['Top_right_x'],
             self.df_square.iloc[index]['Top_right_y']],
            [self.df_square.iloc[index]['Bottom_right_x'],
             self.df_square.iloc[index]['Bottom_right_y']],
            [self.df_square.iloc[index]['Bottom_left_x'],
             self.df_square.iloc[index]['Bottom_left_y']]
        ], dtype=np.float32)

        # Find the homography matrix
        h_matrix, _ = cv2.findHomography(src_points, dst_points)
        return h_matrix

    def _merge_data(self) -> pd.DataFrame:

        self.df_merged = pd.concat([self.df_square,
                                    self.df_monofil,
                                    self.df_transformed_monofil,
                                    self.df_bending_coefficients], axis=1)

        return self.df_merged
        