import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import cv2
from src.post_processing.outlierimputer import OutlierImputer
from src.components.validation import Validation as Val

class DataDLC:
    """Processes DeepLabCut output data and performs geometric transformations and analysis.

    This class handles the loading and processing of DeepLabCut `.h5` tracking files.
    It extracts square and monofilament coordinates, computes bending coefficients,
    applies homography transformations, imputes outliers, and merges all processed
    data into a single DataFrame for downstream analysis.

    Attributes:
        df_square (pd.DataFrame): DataFrame containing the square calibration points.
        df_monofil (pd.DataFrame): DataFrame containing monofilament tracking points.
        df_likelihoods (pd.DataFrame): DataFrame of likelihood values from DLC tracking.
        df_merged (pd.DataFrame): Combined DataFrame of square, monofilament, transformed points, and bending coefficients.
        df_bending_coefficients (pd.Series): Bending coefficients for each frame based on monofilament curvature.
        df_transformed_monofil (pd.DataFrame): Homography-transformed monofilament points.
        homography_points (np.ndarray): Destination points used for computing homography.

    Args:
        h5_file (str): Path to the DeepLabCut-generated `.h5` file containing tracking data.

    Raises:
        AttributeError: If the `.h5` file structure does not match the expected format.
    """
    def __init__(self, h5_file) -> None:
        df = pd.read_hdf(h5_file)
        self.df_square = None
        self.df_monofil = None
        self.df_likelihoods = None
        self.df_merged = None
        self.df_bending_coefficients = None
        self.df_transformed_monofil = None
        self.homography_points = None
        self.assign_homography_points()

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

    def get_avg_likelihoods(self) -> str:
        """Calculates and formats the average likelihoods for all body parts.

        This method computes the overall average likelihood across all body parts and frames,
        as well as the individual average likelihood for each body part. It returns the results
        as a formatted string.

        Returns:
            str: A string containing the overall average likelihood and individual body part averages.
        """
        overall_average = self.df_likelihoods.mean().mean()
        bodypart_average = self.df_likelihoods.mean()

        # Format the bodypart_average DataFrame as a string
        bodypart_average_str = bodypart_average.to_string(index=True, header=False)

        return f"Overall average likelihood: \n{overall_average}\n" + \
            f"Bodypart average likelihoods: \n{bodypart_average_str}"

    def assign_homography_points(self,
                                 start: int = 0,
                                 end: int = 20) -> pd.DataFrame:
        """Assigns four corner points to define a square region for homography transformation.

        This method validates the `start` and `end` parameters to ensure they are positive integers 
        (and not equal), and then constructs a 4-point region in clockwise order: 
        top-left, top-right, bottom-right, bottom-left. The resulting points are stored in the 
        `homography_points` attribute and returned.

        Args:
            start (int, optional): The starting coordinate (inclusive) of the square. Defaults to 0.
            end (int, optional): The ending coordinate (exclusive) of the square. Defaults to 20.

        Returns:
            pd.DataFrame: A NumPy array of shape (4, 2) representing the corner points of the square.

        Raises:
            ValueError: If `start` and `end` are the same.
            TypeError: If `start` or `end` is not an integer.
            ValueError: If `start` or `end` is negative.
        """
        Val.validate_type(start, int, "Start")
        Val.validate_type(end, int, "End")
        Val.validate_positive(start, "Start", zero_allowed=True)
        Val.validate_positive(end, "End", zero_allowed=True)
        if start == end:
            raise ValueError("Start and end points must be different.")

        self.homography_points = np.array([[start, end],
                                           [end, end],
                                           [end, start],
                                           [start, start]], dtype=np.float32)
        return self.homography_points

    def impute_outliers(self,
                        std_threshold: int|float = 2,
                        square: bool = True,
                        filament: bool = False,
                        model_name: str = None) -> None:
        """Detects and imputes outliers in the square or filament dataset.

        This method applies outlier detection and imputation using a statistical threshold on the 
        selected dataset (either square or monofilament). Outliers are identified based on 
        deviation from the mean and replaced using a predefined imputation model.

        Args:
            std_threshold (int | float, optional): The number of standard deviations to use as a 
                threshold for identifying outliers. Defaults to 2.
            square (bool, optional): Whether to impute outliers in the square dataset. 
                Defaults to True.
            filament (bool, optional): Whether to impute outliers in the filament dataset. 
                Defaults to False.
            model_name (str, optional): The name of a custom model to use for imputation. 
                If None, a default model is used.

        Returns:
            None: The function updates the relevant DataFrame in place 
            (`self.df_square` or `self.df_monofil`) with outliers imputed.

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If both `square` and `filament` are True, or if both are False.
        """
        Val.validate_type(std_threshold, (int, float), "STD Threshold")
        Val.validate_positive(std_threshold, "STD Threshold", zero_allowed=True)
        Val.validate_type(square, bool, "Square")
        Val.validate_type(filament, bool, "Filament")
        if model_name: # None is allowed as a default value
            Val.validate_type(model_name, str, "Model Name")

        if square and filament: # if both are True
            raise ValueError("Both square and filament cannot be True.")
        if not square and not filament: # if both are False
            raise ValueError("Either square or filament must be True.")

        # Impute outliers for the square and monofilament points
        if square:
            outlier_imputer = OutlierImputer("latest_square.json")
            self.df_square = outlier_imputer.impute_outliers(self.df_square,
                                                             std_threshold,
                                                             model_name)
            return self.df_square
        elif filament:
            outlier_imputer = OutlierImputer("latest_filament.json")
            self.df_monofil = outlier_imputer.impute_outliers(self.df_monofil,
                                                              std_threshold,
                                                              model_name)
            return self.df_monofil

    def get_bending_coefficients(self) -> pd.Series:
        """Calculates bending coefficients from the monofilament coordinates.

        For each frame in the monofilament dataset, this method computes how much the 
        shape deviates from a straight line by fitting a second-degree polynomial (quadratic) 
        to the centered coordinates. The absolute value of the quadratic coefficient is taken 
        as the bending coefficient.

        Returns:
            pd.Series: A Pandas Series containing the bending coefficient for each frame. 
            This is also stored as an attribute `self.df_bending_coefficients`.

        Notes:
            The bending coefficient reflects the degree of curvature of the monofilament in each frame.
        """

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
        """Applies a homography transformation to the monofilament points.

        For each frame, computes the homography matrix from the square marker points 
        and applies this transformation to the monofilament coordinates. The transformed 
        coordinates represent the monofilament points in the normalized square reference frame.

        Returns:
            pd.DataFrame: A DataFrame containing the transformed monofilament coordinates 
            with columns prefixed by 'tf_' (e.g., 'tf_FR1_x', 'tf_FR1_y', ...).
            The result is also stored as an attribute `self.df_transformed_monofil`.

        Notes:
            - The homography matrix is calculated frame-by-frame.
            - Transformed coordinates are flattened into a 1D array for each frame.
        """
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
                               dst_points: np.ndarray = None) -> np.ndarray:
        """Computes the homography matrix for a given frame index.

        This function calculates the transformation matrix that maps the square points 
        detected in the video frame (`src_points`) to a set of defined destination points 
        (`dst_points`). If `dst_points` is not provided, it uses the predefined 
        `self.homography_points`.

        Args:
            index (int): The index of the frame in the DataFrame to extract square points from.
            dst_points (np.ndarray, optional): A (4, 2) array of destination points to map to. 
                Defaults to `self.homography_points`.

        Returns:
            np.ndarray: A 3x3 homography matrix used to perform a perspective transformation.

        Raises:
            ValueError: If input validation fails (e.g., invalid index or improperly shaped array).

        Notes:
            - The source points are extracted from the square corners in `self.df_square`.
            - The destination points are typically a normalized square area.
        """
        if dst_points is None:
            dst_points = self.homography_points
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
        """Merges all relevant data into a single DataFrame.

        This method combines the square tracking points, monofilament tracking points,
        transformed monofilament points, and bending coefficients into a single merged
        DataFrame (`self.df_merged`). The merge is done horizontally (column-wise).

        Returns:
            pd.DataFrame: A combined DataFrame containing all relevant processed data.

        Notes:
            - Assumes that `self.df_square`, `self.df_monofil`, `self.df_transformed_monofil`, 
            and `self.df_bending_coefficients` are already populated and aligned by index.
            - This method is typically called at the end of the data processing pipeline.
        """

        self.df_merged = pd.concat([self.df_square,
                                    self.df_monofil,
                                    self.df_transformed_monofil,
                                    self.df_bending_coefficients], axis=1)

        return self.df_merged
        