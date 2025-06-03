from scipy.signal import correlate
from scipy.stats import zscore
from src.components.validation import Validation as Val
import pandas as pd
from src.post_processing.dataneuron import DataNeuron
from src.post_processing.datadlc import DataDLC
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MergedData:
    """
    A class for merging and processing DLC and neuron data, performing alignment, 
    gap filling, and cleaning operations.

    This class merges the bending coefficient data from the DLC and spike data from 
    the neuron, aligning them based on cross-correlation. It also performs tasks such as:
        - Merging DLC and neuron data after sequence alignment.
        - Calculating z-scores for the bending coefficients and identifying significant bending events.
        - Filling gaps in the neuron spike data within a specified gap width.
        - Cleaning the merged data based on the bending coefficient and spike events.
        - Splitting the data into different categories based on bending and neuron firing conditions.
        - Saving the processed data in multiple file formats (CSV, Excel).

    Attributes:
        dlc (DataDLC): An instance of the DataDLC class, providing the DLC data.
        neuron (DataNeuron): An instance of the DataNeuron class, providing the neuron data.
        max_gap_fill (int): The maximum gap width for filling neuron spikes (default is 10).
        threshold (float): The z-score threshold for identifying significant bending events (default is 0.1).
        df_merged (pd.DataFrame): The merged DataFrame containing both DLC and neuron data, aligned by time.
        df_merged_cleaned (pd.DataFrame): The cleaned DataFrame filtered based on bending and spike conditions.

    Args:
        dlc (DataDLC): The DataDLC object containing the DLC data.
        neuron (DataNeuron): The DataNeuron object containing the neuron data.
        max_gap_fill (int, optional): The maximum gap size (in rows) for filling neuron spike data (default is 10).
        threshold (float, optional): The z-score threshold for identifying significant bending events (default is 0.1).

    Raises:
        ValueError: If the threshold is not between 0 and 1, or if the max gap fill is not a positive integer.
    """
    def __init__(self,
                 dlc: DataDLC,
                 neuron: DataNeuron,
                 max_gap_fill: int = 10,
                 threshold: float = 0.1) -> None:
        Val.validate_type(dlc, DataDLC, "DLC Object")
        Val.validate_type(neuron, DataNeuron, "Neuron Object")
        Val.validate_type(max_gap_fill, int, "Max Gap Fill")
        Val.validate_positive(max_gap_fill, "Max Gap Fill")
        Val.validate_type(threshold, float, "Threshold")
        Val.validate_float_in_range(threshold, 0, 1, "Threshold")

        self.dlc = dlc
        self.neuron = neuron
        self.max_gap_fill = max_gap_fill
        self.threshold = threshold
        self.df_merged = None
        self._merge()
        self.df_merged_cleaned = None
        self._clean()

    def _merge(self) -> pd.DataFrame:
        """
        Merges the DLC and neuron data after aligning sequences based on the binary bending coefficient
        and neuron spike events.

        This function performs the following steps:
        1. Merges the DLC data (with calculated bending coefficients) and neuron spike data.
        2. Calculates z-scores for the bending coefficients and identifies spikes using a threshold.
        3. Fills gaps in the neuron spike data (Spikes column) by expanding spikes within a maximum gap width.
        4. Aligns the bending binary data with the neuron spike data using cross-correlation.
        5. Shifts the neuron data index to align with the DLC data.
        6. Merges the DLC and neuron data into a single DataFrame.
        7. Fills missing values after the shift for columns like Spikes, Spikes_Filled, and IFF.

        Returns:
            pd.DataFrame: The merged DataFrame containing the DLC and neuron data with aligned timestamps
                        and filled values for Spikes and IFF columns.
        """
        # Merge the DataFrames
        df_dlc = self.dlc._merge_data()
        df_neuron = self.neuron.downsampled_df.copy()

        # Calculate z-scores
        df_dlc['Bending_ZScore'] = zscore(df_dlc['Bending_Coefficient'])
        # Identify spikes based on a z-score threshold
        df_dlc['Bending_Binary'] = (df_dlc['Bending_ZScore'] > self.threshold).astype(int)

        # Fill gaps in neuron Spikes column with dynamic width
        df_neuron['Spikes_Filled'] = df_neuron['Spike'].copy()
        gap_start = None
        for i in range(len(df_neuron)):
            if df_neuron['Spike'][i] == 1:
                if gap_start is not None and (i - gap_start) <= self.max_gap_fill:
                    df_neuron['Spikes_Filled'][gap_start:i] = 1
                gap_start = i + 1
            elif df_neuron['Spike'][i] == 0 and gap_start is None:
                gap_start = i

        # Perform sequence alignment using cross-correlation
        correlation = correlate(
            df_dlc['Bending_Binary'], df_neuron['Spikes_Filled'], mode='full')
        best_shift = correlation.argmax() - (len(df_neuron) - 1)

        # Shift df_neuron index accordingly
        df_neuron = df_neuron.shift(periods=best_shift).reset_index(drop=True)
        # Merge the DataFrames
        self.df_merged = pd.concat([df_dlc, df_neuron], axis=1)

        # After merging, fill the gaps created from shifting
        # Always zero-fill for Spikes and Spikes_Filled
        self.df_merged[['Spike', 'Spikes_Filled']] = \
            self.df_merged[['Spike', 'Spikes_Filled']].fillna(0).astype(int)

        # Fill IFF column based on shift direction
        if best_shift < 0:
            self.df_merged["IFF"].fillna(method='ffill', inplace=True)
            self.df_merged["IFF"].fillna(0, inplace=True)
        else:
            self.df_merged["IFF"].fillna(0, inplace=True)

        return self.df_merged

    def _clean(self) -> pd.DataFrame:
        """
        Clean the data by filtering rows where the bending coefficient is above the threshold
        or the Spikes column is not 0.

        This function applies the following filtering conditions:
        - Include rows where the Bending_Binary column is 1.
        - Include rows where the Spikes column is not 0.

        Returns:
            pd.DataFrame: The cleaned DataFrame containing only the relevant rows based on the filtering conditions.
        """
        # Use the Bending_Binary column instead of recalculating the threshold
        self.df_merged_cleaned = self.df_merged[
            (self.df_merged['Bending_Binary'] == 1) |
            (self.df_merged['Spike'] != 0)]
        return self.df_merged_cleaned

    def threshold_data(self,
                       bending: bool = True,
                       spikes: bool = True) -> pd.DataFrame:
        """
        Return a filtered DataFrame based on the specified conditions:
        - If `bending` is True, include rows where the Bending_Binary column is 1.
        - If `spikes` is True, include rows where the Spikes column is not 0.
        - If both are False, return the unfiltered DataFrame.

        Args:
            bending (bool): If True, include rows where the Bending_Binary column is 1. Default is True.
            spikes (bool): If True, include rows where the Spikes column is not 0. Default is True.

        Returns:
            pd.DataFrame: A filtered DataFrame based on the specified conditions.
        """
        Val.validate_type(bending, bool, "Bending")
        Val.validate_type(spikes, bool, "Spikes")

        if not bending and not spikes:
            # Return the unfiltered DataFrame if both conditions are False
            return self.df_merged

        # Build the filter condition based on the boolean inputs
        condition = pd.Series([False] * len(self.df_merged),
                              index=self.df_merged.index)
        if bending:
            condition |= (self.df_merged['Bending_Binary'] == 1)
        if spikes:
            condition |= (self.df_merged['Spike'] != 0)

        # Return the filtered DataFrame
        return self.df_merged[condition]

    def plotting_split(self) -> tuple:
        """
        Split the data into three parts based on conditions involving bending coefficient and neuron firing.

        The data is split into the following categories:
        - High bending coefficient with neuron firing.
        - High bending coefficient without neuron firing.
        - Low bending coefficient with neuron firing.

        Returns:
            tuple: A tuple containing three DataFrames:
                - High bending coefficient with neuron firing.
                - High bending coefficient without neuron firing.
                - Low bending coefficient with neuron firing.
        """
        high_bend_w_neuron = self.df_merged[
            (self.df_merged['Bending_Binary'] == 1) &
            (self.df_merged['Spike'] >= 1)]
        high_bend_wo_neuron = self.df_merged[
            (self.df_merged['Bending_Binary'] == 1) &
            (self.df_merged['Spike'] == 0)]
        low_bend_w_neuron = self.df_merged[
            (self.df_merged['Bending_Binary'] == 0) &
            (self.df_merged['Spike'] >= 1)]

        return high_bend_w_neuron, high_bend_wo_neuron, low_bend_w_neuron

    def _save_data(self,
                   df: pd.DataFrame,
                   path: str,
                   file_format: str) -> None:
        """
        Save the provided DataFrame to a specified file path and format.

        The function validates the input and saves the DataFrame to the given path in the specified file format.
        Supported formats are CSV and Excel.

        Args:
            df (pd.DataFrame): The DataFrame to be saved.
            path (str): The file path where the DataFrame should be saved.
            file_format (str): The file format for saving the DataFrame. Should be either 'csv' or 'xlsx'.

        Raises:
            ValueError: If an unsupported file format is provided.
        """
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(path, str, "Path")
        Val.validate_type(file_format, str, "File Format")

        if file_format == 'csv':
            Val.validate_path(path, [file_format])
            df.to_csv(path, index=False)
        elif file_format == 'xlsx':
            Val.validate_path(path, [file_format])
            df.to_excel(path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def save_full_data(self,
                       path: str,
                       file_format: str = 'csv') -> None:
        """
        Save the merged DataFrame to a specified file path and format.

        This function calls the `_save_data` method to save the merged DataFrame (`df_merged`) to the 
        specified path in the given file format. The default file format is 'csv', but 'xlsx' is also supported.

        Args:
            path (str): The file path where the DataFrame should be saved.
            file_format (str, optional): The file format for saving the DataFrame. Defaults to 'csv'.

        Raises:
            ValueError: If an unsupported file format is provided.
        """
        self._save_data(self.df_merged, path, file_format)

    def save_cleaned_data(self,
                          path: str,
                          file_format: str = 'csv') -> None:
        """
        Save the cleaned DataFrame to a specified file path and format.

        This function calls the `_save_data` method to save the cleaned DataFrame (`df_merged_cleaned`) to 
        the specified path in the given file format. The default file format is 'csv', but 'xlsx' is also supported.

        Args:
            path (str): The file path where the cleaned DataFrame should be saved.
            file_format (str, optional): The file format for saving the DataFrame. Defaults to 'csv'.

        Raises:
            ValueError: If an unsupported file format is provided.
        """
        self._save_data(self.df_merged_cleaned, path, file_format)
