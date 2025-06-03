import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from src.components.validation import Validation as Val

# Quiet the warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

class DataNeuron:
    """
    A class for handling neuron data, processing spike and time information, 
    and performing operations such as frequency adjustment, spike filling, and IFF calculation.

    This class processes neuron data from aa csv file, validates the required columns, and ensures
    that the data is in a consistent format. It handles various tasks such as:
        - Ensuring the data has valid columns (`Time`, `Spikes`, `Neuron`, `IFF`, etc.)
        - Adjusting the frequency of the data if necessary
        - Calculating Instantaneous Frequency Firing (IFF)
        - Downsampling the data to a desired frequency
        - Filling missing samples when needed

    Attributes:
        df (pd.DataFrame): The DataFrame containing the neuron data, including 'Time', 'Spikes', and optional columns like 'IFF'.
        original_freq (int): The original frequency of the neuron data.
        downsampled_df (pd.DataFrame): The downsampled DataFrame, created if downsampling is required.

    Args:
        xclc_path (str): Path to the Excel file containing neuron data.
        original_freq (int): The original frequency of the data.

    Raises:
        ValueError: If the provided frequency is not a positive integer or if the required columns are not present in the file.
        FileNotFoundError: If the file path does not exist.
    """
    def __init__(self,
                neuron_path: str,
                original_freq: int) -> None:
        """
        Initialize DataNeuron with neuron data from CSV or Excel.

        Args:
            neuron_path (str): Path to the neuron data file (.csv or .xlsx).
            original_freq (int): The original frequency of the data.

        Raises:
            ValueError: If the provided frequency is not a positive integer or if the required columns are not present in the file.
            FileNotFoundError: If the file path does not exist.
        """
        # Accept both CSV and XLSX
        allowed_exts = [".csv", ".xlsx"]
        Val.validate_path_exists(neuron_path)
        Val.validate_path(neuron_path, file_types=allowed_exts)
        Val.validate_type(original_freq, int, "Original Frequency")
        Val.validate_positive(original_freq, "Original Frequency")

        ext = os.path.splitext(neuron_path)[1].lower()
        if ext == ".csv":
            try:
                df = pd.read_csv(neuron_path, sep=",")
            except pd.errors.ParserError:
                df = pd.read_csv(neuron_path, sep=";")
        elif ext == ".xlsx":
            df = pd.read_excel(neuron_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        self.df = df
        self.original_freq = original_freq
        self.downsampled_df = None

        # Define required columns with "OR" groups
        required_columns = [
            ["Time"],  # Time column must exist
            ["Spike", "Neuron"]  # At least one of these must exist
        ]
        # Validate and get column mappings for required columns
        column_mapping = Val.validate_dataframe(
            self.df, required_columns, name="Neuron DataFrame")
        # Rename required columns based on the mapping
        self.df.rename(columns={v: k for k, v in column_mapping.items()},
                       inplace=True)

        # Handle optional columns (e.g., IFF or Freq)
        optional_columns = [["IFF", "Freq"]]
        optional_mapping = Val.validate_dataframe(
            self.df, optional_columns, name="Neuron DataFrame", optional=True)
        # Rename optional columns if they exist
        if optional_mapping:
            self.df.rename(columns={v: k for k, v in optional_mapping.items()},
                           inplace=True)

        # If data is not at a consistent frequency, fill the missing samples
        if self._get_frequency() != original_freq:
            self.fill_samples()
        # If IFF column is missing, calculate it
        if 'IFF' not in self.df.columns:
            self.calculate_iff()

        self.df['IFF'] = pd.to_numeric(self.df['IFF'], errors='coerce')

    def calculate_iff(self) -> None:
        """
        Calculate the Instantaneous Frequency Firing (IFF) for the spike data.

        The IFF is computed as 1 divided by the difference between the current spike's 
        time and the previous spike's time. The IFF is stored in a new column 'IFF' in 
        the dataframe. Any missing IFF values are forward-filled, with any remaining 
        missing values being filled with zero.

        This method modifies the dataframe in place by adding the 'IFF' column.

        Returns:
            None: The function directly modifies the dataframe without returning anything.
        """
        # Create Instantaneous Frequency Firing (IFF):
        # 1 divided by the difference between the current time and last spike time
        spikes_loc = self.df[self.df['Spike'] == 1].index
        self.df["IFF"] = np.nan

        for i in range(1, len(spikes_loc)):
            diff = self.df.loc[spikes_loc[i], 'Time'] - \
                   self.df.loc[spikes_loc[i-1], 'Time']
            # calculate the reciprocal of the difference
            self.df.loc[spikes_loc[i], "IFF"] = 1 / diff

        # fill the NaN values with the previous non-NaN value
        self.df["IFF"].fillna(method='ffill', inplace=True)
        # fill the remaining NaN values with 0
        self.df["IFF"].fillna(0, inplace=True)

    def _get_frequency(self) -> int:
        """
        Calculate the frequency of events based on the time differences between spikes.

        The frequency is calculated as the reciprocal of the mean time difference 
        between consecutive spikes. This method provides an estimate of the event 
        frequency in Hz (spikes per second).

        Returns:
            int: The calculated frequency based on the mean time difference between spikes.
        """
        time_diffs = np.diff(self.df['Time'])
        # Calculate the frequency as the reciprocal of the mean time difference
        current_freq = int(1 / np.mean(time_diffs))
        return current_freq

    def fill_samples(self) -> None:
        """
        Fill missing sample data by generating a complete range of timestamps.

        This method creates a time series with a consistent interval based on the 
        original frequency, and fills in the missing spike data. The `Time` column 
        is rounded to six decimal places to avoid floating-point precision issues. 
        The method also ensures that missing 'IFF' values are propagated forward, 
        filling in any gaps.

        The resulting DataFrame is updated with the filled spike data and IFF values.

        Returns:
            None: The method modifies the DataFrame in place.
        """
        interval = 1 / self.original_freq
        min_time, max_time = 0, self.df['Time'].max()

        # Generate complete range of timestamps
        full_time_range = np.arange(min_time, max_time + interval, interval)
        full_df = pd.DataFrame({'Time': full_time_range})
        full_df['Time'] = full_df['Time'].round(6)
        self.df['Time'] = self.df['Time'].round(6)

        # Merge with original data
        filled_df = full_df.merge(self.df, on='Time', how="left")

        # Fill missing Spikes with 0
        filled_df['Spike'] = filled_df['Spike'].fillna(0).astype(int)

        # Fill IFF column if it exists
        if 'IFF' in filled_df.columns:
            filled_df['IFF'].fillna(method='ffill', inplace=True)
            filled_df['IFF'].fillna(0, inplace=True)

        self.df = filled_df

    def downsample(self,
                   target_freq: int) -> pd.DataFrame:
        """
        Downsample the data to a lower frequency.

        This method reduces the frequency of the 'Spikes' and 'IFF' columns by applying 
        a rolling window technique. The 'Spikes' column is downsampled by summing values 
        within the window, while the 'IFF' column is downsampled by taking the maximum value 
        within the window. The DataFrame is then reduced by selecting every `downsample_factor`-th row.

        Args:
            target_freq (int): The target frequency to downsample the data to. It must be less than or 
                                equal to the original frequency.

        Returns:
            pd.DataFrame: The downsampled DataFrame containing the 'Spikes' and 'IFF' columns at the 
                        specified target frequency.
        
        Raises:
            ValueError: If the target frequency is greater than the original frequency.
        """
        Val.validate_type(target_freq, int, "Target Frequency")
        Val.validate_positive(target_freq, "Target Frequency")
        if target_freq > self.original_freq:
            raise ValueError("Target frequency must be less than or equal to the original frequency.")

        # Calculate the downsampling factor
        downsample_factor = int(self.original_freq / target_freq)

        # Apply a rolling window with a maximum function to preserve binary components
        downsampled_df = pd.DataFrame()

        # Downsample the iff/freq column by picking the maximum value in the window
        downsampled_df['IFF'] = \
            self.df['IFF'].rolling(window=downsample_factor,
                                             min_periods=1).max()

        # Downsample the spikes column by summing the values in the window
        downsampled_df['Spike'] = \
            self.df['Spike'].rolling(window=downsample_factor,
                                                min_periods=1).sum()

        # Downsample the DataFrame by selecting every downsample_factor-th row
        downsampled_df = downsampled_df.iloc[::downsample_factor]

        # Reset the index to ensure it is sequential
        downsampled_df.reset_index(drop=True, inplace=True)

        # Update the DataFrame
        self.downsampled_df = downsampled_df
        return self.downsampled_df

    #! Might not be needed, but keeping for now
    def _fill_downsample_length(self,
                                target_length: int) -> pd.DataFrame:
        """
        Fill the downsampled DataFrame to a specified length.

        This method ensures that the downsampled DataFrame has the desired number of rows, 
        by forward filling missing values in the 'IFF' column and filling missing 'Spikes' values with 0.

        Args:
            target_length (int): The target length to which the DataFrame should be filled.

        Returns:
            pd.DataFrame: The downsampled DataFrame, now extended to the specified length 
                        with forward-filled 'IFF' values and 'Spikes' values set to 0 where necessary.
        
        Raises:
            ValueError: If the target length is not a positive integer.
        """
        Val.validate_type(target_length, int, "Target Length")
        Val.validate_positive(target_length, "Target Length")

        # Fill the data up to a target length by forward filling the data
        self.downsampled_df = self.downsampled_df.reindex(range(target_length))
        self.downsampled_df['Spike'].fillna(0, inplace=True)
        self.downsampled_df['IFF'].ffill(inplace=True)

        return self.downsampled_df
