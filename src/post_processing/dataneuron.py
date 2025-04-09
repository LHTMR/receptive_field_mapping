import pandas as pd
import numpy as np
import cv2
import re
from validation import Validation as Val

class DataNeuron:
    def __init__(self,
                 xclc_path: str,
                 original_freq: int) -> None:
        Val.validate_path(xclc_path, file_types=[".xlsx"])
        Val.validate_type(original_freq, int, "Original Frequency")
        Val.validate_positive(original_freq, "Original Frequency")

        self.df = pd.read_excel(xclc_path)
        self.original_freq = original_freq

        # Define required columns with "OR" groups
        required_columns = [
            ["Time"],  # Time column must exist
            ["Spikes", "Neuron"]  # At least one of these must exist
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

    def calculate_iff(self) -> None:

        # Create Instantaneous Frequency Firing (IFF):
        # 1 divided by the difference between the current time and last spike time
        spikes_loc = self.df[self.df['Spikes'] == 1].index
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
        time_diffs = np.diff(self.df['Time'])
        # Calculate the frequency as the reciprocal of the mean time difference
        current_freq = 1 / np.mean(time_diffs).round()
        return current_freq

    def fill_samples(self) -> None:

        interval = 1 / self.original_freq  # Compute time step based on frequency
        min_time, max_time = 0, self.df['Time'].max()

        # Generate complete range of timestamps
        full_time_range = np.arange(min_time, max_time + interval, interval)

        # Create a new DataFrame with 0s
        full_df = pd.DataFrame({'Time': full_time_range, 'Spikes': 0})

        # Merge with original data, ensuring 1s are preserved
        filled_df = full_df.merge(self.df, on='Time', how="outer", suffixes=('', '_original'))
        filled_df['Spikes'] = filled_df['Spikes' + '_original'].fillna(0).astype(int)
        filled_df = filled_df.drop(columns=['Spikes' + '_original'])

        # Fill IFF column if it exists
        if 'IFF' in filled_df.columns:
            filled_df['IFF'].fillna(method='ffill', inplace=True)
            filled_df['IFF'].fillna(0, inplace=True)

        # Update the DataFrame
        self.df = filled_df

    def downsample(self,
                   target_freq: int) -> pd.DataFrame:
        Val.validate_type(target_freq, int, "Target Frequency")
        Val.validate_positive(target_freq, "Target Frequency")

        # Calculate the downsampling factor
        downsample_factor = int(self.original_freq / target_freq)
        print(downsample_factor)

        # Apply a rolling window with a maximum function to preserve binary components
        downsampled_df = pd.DataFrame()

        # Downsample the time column by selecting the first value in the window
        #downsampled_df['Time'] = \
        #    self.df['Time'].rolling(window=downsample_factor,
        #                                      min_periods=1).min()

        # Downsample the iff/freq column by picking the maximum value in the window
        downsampled_df['IFF'] = \
            self.df['IFF'].rolling(window=downsample_factor,
                                             min_periods=1).max()

        # Downsample the spikes column by summing the values in the window
        downsampled_df['Spikes'] = \
            self.df['Spikes'].rolling(window=downsample_factor,
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
        Val.validate_type(target_length, int, "Target Length")
        Val.validate_positive(target_length, "Target Length")

        # Fill the data up to a target length by forward filling the data
        self.downsampled_df = self.downsampled_df.reindex(range(target_length))
        self.downsampled_df['Spikes'].fillna(0, inplace=True)
        self.downsampled_df['IFF'].ffill(inplace=True)

        return self.downsampled_df
