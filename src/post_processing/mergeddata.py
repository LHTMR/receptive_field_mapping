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
        # Merge the DataFrames
        df_dlc = self.dlc._merge_data()
        df_neuron = self.neuron.downsampled_df.copy()

        # Calculate z-scores
        df_dlc['Bending_ZScore'] = zscore(df_dlc['Bending_Coefficient'])
        # Identify spikes based on a z-score threshold
        df_dlc['Bending_Binary'] = (df_dlc['Bending_ZScore'] > self.threshold).astype(int)

        # Fill gaps in neuron Spikes column with dynamic width
        df_neuron['Spikes_Filled'] = df_neuron['Spikes'].copy()
        gap_start = None
        for i in range(len(df_neuron)):
            if df_neuron['Spikes'][i] == 1:
                if gap_start is not None and (i - gap_start) <= self.max_gap_fill:
                    df_neuron['Spikes_Filled'][gap_start:i] = 1
                gap_start = i + 1
            elif df_neuron['Spikes'][i] == 0 and gap_start is None:
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
        self.df_merged[['Spikes', 'Spikes_Filled']] = \
            self.df_merged[['Spikes', 'Spikes_Filled']].fillna(0).astype(int)

        # Fill IFF column based on shift direction
        if best_shift < 0:
            self.df_merged["IFF"].fillna(method='ffill', inplace=True)
            self.df_merged["IFF"].fillna(0, inplace=True)
        else:
            self.df_merged["IFF"].fillna(0, inplace=True)

        return self.df_merged

    def _clean(self) -> pd.DataFrame:
        """
        Cleaned data where the bending coefficient is above the threshold
        or the Spikes column is not 0.
        """
        # Use the Bending_Binary column instead of recalculating the threshold
        self.df_merged_cleaned = self.df_merged[
            (self.df_merged['Bending_Binary'] == 1) |
            (self.df_merged['Spikes'] != 0)]
        return self.df_merged_cleaned

    def threshold_data(self,
                       bending: bool = True,
                       spikes: bool = True) -> pd.DataFrame:
        """
        Return a filtered DataFrame based on the specified conditions:
        - If `bending` is True, include rows where the Bending_Binary column is 1.
        - If `spikes` is True, include rows where the Spikes column is not 0.
        - If both are False, return the unfiltered DataFrame.
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
            condition |= (self.df_merged['Spikes'] != 0)

        # Return the filtered DataFrame
        return self.df_merged[condition]

    def plotting_split(self) -> tuple:
        """
        Split the data into 3 parts:
        - High bending coefficient with neuron firing
        - High bending coefficient without neuron firing
        - Low bending coefficient with neuron firing
        """
        high_bend_w_neuron = self.df_merged[
            (self.df_merged['Bending_Binary'] == 1) &
            (self.df_merged['Spikes'] >= 1)]
        high_bend_wo_neuron = self.df_merged[
            (self.df_merged['Bending_Binary'] == 1) &
            (self.df_merged['Spikes'] == 0)]
        low_bend_w_neuron = self.df_merged[
            (self.df_merged['Bending_Binary'] == 0) &
            (self.df_merged['Spikes'] >= 1)]

        return high_bend_w_neuron, high_bend_wo_neuron, low_bend_w_neuron

    def _save_data(self,
                   df: pd.DataFrame,
                   path: str,
                   file_format: str) -> None:
        Val.validate_type(df, pd.DataFrame, "DataFrame")
        Val.validate_type(path, str, "Path")
        Val.validate_type(file_format, str, "File Format")
        Val.validate_path(path, [file_format])

        if file_format == 'csv':
            df.to_csv(path, index=False)
        elif file_format == 'xlsx':
            df.to_excel(path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def save_full_data(self,
                       path: str,
                       file_format: str = 'csv') -> None:
        self._save_data(self.df_merged, path, file_format)

    def save_cleaned_data(self,
                          path: str,
                          file_format: str = 'csv') -> None:
        self._save_data(self.df_merged_cleaned, path, file_format)
