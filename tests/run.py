import pandas as pd

mock_data = {
    "Time": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "Spikes": [0, 1, 0, 1, 0, 1],
    "IFF": [0, 0, 0, 5, 5, 5]
}
mock_df = pd.DataFrame(mock_data)
mock_df.to_excel("tests/mock_neuron_data.xlsx", index=False)  # Save in your tests folder