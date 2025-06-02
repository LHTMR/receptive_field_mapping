# Receptive Field Mapping App
## Project Overview
This project is a streamlined application designed to assist neuroscience researchers in mapping receptive fields using synchronized video and nerve recording data. The app takes as input a video recording from a nerve recording session and processes it using DeepLabCut (DLC) to generate pose estimation labels.

By combining these automatically generated labels with the actual nerve signal recordings, the app allows users to easily generate a receptive field (RF) map. This approach simplifies the otherwise complex and manual process of RF mapping, enabling more efficient and reproducible analysis in sensory neuroscience experiments.
## Project Setup Instructions

### 1. Prerequisites

Make sure you have the following installed:

* Git
* Anaconda or Miniconda (Python 3.9+ recommended)
* A trained DeepLabCut model downloaded to your machine.
### 2. Clone the Repository

```bash
git clone https://github.com/GabeLack/receptive_field_mapping_app.git
cd receptive_field_mapping_app
```

### 3. Create the Conda Environment

Create the environment from the provided YAML file:
For Windows:
```bash
conda env create -f RF_MAPPING_win.yml
```
For Linux:
```bash
conda env create -f RF_MAPPING_lnx.yml
```
For MacOS:
```bash
conda env create -f RF_MAPPING_mac.yml
```
This will create a new Conda environment with all the required dependencies.

### 4. Activate the Environment

```bash
conda activate RF_MAPPING
```

Replace `RF_MAPPING` with the name defined at the top of `RF_MAPPING_<platform>.yml` (e.g., `name: RF_MAPPING`).

### 5. Run the Streamlit App
For windows:
```bash
streamlit run app.py
```
For Linux/Mac:
```bash
STREAMLIT_WATCHER_TYPE=none streamlit run app.py
```
After a moment, your default browser should open automatically with the Streamlit interface. If not, look for the local URL (e.g., `http://localhost:8501`) in the terminal.

### 6. Optional: Deactivate or Remove Environment

To deactivate the environment when done:

```bash
conda deactivate
```

To remove the environment later:

```bash
conda remove --name your-env-name --all
```


## Authors
Gabriel Läckgren & Erik Svensson

This app provides a user-friendly interface for integrating DeepLabCut predictions with nerve signal recordings, enabling post-processing and receptive field mapping.

## Credits

This package was created with Cookiecutter and the [andymcdgeo/cookiecutter_streamlit_app](https://github.com/andymcdgeo/cookiecutter-streamlit) project template.

This project is a continuation of [SvenPontus/RF_mapping](https://github.com/SvenPontus/RF_mapping).


## Project structure
receptive_field_mapping_app/
├── app.py                         # Main entry point for launching the Streamlit app
├── README.md
├── RF_MAPPING_lnx.yml             # Conda environment file for Linux
├── RF_MAPPING_mac.yml             # Conda environment file for macOS
├── RF_MAPPING_win.yml             # Conda environment file for Windows
├── assets/                        # Static assets like icons and images
│   ├── bad_bend_example_1.png
│   ├── bad_bend_example_2.png
│   └── ... (other static files)
├── documentation/
├── pages/                         # Additional Streamlit pages
│   ├── 01_Video_Instructions.py
│   ├── 02_Run_Predictions.py
│   └── 03_Post_Processing.py
├── src/                           # Core source code
│   ├── components/
│   │   ├── convert_roi.py
│   │   └── validation.py
│   ├── post_processing/
│   │   ├── datadlc.py
│   │   ├── dataneuron.py
│   │   ├── mergeddata.py
│   │   ├── outlierimputer.py
│   │   ├── plotting_plotly.py
│   │   ├── processing_utils.py
│   └── train_predict/
│       └── dlc_utils.py
└── tests/                         # Unit tests and mocks
    ├── mock_dlc_data.h5
    ├── mock_neuron_data.csv
    ├── run.py
    ├── test_datadlc.py
    ├── test_dataneuron.py
    ├── test_dlc_utils.py
    ├── test_mergedata.py
    ├── test_outlierimputer.py
    ├── test_plotting_plotly.py
    ├── test_processing_utils.py
    ├── test_validation.py

## Functional Overview of Components

### src/train_predict/
This module handles all core DeepLabCut (DLC) related tasks, including:

- **Labeling:** Preparing video frames for manual or automated labeling of body parts.
- **Training:** Creating and updating DLC models using labeled data.
- **Predicting:** Running trained DLC models on new videos to generate pose estimations.
- **Fine-tuning:** Refining existing DLC models by retraining on new or corrected labels.

### src/post_processing/
- **datadlc.py**  
  Handles parsing and managing DLC output files.

- **dataneuron.py**  
  Loads, filters, and pre-processes neural recordings (e.g., CSV spike data).

- **mergeddata.py**  
  Merges DLC labels with neural data based on timestamps or sync signals.

- **outlierimputer.py**  
  Applies imputation to correct DLC outliers (e.g., interpolation or model-based).

- **plotting_plotly.py**  
  Contains plotting functions for receptive fields using Plotly.

- **processing_utils.py**  
  Utilities for calculating receptive field maps and smoothing responses.

### src/components/
- **convert_roi.py**  
  Function to convert and preprocess video based on ROI chosen by user.

- **validation.py**  
  Input validators for uploaded files, ROI formats, or label consistency.

### pages/
- **01_Video_Instructions.py**  
  Page for guiding users through video upload and frame extraction.

- **02_Run_Predictions.py**  
  Interface for running DLC predictions and inspecting results.

- **03_Post_Processing.py**  
  Tools for selecting ROIs, aligning with neuron data, and generating RF maps.

### tests/
Each `test_*.py` module contains unit tests for its respective module in `src/`, using mock data where needed.
