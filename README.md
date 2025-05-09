# Receptive Field Mapping App

## Project Setup Instructions

### 1. Prerequisites

Make sure you have the following installed:

* Git
* Anaconda or Miniconda (Python 3.9+ recommended)

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Replace the URL with your actual repository address.

### 3. Create the Conda Environment

Create the environment from the provided YAML file:

```bash
conda env create -f RF_MAPPING.yml
```

This will create a new Conda environment with all the required dependencies.

### 4. Activate the Environment

```bash
conda activate RF_MAPPING
```

Replace `RF_MAPPING` with the name defined at the top of `full_env.yml` (e.g., `name: RF_MAPPING`).

### 5. Run the Streamlit App

```bash
streamlit run app.py
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


## Created by: Gabriel Läckgren & Erik Svensson

This is a streamlit app for interfacing with DeepLabCut in a simple way to make predictions on a video. Followed up by some post-processing to merge it with neuron data.

## Credits

This package was created with Cookiecutter and the [andymcdgeo/cookiecutter_streamlit_app](https://github.com/andymcdgeo/cookiecutter-streamlit) project template.
