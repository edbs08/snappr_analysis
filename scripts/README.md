# Dataset Preprocessing and Inference Automation

This project contains a collection of Python executable scripts designed to preprocess datasets, automate inference in various ways, and generate training data in the `jsonl` format. The scripts are tailored to work with the Snappr dataset and the OmniGen model.

## Dependencies

Before running the scripts, ensure you have the following dependencies:

1. **Dataset**: This project was tested with the Snappr dataset. Please contact an engineer to obtain the dataset.
2. **OmniGen Model**: Download and configure the open-source OmniGen model from its official repository.

## Setup Instructions

### 1. Create a Virtual Environment

To isolate the project dependencies, create a Python virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```
- On Windows:
  ```bash
  venv\Scripts\activate
  ```

### 2. Install Requirements

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

### Before Running Scripts

Before executing any script, review the paths specified inside the scripts. Ensure they point to the absolute paths of the dataset and the output training `jsonl` file. The paths provided in the scripts are for reference only and may need to be updated based on your system configuration.

### Dataset Analysis

The `dataset_analysis.py` script is a small utility for visualizing pairs of images by reading the JSON file. To run it:

```bash
python dataset_analysis.py
```

### Additional Scripts

Other scripts in this project include:

- **Preprocessing Scripts**: For cleaning and preparing the dataset.
- **Inference Automation Scripts**: For automating inference tasks.
- **Training Data Generation**: For creating the `jsonl` file that maps input images, output images, and text instructions.

Refer to the individual script files for specific usage instructions.
