# Auto-FastCut

Automating fast cuts in Adobe Premiere Pro.

## Prerequisites

To run the script, ensure you have the following installed:

- **Python:** Version 3.6 or higher.
- **Git:** For cloning the repository.
- **Conda:** For environment management.
- **Dataset:** A dataset of audio files organized into `Positive` (audio to detect) and `Negative` (audio to ignore) directories.

## Installation

1. **Clone the Repository:**

    ```sh
    git clone https://github.com/yourusername/auto-fastcut.git
    cd auto-fastcut
    ```

2. **Set Up the Conda Environment:**

    Create and activate a Conda environment using the provided `minecraftAI.yml` file:

    ```sh
    conda env create -f minecraftAI.yml
    conda activate minecraftAI
    ```

3. **Install Required Packages:**

    Ensure the following necessary Python packages are installed:

    ```sh
    pip install tensorflow tensorflow-io soundfile matplotlib colorama
    ```

## Usage

1. **Prepare Your Files:**

    Set up your directories and files to ensure the script runs properly:

    - **Dataset:** Place your dataset into a `Dataset/` directory with `Positive/` and `Negative/` subdirectories.
    - **Model File:** Ensure a pre-trained model file named `model.keras` is located in the root directory.

2. **Run the Training Script:**

    If you need to train the model from scratch, execute the training script. This will save a new `model.keras` file:

    ```bash
    python train.py
    ```

3. **Run the Inference Script:**

    To perform automated fast cuts on an audio file within Adobe Premiere Pro, use the premiere script:

    ```bash
    python premiere.py
    ```

## Notes

- **Premiere Pro Integration:** Works directly with Adobe Premiere Pro sequences. This requires installing the Pymiere extension. Follow the [Pymiere Installation Tutorial](https://github.com/qmasingarbe/pymiere?tab=readme-ov-file#installation) to set this up.
- **Audio Format & Predictions:** The inference script expects the input audio file to be in MP3 format and resamples it to 16kHz mono. The script uses the pre-trained model to predict cut points, printing them with their corresponding timestamps.
- **Dependencies:** Make sure to install all required dependencies and have the Conda environment properly set up to avoid issues. Ensure `model.keras` is correctly generated or downloaded before running the inference script.

---

<div align="center">  
    <img src="https://user-images.githubusercontent.com/40668801/42043955-fbb838a2-7af7-11e8-9795-7f890e871d13.png" alt="Tensorflow Logo" width="50px">
</div>
