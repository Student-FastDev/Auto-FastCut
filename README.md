# Auto-FastCut

**Auto-FastCut** is an AI-based Python script designed to automate fast cuts in Adobe Premiere Pro. This project uses a pre-trained deep learning model to identify and cut specific audio segments from a given audio track. The script integrates with TensorFlow and TensorFlow I/O to process audio files and predict cut points.

## Features

- **Automated Fast Cuts:** Automatically identify and cut audio segments based on a pre-trained model.
- **Audio Processing:** Load and preprocess audio files for model inference.
- **Integration with Adobe Premiere Pro:** Seamlessly work with Adobe Premiere Pro sequences.

## Prerequisites

To run the Auto-FastCut script, you need:

- **Python:** Ensure you have Python 3.6 or higher installed.
- **Dataset:** You need a dataset of audio files organized into `Positive` and `Negative` directories. The `Positive` directory should contain audio files that the model should detect, while the `Negative` directory should contain audio files that are not of interest.

## Installation

1. **Clone the Repository:**

   ```sh
   git clone https://github.com/yourusername/auto-fastcut.git
   cd auto-fastcut
   ```

2. **Set Up the Conda Environment:**

   Create a Conda environment using the provided `minecraftAI.yml` file:

   ```sh
   conda env create -f minecraftAI.yml
   ```

   Activate the Conda environment:

   ```sh
   conda activate minecraftAI
   ```

3. **Install Required Packages:**

   The `minecraftAI.yml` file includes most of the required packages. Ensure that the following packages are installed:

   ```sh
   pip install tensorflow tensorflow-io soundfile matplotlib colorama
   ```

4. **Prepare Your Dataset:**

   Place your dataset into the following structure:

   ```
   Dataset/
   ├── Positive/
   │   └── (positive audio files)
   └── Negative/
       └── (negative audio files)
   ```

5. **Install Pymiere Extension:**
   
   - follow this guide: [Pymiere Installation Tutorial](https://github.com/qmasingarbe/pymiere?tab=readme-ov-file#installation)

6. **Ensure the Following Files Are in Place:**

   - `model.keras`: The pre-trained model file. You can train the model using the provided training script or download a pre-trained model if available.

## Usage

1. **Run the Training Script (if necessary):**

   If you need to train the model from scratch, use the provided training script. Ensure your dataset is correctly set up and run:

   ```sh
   python train.py
   ```

   This will save a model file named `model.keras`.

2. **Run the Inference Script:**

   To perform automated fast cuts on an audio file, use the premiere script:

   ```sh
   python premiere.py
   ```


## Notes

- **Audio File Format:** The inference script expects the input audio file to be in MP3 format and resamples it to 16kHz mono.
- **Model Predictions:** The script uses the pre-trained model to predict cut points in the audio file. Predictions are printed with their corresponding timestamps.
- **Dependencies:** Make sure to install all required dependencies and have the Conda environment properly set up to avoid issues.

## Contributing

Feel free to contribute to the project by submitting issues, bug reports, or pull requests. For significant changes, please open an issue first to discuss what you would like to change.

**Auto-FastCut** is a project developed to streamline video editing processes using AI. Happy cutting!

<div align="center">  
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Adobe_Premiere_Pro_CC_icon.svg/512px-Adobe_Premiere_Pro_CC_icon.svg.png?20210729021549" alt="Premiere Logo" width="50px">
<img src="https://user-images.githubusercontent.com/40668801/42043955-fbb838a2-7af7-11e8-9795-7f890e871d13.png" alt="Tensorflow Logo" width="50px">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Adobe_Creative_Cloud_rainbow_icon.svg/1050px-Adobe_Creative_Cloud_rainbow_icon.svg.png" alt="Adobe Logo" width="50px">  
</div>  
