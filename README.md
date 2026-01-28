# Hand Detection Project

## Overview
This **reel-time hands tracker project** using MediaPipe, allows you to control your computer's control only by hands. While the right hand can be used to control your mouse, your left hand can be assigned to some key controllers thanks to a basic MLP classifier.

## Features
1) **Real-time classifier** with features to help you build your own MLPClassifier model with any hand-sign you would like :
    - **Collecting** images to train your model with a simple python script
    - **Labelling** images to train your model with tkinter librairy
    - **Training** the model with your own data throught the dedicated jupyter Notebook

2) **Mouse control** with your eyes or hands
    - **Basic hand's rotation detection** to move your cursor up and down or sideways direclty implemented in the main python script
    - **Eye movement detection** with Mediapipe to move your cursor with your gaze. *Note that this feature is not implemented direclty in the main python script and is to be explored in a dedicated one*

## Installation

### Prerequisites
    - Python 3.8+
    - pip

### Steps
    1. Clone the repository:
        ```bash
        git clone <repository-url>
        cd MediaPipe
        ```

    2. Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```

    3. Verify installation:
        ```bash
        python -c "import mediapipe; print(mediapipe.__version__)"
        ```

    **Note:** tkinter is included with Python 3.8+. If you're on macOS, you may need to install it separately via Homebrew: `brew install python-tk@3.x`

## Project Structure
```
HandTracker/
├── data/       #this is the data folder structured when filled
│   ├── images/
│   ├── landmarks/
│   └── labels.json/
├── models/
│   ├── mlp_classifier.ipynb
│   ├── scaler.pkl
│   └── mlp_classifier.pkl
├── scripts/
│   ├── collecting_signs.py
│   ├── labelling_dataset.py
│   ├── hands_obj.py
│   └── main.py       #tracker file
├── README.md
└── requirements.txt
```

## Getting Started

Once installations are complete, you can simply start the hand-tracker with the command `python main.py`. The current classification runs with a model already included in the repository, however you can easily create your own model if desired.

## Demo
A **demo video** is available direclty in this repository