# Object Detection and Tracking System

This project implements an object detection and tracking system using YOLOv8 and STRONGSORT algorithms. The system can detect, track, and count objects in a video feed from a camera. Additionally, it allows users to define custom counting lines and areas using mouse clicks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SpawnedNPC/PersonCountingApp.git
    cd PersonCountingApp
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dependencies

- `cv2`
- `ultralytics`
- `supervision`
- `numpy`
- `torch`
- `strong_sort`

Ensure you have installed all the necessary dependencies listed in `requirements.txt`.

After installing the requirements follow the `steps to setup.docx` for further setup.

## Usage

1. Start the program:
    - If you want to use yolov8s.pt (you can replace with any other yolov8 model including custom model) and YOLOv8's object tracking:

        ```bash
        python3 Counting_appV2.py
        ```
    - If you want to use yolov8s.pt (you can replace with any other yolov8 model including custom model) and StrongSORT object tracking:

        ```bash
        python3 Counting_appV3.py
        ```
    - If you are using Stereo Lab's ZED camera:
        ```bash
        python3 Counting_appV4.py
        ```

2. The video feed will open in a new window. Use the following keys for interaction:
    - **Left Click**: Select coordinates for counting lines or areas.
    - **l**: Initialize line counter.
    - **a**: Initialize area counter.
    - **ESC**: Exit the program.
    - **Backspace**: Remove previous points.
    - **c**: Clear all counters and points.

3. Observe the counting results in the video feed.

## Features

- **Object Detection**: Detects objects using YOLOv8.
- **Object Tracking**: Tracks detected objects using STRONGSORT.
- **Line Counting**: Counts objects crossing user-defined lines.
- **Area Counting**: Counts objects within user-defined areas.


## Acknowledgements

Special thanks to the authors of the following packages for their excellent work:
- [ultralytics](https://github.com/ultralytics/yolov5)
- [supervision](https://github.com/roboflow/supervision)
- [STRONGSORT](https://github.com/dyabel/strong_sort)
