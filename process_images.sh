#!/bin/bash

FOLDER_PATH="$1"

python process_images.py --image_folder "$FOLDER_PATH"
python mtcnn_detector.py --image_folder "$FOLDER_PATH"