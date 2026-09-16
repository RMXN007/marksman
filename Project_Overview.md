# Airgun Target Analysis: Bullet Hole Detection

## 🎯 Project Overview
This project focuses on building a computer vision system designed to detect and localize extremely small airgun bullet impacts on paper targets. It was developed as a rapid, end-to-end college project demonstration.

## 📖 The Purpose & Use Case
In a live shooting range environment, specifically for "Marksman" competitions or training, it is crucial to accurately locate where bullets hit the target to calculate a shooter's score based on standard ISSF target rings. 

The unique challenge in this project is the size and nature of the objects being detected: 4.5mm / .177 caliber air rifle or air pistol pellets leave tiny holes. Furthermore, the model must handle varying environmental factors, such as:
- Lighting condition changes
- Shadows overlapping the holes
- Varying target paper textures
- Color contrast differences (white tears on a black bullseye vs. dark shadows on white paper)

## 📊 Dataset Preparation
The model was trained on a custom dataset collected directly from the shooting range.
- **Total Images:** 518 raw target images
- **Total Annotations:** 4,042 individual bullet holes
- **Data Split:** The dataset was automatically structured into an 80% Training (414 images) and 20% Validation (104 images) split.
- **Classes:** A single class (`0: bullet_hole`) was defined in the `data.yaml` configuration.

Because the initial annotations were generated via mathematical heuristics (OpenCV edge/shadow detection) rather than flawless manual labeling, the dataset contains a small margin of error (e.g., overlapping holes or printed numbers on the target occasionally confused as holes).

## 🧠 Model & Training Process
To achieve a fast and practical working model for a demonstration, the project utilizes the **YOLO (You Only Look Once)** architecture.

- **Model Used:** YOLOv8 Nano (`yolov8n.pt`). This is the smallest and fastest model in the Ultralytics lineup, heavily optimized for speed while maintaining respectable accuracy.
- **Transfer Learning:** Instead of training from scratch, the model loaded pre-trained weights to speed up convergence.
- **Hardware Optimization:** Because the system was trained on a CPU-only setup, the image sizes were downscaled from their original 1600x1200 resolution to 320x320. 
- **Training Duration:** The model was trained for 10 Epochs, resulting in a training time of roughly 14 minutes.
- **Augmentations:** To make the model robust against a relatively small dataset, subtle augmentations were applied (brightness adjustments, translations, scaling, and rotations).

## 📈 Results
Despite the downscaled image sizes for fast CPU processing and the preliminary nature of the dataset labels, the model successfully learned the visual characteristics of bullet holes and performs practical detection suitable for a college presentation.

- **Final Model Weights:** `runs/bullet_hole_detection-2/weights/best.pt`
- **Output:** The model predicts bounding box coordinates along with a confidence score for every bullet hole it identifies on a given target.
