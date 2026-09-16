# 🚀 Bullet Hole Detection Application Guide

This guide will walk you through how to start and use the interactive web application built for the Bullet Hole Detection project.

## 🛠 Prerequisites
Before running the application, ensure that you have Python installed, along with the required libraries. If you haven't installed them yet, you can do so by opening your **Command Prompt** or **PowerShell** and running:
```bash
python -m pip install ultralytics streamlit pandas opencv-python
```

---

## 💻 Starting the Application

The application is built using **Streamlit**, which creates a local web server to host the interactive interface.

1. Open your **Command Prompt** or **PowerShell**.
2. Run the following command to launch the app:
   ```bash
   python -m streamlit run "C:\Users\acer\.gemini\antigravity-ide\brain\7d9a5e22-b95f-409a-bcef-008960aa871c\scratch\app.py"
   ```
3. A new tab will automatically open in your default web browser (usually at `http://localhost:8501`).

---

## 🎯 Using the Web Application

Once the web page has loaded in your browser, using the tool is simple:

### 1. Verify the Model
At the top of the page, there is a text box containing the path to the trained model (`best.pt`). 
- If the model is found successfully, you will see a green **"Model loaded successfully!"** message.
- If it throws a warning, verify that the path to `best.pt` is correct for your PC.

### 2. Upload a Target Image
- Click on the **"Browse files"** button (or drag and drop an image into the box).
- Select any `.jpg` or `.png` image of an airgun target from your computer.

### 3. View the Results
As soon as the image uploads, the YOLO model will automatically run its inference in the background. 
Within seconds, the application will display:
- **The Annotated Image:** Your uploaded target image, but with red bounding boxes drawn around every detected bullet hole.
- **Total Count:** A large text readout of the total number of bullet holes detected on the paper.
- **Detection Details Table:** A detailed table listing the exact pixel coordinates (X1, Y1, X2, Y2) of every single bounding box, along with the model's confidence percentage for that specific hole.

---

## ⌨️ Advanced: Command-Line Inference

If you prefer to test images without the graphical user interface, you can use the command-line inference script.

Open Command Prompt and run:
```bash
python "C:\Users\acer\.gemini\antigravity-ide\brain\7d9a5e22-b95f-409a-bcef-008960aa871c\scratch\inference.py" --image "PATH_TO_YOUR_IMAGE.jpg" --model "C:\Users\acer\Desktop\anu project\runs\bullet_hole_detection-2\weights\best.pt"
```

*Make sure to replace `"PATH_TO_YOUR_IMAGE.jpg"` with the actual location of an image you want to test!* This will output the bounding box coordinates directly into your terminal and save a result image locally.
