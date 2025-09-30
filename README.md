# 🧠 CNN-Based Stress Detection Project


## Project Structure

```python
project_2/
│
├── archive/
│   ├── train/               # Training images organized in class folders
│   └── test/                # Testing images organized in class folders
│
├── src/
│   ├── data_loader.py       # Loads dataset and applies augmentation
│   ├── evaluate.py          # Evaluates model performance
│   ├── model.py             # CNN architecture / transfer learning
│   ├── train.py             # Training script (from scratch)
│   ├── predict.py           # Prediction script for new images
│   └── train_transfer.py    # Transfer learning training script
│
├── saved_models/
│   ├── cnn_stress_model.h5  # Trained CNN model
│
├── app.py                   # Streamlit web application
├── run.py                   # Optional script to run app or pipeline
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation



## 📌 Project Overview

This project is a **facial expression-based stress detection system**. The goal is to automatically detect stress levels in a person using a deep learning model trained on facial images.  

We built a **Convolutional Neural Network (CNN)** to classify facial expressions into emotions like *happy, angry, sad, neutral*, etc., which helps estimate stress.  

The model is deployed as a **web application using Streamlit**, so users can upload images and instantly see predictions.

🔗 [Live demo of the project](https://cnnstressdetectionproject-lgdtggbwp6s2njmp5qpadg.streamlit.app//)


## Workflow
![Workflow Diagram](workflow.png)

---

## 🔍 What We Did Step by Step

1. **Data Collection**
   - Collected facial expression images from a dataset.
   - Images were labeled with emotions.

2. **Data Preprocessing**
   - Converted images to **grayscale** for model training.
   - Applied **data augmentation** (like rotation, zoom, flips) to increase dataset size and make the model robust.

3. **Model Development**
   - Built a **CNN from scratch** to classify emotions.
   - Used layers like Conv2D, MaxPooling, and Dense.
   - Trained the model on the preprocessed images.

4. **Training and Evaluation**
   - Trained the model using 50+ epochs.
   - Monitored validation accuracy and used **EarlyStopping** to prevent overfitting.
   - Achieved around **60% accuracy** on original grayscale images.

5. **Improving Accuracy**
   - Applied **data augmentation** to increase dataset variability.
   - Experimented with **transfer learning** using pre-trained models (like VGG16) to leverage existing knowledge and improve performance.

6. **Prediction**
   - Built a Python script to load the trained model and predict stress from **new images**.
   - Converted color images to grayscale if necessary to match training format.

7. **Deployment**
   - Used **Streamlit** to create an interactive web app.
   - Users can upload images and get **real-time stress predictions**.
   - Pushed the project to **GitHub** and used **Git LFS** for storing the trained model (`.h5` file).
   - Deployed successfully on **Streamlit Cloud**.

---

## 🛠 Technologies Used

- **Python 3.x**
- **Deep Learning**: TensorFlow / Keras
- **Web Deployment**: Streamlit
- **Data Processing**: NumPy, OpenCV, PIL
- **Version Control**: Git & GitHub (with Git LFS for large files)

---

## 🚀 How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Debbatisudheer/cnn_stress_detection_project.git
   cd cnn_stress_detection_project

   
Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Run the Streamlit app:


streamlit run app.py
🎯 Key Features
Real-time stress detection from facial expressions

Handles image uploads easily

Uses CNN and transfer learning for better accuracy

Interactive web app with Streamlit

📄 License
This project is licensed under MIT License - see LICENSE for details.







