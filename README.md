# Brain-Tumor-Classification

## Project Description

The Brain Tumor Classification web app is an end-to-end machine learning solution designed to classify brain tumors from MRI scans. The project involves building a complete pipeline, from data preprocessing and model training to deploying a web app for model inference. The goal is to assist medical professionals in diagnosing brain tumors by providing accurate and reliable predictions.

## Features

- **Image Upload:** Upload MRI scan images in JPG, PNG, or JPEG formats.
- **Model Selection:** Choose between a custom CNN model and a transfer learning model (Xception).
- **Prediction:** Classify brain tumors into four categories: Glioma, Meningioma, No Tumor, and Pituitary.
- **Saliency Maps:** Generate and display saliency maps to highlight the regions of the MRI scan that the model focuses on for predictions.
- **Confidence Scores:** Display the confidence scores for each prediction.
- **Explanations:** Provide explanations for the model's predictions using a generative AI model.

## Tech Stack Used

- **Python:** Core programming language for the project.
- **Pandas:** Library for data manipulation and analysis.
- **Matplotlib:** Visualization library for plotting graphs and understanding data distribution.
- **scikit-learn:** Library for machine learning algorithms and model evaluation.
- **Streamlit:** Framework for building and deploying the web app.
- **TensorFlow:** Open-source library for machine learning and deep learning.
- **OpenCV (cv2):** Library for computer vision tasks.
- **Plotly:** Interactive graphing library.

## How to Run the Project

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the [Kaggle dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and unzip it.
4. In the `data/` run the Jupyter Notebook file to get the model.
5. Run the Streamlit app with `streamlit run app.py`.
6. Access the web app at `http://localhost:8501` in your web browser.

## Project Structure

- `app.py`: Streamlit app script to serving the model.
- `data/`: Directory containing the dataset.
- `models/`: Directory where the jupyter notebook was used for model development and saved trained models.
- `.streamlit/`: Directory containing the theme configuration.
- `requirements.txt`: List of dependencies.
