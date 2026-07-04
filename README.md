# 🌾 Rice Grain Classification

An end-to-end computer vision and machine learning project for classifying rice grain images into five different varieties.

## 📌 Project Overview

Rice variety identification is important for food quality assessment, agricultural processing and grain classification.

This project develops an image-based machine learning system that classifies rice grains into five different varieties.

The project includes:

- Image preprocessing using OpenCV
- Image resizing and feature extraction
- Feature scaling
- Support Vector Machine (SVM) classification
- CNN experimentation
- Model serialization using Joblib
- Real-image testing
- Interactive Streamlit web application

## 🌾 Rice Varieties

The model classifies rice grains into:

- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

## 📂 Project Structure

```text
rice-grain-classification/
│
├── models/
│   ├── rice_svm_model.pkl
│   ├── rice_scaler.pkl
│   └── rice_cnn_model.h5
│
├── notebooks/
│   ├── svm.ipynb
│   └── cnn.ipynb
│
├── sample_images/
│   ├── test_rice2.jpg
│   └── test_rice3.png
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## ⚙️ Machine Learning Pipeline

The project follows the following workflow:

1. Load rice grain images
2. Resize images to 64 × 64 pixels
3. Convert images into numerical arrays
4. Flatten image pixels into feature vectors
5. Apply feature scaling
6. Train the SVM classifier
7. Save the trained model and scaler
8. Load the saved model for inference
9. Predict rice varieties from unseen images
10. Deploy the model using Streamlit

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- TensorFlow
- Streamlit
- Joblib
- Jupyter Notebook

## 🚀 Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Navigate to the project directory:

```bash
cd rice-grain-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your web browser.

Upload a rice grain image and click the prediction button to classify the rice variety.

## 📊 Model Output

The trained model predicts one of the following classes:

```text
Arborio
Basmati
Ipsala
Jasmine
Karacadag
```

## 📁 Dataset

The dataset contains images of five rice varieties:

- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

The complete dataset is excluded from this repository because of its large size.

## 🔮 Future Improvements

- Transfer learning using EfficientNet or MobileNet
- Model confidence scores
- Batch image prediction
- Improved image preprocessing
- Cloud deployment
- Mobile application integration

## 👨‍💻 Author

**Pratik Das**

M.Tech  
IIT Kharagpur

## 📄 License

This project is licensed under the MIT License.

## 🚀 Live Demo

Try the deployed application here:

**[Launch Rice Grain Classification App](https://rice-grain-classification-pratik.streamlit.app)**
