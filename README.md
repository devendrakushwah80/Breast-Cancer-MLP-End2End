# 🧠 Breast Cancer Classification using MLP (End-to-End ML Pipeline)

## 📌 Project Overview

This project builds a complete end-to-end Machine Learning pipeline using a Multi-Layer Perceptron (MLP) Neural Network to classify breast cancer tumors as **Malignant** or **Benign**.

The dataset used is the Breast Cancer Wisconsin dataset from sklearn.

---

## 🎯 Objectives

- Perform detailed Exploratory Data Analysis (EDA)
- Visualize feature distributions and correlations
- Detect and remove outliers (IQR Method)
- Perform feature engineering (remove highly correlated features)
- Build an end-to-end ML pipeline
- Apply hyperparameter tuning using GridSearchCV
- Evaluate model using all major classification metrics

---

## 📊 Dataset Information

- Total Samples: 569
- Features: 30 numerical features
- Target Classes:
  - 0 → Malignant
  - 1 → Benign

---

## 🔎 Exploratory Data Analysis (EDA)

- Dataset shape & statistical summary
- Class distribution visualization
- Correlation heatmap
- Feature histograms
- Outlier detection using IQR

---

## ⚙️ Feature Engineering

- Removed highly correlated features (> 0.9)
- Outlier filtering using IQR
- Feature scaling using StandardScaler

---

## 🏗️ Machine Learning Pipeline

Pipeline includes:

1. StandardScaler
2. MLPClassifier
3. GridSearchCV for hyperparameter tuning

Hyperparameters tuned:
- Hidden layer sizes
- Activation function
- Alpha (regularization)
- Learning rate strategy

---

## 📈 Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

---

## 🏆 Results

Typical performance:

- Accuracy: 96–99%
- ROC-AUC: 0.98+
- Very low False Negative rate

---

## 📦 Project Structure

```
Breast-Cancer-MLP-End2End/
│
├── MPL_Cancer_dataset.ipynb
├── README.md
├── requirements.txt
```

---

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the notebook

---

## 🧠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📌 Future Improvements

- Add PCA for dimensionality reduction
- Compare with Logistic Regression & SVM
- Deploy using Streamlit
- Convert to Deep Learning (TensorFlow / PyTorch)

---

## 👨‍💻 Author

Devendra Kushwah  
Machine Learning & AI Enthusiast
