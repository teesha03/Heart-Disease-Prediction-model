# ❤️ Heart Disease Prediction using AWS SageMaker

This project aims to build, train, and deploy a heart disease prediction model using multiple machine learning algorithms, with final model training and deployment on AWS SageMaker.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [AWS SageMaker Integration](#aws-sagemaker-integration)
- [Results](#results)
- [Setup Instructions](#setup-instructions)
- [License](#license)

---

## 📖 Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help reduce mortality. This project uses a dataset of clinical parameters and applies machine learning models to predict the presence of heart disease.

---

## 💻 Tech Stack

- **Languages & Libraries**:
  - Python, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Yellowbrick
- **AWS Services**:
  - SageMaker
  - S3
  - IAM
- **Others**:
  - Jupyter Notebook / Google Colab
  - Git, GitHub

---

## 📊 Exploratory Data Analysis

- Gender-wise distribution of heart disease
- Age vs Max Heart Rate scatter plot
- Correlation heatmap between clinical features

Visuals are generated using `matplotlib`, `seaborn`, and custom color palettes to highlight patterns.

---

## 🛠️ Data Preprocessing

- Converted categorical features to dummy variables (`cp`, `thal`, `slope`)
- Normalized features using **MinMaxScaler**
- Split dataset into 80:20 (train:test)
- Saved processed data as CSV and uploaded to **AWS S3**

---

## 🤖 Machine Learning Models

The following models were trained and compared:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

Each model’s accuracy was evaluated and compared using `accuracy_score`, `classification_report`, and confusion matrices.

---

## ☁️ AWS SageMaker Integration

- Preprocessed datasets are uploaded to **Amazon S3**
- Used **SKLearn Estimator** to train a `RandomForestClassifier` model
- Script (`script.py`) performs:
  - Argument parsing for data paths
  - Model training
  - Evaluation
  - Model serialization using `joblib`
- Trained on instance type: `ml.m5.large` (or `ml.c5.2xlarge`)
- Used Spot Instances to reduce cost

---

## 📈 Results

| Model                 | Accuracy (%) |
|----------------------|--------------|
| Logistic Regression  | ~86%         |
| K-Nearest Neighbors  | ~85%         |
| Support Vector Machine | ~87%       |
| Decision Tree        | ~84%         |
| Random Forest        | **~90%**     |

> Random Forest performed best and was chosen for SageMaker deployment.

---

## 🚀 Setup Instructions

1. **Install Required Libraries**:
    ```bash
    pip install pandas numpy seaborn matplotlib yellowbrick sagemaker boto3
    ```

2. **Configure AWS CLI**:
    ```bash
    aws configure
    ```

3. **Upload Dataset**:
    - Place `heart.csv` in your working directory (or Colab's `/content/`)
    - Data will be uploaded to S3 programmatically.

4. **Run Notebook or Python Script**:
    - For local testing, run all cells in Jupyter Notebook or Colab
    - For training in SageMaker, make sure your IAM role has full SageMaker, S3, and CloudWatch access

5. **Model Script**:
    - The file `script.py` handles training and saving the model
    - Trained using `sagemaker.sklearn.estimator.SKLearn`

---

## 🔒 Note

- Ensure your **AWS_ACCESS_KEY_ID** and **AWS_SECRET_ACCESS_KEY** are not hardcoded in the script.
- Use environment variables or IAM roles attached to your SageMaker notebook/lab for credentials.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🔗 GitHub Repository

📁 [View on GitHub](https://github.com/teesha03/BroadBand_Billing)
