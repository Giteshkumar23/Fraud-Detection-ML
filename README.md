
# Fraud Detection Using Machine Learning

A machine-learning project that detects potentially fraudulent credit-card transactions by comparing four algorithms: Logistic Regression, Decision Tree, Random Forest, and Isolation Forest.

## 1. Project Overview

This project demonstrates an end-to-end machine-learning workflow for credit-card fraud detection, including data inspection, model training, evaluation, feature-importance analysis, model persistence, and a local Flask prediction API.

**Project type:** Educational machine-learning prototype  
**Primary model:** Random Forest  
**API framework:** Flask

## 2. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Flask
- Joblib

## 3. Dataset

The project uses the Credit Card Fraud Detection dataset distributed through Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download the dataset and place the CSV file at:

`data/creditcard.csv`

The dataset contains 284,807 transactions and 31 columns. The `Class` column is the target:

- `0`: Legitimate transaction
- `1`: Fraudulent transaction

The dataset is highly imbalanced. Do not use accuracy alone to judge model performance.

## 4. Project Structure

```text
Fraud-Detection-ML/
├── data/
│   └── creditcard.csv
├── models/
│   └── random_forest.joblib
├── reports/
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── model_comparison.csv
│   └── model_comparison.png
├── scr/
│   ├── check_data.py
│   ├── train_models.py
│   ├── plot_feature_importance.py
│   ├── plot_model_comparison.py
│   └── test_api.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Prerequisites

- Python installed on your computer
- pip package installer
- A terminal such as PowerShell or Command Prompt

Use the Python version and package versions compatible with `requirements.txt`.

## 6. Installation and Setup

Open a terminal in the project root directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Ensure the dataset is available at `data/creditcard.csv` before running the data inspection or training scripts.

## 7. Inspect the Dataset

```bash
python scr/check_data.py
```

This script checks the dataset structure, target distribution, and missing values.

## 8. Train and Evaluate the Models

```bash
python scr/train_models.py
```

The training script compares Logistic Regression, Decision Tree, Random Forest, and Isolation Forest. It saves the trained Random Forest model and evaluation outputs according to the implementation.

## 9. Generate Visualizations

Run:

```bash
python scr/plot_feature_importance.py
python scr/plot_model_comparison.py
```

The generated reports include feature-importance and model-comparison visualizations.

## 10. Run the Flask API

Start the application:

```bash
python app.py
```

Open the following URL in your browser:

http://127.0.0.1:5000/

The root endpoint provides API status information.

### Prediction Endpoint

- Method: `POST`
- Endpoint: `/predict`
- Content type: `application/json`

The request must contain all 30 model input features: `Time`, `V1` through `V28`, and `Amount`.

The response includes the predicted class, a human-readable result, and an estimated fraud probability.

## 11. Test the API

Keep the Flask server running. Open a second terminal, activate the virtual environment, and run:

```bash
python scr/test_api.py
```

The test script sends a historical transaction to the local API and checks the response.

## 12. Model Evaluation

In the recorded experiment, Random Forest achieved the highest fraud-class F1-score among the four tested models.

| Metric | Random Forest |
|---|---:|
| Accuracy | 0.9994 |
| Fraud precision | 0.8316 |
| Fraud recall | 0.8061 |
| Fraud F1-score | 0.8187 |

These values are from one stratified train-test split and are not a guarantee of future performance.

## 13. Limitations

- The project uses a historical dataset and a single train-test split.
- The input features `V1` through `V28` are anonymized.
- Feature importance provides a global view rather than a complete explanation of an individual prediction.
- The Flask API is a local prototype and is not integrated with a live payment system.
- Fraud probabilities may require calibration and additional validation.

## 14. Future Improvements

- Repeated stratified cross-validation
- Precision-recall curve analysis and threshold tuning
- SHAP-based explanations
- Automated API tests
- Monitoring and evaluation on newer data
- Security hardening for any future deployment

## 15. Disclaimer

This project is intended for educational and demonstration purposes. It must not be used as a production financial fraud-detection system without appropriate validation, security controls, monitoring, and domain-specific review.