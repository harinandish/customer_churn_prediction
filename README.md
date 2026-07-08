# 📊 Customer Churn Prediction

A complete machine learning pipeline that predicts which customers are likely to cancel their subscription, based on demographic, usage, and engagement features. Built with a synthetic dataset designed to mimic realistic churn patterns.

## 📌 Overview

Customer churn — when a subscriber cancels their service — is costly for subscription-based businesses. This project builds an end-to-end pipeline that:
1. Generates a realistic synthetic customer dataset
2. Engineers and preprocesses demographic, usage, and engagement features
3. Compares multiple ML models using cross-validation
4. Selects the best model based on ROC-AUC
5. Evaluates final performance with a confusion matrix and ROC curve
6. Saves the trained pipeline for future predictions on new customers

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, joblib
- **Approach:** Full ML pipeline (preprocessing + model, saved as one object)

## 📂 Project Structure
```
customer-churn-prediction/
│
├── data/
│   └── customer_churn.csv        # Synthetic dataset (generated)
├── models/
│   └── churn_model.pkl            # Trained pipeline (generated)
├── output/
│   ├── confusion_matrix.png       # Generated after training
│   ├── roc_curve.png              # Generated after training
│   └── predictions.csv            # Generated after running predict.py
├── generate_dataset.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Usage

### Step 1: Generate the synthetic dataset
```bash
python generate_dataset.py
```
This creates `data/customer_churn.csv` with 3,000 customer records and an engineered churn label based on tenure, usage, engagement, and subscription features.

### Step 2: Train and evaluate the model
```bash
python train.py
```
This will:
- Preprocess numeric and categorical features (scaling + one-hot encoding)
- Run 5-fold cross-validation comparing Logistic Regression, Random Forest, and Gradient Boosting
- Select the best model by ROC-AUC
- Evaluate on a held-out test set
- Save the confusion matrix and ROC curve to `output/`
- Save the trained pipeline to `models/churn_model.pkl`

### Step 3: Predict on new customers
```bash
python predict.py --input data/customer_churn.csv --output output/predictions.csv
```
Loads the saved pipeline and outputs a churn prediction and probability for each customer record.

## 🧠 Features Used

| Category | Features |
|---|---|
| **Demographic** | age, gender, region |
| **Account** | tenure_months, subscription_type, monthly_charges |
| **Usage** | avg_sessions_per_week, avg_session_duration_min, support_tickets_raised |
| **Engagement** | email_open_rate, app_login_frequency, has_autopay, used_promo_offer |

## 📈 Learning Outcomes
- Feature engineering across demographic, usage, and engagement data
- Building preprocessing pipelines with `ColumnTransformer` (scaling + one-hot encoding)
- Model selection via cross-validation across multiple algorithms
- Evaluating classifiers with ROC-AUC, confusion matrices, and classification reports
- Packaging preprocessing and model together in a single deployable pipeline

## 📊 Model Performance
The pipeline compares Logistic Regression, Random Forest, and Gradient Boosting using 5-fold stratified cross-validation on ROC-AUC, then evaluates the best-performing model on a held-out test set. Exact scores will vary slightly by run since the dataset is synthetically generated with randomized noise.

## 🚧 Known Limitations
- Dataset is synthetic, so patterns are simplified compared to real-world churn behavior
- Class imbalance (churners are a minority) affects recall for the churn class
- No hyperparameter tuning is performed beyond default model comparison — a natural next step (e.g. GridSearchCV) could improve results

## 📜 License
This project is for educational purposes, built as part of a hands-on data science and machine learning pipeline learning path.

## 🙋 Author
Mahendra
