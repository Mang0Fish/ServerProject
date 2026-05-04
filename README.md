# EZPredict - FastAPI ML Training Server

EZPredict is a FastAPI backend server that allows authenticated users to upload CSV files, train machine learning models, save trained models, and make predictions using saved models.

The project also includes JWT authentication, a token-based usage system, model metadata storage, logging, and a simple Streamlit dashboard for viewing user token balances.

---

## Features

- User signup and JWT login
- Token-based usage system
- CSV upload for model training
- Feature selection during training
- Multiple model types:
  - Logistic Regression
  - Linear Regression
  - Random Forest
  - SVM
  - CatBoost
- Model evaluation metrics
- Model metadata saved as JSON
- Prediction using saved models
- Model listing endpoint
- Admin user management
- Streamlit dashboard for user tokens
- Server logging to `server.log`

---

## Tech Stack

- Python 3.10+
- FastAPI
- Pydantic
- PostgreSQL
- psycopg2
- Pandas
- Scikit-learn
- CatBoost
- Joblib
- PyJWT
- Streamlit

---

## Project Structure

```text
.
├── main.py
├── models.py
├── bl.py
├── dal.py
├── auth.py
├── security.py
├── dashboard.py
├── routers/
│   └── token.py
├── ml/
│   ├── utils.py
│   ├── training.py
│   ├── preprocessing.py
│   ├── validation.py
│   └── ml_models/
├── requirements.txt
└── README.md