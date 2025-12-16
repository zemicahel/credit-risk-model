import pandas as pd
import mlflow
from mlflow.sklearn import log_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from src.data_processing import prepare_training_data, build_preprocessor
import joblib
import os

def train_model():
    # 1. Prepare Data
    print("Loading and preparing data...")
    
    data_path = 'data/raw/data.csv'
    # Fallback to absolute path just in case
    if not os.path.exists(data_path):
        data_path = '/Users/mac/Documents/kaim/credit-risk-model/data/raw/data.csv'
    
    if not os.path.exists(data_path):
        print(f"CRITICAL ERROR: Data file not found at {data_path}")
        return

    data = prepare_training_data(data_path)
    
    # --- FIX IS HERE ---
    # We only drop 'is_high_risk'. 
    # 'Cluster' was already removed inside data_processing.py
    X = data.drop(columns=['is_high_risk']) 
    y = data['is_high_risk']

    # Define feature groups
    categorical_features = ['ProductCategory', 'Channel']
    numeric_features = [col for col in X.columns if col not in categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Setup MLflow
    mlflow.set_experiment("Credit_Risk_Model_Experiment")

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    # Define Hyperparameters
    param_grids = {
        "LogisticRegression": {'classifier__C': [0.1, 1, 10]},
        "RandomForest": {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [5, 10]}
    }

    best_overall_model = None
    best_overall_auc = 0

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Build Full Pipeline
            preprocessor = build_preprocessor(numeric_features, categorical_features)
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])

            # Grid Search
            grid_search = GridSearchCV(clf, param_grids[model_name], cv=3, scoring='roc_auc')
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            
            # Predict
            y_pred = best_model.predict(X_test)
            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(X_test)[:, 1]
            else:
                y_prob = [0] * len(y_test)

            # Metrics
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            auc = float(roc_auc_score(y_test, y_prob))

            # Log to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc})
            log_model(best_model, "model")

            print(f"{model_name} - AUC: {auc:.4f}")

            if auc > best_overall_auc:
                best_overall_auc = auc
                best_overall_model = best_model

    if best_overall_model:
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_overall_model, "models/best_model.pkl")
        print(f"Best model saved with AUC: {best_overall_auc}")

if __name__ == "__main__":
    train_model()