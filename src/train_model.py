import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

# Load train and test data
train_df = pd.read_csv("artifacts/train.csv")
test_df = pd.read_csv("artifacts/test.csv")

# Split features and target
X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_test = test_df.drop("ProdTaken", axis=1)
y_test = test_df["ProdTaken"]

# Encode categorical columns
categorical_columns = X_train.select_dtypes(include="object").columns

label_encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()

    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])

    label_encoders[col] = encoder

# XGBoost model
model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Metrics
print("Best Parameters:")
print(grid_search.best_params_)

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_prob))

# Save model files
joblib.dump(best_model, "best_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(list(X_train.columns), "feature_columns.pkl")

print("\nBest model saved successfully.")
