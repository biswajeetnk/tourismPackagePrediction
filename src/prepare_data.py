import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/tourism.csv")

# Remove unnecessary column
df = df.drop(columns=["CustomerID"])

# Split features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Combine features and target
train_df = X_train.copy()
train_df["ProdTaken"] = y_train

test_df = X_test.copy()
test_df["ProdTaken"] = y_test

# Create artifacts folder
os.makedirs("artifacts", exist_ok=True)

# Save datasets
train_df.to_csv("artifacts/train.csv", index=False)
test_df.to_csv("artifacts/test.csv", index=False)

print("Train and test datasets saved successfully.")
