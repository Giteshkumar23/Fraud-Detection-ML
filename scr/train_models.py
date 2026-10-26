
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)

# 1. Load dataset
df = pd.read_csv("data/creditcard.csv")

# 2. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 4. Build a preprocessing and model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

# 5. Train the model
print("Training Logistic Regression...")
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

# 7. Evaluate results
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0,
))

print(
    "Average Precision Score:",
    round(average_precision_score(y_test, y_scores), 4),
)

print("\nModel training completed!")



from sklearn.tree import DecisionTreeClassifier

print("\n" + "=" * 40)
print("Training Decision Tree...")

tree_model = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=8,
    random_state=42,
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, tree_pred))

print("\nDecision Tree Classification Report:")
print(classification_report(
    y_test,
    tree_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0,
))

from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 40)
print("Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    rf_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0,
))

from sklearn.ensemble import IsolationForest

print("\n" + "=" * 40)
print("Training Isolation Forest...")

iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.00173,
    random_state=42,
    n_jobs=-1,
)

iso_model.fit(X_train)

# Isolation Forest predicts:
#  1 = normal
# -1 = anomaly
iso_raw = iso_model.predict(X_test)

# Convert to the dataset's labels:
#  0 = legitimate
#  1 = fraud
iso_pred = (iso_raw == -1).astype(int)

print("\nIsolation Forest Confusion Matrix:")
print(confusion_matrix(y_test, iso_pred))

print("\nIsolation Forest Classification Report:")
print(classification_report(
    y_test,
    iso_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0,
))


import os
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Create folders if they do not exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Compare all four models
predictions = {
    "Logistic Regression": y_pred,
    "Decision Tree": tree_pred,
    "Random Forest": rf_pred,
    "Isolation Forest": iso_pred,
}

results = []

for name, predictions_array in predictions.items():
    results.append({
        "Model": name,
        "Accuracy": round(
            accuracy_score(y_test, predictions_array), 4
        ),
        "Fraud Precision": round(
            precision_score(
                y_test, predictions_array, zero_division=0
            ), 4
        ),
        "Fraud Recall": round(
            recall_score(
                y_test, predictions_array, zero_division=0
            ), 4
        ),
        "Fraud F1 Score": round(
            f1_score(
                y_test, predictions_array, zero_division=0
            ), 4
        ),
    })

comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="Fraud F1 Score", ascending=False
)

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(comparison.to_string(index=False))

# Save the comparison report
comparison.to_csv(
    "reports/model_comparison.csv", index=False
)

# Save Random Forest as our current candidate model
joblib.dump(rf_model, "models/random_forest.joblib")

# Save Random Forest feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_,
}).sort_values("Importance", ascending=False)

importance.to_csv(
    "reports/feature_importance.csv", index=False
)

print("\nComparison saved to reports/model_comparison.csv")
print("Model saved to models/random_forest.joblib")
print("Feature importance saved to reports/feature_importance.csv")



