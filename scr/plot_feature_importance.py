
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load feature importance generated during model training
df = pd.read_csv("reports/feature_importance.csv")

# Select the 15 most important features
top_features = df.sort_values(
    "Importance", ascending=False
).head(15).sort_values("Importance")

# Create the graph
plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features - Fraud Detection Random Forest")
plt.tight_layout()

# Save the graph
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/feature_importance.png", dpi=300)
plt.show()

print("Graph saved to reports/feature_importance.png")
