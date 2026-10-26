
import pandas as pd
import matplotlib.pyplot as plt

# Load the model comparison results
df = pd.read_csv("reports/model_comparison.csv")

# Compare fraud precision, recall and F1 score
metrics = [
    "Fraud Precision",
    "Fraud Recall",
    "Fraud F1 Score"
]

ax = df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Fraud Detection Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Model")
ax.set_ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.legend(loc="best")
plt.tight_layout()

plt.savefig(
    "reports/model_comparison.png",
    dpi=300
)
plt.show()

print("Model comparison graph saved successfully!")
