
import pandas as pd

# Load the transaction dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five transactions:")
print(df.head())

print("\nFraud distribution:")
print(df["Class"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())
