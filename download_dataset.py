from ucimlrepo import fetch_ucirepo
import pandas as pd

dataset = fetch_ucirepo(id=601)

X = dataset.data.features
y = dataset.data.targets

df = pd.concat([X, y], axis=1)

df.to_csv("data/raw/ai4i2020.csv", index=False)

print(f"Dataset saved to data/raw/ai4i2020.csv")
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nTarget distribution:\n{df['Machine failure'].value_counts()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
