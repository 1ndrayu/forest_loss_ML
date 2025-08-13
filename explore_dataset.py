import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('forest_loss_dataset_10000.csv')

print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"{i+1:2d}. {col}")

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nTarget variable distribution:")
print(df['forest_loss_next_period'].value_counts())
print(f"Target balance: {df['forest_loss_next_period'].value_counts(normalize=True)}")

print(f"\nSample statistics:")
print(df.describe())

print(f"\nProtected area flag distribution:")
print(df['protected_area_flag'].value_counts())

print(f"\nPast loss events distribution:")
print(df['past_loss_events'].value_counts())
