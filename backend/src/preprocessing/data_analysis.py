# EDA -Exploratory Data Analysis

import pandas as pd

df = pd.read_csv("src\\data\\insurance.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isna().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nCategorical columns:")
for col in df.select_dtypes('object'):
    print(f"\ncol")
    print(df[col].value_counts())


# Remove duplicate 
print("Before: ", df.shape)

df_clean = df.drop_duplicates()

print("After:", df_clean.shape)

