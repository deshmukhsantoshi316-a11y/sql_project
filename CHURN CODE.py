
# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. LOAD DATA

df = pd.read_csv('churn.csv')

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary:")
print(df.describe())

# 3. DATA CLEANING

# Convert TotalCharges to numeric 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert Churn to numeric
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Check null values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. FEATURE UNDERSTANDING

print("\nUnique Values:")
for col in df.columns:
    print(col, ":", df[col].nunique())

# 5. EXPLORATORY DATA ANALYSIS

# Overall Churn Rate
churn_rate = df['Churn'].mean() * 100
print("\nChurn Rate: ", churn_rate)

# Churn by Contract
print("\nChurn by Contract:")
print(df.groupby('Contract')['Churn'].mean()*100)

# Churn by Payment Method
print("\nChurn by Payment Method:")
print(df.groupby('PaymentMethod')['Churn'].mean()*100)

# Churn by Monthly Charges
print("\nMonthly Charges vs Churn:")
print(df.groupby('Churn')['MonthlyCharges'].mean())

# Churn by Tenure
print("\nTenure vs Churn:")
print(df.groupby('Churn')['tenure'].mean())

# 6. DATA VISUALIZATION

# Churn Count
plt.figure()
sns.countplot(x='Churn', data=df)
plt.title("Churn Count")
plt.show()

# Contract vs Churn
plt.figure()
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract")
plt.xticks(rotation=30)
plt.show()

# Payment Method vs Churn
plt.figure()
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title("Churn by Payment Method")
plt.xticks(rotation=45)
plt.show()

# Monthly Charges Distribution
plt.figure()
sns.histplot(df['MonthlyCharges'], kde=True)
plt.title("Monthly Charges Distribution")
plt.show()

# Boxplot Monthly Charges vs Churn
plt.figure()
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# 7. CORRELATION HEATMAP

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 8. FINAL INSIGHTS (PRINT)

print("\n--------FINAL INSIGHTS ------")
print("1. Month-to-month customers have higher churn")
print("2. High monthly charges customers churn more")
print("3. Low tenure customers are at high risk")
print("4. Some payment methods have higher churn")


print("\nProject Completed Successfully")




