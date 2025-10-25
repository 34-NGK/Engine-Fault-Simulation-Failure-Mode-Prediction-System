
import pandas as pd
import numpy as np

# Load and normalize column names for consistency
df = pd.read_csv("../data/engine_failure_dataset.csv")
df.columns = [col.strip().replace(" ", "_").title() for col in df.columns]

print("Dataset Loaded Successfully")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# === 2. Initial Quality Checks ===
print("Missing Values per Feature:\n", df.isnull().sum(), "\n")
print("Duplicate Records:", df.duplicated().sum(), "\n")

# === 3. Remove Duplicates ===
df = df.drop_duplicates()
print(f"Duplicates Removed. New shape: {df.shape}\n")

# === 4. Handle Missing Values ===
# Numeric columns → fill with median; Categorical → fill with mode
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(" Missing values handled using median/mode imputation.\n")

# === 5. Outlier Detection & Treatment (IQR Method) ===
def remove_outliers_iqr(data, columns):
    cleaned = data.copy()
    for col in columns:
        Q1 = cleaned[col].quantile(0.25)
        Q3 = cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((cleaned[col] < lower) | (cleaned[col] > upper)).sum()
        if outliers > 0:
            print(f"Removing {outliers} outliers from '{col}'...")
            cleaned[col] = np.where(cleaned[col] < lower, lower,
                                    np.where(cleaned[col] > upper, upper, cleaned[col]))
    return cleaned

df = remove_outliers_iqr(df, num_cols)
print("Outlier thresholds applied (IQR Method).\n")

# === 6. Data Type Enforcement ===
# Convert any numeric-like strings to numeric
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("🔧 Data types standardized:\n", df.dtypes, "\n")

# === 7. Statistical Summary ===
print("System Capability Statistical Overview:\n")
print(df.describe().T)

# === 8. Failure Mode Verification ===
failure_col = "Failuremode" if "Failuremode" in df.columns else "FailureMode"
if failure_col in df.columns:
    print("\n Failure Modes Distribution:\n", df[failure_col].value_counts())
else:
    print("\n  No 'FailureMode' column detected. Verify label field name.")

# === 9. Save Cleaned Data ===
df.to_csv("../data/dataset_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved'")