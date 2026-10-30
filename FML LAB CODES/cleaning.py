import pandas as pd
import numpy as np

# ⃣ Create sample dataset with missing values
data = {
"Name": ["Alice", "Bob", "Charlie", "David", "Eva"], "Age": [23, np.nan, 22, 28, np.nan],
"City": ["New York", "London", np.nan, "Paris", "Berlin"]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("\n")

# ⃣ Check for null values
print("Null values count in each column:")
print(df.isnull().sum())
print("\n")

# ⃣ Drop rows with missing values
df_clean = df.dropna()
print("DataFrame after dropping rows with missing values:")
print(df_clean)
print("\n")

# ⃣ Fill missing values with 0
df_filled = df.fillna(0)
print("DataFrame after filling missing values with 0:")
print(df_filled)
print("\n")

# ⃣ Fill missing 'Age' values with mean of the column
df_mean_filled = df.copy()
df_mean_filled["Age"] = df_mean_filled["Age"].fillna(df_mean_filled["Age"].mean())
print("DataFrame after filling 'Age' with mean value:")
print(df_mean_filled)
print("\n")

df_city_filled = df.copy()
df_city_filled["City"] = df_city_filled["City"].fillna("Unknown")
print("DataFrame after filling 'City' with 'Unknown':")
print(df_city_filled)
print("/n")