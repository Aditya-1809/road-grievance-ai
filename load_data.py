import pandas as pd

# Load roads dataset
df = pd.read_csv("roads.csv")

# Show first 5 rows
print(df.head())

# Show total complaints
print("Total complaints:", len(df))