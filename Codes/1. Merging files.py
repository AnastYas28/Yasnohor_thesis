import sys
import subprocess

# Install pandas
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
print("Pandas installed successfully!")

import pandas as pd

# Paths to your CSV files
file1_path = "/Users/anyas/Desktop/Thesis/data 2017.csv"
file2_path = "/Users/anyas/Desktop/Thesis/data 2021.csv"
output_path = "/Users/anyas/Desktop/Thesis/data 2017-2021.csv"

# List of columns to keep
columns_to_keep = [
    "economy", "economycode", "regionwb", "pop_adult", "wpid_random", "wgt", "female", "age", "educ", "inc_q", "emp_in",
    "account_fin", "account_mob", "account", "borrowed", "saved", "receive_wages", "receive_transfers", "receive_pension",
    "receive_agriculture", "pay_utilities", "remittances", "mobileowner", "fin2", "fin4", "fin5", "fin6", "fin7", "fin8",
    "fin9", "fin10", "fin11a", "fin11b", "fin11c", "fin11d", "fin11e", "fin11f", "fin11g", "fin11h", "fin14a", "fin14b",
    "fin14c", "fin16", "fin17a", "fin17b", "fin20", "fin22a", "fin22b", "fin22c", "fin24", "fin26", "fin27c1", "fin27c2",
    "fin28", "fin29c1", "fin29c2", "fin30", "fin31a", "fin31b", "fin31c", "fin32", "fin33", "fin34a", "fin34b", "fin35",
    "fin37", "fin38", "fin39a", "fin39b", "fin42", "fin43a", "fin43b", "fin45"
]

# Try different encodings
try:
    df1 = pd.read_csv(file1_path, usecols=columns_to_keep, encoding="utf-8")
    df2 = pd.read_csv(file2_path, usecols=columns_to_keep, encoding="utf-8")
except UnicodeDecodeError:
    print("UTF-8 failed, trying ISO-8859-1...")
    df1 = pd.read_csv(file1_path, usecols=columns_to_keep, encoding="ISO-8859-1")
    df2 = pd.read_csv(file2_path, usecols=columns_to_keep, encoding="ISO-8859-1")

# Add Year column
df1["Year"] = 2017
df2["Year"] = 2021

# Merge the two DataFrames
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged file as UTF-8
merged_df.to_csv(output_path, index=False, encoding="utf-8")

print("✅ Merging complete! File saved at:", output_path)