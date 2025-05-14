import pandas as pd

# Load your dataset
file_path = "/Users/anyas/Desktop/Thesis/data 2017-2021.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# Define recoding mappings
recoding_dict = {
    "female": {1: "1", 2: "0"},
    "emp_in": {1: "1", 2: "0"},
    "educ": {1: "0", 2: "0", 3: "1"},
    "fin16": {1: "1", 2: "0", 3: "NA", 4: "NA"},
    "fin17a": {1: "1", 2: "0", 3: "NA", 4: "NA"},
    "fin17b": {1: "1", 2: "0", 3: "NA", 4: "NA"},
    "fin20": {1: "1", 2: "0", 3: "NA", 4: "NA"},
    "fin22a": {1: "1", 2: "0", 3: "NA", 4: "NA"},
    "fin22b": {1: "1", 2: "0", 3: "NA", 4: "NA"},
    "fin22c": {1: "1", 2: "0", 3: "NA", 4: "NA"},
    "fin32": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin33": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin34a": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin34b": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin35": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin37": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin38": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin39a": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin39b": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin42": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin43a": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin43b": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"receive_wages": {1: "1", 2: "1", 3: "1", 4: "0", 5: "NA"},
"receive_transfers": {1: "1", 2: "1", 3: "1", 4: "0", 5: "NA"},
"receive_pension": {1: "1", 2: "1", 3: "1", 4: "0", 5: "NA"},
"receive_agriculture": {1: "1", 2: "1", 3: "1", 4: "0", 5: "NA"},
"mobileowner": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin5": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin6": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin14a": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin14b": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin31b": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin2": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin4": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin7": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin8": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin9": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin10": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin11a": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin11b": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin11c": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin11d": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin11e": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin11f": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin11g": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin11h": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin31a": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin26": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin27c1": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin27c2": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin28": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin29c1": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin29c2": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin30": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"fin31c": {1: "1", 2: "0", 3: "NA", 4: "NA"},
"remittances": {1: "1", 2: "1", 3: "1", 4: "1", 5: "0", 6: "NA"},
"pay_utilities": {1: "1", 2: "0", 3: "0", 4: "0", 5: "NA"},
}

# Splitting variables into separate binary variables
df["worried_old_age"] = df["fin45"].map({1: "1", 2: "0", 3: "0", 4: "0", 5: "NA", 6: "NA"})
df["worried_medical_costs"] = df["fin45"].map({1: "0", 2: "1", 3: "0", 4: "0", 5: "NA", 6: "NA"})
df["worried_monthly_expenses"] = df["fin45"].map({1: "0", 2: "0", 3: "1", 4: "0", 5: "NA", 6: "NA"})
df["worried_education_fees"] = df["fin45"].map({1: "0", 2: "0", 3: "0", 4: "1", 5: "NA", 6: "NA"})

df["inc_q_1"] = df["inc_q"].map({1: "1", 2: "0", 3: "0", 4: "0", 5: "0"})
df["inc_q_2"] = df["inc_q"].map({1: "0", 2: "1", 3: "0", 4: "0", 5: "0"})
df["inc_q_3"] = df["inc_q"].map({1: "0", 2: "0", 3: "1", 4: "0", 5: "0"})
df["inc_q_4"] = df["inc_q"].map({1: "0", 2: "0", 3: "0", 4: "1", 5: "0"})
df["inc_q_5"] = df["inc_q"].map({1: "0", 2: "0", 3: "0", 4: "0", 5: "1"})

df["fin14c_online"] = df["fin14c"].map({1: "1", 2: "0", 3: "0", 4: "NA", 5: "NA"})
df["fin14c_cash"] = df["fin14c"].map({1: "0", 2: "1", 3: "0", 4: "NA", 5: "NA"})
df["fin14c_both"] = df["fin14c"].map({1: "0", 2: "0", 3: "1", 4: "NA", 5: "NA"})

df["fin24_savings"] = df["fin24"].map({1: "1", 2: "0", 3: "0", 4: "0", 5: "0", 6: "0", 7:"0", 8: "NA", 9: "NA"})
df["fin24_family"] = df["fin24"].map({1: "0", 2: "1", 3: "0", 4: "0", 5: "0", 6: "0", 7:"0", 8: "NA", 9: "NA"})
df["fin24_work"] = df["fin24"].map({1: "0", 2: "0", 3: "1", 4: "0", 5: "0", 6: "0", 7:"0", 8: "NA", 9: "NA"})
df["fin24_borrowings"] = df["fin24"].map({1: "0", 2: "0", 3: "0", 4: "1", 5: "0", 6: "0", 7:"0", 8: "NA", 9: "NA"})
df["fin24_sale"] = df["fin24"].map({1: "0", 2: "0", 3: "0", 4: "0", 5: "1", 6: "0", 7:"0", 8: "NA", 9: "NA"})
df["fin24_other"] = df["fin24"].map({1: "0", 2: "0", 3: "0", 4: "0", 5: "0", 6: "1", 7:"0", 8: "NA", 9: "NA"})
df["fin24_no"] = df["fin24"].map({1: "0", 2: "0", 3: "0", 4: "0", 5: "0", 6: "0", 7:"1", 8: "NA", 9: "NA"})

# Apply recoding to each variable in the dictionary
for column, mapping in recoding_dict.items():
    if column in df.columns:  # Ensure the column exists before recoding
        df[column] = df[column].map(mapping)

# Save the updated dataset
output_path = "/Users/anyas/Desktop/Thesis/data_recoded.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print("✅ Recoding complete! File saved at:", output_path)