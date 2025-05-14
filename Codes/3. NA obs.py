import pandas as pd

# Load your dataset (if not already loaded)
df = pd.read_csv("/Users/anyas/Desktop/Thesis/data_recoded.csv")

# Variables you want to check
vars_to_check = ["remittances", "receive_wages", "receive_transfers", "receive_pension",
                 "receive_agriculture", "pay_utilities", "fin14a", "fin14b", "fin14c",
                 "fin5", "fin6", "fin34a", "fin34b", "fin35", "fin39a", "fin39b",
                 "fin43a", "fin43b", "fin27c1", "fin27c2", "fin29c1", "fin29c2",
                 "fin31a", "fin31b", "fin31c" ]

print("Overall Share of Non-NA Observations:")
for var in vars_to_check:
    non_na = df[var].notna().sum()
    total = len(df)
    share = non_na / total
    print(f"{var}: {non_na} non-NA out of {total} ({share:.2%})")

print("Share of Non-NA Observations Per Country:")
# Group by country and calculate share per variable
country_stats = df.groupby("economy")[vars_to_check].apply(lambda x: x.notna().sum() / len(x))

# Multiply by 100 to show as percentages if you want
country_stats = (country_stats * 100).round(2)

# Display
print(country_stats)
