import pandas as pd

# Define file paths at the top of the script
input_file = r"/Users/anyas/Desktop/Thesis/data_recoded.csv"
output_file = r"/Users/anyas/Desktop/Thesis/data_cleaned.csv"

# Load your dataset
df = pd.read_csv(input_file)

# First rename the columns according to the mapping
column_mapping = {
    'saved': 'saved',
    'fin17a': 'saved_account',
    'fin16': 'saved_retirement',
    'fin2': 'has_debit_card',
    'fin7': 'has_credit_card',
    'female': 'female',
    'age': 'age',
    'educ': 'higher_educ',
    'emp_in': 'employed',
    'inc_q_1': 'inc_quint1',
    'inc_q_2': 'inc_quint2',
    'inc_q_3': 'inc_quint3',
    'inc_q_4': 'inc_quint4',
    'inc_q_5': 'inc_quint5',
    'fin32': 'recv_wage',
    'fin37': 'recv_govt_trans',
    'fin38': 'recv_pension',
    'borrowed': 'borrowed',
    'mobileowner': 'has_mobile',
    'fin30': 'paid_utility',
    'fin14a': 'paid_bills_online',
    'fin14b': 'bought_online',
    'Year': 'year'
}

df.rename(columns=column_mapping, inplace=True)

# Keep only the renamed columns plus 'economycode'
columns_to_keep = list(column_mapping.values()) + ['economycode']

# Create a new dataframe with only the selected columns
df_selected = df[columns_to_keep]

# List of countries to exclude because of problems they cause for regressions (due to NAs)
countries_to_exclude = [
    'TTO',  # Trinidad and Tobago
    'MOZ',  # Mozambique
    'BLR',  # Belarus
    'SWZ',  # Eswatini
    'LUX',  # Luxembourg
    'MNE',  # Montenegro
    'LBY',  # Libya
    'KWT',  # Kuwait
    'BHR',  # Bahrain
    'ARE',  # United Arab Emirates
    'ISL',  # Iceland
    'JAM'   # Jamaica
]

# Filter out the excluded countries
df_filtered = df_selected[~df_selected['economycode'].isin(countries_to_exclude)]

# Print info about how many rows were filtered out
original_count = len(df_selected)
filtered_count = len(df_filtered)
removed_count = original_count - filtered_count

print(f"Original dataset: {original_count} rows")
print(f"Filtered dataset: {filtered_count} rows")
print(f"Removed {removed_count} rows ({(removed_count/original_count)*100:.2f}% of data)")

# Save the filtered dataset
df_filtered.to_csv(output_file, index=False)
