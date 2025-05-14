import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define all paths at the top of the code
INPUT_FILE_PATH = r"/Users/anyas/Desktop/Thesis/data_cleaned.csv"
OUTPUT_STATS_PATH = r"/Users/anyas/Desktop/Thesis/descriptive_overall.csv"
OUTPUT_CORR_PATH = r"/Users/anyas/Desktop/Thesis/correlation_matrix_overall.csv"
OUTPUT_COUNTRY_MEANS_PATH = r"/Users/anyas/Desktop/Thesis/country_means.csv"
OUTPUT_FILTERED_DATA_PATH = r"/Users/anyas/Desktop/Thesis/data_for_regressions.csv"
OUTPUT_PLOTS_PREFIX = r"/Users/anyas/Desktop/Thesis/scatter_"

# Minimum credit card ownership threshold (can be adjusted by the user)
MIN_CREDIT_CARD_THRESHOLD = 0.10  # 10% threshold

# Load your cleaned dataset
data_cleaned = pd.read_csv(INPUT_FILE_PATH)

# ✅ Filter countries by credit card ownership threshold
# First calculate the country means to apply the filter
country_means_all = data_cleaned.groupby('economycode')['has_credit_card'].mean().reset_index()

# Get list of countries that meet the threshold
countries_above_threshold = country_means_all[country_means_all['has_credit_card'] >= MIN_CREDIT_CARD_THRESHOLD][
    'economycode'].tolist()

# Filter the dataset to include only countries above threshold
data_filtered = data_cleaned[data_cleaned['economycode'].isin(countries_above_threshold)]

print(f"🔹 Filtered out countries with less than {MIN_CREDIT_CARD_THRESHOLD * 100}% credit card ownership")
print(f"   - Original dataset: {len(data_cleaned['economycode'].unique())} countries")
print(f"   - Filtered dataset: {len(countries_above_threshold)} countries")
print(f"   - Countries removed: {len(data_cleaned['economycode'].unique()) - len(countries_above_threshold)}")

# Save filtered data for regressions
data_filtered.to_csv(OUTPUT_FILTERED_DATA_PATH, index=False)
print(f"✅ Filtered data saved to {OUTPUT_FILTERED_DATA_PATH}")

# ✅ Overall descriptive statistics (filtered dataset)
overall_stats = data_filtered.describe(include='all').T[["count", "mean", "std", "min", "max"]]
print("🔹 Overall Descriptive Statistics (Filtered Dataset):")
print(overall_stats)

# Saving descriptive statistics
overall_stats.to_csv(OUTPUT_STATS_PATH)

# ✅ Correlation matrix for numerical variables
# Select only numeric columns for correlation analysis
numeric_data = data_filtered.select_dtypes(include=[np.number])

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

# Save correlation matrix to CSV
correlation_matrix.to_csv(OUTPUT_CORR_PATH)

print("✅ Correlation analysis completed and saved to files.")

# ✅ Calculate average per country for specified variables
# Main explanatory variable: has_credit_card
# Dependent variables: saved, saved_account, saved_retirement
variables_of_interest = ['has_credit_card', 'saved', 'saved_account', 'saved_retirement']

# Group by economycode (country code) and calculate means for variables of interest
country_means = data_filtered.groupby('economycode')[variables_of_interest].mean().reset_index()

# Save country means to CSV
country_means.to_csv(OUTPUT_COUNTRY_MEANS_PATH, index=False)
print("✅ Country means calculated and saved to file.")
print(country_means.head())

# ✅ Create scatter plots for each dependent variable vs has_credit_card
# Define dependent variables
dependent_vars = ['saved', 'saved_account', 'saved_retirement']

# Create scatter plots
for dep_var in dependent_vars:
    plt.figure(figsize=(10, 6))

    # Create scatter plot
    plt.scatter(country_means['has_credit_card'], country_means[dep_var], alpha=0.7)

    # Add regression line
    sns.regplot(x='has_credit_card', y=dep_var, data=country_means,
                scatter=False, line_kws={"color": "red"})

    # Add labels for each country using economycode
    for i, row in country_means.iterrows():
        plt.annotate(row['economycode'],
                     (row['has_credit_card'], row[dep_var]),
                     textcoords="offset points",
                     xytext=(0, 7),
                     ha='center')

    # Set title and labels
    plt.title(f'Relationship between Credit Card Ownership and {dep_var} by Country Code')
    plt.xlabel('Credit Card Ownership Rate')
    plt.ylabel(f'{dep_var} Rate')

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Calculate correlation
    corr = country_means['has_credit_card'].corr(country_means[dep_var])
    plt.annotate(f'Correlation: {corr:.2f}',
                 xy=(0.05, 0.95),
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Save figure
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS_PREFIX}{dep_var}_vs_credit_card.png", dpi=300)
    plt.close()

print("✅ Scatter plots created and saved to files.")
