import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
import os

# File Paths
INPUT_CSV_PATH = r"/Users/anyas/Desktop/Thesis/data_for_regressions.csv"

# Define the exact output file path for EACH dependent variable
OUTPUT_FILE_PATHS = {
    'saved': r"/Users/anyas/Desktop/Thesis/regression_results_per_country_saved.csv",
    'saved_account': r"/Users/anyas/Desktop/Thesis/regression_results_per_country_saved_account.csv",
    'saved_retirement': r"/Users/anyas/Desktop/Thesis/regression_results_per_country_saved_retirement.csv",
}


# --- Configuration ---
# Suppress potential ConvergenceWarning and PerfectSeparationWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', PerfectSeparationWarning)

# Specify Dependent Variables (must be binary 0/1 for logit)
# IMPORTANT: Ensure every variable listed here has a corresponding entry in OUTPUT_FILE_PATHS above
dependent_vars = [
    'saved',
    'saved_account',
    'saved_retirement'
]

# Specify Explanatory Variables
explanatory_vars = [
    'has_credit_card',
    'female',
    'age',
    'higher_educ',
    'employed',
#   'inc_quint1', # IMPORTANT: we need to remove one of the levels when levels represent all possible outcomes, usually we remove the lowest
    'inc_quint2',
    'inc_quint3',
    'inc_quint4',
    'inc_quint5',
    'recv_wage',
    'recv_govt_trans',
    'recv_pension',
    'borrowed',
    'has_mobile',
    'paid_utility',
    'paid_bills_online',
    'bought_online',
]

# Specify key column names
country_var = 'economycode'
year_var = 'year' # Set to None if you don't want year controls

# --- Validate Configuration ---
missing_paths = [dv for dv in dependent_vars if dv not in OUTPUT_FILE_PATHS]
if missing_paths:
    print("ERROR: Output file paths are not defined for the following dependent variables in OUTPUT_FILE_PATHS:")
    for mp in missing_paths:
        print(f" - {mp}")
    print("Please define the full path for each output file at the top of the script.")
    exit()


# --- Data Loading ---
print(f"Loading data from: {INPUT_CSV_PATH}")
try:
    data_cleaned = pd.read_csv(INPUT_CSV_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: File not found at {INPUT_CSV_PATH}. Please check the path.")
    exit()
except Exception as e:
    print(f"ERROR loading data: {e}")
    exit()

# --- Data Preparation ---
print("Preparing data (checking required columns)...")
required_cols_list = list(set(
    dependent_vars +
    explanatory_vars +
    [country_var] +
    ([year_var] if year_var else [])
))

missing_cols = [col for col in required_cols_list if col not in data_cleaned.columns]
if missing_cols:
    print(f"\nERROR: The following required columns are missing from the CSV: {missing_cols}")
    exit()

if year_var and year_var in data_cleaned.columns:
    print(f"Variable '{year_var}' found.")
else:
     print(f"Year variable '{year_var}' not found or not specified. Proceeding without year controls.")
     year_var = None

if country_var not in data_cleaned.columns:
    print(f"ERROR: Country variable '{country_var}' not found in the CSV.")
    exit()

print("Data preparation checks complete.")


# --- Running Logistic Regressions by Country ---

results_storage = {} # Structure: {country: {dv: {'OR': float, 'Lower_CI': float, 'Upper_CI': float, 'Status': str}}}
countries = data_cleaned[country_var].unique()
countries = sorted([c for c in countries if pd.notna(c)])

print(f"\nFound {len(countries)} unique countries. Running regressions for each...")

# --- Loop through countries and DVs ---
for country in countries:
    results_storage[country] = {}
    country_df = data_cleaned[data_cleaned[country_var] == country].copy()

    if country_df.empty:
        continue

    for dv in dependent_vars:
        cols_for_model = [dv] + explanatory_vars + ([year_var] if year_var else [])
        cols_for_model = [col for col in cols_for_model if col in country_df.columns]
        df_model_ready = country_df[cols_for_model].dropna()

        n_obs = len(df_model_ready)
        num_potential_predictors = len(explanatory_vars) + (1 if year_var and year_var in df_model_ready.columns and df_model_ready[year_var].nunique() > 1 else 0)
        min_obs_needed = num_potential_predictors + 5

        if n_obs < min_obs_needed:
             results_storage[country][dv] = {'Status': 'Insufficient N'}
             continue
        if dv not in df_model_ready.columns or df_model_ready[dv].nunique() < 2:
             results_storage[country][dv] = {'Status': 'No DV Variation'}
             continue

        current_explanatory_parts = [var for var in explanatory_vars if var in df_model_ready.columns]
        current_formula_parts = current_explanatory_parts
        use_year_control = False

        if year_var and year_var in df_model_ready.columns:
             current_formula_parts.append(f"C({year_var})")
             use_year_control = True

        if not current_explanatory_parts:
             results_storage[country][dv] = {'Status': 'No Expl Vars'}
             continue

        formula = f"{dv} ~ {' + '.join(current_formula_parts)}"

        try:
            if use_year_control:
                 if year_var in df_model_ready.columns:
                     df_model_ready[year_var] = df_model_ready[year_var].astype('category')
                 else:
                     formula = f"{dv} ~ {' + '.join(current_explanatory_parts)}"

            model = smf.logit(formula, data=df_model_ready).fit(disp=False)

            if 'has_credit_card' in model.params.index:
                param = model.params['has_credit_card']
                odds_ratio = np.exp(param)
                conf = model.conf_int()
                if 'has_credit_card' in conf.index:
                     log_odds_ci = conf.loc['has_credit_card']
                     lower_ci = np.exp(log_odds_ci[0])
                     upper_ci = np.exp(log_odds_ci[1])
                     results_storage[country][dv] = {
                         'OR': odds_ratio, 'Lower_CI': lower_ci, 'Upper_CI': upper_ci
                     }
                else:
                     results_storage[country][dv] = {'Status': 'CI Calc Error'}
            else:
                 if 'has_credit_card' in current_explanatory_parts:
                     results_storage[country][dv] = {'Status': 'Not Estimated (Dropped)'}
                 else:
                     results_storage[country][dv] = {'Status': 'Not Estimated (Missing/Constant)'}

        except Exception as e:
            # error_type = type(e).__name__ # Keep for debugging if needed
            results_storage[country][dv] = {'Status': 'Fit/CI Error'}

    country_index = countries.index(country) + 1
    if country_index % 25 == 0 or country_index == len(countries):
         print(f"  Processed {country_index}/{len(countries)} countries...")


print("\n--- Regression runs finished ---")

# --- Assembling and Saving Final Tables (One per DV) ---
print("\nAssembling and saving final result tables...")

output_columns = ['Lower 95', 'OR', 'Higher 95']
all_saved_successfully = True

for dv_name in dependent_vars:
    print(f"\nProcessing table for Dependent Variable: {dv_name}")
    dv_table = pd.DataFrame(index=countries, columns=output_columns, dtype=object)
    dv_table.index.name = 'Country'

    for country_code in countries:
        result = results_storage.get(country_code, {}).get(dv_name, None)
        if (result and 'OR' in result and 'Lower_CI' in result and 'Upper_CI' in result
                and pd.notna(result['OR']) and 'Status' not in result):
            dv_table.loc[country_code, 'OR'] = f"{result['OR']:.3f}"
            dv_table.loc[country_code, 'Lower 95'] = f"{result['Lower_CI']:.3f}"
            dv_table.loc[country_code, 'Higher 95'] = f"{result['Upper_CI']:.3f}"
        else:
            dv_table.loc[country_code, :] = 'NA'

    # --- Get the pre-defined output path for this DV ---
    # Path validation happened at the start, so dv_name should be in the dictionary
    output_csv_path = OUTPUT_FILE_PATHS[dv_name]

    print(f"--- Results Table: {dv_name} ---")
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 150):
        print(dv_table)

    print(f"Saving table for '{dv_name}' to: {output_csv_path}")
    try:
        dv_table.to_csv(output_csv_path)
        print(f"Successfully saved {output_csv_path}")
    except Exception as e:
        print(f"ERROR saving table for '{dv_name}' to CSV: {e}")
        all_saved_successfully = False

# --- Final Summary ---
if all_saved_successfully:
     print("\nAll result tables saved successfully.")
else:
     print("\nWarning: One or more result tables could not be saved.")

print("\n--- Script Finished ---")
