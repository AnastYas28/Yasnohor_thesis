import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings # To manage potential warnings

# File Paths
input_csv_path = r"/Users/anyas/Desktop/Thesis/data_for_regressions.csv"
output_or_csv_path = r"/Users/anyas/Desktop/Thesis/regression_table_full_data.csv"

# Suppress potential ConvergenceWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# --- Configuration ---

# Specify Dependent Variables (must be binary 0/1 for logit)
dependent_vars = [
    'saved',
    'saved_account',
    'saved_retirement'
]

# Specify Explanatory Variables (main variables for table output)
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


# Define Fixed Effects Variables
fe_vars = ['economycode', 'year']
# Define Clustering Variable
cluster_var = 'economycode'

# --- Data Loading ---
print(f"Loading data from: {input_csv_path}")
try:
    data_cleaned = pd.read_csv(input_csv_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: File not found at {input_csv_path}. Please check the path.")
    exit()
except Exception as e:
    print(f"ERROR loading data: {e}")
    exit()

# --- Data Preparation ---
print("Preparing categorical variables...")
try:
    for var in [cluster_var] + fe_vars:
         if var in data_cleaned.columns:
              data_cleaned[var] = data_cleaned[var].astype("category")
         else:
              raise KeyError(f"Variable '{var}' needed for FE/clustering not found.")
    print("Categorical variables prepared.")
    required_cols = list(set([dv for dv in dependent_vars if dv in data_cleaned.columns] + explanatory_vars + [cluster_var] + fe_vars))
    missing_cols = [col for col in required_cols if col not in data_cleaned.columns]
    if missing_cols:
        print(f"\nERROR: The following required columns are missing from the CSV: {missing_cols}")
        exit()
except KeyError as e:
    print(f"ERROR: Column {e} not found during preparation. Check CSV column names and config.")
    exit()
except Exception as e:
    print(f"ERROR during data preparation: {e}")
    exit()


# --- Running Logistic Regressions with Pre-filtering and Clustered SEs ---
models = {}
model_stats = {}

print(f"\nRunning regressions with pre-filtering and SEs clustered by '{cluster_var}'...")
explanatory_formula_part = " + ".join(explanatory_vars)
fixed_effects_formula_part = " + ".join([f"C({fe})" for fe in fe_vars])
formula_base = f"{explanatory_formula_part} + {fixed_effects_formula_part}"

all_models_successful = True
for dv in dependent_vars:
    print(f"  Processing dependent variable: {dv}")
    print(f"    Pre-filtering data for model '{dv}'...")
    cols_for_model = list(set([dv] + explanatory_vars + [cluster_var] + fe_vars))
    df_temp = data_cleaned[cols_for_model]
    df_model_ready = df_temp.dropna()

    if df_model_ready.empty:
        print(f"    ERROR: No non-missing observations remain for model '{dv}' after filtering. Skipping.")
        models[dv] = None
        model_stats[dv] = {}
        all_models_successful = False
        continue

    print(f"    Filtered data has {len(df_model_ready)} observations.")
    cluster_groups_aligned = df_model_ready[cluster_var]

    formula = f"{dv} ~ {formula_base}"
    print(f"    Fitting model: {formula}")
    try:
        model = smf.logit(formula, data=df_model_ready).fit(
            disp=False,
            cov_type='cluster',
            cov_kwds={'groups': cluster_groups_aligned},
            use_t=False
        )
        models[dv] = model
        print(f"    Regression for '{dv}' completed.")
    except Exception as e:
        print(f"    ERROR during model fit for '{dv}': {e}")
        models[dv] = None
        model_stats[dv] = {}
        all_models_successful = False
        continue

    print(f"    Calculating statistics for '{dv}'...")
    try:
        n_obs = int(model.nobs)
        pseudo_r2 = model.prsquared
        llf = model.llf
        llnull = model.llnull
        k = len(model.params)
        adj_pseudo_r2 = 1 - (llf - k) / llnull if llnull != 0 else np.nan
        num_clusters = cluster_groups_aligned.nunique()

        model_stats[dv] = {
            'N': n_obs,
            'Pseudo R2': pseudo_r2,
            'Adj. Pseudo R2': adj_pseudo_r2,
            'Num. Clusters': num_clusters # Store the count of clusters
        }
        print(f"    Stats calculated for '{dv}'.")
    except Exception as e:
        print(f"    ERROR calculating statistics for '{dv}': {e}")
        model_stats[dv] = {}


print(f"\nRegressions attempted. Overall success status may vary per model.\n")

# --- Calculating and Presenting Odds Ratios & Stats ---

or_tables = {}
valid_models_count = 0

for dv, model in models.items():
    if model is None or dv not in model_stats or not model_stats[dv]:
        print(f"Skipping table generation for {dv} due to previous errors.")
        continue

    print(f"  Generating table section for {dv}...")
    try:
        valid_exp_vars = [var for var in explanatory_vars if var in model.params.index]
        if len(valid_exp_vars) != len(explanatory_vars):
            missing_exp_vars = [var for var in explanatory_vars if var not in model.params.index]
            print(f"    WARNING for {dv}: Explanatory variables missing from model results: {missing_exp_vars}")
        if not valid_exp_vars:
             print(f"    ERROR for {dv}: No valid explanatory variables found. Skipping.")
             continue

        params = model.params.loc[valid_exp_vars]
        p_vals = model.pvalues.loc[valid_exp_vars]
        conf_int = model.conf_int().loc[valid_exp_vars]

        odds_ratios = np.exp(params)
        conf_int_or = np.exp(conf_int)
        conf_int_or.columns = ['OR CI 95% Lower', 'OR CI 95% Upper']

        or_df = pd.DataFrame({'OddsRatio': odds_ratios, 'PValue': p_vals})
        or_df = pd.concat([or_df, conf_int_or], axis=1)

        stars = pd.cut(p_vals, bins=[-np.inf, 0.001, 0.01, 0.05, np.inf], labels=['***', '**', '*', ''])
        or_df['Odds Ratio (OR)'] = or_df['OddsRatio'].round(4).astype(str) + stars.astype(str)

        final_or_df = or_df[['Odds Ratio (OR)', 'OR CI 95% Lower', 'OR CI 95% Upper']].copy()

        # --- Create FE and Info Rows ---
        fe_rows = pd.DataFrame({
            'Odds Ratio (OR)': ['YES', 'YES'],
            'OR CI 95% Lower': ['YES', 'YES'],
            'OR CI 95% Upper': ['YES', 'YES']
        }, index=['Country FE', 'Year FE'])

        cluster_row = pd.DataFrame({
            'Odds Ratio (OR)': ['YES'], 'OR CI 95% Lower': ['YES'], 'OR CI 95% Upper': ['YES']
        }, index=['Clustered St.Er.']) # Changed label slightly for consistency

        # --- Create individual Stat Rows with desired labels ---
        stats = model_stats[dv]
        individuals_row = pd.DataFrame({'Odds Ratio (OR)': [f"{stats['N']}"],
                                        'OR CI 95% Lower': [''], 'OR CI 95% Upper': ['']},
                                       index=['Individuals'])
        # Use the Num. Clusters count but label the row 'Countries'
        countries_row = pd.DataFrame({'Odds Ratio (OR)': [f"{stats['Num. Clusters']}"],
                                      'OR CI 95% Lower': [''], 'OR CI 95% Upper': ['']},
                                     index=['Countries']) # Use desired label
        pseudo_r2_row = pd.DataFrame({'Odds Ratio (OR)': [f"{stats['Pseudo R2']:.4f}"],
                                       'OR CI 95% Lower': [''], 'OR CI 95% Upper': ['']},
                                      index=['Pseudo R2'])
        adj_pseudo_r2_row = pd.DataFrame({'Odds Ratio (OR)': [f"{stats['Adj. Pseudo R2']:.4f}"],
                                           'OR CI 95% Lower': [''], 'OR CI 95% Upper': ['']},
                                          index=['Adj. Pseudo R2'])

        # --- Combine rows in the specified order ---
        final_df_for_model = pd.concat([
            final_or_df,
            fe_rows,
            cluster_row,
            individuals_row,
            countries_row,        # Now in the requested position
            pseudo_r2_row,
            adj_pseudo_r2_row
        ], axis=0)

        or_tables[dv] = final_df_for_model
        valid_models_count += 1
        print(f"    Table section for {dv} generated successfully.")

    except KeyError as e:
        print(f"    ERROR generating table section for {dv}: Key error - {e}.")
    except Exception as e:
        print(f"    ERROR generating table section for {dv}: {e}")

# --- Final Output Generation ---
if valid_models_count > 0 and or_tables:
     combined_table = pd.concat(or_tables, axis=1)
     print(f"\n--- Combined Logit Results (Clustered SEs via Pre-filtering) and Statistics ({valid_models_count} models) ---")
     with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 150):
         print(combined_table)

     print(f"\nSaving combined table to: {output_or_csv_path}")
     try:
         combined_table.to_csv(output_or_csv_path)
         print("Combined table saved successfully.")
     except Exception as e:
         print(f"ERROR saving table to CSV: {e}")
else:
     print("\nNo valid model results available to generate or save the final table.")

print("\n--- Script Finished ---")
