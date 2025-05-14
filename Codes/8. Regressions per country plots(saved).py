import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker # Import ticker
import os

# ===================== TEXT INPUTS =====================
# Main title for the figure
MAIN_TITLE = 'Odds Ratios for association between "Has a Credit Card" and "Made Savings"'
MAIN_SUBTITLE = 'Across Countries (Sorted by OR, Highest First) with 95% Confidence Intervals'

# Legend labels
SIGNIFICANT_LABEL = 'Statistically Significant at 95% CI'
NONSIGNIFICANT_LABEL = 'Non-Statistically Significant at 95% CI'

# ===================== PATHS =====================
# File paths
INPUT_FILE = r"/Users/anyas/Desktop/Thesis/regression_results_per_country_saved.csv"
OUTPUT_FILE = r"/Users/anyas/Desktop/Thesis/visualization_per_country_saved.png"

# ===================== VISUAL SETTINGS =====================
# Number of groups to split countries into
NUM_GROUPS = 2

# REMOVED previous capping settings, replaced by roundup logic below

# Figure settings
FIG_WIDTH = 18      # inches
FIG_HEIGHT = 14     # inches # Reverted height, adjust if needed for 2 groups
DPI = 300           # dots per inch

# Colors
SIGNIFICANT_COLOR = "#1F77B4"     # Blue for significant
NONSIGNIFICANT_COLOR = "#D62728"   # Red for non-significant
REFERENCE_LINE_COLOR = "#555555"   # Dark gray for reference line
GRID_COLOR = "#CCCCCC"            # Light gray for grid

# Visual elements
REFERENCE_LINE_STYLE = "--"       # Dashed line
GRID_ALPHA = 0.3                  # Grid transparency
POINT_SIZE = 80                   # Size of OR points
LINE_WIDTH = 2                    # Width of CI lines
CAP_LENGTH = 0.2                  # Length of CI end caps

# Text settings
TITLE_SIZE = 16                   # Main title font size
AXIS_LABEL_SIZE = 12              # Axis label font size
LEGEND_FONT_SIZE = 12             # Legend font size

# Layout settings
GRID_WSPACE = 0.3                 # Width space between subplots
GRID_HSPACE = 0.3                 # Height space between subplots
LEFT_MARGIN = 0.15                # Left margin for y-axis labels

# ===================== FUNCTIONS =====================

def load_data(file_path):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(df)} countries.")

        required_cols = ['Country', 'OR', 'Lower 95', 'Higher 95']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing required columns in input file: {missing}")
            return None

        df['Significant'] = ~((df['Lower 95'] <= 1) & (df['Higher 95'] >= 1))
        df['Color'] = df['Significant'].apply(
            lambda x: SIGNIFICANT_COLOR if x else NONSIGNIFICANT_COLOR
        )

        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        print("Please ensure the INPUT_FILE path is correct.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def split_into_groups(df, num_groups):
    """Split sorted dataframe into groups for multiple plots."""
    total_countries = len(df)
    groups = np.array_split(df, num_groups)
    print(f"Split data into {len(groups)} groups.")
    return groups

# Uses the calculated axis_max (previously padded_max)
def plot_group(ax, group_data, axis_max):
    """Plot a single group of countries (assumes group_data is sorted)."""
    countries = group_data['Country'].tolist()
    n_countries = len(countries)
    positions = range(n_countries - 1, -1, -1)

    for i, (_, row) in enumerate(group_data.iterrows()):
        pos = n_countries - 1 - i
        ax.plot(
            [row['Lower 95'], row['Higher 95']], [pos, pos],
            color=row['Color'], linewidth=LINE_WIDTH, zorder=1
        )
        ax.plot(
            [row['Lower 95'], row['Lower 95']], [pos - CAP_LENGTH/2, pos + CAP_LENGTH/2],
            color=row['Color'], linewidth=LINE_WIDTH, zorder=1
        )
        ax.plot(
            [row['Higher 95'], row['Higher 95']], [pos - CAP_LENGTH/2, pos + CAP_LENGTH/2],
            color=row['Color'], linewidth=LINE_WIDTH, zorder=1
        )
        ax.scatter(
            row['OR'], pos, color=row['Color'], edgecolor='black', s=POINT_SIZE, zorder=2
        )

    ax.axvline(
        x=1, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
        linewidth=1, alpha=0.7, zorder=0
    )

    # --- X-AXIS CONFIGURATION ---
    # Set x-axis limits: min 0, max based on global data rounded up to nearest 0.5
    ax.set_xlim(left=0, right=axis_max) # Use the calculated axis_max

    # Set x-axis ticks to integers only, stepping by 1
    # Determine the highest integer needed based on the axis_max
    max_tick_value = int(np.ceil(axis_max))
    # Generate integer ticks from 0 up to max_tick_value
    integer_ticks = np.arange(0, max_tick_value + 1, 1)
    ax.set_xticks(integer_ticks)
    # --- END X-AXIS CONFIGURATION ---

    ax.set_yticks(positions)
    ax.set_yticklabels(countries)

    ax.tick_params(axis='y', which='both', left=False, labelleft=True)
    # Keep x-axis label rotation at 0 (horizontal)
    ax.tick_params(axis='x', rotation=0)

    ax.grid(True, axis='x', linestyle='--', alpha=GRID_ALPHA, color=GRID_COLOR, zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=AXIS_LABEL_SIZE)


def create_visualization(groups, df):
    """Create the complete visualization with multiple groups."""
    if len(groups) <= 2:
        rows, cols = 1, len(groups)
    else:
        rows = 2
        cols = int(np.ceil(len(groups) / 2))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

    # --- CALCULATE X-AXIS MAX (Round Up to Nearest 0.5) ---
    global_max_ci = df['Higher 95'].max()

    # Round the global max upward to the nearest 0.5 using ceil(x*2)/2
    axis_max = np.ceil(global_max_ci * 2) / 2

    print(f"Global max CI found: {global_max_ci:.2f}. Setting x-axis max (rounded up to nearest 0.5) to: {axis_max}")
    # --- END CALCULATION ---

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    gs = gridspec.GridSpec(
        rows, cols, figure=fig, wspace=GRID_WSPACE, hspace=GRID_HSPACE, left=LEFT_MARGIN
    )

    for i, group_data in enumerate(groups):
        if group_data.empty:
            continue
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        # Pass the calculated axis_max to the plotting function
        plot_group(ax, group_data, axis_max)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SIGNIFICANT_COLOR, markersize=10, label=SIGNIFICANT_LABEL),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NONSIGNIFICANT_COLOR, markersize=10, label=NONSIGNIFICANT_LABEL)
    ]
    fig.legend(
        handles=legend_elements, loc='lower center', ncol=2,
        fontsize=LEGEND_FONT_SIZE, frameon=True, bbox_to_anchor=(0.5, 0.02)
    )

    plt.suptitle(
        f'{MAIN_TITLE}\n{MAIN_SUBTITLE}', fontsize=TITLE_SIZE, fontweight='bold', y=0.98
    )

    fig.subplots_adjust(left=LEFT_MARGIN, bottom=0.1, right=0.95, top=0.92, wspace=GRID_WSPACE, hspace=GRID_HSPACE)

    return fig

# ===================== MAIN EXECUTION =====================

def main():
    """Main execution function."""
    print("Starting odds ratio visualization...")

    df = load_data(INPUT_FILE)
    if df is None:
        print("Error loading data. Exiting.")
        return

    print(f"Sorting {len(df)} countries by Odds Ratio (descending)...")
    df = df.sort_values(by='OR', ascending=False).reset_index(drop=True)

    groups = split_into_groups(df, NUM_GROUPS)

    fig = create_visualization(groups, df)

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
         print(f"Creating output directory: {output_dir}")
         os.makedirs(output_dir)

    try:
        plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
        print(f"Visualization saved to {os.path.abspath(OUTPUT_FILE)}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        return

    # plt.show()
    print("Visualization process completed successfully!")

if __name__ == "__main__":
    main()