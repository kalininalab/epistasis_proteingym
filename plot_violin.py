import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load the data
data = pd.read_csv('best_models.csv')

# List of selected columns (datasets) to include in the plot
selected_columns = [
    "D7PM05_CLYGR_Somermeyer_2022_all", "D7PM05_CLYGR_Somermeyer_2022_1std", "D7PM05_CLYGR_Somermeyer_2022_2std",
    "F7YBW8_MESOW_Aakre_2015_all", "F7YBW8_MESOW_Aakre_2015_1std", "F7YBW8_MESOW_Aakre_2015_2std",
    "GFP_AEQVI_Sarkisyan_2016_all", "GFP_AEQVI_Sarkisyan_2016_1std", "GFP_AEQVI_Sarkisyan_2016_2std"
]

# Add the 'model' column to the selected columns for context
columns_to_plot = ["model"] + selected_columns

# Filter the DataFrame
filtered_data = data[columns_to_plot]
# Take the absolute value of all numerical columns
filtered_data[selected_columns] = filtered_data[selected_columns].abs()
# Remove rows which have 'LinearRegression' in the 'model' column
filtered_data = filtered_data[~filtered_data['model'].str.contains('LinearRegression')]
# Melt the filtered DataFrame into long format for plotting
df_melted = filtered_data.melt(id_vars=["model"], var_name="Dataset", value_name="Spearman correlation")

# Set up the plot
plt.figure(figsize=(8, 6))

# Define a custom color palette
palette = [
    "#ff7051", "#ff7051", "#ff7051",  # Colors for D7PM05_CLYGR (orange shades)
    "#709ac6", "#709ac6", "#709ac6",  # Colors for F7YBW8_MESOW (blue shades)
    "#7cc670", "#7cc670", "#7cc670"   # Colors for GFP_AEQVI (green shades)
]

# Create the violin plot
sns.violinplot(
    data=df_melted,
    x="Dataset",
    y="Spearman correlation",
    palette=palette,
    scale="width",
    inner="box"
)

# Customize the plot
plt.ylabel("Spearman correlation", fontsize=14)

# Custom x-tick labels
custom_labels = [
    "all", "1 std", "2 std",
    "all", "1 std", "2 std",
    "all", "1 std", "2 std"
]
plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=45, fontsize=14, ha='right')

# Remove x-axis label
plt.xlabel(None)

# Add a legend for the colors
legend_elements = [
    Patch(facecolor="#ff7051", edgecolor='k', label="GFP from C. gregaria"),
    Patch(facecolor="#709ac6", edgecolor='k', label="Addition module antidote protein"),
    Patch(facecolor="#7cc670", edgecolor='k', label="GFP from A. victoria")
]
plt.legend(handles=legend_elements, loc="upper center", fontsize=12, title_fontsize=14)

# Adjust layout and show the plot
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('figures/paper/2.png', dpi=300)
plt.show()
