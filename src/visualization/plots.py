import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_individual_complexity_differences(df):
    # Plot for dataset_complexity_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='dataset_complexity_difference', data=df)
    plt.title('Differences in Dataset Complexity (global - mean_folds)')
    plt.xlabel('Measure')
    plt.ylabel('Dataset Complexity Difference')
    plt.show()

    # Plot for majority_class_complexity_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='majority_class_complexity_difference', data=df)
    plt.title('Differences in Majority Class Complexity (global - mean_folds)')
    plt.xlabel('Measure')
    plt.ylabel('Majority Class Complexity Difference')
    plt.show()

    # Plot for minority_class_complexity_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='minority_class_complexity_difference', data=df)
    plt.title('Differences in Minority Class Complexity (global - mean_folds)')
    plt.xlabel('Measure')
    plt.ylabel('Minority Class Complexity Difference')
    plt.show()

    # Plot for most_complex_class_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='most_complex_class_difference', data=df)
    plt.title('Differences in Most Complex Class (global - mean_folds)')
    plt.xlabel('Measure')
    plt.ylabel('Most Complex Class Difference')
    plt.show()

    # Plot for least_complex_class_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='least_complex_class_difference', data=df)
    plt.title('Differences in Least Complex Class (global - mean_folds)')
    plt.xlabel('Measure')
    plt.ylabel('Least Complex Class Difference')
    plt.show()


def plot_line_correlations(df, score_metric):
    """
    Plot line graphs of correlations for each complexity metric by k, method, and complexity_metric.
    Highlights and annotates the maximum value for each series with arrows pointing at a 30-degree downward angle.
    The highest value among all series is highlighted in bold separately for kdn and ddn.
    The legend is always positioned at the bottom left.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot. Must include columns: 'k', 'method', 'complexity_metric', 
        and the various correlation metrics to be visualized.

    Returns:
    --------
    None
    """
    # List of correlation columns to plot
    correlation_columns = [
        'corr_dataset_complexity', 
        'corr_majority_class_complexity', 
        'corr_minority_class_complexity', 
        'corr_most_complex_class', 
        'corr_least_complex_class'
    ]

    # Plot each complexity correlation as a line plot
    for col in correlation_columns:
        plt.figure(figsize=(14, 8))
        # Create the line plot
        ax = sns.lineplot(
            data=df, 
            x='k', 
            y=col, 
            hue='method', 
            style='complexity_metric', 
            markers=True, 
            dashes=False
        )

        # Track if the maximum has been bolded for kdn and ddn
        max_bolded_kdn = False
        max_bolded_ddn = False

        # Iterate over each group to find and annotate the maximum value
        for (method, metric), group_data in df.groupby(['method', 'complexity_metric']):
            # Find the maximum value and its corresponding k
            max_idx = group_data[col].idxmax()
            max_k = group_data.loc[max_idx, 'k']
            max_val = group_data.loc[max_idx, col]

            # Find the maximum value for each complexity_metric separately
            max_val_kdn = df[(df['complexity_metric'] == 'kdn') & (df['k'] == max_k)][col].max()
            max_val_ddn = df[(df['complexity_metric'] == 'ddn') & (df['k'] == max_k)][col].max()

            # Determine if the current value is the maximum for its respective complexity_metric
            is_bold_kdn = (metric == 'kdn') and (max_val == max_val_kdn) and not max_bolded_kdn
            is_bold_ddn = (metric == 'ddn') and (max_val == max_val_ddn) and not max_bolded_ddn

            # Mark as bold only once for each method
            if is_bold_kdn:
                max_bolded_kdn = True
            if is_bold_ddn:
                max_bolded_ddn = True

            # Annotate the maximum value on the plot
            ax.annotate(
                f'{max_val:.3f}', 
                xy=(max_k, max_val), 
                xytext=(max_k + 0.4, max_val - 0.03),  # Position slightly down and more to the right
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),  # Arrow pointing to the point
                fontsize=10, 
                color='black',
                ha='left',  # Horizontal alignment
                va='top',   # Vertical alignment to the top of the text
                fontweight='bold' if is_bold_kdn or is_bold_ddn else 'normal'  # Bold if it is the max for kdn or ddn
            )
        
        # Set plot titles and labels
        plt.title(f'Correlation of {col.replace("_", " ").title()} with Performance ({score_metric})')
        plt.ylabel('Spearman Correlation')
        plt.xlabel('k')
        plt.legend(title='Metric (Global vs. Mean Folds)', loc='lower left')  # Legend always at the bottom left
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()


def plot_score_differences(df, performance_metric):
    """
    Plot a boxplot comparing kdn vs ddn for the differences between score and 1 - complexity
    for two complexity metrics across different k values for a specific performance metric.
    Each method (kdn vs ddn) is represented by distinct shades of the same color.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the calculated differences and performance data.
        It must include 'performance_metric', 'k', 'metric_y', 'diff_score_minority_class_complexity', 
        and 'diff_score_most_complex_class' columns.

    performance_metric : str
        The performance metric to filter and visualize (e.g., 'accuracy_score').

    Returns:
    --------
    None
    """
    # Filter the DataFrame for the selected performance metric
    df_filtered = df[df['metric_x'] == performance_metric]

    # Melt the DataFrame to have one column for the differences
    df_melted = df_filtered.melt(
        id_vars=['k', 'metric_y'], 
        value_vars=['diff_score_minority_class_complexity', 'diff_score_most_complex_class'],
        var_name='Complexity_Type', 
        value_name='Difference'
    )

    # Add a combined column for hue to differentiate between kdn and ddn
    df_melted['Complexity_Method'] = df_melted['Complexity_Type'] + '_' + df_melted['metric_y']

    # Define the custom order for the hue categories
    hue_order = [
        'diff_score_minority_class_complexity_kdn', 
        'diff_score_most_complex_class_kdn', 
        'diff_score_minority_class_complexity_ddn', 
        'diff_score_most_complex_class_ddn'
    ]

    # Create a color palette with different shades for kdn and ddn
    palette = {
        'diff_score_minority_class_complexity_kdn': 'lightblue', 
        'diff_score_most_complex_class_kdn': 'skyblue',
        'diff_score_minority_class_complexity_ddn': 'lightcoral', 
        'diff_score_most_complex_class_ddn': 'red'
    }

    # Plotting the boxplot
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(
        data=df_melted, 
        x='k', 
        y='Difference', 
        hue='Complexity_Method', 
        order=None,  # No need to set order for x-axis
        hue_order=hue_order,  # Set custom order for hue
        palette=palette
    )
    
    # Set y-axis ticks to show every 0.1 increment
    y_min, y_max = ax.get_ylim()  # Get the current y-axis limits
    ax.set_yticks(np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 0.1, 0.1))


    # Set plot titles and labels
    plt.title(f'Differences Between Score and 1 - Complexity for {performance_metric}')
    plt.xlabel('k')
    plt.ylabel('Difference: Score - (1 - Complexity)')
    plt.legend(title='Complexity Type and Method', loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()
    

def plot_score_differences_vars(df_merged, diff='diff_score_most_complex_class', x_var='best_method', y_title='Minority Class complexity Difference'):
    """
    Plots score differences using Seaborn.

    Parameters:
    -----------
    df_merged : pd.DataFrame
        Merged DataFrame containing all the necessary information for plotting.
    diff : str, optional (default='diff_score_most_complex_class')
        Column name for score differences.
    x_var : str, optional (default='best_method')
        Column name for x-axis variable.

    Returns:
    --------
    None
        The function returns None as it directly creates plots.
    """
    
    # Create violin plots for each unique metric
    for metric in df_merged['metric_x'].unique():
        g = sns.catplot(
            data=df_merged[df_merged['metric_x'] == metric],
            x=x_var,
            y=diff,
            hue="metric_y",
            kind="box",
           # inner="stick",
           # split=True,
            palette="pastel",
            legend_out=True  # Move legend outside the plot for better visibility
        )
        g.fig.suptitle(f"Score differences for {metric}", y=1.02)  # Set title above the plot

        # Get the legend and set its title and position
        legend = g._legend
        legend.set_title("Complexity Measure")
        legend.set_bbox_to_anchor((1, 0.9))  # Adjust position to upper right corner
        
        # Customize the legend appearance
        legend.get_frame().set_facecolor('white')  # Set background color to white
        legend.get_frame().set_edgecolor('grey')   # Set border color to grey
        legend.get_frame().set_linewidth(1.2)      # Set border width
        legend.get_frame().set_alpha(0.7)          # Make the legend background semi-transparent
        legend.set_frame_on(True)                  # Ensure that the frame is turned on
        
        # Add a horizontal line at y=0
        plt.axhline(0, color='black', linestyle='--', linewidth=1)

        plt.ylabel(f'{y_title}')
        plt.xlabel('Class Proportion')

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
# Function to plot residuals comparison (boxplot)
def plot_residuals_comparison(df_resid):
    plt.figure(figsize=(6, 6))
    g = sns.catplot(
        data=df_resid,
        x="metric_y",
        y="resid",
        kind="box",
        palette="pastel",
        legend_out=True
    )
    g.fig.suptitle("Residuals comparison", y=1.02)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylabel("Residuals")
    plt.xlabel("Complexity Measure")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot Observed vs Predicted for a group
def plot_observed_vs_predicted(y_test, y_pred, group_label):
    # Get colors for the scatter plots from the "pastel" palette
    colors = sns.color_palette("pastel", 2)
    if group_label == "ddn":
        color = colors[0]
    else:
        color = colors[1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color=color, edgecolor='k')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    plt.xlabel("Observed Score")
    plt.ylabel("Predicted Score")
    plt.title(f"Observed vs Predicted ({group_label})")
    plt.tight_layout()
    plt.show()

def plot_complexity_differences(df_differences):
    # Filter columns representing complexity differences
    complexity_diff_columns = [col for col in df_differences.columns if col.endswith("_difference")]

    # Melt DataFrame to long format
    df_melted = df_differences.melt(
        id_vars=["metric"],
        value_vars=complexity_diff_columns,
        var_name="Complexity Measure",
        value_name="Difference"
    )

    # Custom labels and order
    custom_labels = {
        "dataset_complexity_difference": "Dataset",
                "majority_class_complexity_difference": "Majority Class",
        "least_complex_class_difference": "Least Complex",
        "minority_class_complexity_difference": "Minority Class",
        "most_complex_class_difference": "Most Complex"
    }

    # Apply labels
    df_melted["Complexity Measure"] = df_melted["Complexity Measure"].map(custom_labels)

    # Replace metric labels
    df_melted["metric"] = df_melted["metric"].replace({"ddn": "DDN", "kdn": "kDN"})

    # Set categorical order
    order = list(custom_labels.values())
    df_melted["Complexity Measure"] = pd.Categorical(df_melted["Complexity Measure"], categories=order, ordered=True)

    # Define color palette
    pastel_colors = sns.color_palette("pastel")
    custom_palette = {"kDN": pastel_colors[0], "DDN": pastel_colors[1]}

    # Plot boxplot
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=df_melted,
        x="Complexity Measure",
        y="Difference",
        hue="metric",
        palette=custom_palette,
        order=order
    )

    # Customize plot
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("")
    plt.ylabel("")
    plt.ylim(-0.1, 0.1)

    # Set legend title
    ax.legend(title="Complexity Measure")
    ax.tick_params(axis='both', labelsize=12)

    plt.show()

def plot_line_correlations_new(df, score_metric, method_labels={"global": "Global", "mean_folds": "Mean Folds"}):
    """
    Plot line graphs of correlations for each complexity metric by k, method, and complexity_metric.
    Colors represent complexity metrics (kDN, DDN), and line styles represent methods (Global, Mean Folds).
    The maximum value of each line is highlighted with a black dot.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing columns: 'k', 'method', 'complexity_metric', and correlation metrics.
    score_metric : str
        Performance metric name for plot titles.
    method_labels : dict, optional
        Dictionary mapping original method names to custom labels.

    Returns:
    --------
    None
    """

    # List of correlation columns to plot
    correlation_columns = [
        'corr_dataset_complexity', 
        'corr_majority_class_complexity', 
        'corr_minority_class_complexity', 
        'corr_most_complex_class', 
        'corr_least_complex_class'
    ]

    # Replace method labels BEFORE plotting
    df = df.copy()
    df["method"] = df["method"].replace(method_labels)
    
    # Replace metric labels
    df["complexity_metric"] = df["complexity_metric"].replace({"ddn": "DDN", "kdn": "kDN"})

    pastel_colors = sns.color_palette("pastel")
    custom_palette = {"kDN": pastel_colors[0], "DDN": pastel_colors[1]}

    for col in correlation_columns:
        plt.figure(figsize=(14, 8))

        # Create line plot
        ax = sns.lineplot(
            data=df,
            x='k',
            y=col,
            hue='complexity_metric',
            style='method',
            palette=custom_palette,
            markers=True,
            dashes={'Global': '', 'Mean Folds': (5, 5)},
            hue_order=["kDN", "DDN"],
            style_order=['Global', 'Mean Folds']
        )

        # Mark the highest value on each line with a black dot
        for (complexity_metric, method), group_data in df.groupby(['complexity_metric', 'method']):
            max_idx = group_data[col].idxmax()
            max_k = group_data.loc[max_idx, 'k']
            max_val = group_data.loc[max_idx, col]

            ax.plot(max_k, max_val, marker='*', color='darkgoldenrod', markersize=10, markeredgecolor='black', markeredgewidth=0.5)

        print(f'Correlation of {(col.replace("corr", "")).replace("_", " ").title()} with Performance ({score_metric})')
        # Customize plot
        #plt.title(f'Correlation of {(col.replace("corr", "")).replace("_", " ").title()} with Performance ({score_metric})')
        plt.ylabel('Spearman Correlation')
        plt.xlabel('k')
        plt.grid(axis='y', linestyle='--', alpha=1)
        

        # Obtaining the legend handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # Extracting the first two handles and labels
        clean_handles = [handles[1], handles[2], handles[4], handles[5]]
        clean_labels = ['kDN', 'DDN', 'Global', 'Mean Folds']

        # Creating the legend
        ax.legend(clean_handles, clean_labels, title='', loc='lower left')

        plt.show()
