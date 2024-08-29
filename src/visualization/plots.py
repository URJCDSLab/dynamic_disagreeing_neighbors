import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_individual_complexity_differences(df):
    # Plot for dataset_complexity_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='dataset_complexity_difference', data=df)
    plt.title('Differences in Dataset Complexity (global - mean_folds) by metric')
    plt.xlabel('Metric')
    plt.ylabel('Dataset Complexity Difference')
    plt.show()

    # Plot for majority_class_complexity_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='majority_class_complexity_difference', data=df)
    plt.title('Differences in Majority Class Complexity (global - mean_folds) by metric')
    plt.xlabel('Metric')
    plt.ylabel('Majority Class Complexity Difference')
    plt.show()

    # Plot for minority_class_complexity_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='minority_class_complexity_difference', data=df)
    plt.title('Differences in Minority Class Complexity (global - mean_folds) by metric')
    plt.xlabel('Metric')
    plt.ylabel('Minority Class Complexity Difference')
    plt.show()

    # Plot for most_complex_class_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='most_complex_class_difference', data=df)
    plt.title('Differences in Most Complex Class (global - mean_folds) by metric')
    plt.xlabel('Metric')
    plt.ylabel('Most Complex Class Difference')
    plt.show()

    # Plot for least_complex_class_difference
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='metric', y='least_complex_class_difference', data=df)
    plt.title('Differences in Least Complex Class (global - mean_folds) by metric')
    plt.xlabel('Metric')
    plt.ylabel('Least Complex Class Difference')
    plt.show()


def plot_line_correlations(df):
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
        plt.title(f'Correlation of {col.replace("_", " ").title()} with Performance (scaled_mcc_score)')
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
