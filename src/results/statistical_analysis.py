
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro, spearmanr


def perform_paired_tests(df, remove_outliers=False, return_outliers=False):
    """
    Perform paired statistical tests (t-test or Wilcoxon) on specified columns of the DataFrame 
    with an option to remove outliers and return them.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to be tested. Must include the columns specified in `columns_to_test`.
    remove_outliers : bool, optional
        If True, outliers will be removed before performing the tests. Default is False.
    return_outliers : bool, optional
        If True, returns a DataFrame with the original records of the outliers detected during the analysis. Default is False.

    Returns:
    --------
    pandas.DataFrame (optional)
        A DataFrame containing the original records of the outliers detected for each column tested, if `return_outliers` is True.
    """
    # Define the columns to test
    columns_to_test = [
        ('dataset_complexity_global', 'dataset_complexity_mean_folds'),
        ('majority_class_complexity_global', 'majority_class_complexity_mean_folds'),
        ('minority_class_complexity_global', 'minority_class_complexity_mean_folds'),
        ('most_complex_class_global', 'most_complex_class_mean_folds'),
        ('least_complex_class_global', 'least_complex_class_mean_folds')
    ]

    def interpret_significance(p_value):
        """
        Interpret the significance level of a p-value.

        Parameters:
        -----------
        p_value : float
            The p-value from a statistical test.

        Returns:
        --------
        str
            A string describing the significance level of the p-value.
        """
        if p_value < 0.01:
            return "The difference is highly significant at the 99% confidence level (p < 0.01). There is strong evidence that the two measures are different."
        elif p_value < 0.05:
            return "The difference is significant at the 95% confidence level (p < 0.05). It is likely that the two measures are different."
        elif p_value < 0.10:
            return "The difference is moderately significant at the 90% confidence level (p < 0.10). There is some evidence that the two measures are different."
        else:
            return "The difference is not statistically significant. There is insufficient evidence to conclude that the two measures are different."

    def remove_outliers_iqr(data, original_data):
        """
        Remove outliers from a data series based on the Interquartile Range (IQR) and return original records.

        Parameters:
        -----------
        data : pandas.Series or numpy.ndarray
            The data from which to remove outliers.
        original_data : pandas.DataFrame
            The original DataFrame containing the complete records of the data.

        Returns:
        --------
        tuple
            A tuple containing:
            - filtered_data: Data with outliers removed.
            - outliers_records: The original records corresponding to the detected outliers.
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        outliers_idx = data[(data < lower_bound) | (data > upper_bound)].index
        outliers_records = original_data.loc[outliers_idx]
        return filtered_data, outliers_records

    # DataFrame to collect outliers if required
    outliers_list = []

    # Perform paired tests for each method and each set of columns
    for metric in df['metric'].unique():
        metric_data = df[df['metric'] == metric]
        print(f"\nResults for measure: {metric}")

        for col_global, col_mean_folds in columns_to_test:
            data_global = metric_data[col_global].copy()
            data_mean_folds = metric_data[col_mean_folds].copy()

            # Remove outliers if specified
            if remove_outliers:
                data_global, outliers_global = remove_outliers_iqr(data_global, metric_data)
                data_mean_folds, outliers_mean_folds = remove_outliers_iqr(data_mean_folds, metric_data)

                # Collect original outliers if requested
                if return_outliers:
                    outliers_list.append(outliers_global)
                    outliers_list.append(outliers_mean_folds)

                # Align both arrays to have the same length after outlier removal
                min_length = min(len(data_global), len(data_mean_folds))
                data_global = data_global[:min_length]
                data_mean_folds = data_mean_folds[:min_length]

            # Test for normality using Shapiro-Wilk test
            shapiro_test_global = shapiro(data_global)
            shapiro_test_mean_folds = shapiro(data_mean_folds)
            
            if shapiro_test_global.pvalue > 0.05 and shapiro_test_mean_folds.pvalue > 0.05:
                # Both distributions are normal, perform paired t-test
                t_stat, p_val = ttest_rel(data_global, data_mean_folds)
                test_name = "Paired t-test"
            else:
                # Distributions are not normal, perform Wilcoxon signed-rank test
                t_stat, p_val = wilcoxon(data_global, data_mean_folds)
                test_name = "Wilcoxon signed-rank test"

            significance = interpret_significance(p_val)
            print(f"Comparison: {col_global} vs {col_mean_folds} - {test_name} (Outliers {'removed' if remove_outliers else 'included'}):")
            print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.3f}\n{significance}\n")

    # Return original outlier records if requested
    if return_outliers and outliers_list:
        outliers_df = pd.concat(outliers_list).drop_duplicates()
        return outliers_df
            
            
def calculate_spearman_correlations(df_performance, df_complexity):
    # Store the correlation results for each metric in df_performance
    results = {}

    # Loop through each unique metric in df_performance
    for performance_metric in df_performance['metric'].unique():
        # Filter df_performance for the current metric
        df_filtered_performance = df_performance[df_performance['metric'] == performance_metric]
        
        # Merge the filtered df_performance with df_complexity on 'dataset'
        merged_df = pd.merge(df_complexity, df_filtered_performance, on='dataset', how='inner')
        
        # Store the correlations for each combination of method, k, and metric in df_complexity (global or mean_folds)
        correlations = []

        for method in merged_df['method'].unique():
            for k in merged_df['k'].unique():
                for metric in merged_df['metric_x'].unique():  # 'metric_x' distinguishes between global and mean_folds in df_complexity
                    # Filter the data for the current method, k, and metric (global or mean_folds)
                    filtered_data = merged_df[(merged_df['method'] == method) & 
                                              (merged_df['k'] == k) & 
                                              (merged_df['metric_x'] == metric)]
                    
                    if not filtered_data.empty:
                        # Calculate Spearman correlations for each complexity variable with the performance score
                        corr_dataset_complexity, _ = spearmanr(1-filtered_data['dataset_complexity'], filtered_data['score'])
                        corr_majority_class_complexity, _ = spearmanr(1-filtered_data['majority_class_complexity'], filtered_data['score'])
                        corr_minority_class_complexity, _ = spearmanr(1-filtered_data['minority_class_complexity'], filtered_data['score'])
                        corr_most_complex_class, _ = spearmanr(1-filtered_data['most_complex_class'], filtered_data['score'])
                        corr_least_complex_class, _ = spearmanr(1-filtered_data['least_complex_class'], filtered_data['score'])

                        correlations.append({
                            'method': method, # Specifies whether it's global or mean_folds
                            'k': k,
                            'complexity_metric': metric,  
                            'performance_metric': performance_metric,  # Metric from df_performance (f1_score, etc.)
                            'corr_dataset_complexity': corr_dataset_complexity,
                            'corr_majority_class_complexity': corr_majority_class_complexity,
                            'corr_minority_class_complexity': corr_minority_class_complexity,
                            'corr_most_complex_class': corr_most_complex_class,
                            'corr_least_complex_class': corr_least_complex_class
                        })

        # Convert the list of correlations to a DataFrame and store it with the performance metric as the key
        results[performance_metric] = pd.DataFrame(correlations)

    return results


def summarize_correlations(df):
    # Group by method, complexity metric, and performance metric to calculate summary statistics
    summary = df.groupby(['method', 'complexity_metric']).agg(
        mean_corr_dataset=('corr_dataset_complexity', 'mean'),
        std_corr_dataset=('corr_dataset_complexity', 'std'),
        mean_corr_majority=('corr_majority_class_complexity', 'mean'),
        std_corr_majority=('corr_majority_class_complexity', 'std'),
        mean_corr_minority=('corr_minority_class_complexity', 'mean'),
        std_corr_minority=('corr_minority_class_complexity', 'std'),
        mean_corr_most_complex=('corr_most_complex_class', 'mean'),
        std_corr_most_complex=('corr_most_complex_class', 'std'),
        mean_corr_least_complex=('corr_least_complex_class', 'mean'),
        std_corr_least_complex=('corr_least_complex_class', 'std'),
    ).reset_index()

    return summary


def calculate_statistical_differences(df, performance_metric, k=1):
    """
    Calculate the statistical differences between diff_score_minority_class_complexity and 
    diff_score_most_complex_class for a specified k value, comparing each method (kdn and ddn) 
    depending on the performance metric.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the performance and complexity data. 
        Must include 'k', 'metric_x', 'metric_y', 'diff_score_minority_class_complexity',
        and 'diff_score_most_complex_class' columns.

    performance_metric : str
        The performance metric to filter and analyze (e.g., 'accuracy_score').
        
    k : int, optional (default=1)
        The value of k to filter and analyze.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the statistical results of the tests.
    str
        A textual explanation of the test results, indicating if there are significant differences or not.
    """
    # Filter data for the specified k value and the selected performance metric
    df_filtered = df[(df['k'] == k) & (df['metric_x'] == performance_metric)]

    # Initialize a list to store results
    results = []
    explanation = ""

    # Loop over the methods 'kdn' and 'ddn'
    for method in ['kdn', 'ddn']:
        method_data = df_filtered[df_filtered['metric_y'] == method]

        # Extract the differences
        diff_minority = method_data['diff_score_minority_class_complexity']
        diff_most_complex = method_data['diff_score_most_complex_class']

        # Perform normality test (Shapiro-Wilk test)
        stat, p_normality = shapiro(diff_minority - diff_most_complex)

        # Choose the statistical test based on the normality of the data
        if p_normality > 0.05:
            # Use paired t-test if normal
            test_stat, p_value = ttest_rel(diff_minority, diff_most_complex)
            test_used = 'Paired t-test'
        else:
            # Use Wilcoxon signed-rank test if not normal
            test_stat, p_value = wilcoxon(diff_minority, diff_most_complex)
            test_used = 'Wilcoxon signed-rank test'

        # Interpret the results
        if p_value < 0.01:
            interpretation = "There is a highly significant difference at the 99% confidence level (p < 0.01)."
        elif p_value < 0.05:
            interpretation = "There is a significant difference at the 95% confidence level (p < 0.05)."
        elif p_value < 0.10:
            interpretation = "There is a moderately significant difference at the 90% confidence level (p < 0.10)."
        else:
            interpretation = "There is no statistically significant difference between the two measures."

        # Store the results
        results.append({
            'method': method,
            'performance_metric': performance_metric,
            'k': k,
            'test_used': test_used,
            'test_statistic': test_stat,
            'p_value': p_value,
            'normality_p_value': p_normality,
            'normality_test': 'Shapiro-Wilk'
        })

        # Add interpretation to the explanation
        explanation += (f"\nFor method {method} using {test_used}: {interpretation} "
                        f"Test statistic: {test_stat:.3f}, p-value: {p_value:.3f}.")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df, explanation
