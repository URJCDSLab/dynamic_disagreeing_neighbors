
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
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        print(f"\nResults for method: {method}")

        for col_global, col_mean_folds in columns_to_test:
            data_global = method_data[col_global].copy()
            data_mean_folds = method_data[col_mean_folds].copy()

            # Remove outliers if specified
            if remove_outliers:
                data_global, outliers_global = remove_outliers_iqr(data_global, method_data)
                data_mean_folds, outliers_mean_folds = remove_outliers_iqr(data_mean_folds, method_data)

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
                        corr_most_complex_value, _ = spearmanr(1-filtered_data['most_complex_value'], filtered_data['score'])
                        corr_least_complex_value, _ = spearmanr(1-filtered_data['least_complex_value'], filtered_data['score'])

                        correlations.append({
                            'method': method,
                            'k': k,
                            'complexity_metric': metric,  # Specifies whether it's global or mean_folds
                            'performance_metric': performance_metric,  # Metric from df_performance (f1_score, etc.)
                            'corr_dataset_complexity': corr_dataset_complexity,
                            'corr_majority_class_complexity': corr_majority_class_complexity,
                            'corr_minority_class_complexity': corr_minority_class_complexity,
                            'corr_most_complex_value': corr_most_complex_value,
                            'corr_least_complex_value': corr_least_complex_value
                        })

        # Convert the list of correlations to a DataFrame and store it with the performance metric as the key
        results[performance_metric] = pd.DataFrame(correlations)

    return results
