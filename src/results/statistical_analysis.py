
import pandas as pd
from scipy.stats import ttest_rel, spearmanr


def perform_paired_t_tests(df):
    # Define the columns to test
    columns_to_test = [
        ('dataset_complexity_global', 'dataset_complexity_mean_folds'),
        ('majority_class_complexity_global', 'majority_class_complexity_mean_folds'),
        ('minority_class_complexity_global', 'minority_class_complexity_mean_folds'),
        ('most_complex_class_global', 'most_complex_class_mean_folds'),
        ('least_complex_class_global', 'least_complex_class_mean_folds')
    ]

    # Function to interpret the significance level
    def interpret_significance(p_value):
        if p_value < 0.01:
            return "The difference is highly significant at the 99% confidence level (p < 0.01). There is strong evidence that the two measures are different."
        elif p_value < 0.05:
            return "The difference is significant at the 95% confidence level (p < 0.05). It is likely that the two measures are different."
        elif p_value < 0.10:
            return "The difference is moderately significant at the 90% confidence level (p < 0.10). There is some evidence that the two measures are different."
        else:
            return "The difference is not statistically significant. There is insufficient evidence to conclude that the two measures are different."

    # Perform paired t-tests for each method and each set of columns
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        print(f"\nResults for method: {method}")
        for col_global, col_mean_folds in columns_to_test:
            t_stat, p_val = ttest_rel(method_data[col_global], method_data[col_mean_folds])
            significance = interpret_significance(p_val)
            print(f"Comparison: {col_global} vs {col_mean_folds} - t-statistic: {t_stat:.3f}, p-value: {p_val:.3f}\n{significance}\n")


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
