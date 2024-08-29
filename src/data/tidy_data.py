
import pandas as pd
import json
import os


def df_description(df_path='data/preprocessed'):
    
    exp_data = sorted([i.replace('.parquet', '') for i in os.listdir(df_path) if i != '.gitkeep'])
    
    dfs = pd.DataFrame(columns=['instances', 'n_features', 'class_prop'], index=exp_data, data = [])

    for exp in exp_data:
        try:
            X = pd.read_parquet(f'{df_path}/{exp}.parquet')
            dfs.loc[exp, 'instances'] = X.shape[0]
            dfs.loc[exp, 'n_features'] = X.shape[1]-1
            dfs.loc[exp, 'class_prop'] = round(min(X['y'].value_counts()/X.shape[0]), 3)
        except :
            pass
        
    dfs.sort_index(inplace=True)
    return dfs


def performance_df():
    data = []

    metrics = ['accuracy_score','f1_score', 'gps_score', 'scaled_mcc_score']
    experiments = [
'a9a', 
        'appendicitis', 
        'australian', 
        'backache', 
        'banknote', 
        'breastcancer',
        'bupa', 
        'cleve', 
        'cod-rna', 
        'colon-cancer', 
        'diabetes', 
        'flare', 
        'fourclass',
        'german_numer', 
        'haberman', 
        'heart', 
        'housevotes84', 
        'ilpd', 
        'ionosphere',
        'kr_vs_kp', 
        'liver-disorders', 
        'mammographic', 
        'mushroom', 
        'r2', 
        'sonar',
        'splice', 
        'svmguide1', 
        'svmguide3', 
        'transfusion', 
        'w1a', 
        'w2a', 
        'w3a', 
        'w4a', 
        'w5a', 
        'w6a', 
        'w7a', 
        'w8a',
        'yeast1',
        'ecoli-0-1-4-7_vs_2-3-5-6',
        'yeast3',
        'analcatdata_lawsuit',
        'vehicle3',
        'spect',
        'yeast-0-2-5-7-9_vs_3-6-8',
        'page-blocks-1-3_vs_4',
        'ecoli-0-1-4-6_vs_5',
        'ecoli3',
        'ecoli4',
        'vehicle1',
        'page-blocks0',
        'vehicle0',
        'yeast6',
        'glass6',
        'yeast4',
        'glass2',
        'yeast5',
        'glass4',
        'ecoli1',
        'new-thyroid1',
        'ecoli2',
        'glass0',
        'dermatology-6',
        'glass1',
        'newthyroid2',
        'vehicle2'
    ]

    for metric in metrics:
        for experiment in experiments:
            file_path = f'results/performance/{metric}/{experiment}.json'
            
            # Verificar si el archivo existe
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as fin:
                        performance = json.load(fin)
                    # Agregar datos a la lista
                    data.append({
                        'dataset': experiment,
                        'metric': metric,
                        'score': performance[experiment]['best_method']['cv_score'],
                        'score_sd': performance[experiment]['best_method']['cv_score_sd']
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    # Manejar el error en caso de un archivo JSON corrupto o formato inesperado
                    print(f"Error leyendo o procesando el archivo {file_path}: {e}")

    # Convertir la lista de diccionarios en un DataFrame
    df = pd.DataFrame(data, columns=['dataset', 'metric', 'score', 'score_sd'])
    
    return df

def extract_complexity_df():
    data = []   

    datasets = [
        'a9a', 
        'appendicitis', 
        'australian', 
        'backache', 
        'banknote', 
        'breastcancer',
        'bupa', 
        'cleve', 
        'cod-rna', 
        'colon-cancer', 
        'diabetes', 
        'flare', 
        'fourclass',
        'german_numer', 
        'haberman', 
        'heart', 
        'housevotes84', 
        'ilpd', 
        'ionosphere',
        'kr_vs_kp', 
        'liver-disorders', 
        'mammographic', 
        'mushroom', 
        'r2', 
        'sonar',
        'splice', 
        'svmguide1', 
        'svmguide3', 
        'transfusion', 
        'w1a', 
        'w2a', 
        'w3a', 
        'w4a', 
        'w5a', 
        'w6a', 
        'w7a', 
        'w8a',
        'yeast1',
        'ecoli-0-1-4-7_vs_2-3-5-6',
        'yeast3',
        'analcatdata_lawsuit',
        'vehicle3',
        'spect',
        'yeast-0-2-5-7-9_vs_3-6-8',
        'page-blocks-1-3_vs_4',
        'ecoli-0-1-4-6_vs_5',
        'ecoli3',
        'ecoli4',
        'vehicle1',
        'page-blocks0',
        'vehicle0',
        'yeast6',
        'glass6',
        'yeast4',
        'glass2',
        'yeast5',
        'glass4',
        'ecoli1',
        'new-thyroid1',
        'ecoli2',
        'glass0',
        'dermatology-6',
        'glass1',
        'newthyroid2',
        'vehicle2'
    ]

    metrics = ['kdn', 'ddn']
    methods = ['mean_folds', 'global']
    classes = ['global', 'class 0', 'class 1']

    for dataset in datasets:
        file_path = f'results/complexity/{dataset}.json'
        
        # Check if the file exists
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as fin:
                    performance = json.load(fin)
                
                for k in range(1, 12):  # k from 1 to 11
                    k_str = str(k)
                    if k_str in performance.get(dataset, {}):
                        for metric in metrics:
                            for method in methods:
                                complexity_data = performance[dataset][k_str].get(method, {}).get(metric, {})
                                
                                # Get the complexity values
                                majority_complexity = complexity_data.get('class 0', None)
                                minority_complexity = complexity_data.get('class 1', None)
                                
                                # Append the data
                                data.append({
                                    'dataset': dataset,
                                    'k': k,
                                    'metric': metric,
                                    'method': method,
                                    'dataset_complexity': complexity_data.get('global', None),
                                    'majority_class_complexity': majority_complexity,
                                    'minority_class_complexity': minority_complexity
                                })

            except (json.JSONDecodeError, KeyError) as e:
                # Handle the error in case of a corrupt JSON file or unexpected format
                print(f"Error reading or processing file {file_path}: {e}")

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data, columns=[
        'dataset', 'k', 'metric', 'method', 
        'dataset_complexity', 'majority_class_complexity', 'minority_class_complexity'
    ])

    # Create columns for the max and min complexity between majority and minority classes
    df['most_complex_class'] = df[['majority_class_complexity', 'minority_class_complexity']].max(axis=1)
    df['least_complex_class'] = df[['majority_class_complexity', 'minority_class_complexity']].min(axis=1)
    
    return df


def calculate_differences(df):
    # Pivot the DataFrame to have `global` and `mean_folds` as columns for easy difference calculation
    pivoted_df = df.pivot_table(
        index=['dataset', 'k', 'metric'],
        columns='method',
        values=['dataset_complexity', 'majority_class_complexity', 'minority_class_complexity']
    ).reset_index()

    # Flatten multi-index columns correctly
    pivoted_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivoted_df.columns.values]

    # Calculate the differences for each complexity metric (global - mean_folds)
    pivoted_df['dataset_complexity_difference'] = pivoted_df['dataset_complexity_global'] - pivoted_df['dataset_complexity_mean_folds']
    pivoted_df['majority_class_complexity_difference'] = pivoted_df['majority_class_complexity_global'] - pivoted_df['majority_class_complexity_mean_folds']
    pivoted_df['minority_class_complexity_difference'] = pivoted_df['minority_class_complexity_global'] - pivoted_df['minority_class_complexity_mean_folds']

    # Calculate the most complex and least complex classes for global and mean_folds
    pivoted_df['most_complex_class_global'] = pivoted_df[['majority_class_complexity_global', 'minority_class_complexity_global']].max(axis=1)
    pivoted_df['least_complex_class_global'] = pivoted_df[['majority_class_complexity_global', 'minority_class_complexity_global']].min(axis=1)

    pivoted_df['most_complex_class_mean_folds'] = pivoted_df[['majority_class_complexity_mean_folds', 'minority_class_complexity_mean_folds']].max(axis=1)
    pivoted_df['least_complex_class_mean_folds'] = pivoted_df[['majority_class_complexity_mean_folds', 'minority_class_complexity_mean_folds']].min(axis=1)

    # Calculate the differences for the most and least complex classes
    pivoted_df['most_complex_class_difference'] = pivoted_df['most_complex_class_global'] - pivoted_df['most_complex_class_mean_folds']
    pivoted_df['least_complex_class_difference'] = pivoted_df['least_complex_class_global'] - pivoted_df['least_complex_class_mean_folds']

    return pivoted_df


def calculate_score_differences(df_performance, df_complexity):
    """
    Calculate the differences between the score and 1 - complexity for minority_class_complexity and most_complex_class.
    Merges performance data with complexity data and computes the differences.

    Parameters:
    -----------
    df_performance : pandas.DataFrame
        The DataFrame containing the performance scores. Must include 'dataset' and 'score' columns.

    df_complexity : pandas.DataFrame
        The DataFrame containing the complexity data. Must include 'dataset', 'method', 'minority_class_complexity',
        and 'most_complex_class'.

    Returns:
    --------
    pd.DataFrame
        A merged DataFrame with calculated differences between the score and 1 - complexity for the relevant columns.
    """
    # Filter to keep only the 'global' method complexities
    df_complexity_filtered = df_complexity[df_complexity['method'] == 'global'].copy()

    # Drop unnecessary columns
    df_complexity_filtered.drop(
        columns=['method', 'dataset_complexity', 'majority_class_complexity', 'least_complex_class'], 
        inplace=True
    )

    # Merge the performance and complexity data
    df_combined = pd.merge(df_performance, df_complexity_filtered, on='dataset', how='inner')

    # Calculate the differences between the score and 1 - complexity
    df_combined['diff_score_minority_class_complexity'] = df_combined['score'] - (1 - df_combined['minority_class_complexity'])
    df_combined['diff_score_most_complex_class'] = df_combined['score'] - (1 - df_combined['most_complex_class'])

    return df_combined
