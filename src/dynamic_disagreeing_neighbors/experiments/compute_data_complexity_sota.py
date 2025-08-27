import os
import json
import warnings

import numpy as np
import pandas as pd

from dynamic_disagreeing_neighbors.utils import get_store_name, NpEncoder
from pyhard.measures import ClassificationMeasures

warnings.filterwarnings('ignore')

def calculate_aggregates(complexity_array, y):
    """Helper function to calculate global, class 0, and class 1 average complexity."""
    # Ensure complexity_array is a numpy array for boolean indexing
    complexity_array = np.array(complexity_array)
    y_class0_mask = (y == 0)
    y_class1_mask = (y == 1)

    # Calculate means, handling cases where a class might be empty in a fold
    mean_global = np.nanmean(complexity_array)
    mean_class0 = np.nanmean(complexity_array[y_class0_mask]) if y_class0_mask.any() else np.nan
    mean_class1 = np.nanmean(complexity_array[y_class1_mask]) if y_class1_mask.any() else np.nan

    return {
        'global': mean_global,
        'class 0': mean_class0,
        'class 1': mean_class1
    }

def process_sota_complexity_measures(evaluate=True):
    """
    Runs a new, simplified experiment to calculate SOTA complexity measures.
    This only computes the 'global' complexity on the full dataset.
    """
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

    results_folder = 'results/complexity_sota'
    os.makedirs(results_folder, exist_ok=True)

    for experiment in datasets:
        print(f"--- Processing experiment: {experiment} ---")
        
        results_file = get_store_name(experiment, results_folder)

        if os.path.exists(results_file) and evaluate:
            print(f"Skipping {experiment}, results already exist.")
            continue

        try:
            data = pd.read_parquet(f'data/preprocessed/{experiment}.parquet')
            X = data.drop(columns=['y']).values
            y = data.y.values
        except Exception as e:
            print(f"Could not load data for {experiment}. Error: {e}")
            continue

        # This object calculates many measures at once
        cm = ClassificationMeasures(data, target_col='y')
        
        # Dictionary to store all results for this dataset
        exp_results = {}

        print("Calculating State-of-the-Art measures...")
        # --- Calculate New SOTA Measures (k-independent) ---
        exp_results['N1'] = calculate_aggregates(cm.borderline_points(), y)
        exp_results['N2'] = calculate_aggregates(cm.intra_extra_ratio(), y)
        exp_results['F1'] = calculate_aggregates(cm.f1(), y)
        exp_results['CLD'] = calculate_aggregates(cm.class_likeliood_diff(), y)

        # Save results for this experiment
        with open(results_file, 'w') as fout:
            json.dump(exp_results, fout, indent=4, cls=NpEncoder)
        
        print(f"Finished {experiment}. Results saved to {results_file}\n")

if __name__ == "__main__":
    # Set evaluate=False if you want to overwrite and re-run all experiments
    process_sota_complexity_measures(evaluate=True)