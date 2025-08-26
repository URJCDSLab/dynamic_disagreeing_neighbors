import os
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dynamic_disagreeing_neighbors.utils import get_store_name, NpEncoder
from dynamic_disagreeing_neighbors.model.instance_hardness import *
from dynamic_disagreeing_neighbors.model.ddn import *

def process_experiments(max_k = 11, evaluate=True):
    for experiment in [
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
    ]:
        print(f'Experiment: {experiment}\n')

        results_folder = 'results/complexity'
        os.makedirs(results_folder, exist_ok=True)

        results_file = get_store_name(experiment, results_folder)

        if os.path.exists(results_file) and evaluate:
            # Load existing results if in evaluation mode
            with open(results_file, 'r') as fin:
                exp_info = json.load(fin)
        else:
            # Initialize the dictionary to store the results (overwrite mode)
            exp_info = {}

        data = pd.read_parquet(f'data/preprocessed/{experiment}.parquet')

        X = data.drop(columns=['y']).values
        y = data.y.values        

        skf = StratifiedKFold(n_splits=5)

        # Process each value of k
        for k in range(1, max_k):
            if evaluate and str(k) in exp_info.get(experiment, {}):
                print(f"Skipping k={k} for experiment {experiment}, already processed.")
                continue

            print(f'Processing k={k} for experiment {experiment}')

            # Ensure the experiment and k structure exists in the results
            if experiment not in exp_info:
                exp_info[experiment] = {}
            exp_info[experiment][k] = {'folds': {'kdn': {'global': [], 'class 0': [], 'class 1': []},
                                                 'ddn': {'global': [], 'class 0': [], 'class 1': []}},
                                       'mean_folds': {},
                                       'global': {}}

            for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                print('Fold:', i)
                complexity_kdn, _ = kdn_score(X[train_index], y[train_index], k)
                global_complexity_kdn = np.mean(complexity_kdn)
                class0_complexity_kdn = np.mean(complexity_kdn[y[train_index] < 1])
                class1_complexity_kdn = np.mean(complexity_kdn[y[train_index] > 0])
                print(f'complexity_kdn {global_complexity_kdn}')
                
                ddn = DDN(k=k)
                ddn.fit(X[train_index], y[train_index])
                complexity_ddn = ddn.complexity
                global_complexity_ddn = np.mean(complexity_ddn)
                class0_complexity_ddn = np.mean(complexity_ddn[y[train_index] < 1])
                class1_complexity_ddn = np.mean(complexity_ddn[y[train_index] > 0])
                print(f'complexity_ddn {global_complexity_ddn}')
                
                # Store the fold results
                exp_info[experiment][k]['folds']['kdn']['global'].append(global_complexity_kdn)
                exp_info[experiment][k]['folds']['kdn']['class 0'].append(class0_complexity_kdn)
                exp_info[experiment][k]['folds']['kdn']['class 1'].append(class1_complexity_kdn)
                
                exp_info[experiment][k]['folds']['ddn']['global'].append(global_complexity_ddn)
                exp_info[experiment][k]['folds']['ddn']['class 0'].append(class0_complexity_ddn)
                exp_info[experiment][k]['folds']['ddn']['class 1'].append(class1_complexity_ddn)

            # Calculate the mean of the fold results
            exp_info[experiment][k]['mean_folds']['kdn'] = {
                'global': np.mean(exp_info[experiment][k]['folds']['kdn']['global']),
                'class 0': np.mean(exp_info[experiment][k]['folds']['kdn']['class 0']),
                'class 1': np.mean(exp_info[experiment][k]['folds']['kdn']['class 1'])
            }

            exp_info[experiment][k]['mean_folds']['ddn'] = {
                'global': np.mean(exp_info[experiment][k]['folds']['ddn']['global']),
                'class 0': np.mean(exp_info[experiment][k]['folds']['ddn']['class 0']),
                'class 1': np.mean(exp_info[experiment][k]['folds']['ddn']['class 1'])
            }

            # Calculate the global results (without folding)
            complexity_kdn_global, _ = kdn_score(X, y, k)
            exp_info[experiment][k]['global']['kdn'] = {
                'global': np.mean(complexity_kdn_global),
                'class 0': np.mean(complexity_kdn_global[y < 1]),
                'class 1': np.mean(complexity_kdn_global[y > 0])
            }

            ddn_global = DDN(k=k)
            ddn_global.fit(X, y)
            complexity_ddn_global = ddn_global.complexity
            exp_info[experiment][k]['global']['ddn'] = {
                'global': np.mean(complexity_ddn_global),
                'class 0': np.mean(complexity_ddn_global[y < 1]),
                'class 1': np.mean(complexity_ddn_global[y > 0])
            }

        # Save results after processing each experiment
        with open(results_file, 'w') as fout:
            json.dump(exp_info, fout, indent=3, cls=NpEncoder)

# Execute the function when the script is run directly
if __name__ == "__main__":
    process_experiments(max_k = 11, evaluate=True)  # Change to False if you want to overwrite existing results
