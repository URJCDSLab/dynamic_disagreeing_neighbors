import os
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.utils import get_store_name, NpEncoder
from src.model.instance_hardness import *
from src.model.ddn import *

for experiment in [
# 'a9a',
'appendicitis',
# 'australian',
# 'backache',
# 'banknote',
# 'breastcancer',
# 'bupa',
# 'cleve',
# 'cod-rna',
# 'colon-cancer',
# 'diabetes',
# 'flare',
# 'fourclass',
# 'german_numer',
# 'haberman',
# 'heart',
# 'housevotes84',
# 'ilpd',
# 'ionosphere',
# 'kr_vs_kp',
# 'liver-disorders',
# 'mammographic',
# 'mushroom',
# 'r2',
# 'sonar',
# 'splice',
# 'svmguide1',
# 'svmguide3',
# 'transfusion',
# 'w1a',
# 'w2a',
# 'w3a',
# 'w4a',
# 'w5a',
# 'w6a',
# 'w7a',
# 'w8a'
]:
    print(f'Experiment: {experiment}\n')

    results_folder = 'results/complexity'

    os.makedirs(results_folder, exist_ok=True)

    data = pd.read_parquet(f'data/preprocessed/{experiment}.parquet')

    X = data.drop(columns=['y']).values
    y = data.y.values

    max_k = 9

    skf = StratifiedKFold(n_splits=5)
    
    # Initialize the dictionary to store the results
    exp_info = {experiment: {k: {'folds': {'kdn': {'global': [], 'class 0': [], 'class 1': []},
                                           'ddn': {'global': [], 'class 0': [], 'class 1': []}},
                                 'mean_folds': {},  # Para almacenar la media de cada fold
                                 'global': {}} for k in range(1, 8)}}
        
    for k in range(1, 8):
        print(k)
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
            
            # Almacenar los resultados en las listas correspondientes
            exp_info[experiment][k]['folds']['kdn']['global'].append(global_complexity_kdn)
            exp_info[experiment][k]['folds']['kdn']['class 0'].append(class0_complexity_kdn)
            exp_info[experiment][k]['folds']['kdn']['class 1'].append(class1_complexity_kdn)
            
            exp_info[experiment][k]['folds']['ddn']['global'].append(global_complexity_ddn)
            exp_info[experiment][k]['folds']['ddn']['class 0'].append(class0_complexity_ddn)
            exp_info[experiment][k]['folds']['ddn']['class 1'].append(class1_complexity_ddn)

        # Calcular la media de los resultados por fold
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

        # Calcular los resultados globales (sin dividir en folds)
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

    with open(get_store_name(experiment, results_folder), 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)