import os
import pandas as pd
import warnings
import json
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.utils import named_scorer, get_store_name, NpEncoder, scaled_mcc

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
    'w8a'
]:
    print(f'Experiment: {experiment}\n')
    
    score = named_scorer(scaled_mcc, 'scaled_mcc_score', greater_is_better=True)

    results_folder = f'results/performance/{score.__name__}'
    os.makedirs(results_folder, exist_ok=True)

    data = pd.read_parquet(f'data/{experiment}.parquet')

    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))
    y = data.y.values

    y[y == -1] = 0
    if y.sum() > len(y) - y.sum():
        y = abs(y - 1)
    y = y.astype(int)
    rng_seed = 1234

    methods_mapping = {
        'SVC': SVC,
        'KNeighborsClassifier': KNeighborsClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier
    }

    methods = [
        [SVC, {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf'], 'random_state': [rng_seed]}],
        [KNeighborsClassifier, {'n_neighbors': list(range(1, 13, 2)), 'n_jobs': [-1]}],
        [RandomForestClassifier, {'max_features': [None, 'sqrt', 'log2'], 'n_estimators': [100, 300, 500], 'max_depth': [5, 10, None],'random_state': [rng_seed]}],
        [GradientBoostingClassifier, {'max_features': [None, 'sqrt', 'log2'], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.2, 0.3], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}]
    ]

    try:
        with open(f'results/errors/{experiment}.json', 'r') as fin:
            exp_info = json.load(fin)
    except:
        exp_info = {experiment: {}}
        best_method = None
        best_params = None
        best_score_sd = None
        best_score = 0

        exp_info[experiment]['all_models'] = []

        for method, grid_params in methods:
            print(f'Testing: {str(method())[:-2]} \n')

            clf = GridSearchCV(method(), grid_params, scoring=score, n_jobs=-1, cv=5, verbose=2)
            clf.fit(X, y)

            model_info = {
                'method': str(method())[:-2],
                'params': clf.best_params_,
                'cv_score': clf.best_score_,
                'cv_score_sd': clf.cv_results_['std_test_score'][clf.best_index_]
            }
            print(f'Best method: {str(method())[:-2]} - CV score: {clf.best_score_}±{clf.cv_results_} \n')

            exp_info[experiment]['all_models'].append(model_info)

            if clf.best_score_ > best_score:
                best_method = model_info['method']
                best_params = model_info['params']
                best_score = model_info['cv_score']
                best_score_sd = model_info['cv_score_sd']

        exp_info[experiment]['best_method'] = {
            'method': best_method,
            'params': best_params,
            'cv_score': best_score,
            'cv_score_sd': best_score_sd
        }

        print(f'Best method: {best_method} - CV score: {best_score}±{best_score_sd} \n')

    # Store information about the best model
    clf = methods_mapping[exp_info[experiment]['best_method']['method']](**exp_info[experiment]['best_method']['params'])

    print(exp_info)

    with open(get_store_name(experiment, results_folder), 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)
