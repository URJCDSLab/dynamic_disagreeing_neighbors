import os
import pandas as pd
import warnings
import json
warnings.filterwarnings('ignore') # Ignore warnings to keep the output clean

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.utils import named_scorer, get_store_name, NpEncoder # Importing custom utilities

# Loop through all the specified experiments (datasets)
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
    print(f'Experiment: {experiment}\n') # Print the current experiment being processed
    
    # Define the scoring method (accuracy in this case)
    score = named_scorer(accuracy_score, 'accuracy_score', greater_is_better=True)

    # Create the results folder if it doesn't exist
    results_folder = f'results/performance/{score.__name__}'
    os.makedirs(results_folder, exist_ok=True)

    # Load the preprocessed dataset for the current experiment
    data = pd.read_parquet(f'data/preprocessed/{experiment}.parquet')

    # Split the dataset into features (X) and target (y)
    X = data.drop(columns=['y']).values
    y = data.y.values

    rng_seed = 1234 # Set a random seed for reproducibility

    # Define a mapping of model names to their classes
    methods_mapping = {
        'SVC': SVC,
        'KNeighborsClassifier': KNeighborsClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier
    }

    # Define the models and their hyperparameters to be tested
    methods = [
        [SVC, {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf'], 'random_state': [rng_seed]}],
        [KNeighborsClassifier, {'n_neighbors': list(range(1, 13, 2)), 'n_jobs': [-1]}],
        [RandomForestClassifier, {'max_features': [None, 'sqrt', 'log2'], 'n_estimators': [100, 300, 500], 'max_depth': [5, 10, None],'random_state': [rng_seed]}],
        [GradientBoostingClassifier, {'max_features': [None, 'sqrt', 'log2'], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.2, 0.3], 'n_estimators': [100, 300, 500], 'random_state': [rng_seed]}]
    ]

    try:
        # Try to load existing results for the current experiment (if they exist)
        file_path = os.path.join(results_folder, f'{experiment}.json')
        with open(file_path, 'r') as fin:
            exp_info = json.load(fin)
    except:
        # If no existing results, initialize empty variables for tracking the best model
        exp_info = {experiment: {}}
        best_method = None
        best_params = None
        best_score_sd = None
        best_score = 0

        exp_info[experiment]['all_models'] = []

        # Loop through each model and its corresponding hyperparameters
        for method, grid_params in methods:
            print(f'Testing: {str(method())[:-2]} \n') # Print the model being tested
            
            # Perform grid search with cross-validation for hyperparameter tuning
            clf = GridSearchCV(method(), grid_params, scoring=score, n_jobs=-1, cv=5, verbose=2)
            clf.fit(X, y)

            # Store the results for the current model
            model_info = {
                'method': str(method())[:-2],
                'params': clf.best_params_,
                'cv_score': clf.best_score_,
                'cv_score_sd': clf.cv_results_['std_test_score'][clf.best_index_]
            }
            print(f'Best method: {str(method())[:-2]} - CV score: {clf.best_score_}±{clf.cv_results_} \n')

            exp_info[experiment]['all_models'].append(model_info)

            # Update the best model if the current model's score is higher
            if clf.best_score_ > best_score:
                best_method = model_info['method']
                best_params = model_info['params']
                best_score = model_info['cv_score']
                best_score_sd = model_info['cv_score_sd']

        # Save the best model information
        exp_info[experiment]['best_method'] = {
            'method': best_method,
            'params': best_params,
            'cv_score': best_score,
            'cv_score_sd': best_score_sd
        }

        print(f'Best method: {best_method} - CV score: {best_score}±{best_score_sd} \n')

    # Store information about the best model
    clf = methods_mapping[exp_info[experiment]['best_method']['method']](**exp_info[experiment]['best_method']['params'])

    print(exp_info) # Print the experiment info

    # Save the experiment information as a JSON file
    with open(get_store_name(experiment, results_folder), 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)
