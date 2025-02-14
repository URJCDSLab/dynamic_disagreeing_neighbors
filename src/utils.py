import json
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_store_name(experiment, results_folder):
    return os.path.join(results_folder, f'{experiment}.json')


def scaled_mcc(y_true, y_pred):
    matthews_corrcoef_scaled = (matthews_corrcoef(y_true, y_pred) + 1)/2
    return matthews_corrcoef_scaled


def gps_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Evitar división por cero usando una pequeña constante en el denominador
    eps = 1e-10

    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    ppv = tp / (tp + fp + eps)
    npv = tn / (tn + fn + eps)

    gps_num = 4 * ppv * recall * specificity * npv
    gps_denom = (ppv * recall * npv) + (ppv * recall * specificity) + (npv * specificity * ppv) + (npv * specificity * recall)

    gps = gps_num / (gps_denom + eps)
    return gps


def named_scorer(score_func, name, greater_is_better=True, **kwargs):
    scorer = make_scorer(score_func, greater_is_better=greater_is_better, **kwargs)
    scorer.__name__ = name
    return scorer


# Function to compute residuals
def compute_residuals(y_test, y_pred):
    return y_test - y_pred

# Function to train and evaluate the model with cross-validation
def train_and_evaluate_model(data, label):
    # Select predictor variables and target variable
    X = data[["dataset_complexity", "minority_class_complexity", "metric_x"]]
    # Convert categorical variable 'metric_x' into dummy variables
    X = pd.get_dummies(X, columns=["metric_x"], drop_first=True)
    y = data["score"]
    
    # Split the data into training (70%) and test (30%), stratifying by 'metric_x'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=data['metric_x'], random_state=42
    )
    
    # Define a pipeline that includes feature selection and the decision tree regressor
    pipeline = Pipeline([
        ('select', SelectKBest(score_func=f_regression)),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])
    
    # Define the grid of hyperparameters to tune:
    # - 'select__k': number of features to select (or 'all' to use all features)
    # - 'regressor__max_depth': maximum depth of the tree
    param_grid = {
        'select__k': [2, 'all'],
        'regressor__max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    
    # Configure GridSearchCV with 5-fold cross-validation
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Fit GridSearchCV on the training set
    grid.fit(X_train, y_train)
    
    print(f"[{label}] Best parameters:", grid.best_params_)
    print(f"[{label}] Best validation score (neg MSE):", grid.best_score_)
    
    # Select the best model found and predict on the test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    return best_model, X_test, y_test, y_pred

