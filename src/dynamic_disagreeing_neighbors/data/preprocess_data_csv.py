import os
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# List of experiments (datasets)
experiments = os.listdir('data/raw_csv')

experiments = [experiment.removesuffix('.csv') for experiment in experiments]

# Directory to save preprocessed datasets
preprocessed_dir = 'data/preprocessed'
os.makedirs(preprocessed_dir, exist_ok=True)

for experiment in experiments:
    print(f'Experiment: {experiment}\n')

    # Load the dataset
    data = pd.read_csv(f'data/raw_csv/{experiment}.csv', header=0, index_col=0)

    # Preprocessing the data
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))  # Scale the features
    y = data.y.values

    # Replace all instances of -1 in y with 0 to standardize the binary class labels to 0 and 1.
    y[y == -1] = 0

    # If the number of 1s in y exceeds the number of 0s, invert the class labels.
    # This ensures that 1 represents the minority class.
    if y.sum() > len(y) - y.sum():
        y = abs(y - 1)

    # Convert y to integers to ensure the class labels are stored as integers.
    y = y.astype(int)

    # Save preprocessed data in the new directory with the same name
    preprocessed_data = pd.DataFrame(X, columns=data.columns.drop('y'))
    preprocessed_data['y'] = y
    preprocessed_data.to_parquet(f'{preprocessed_dir}/{experiment}.parquet')
