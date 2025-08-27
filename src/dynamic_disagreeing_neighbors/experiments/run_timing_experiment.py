import time
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from dynamic_disagreeing_neighbors.model.ddn import DDN
from dynamic_disagreeing_neighbors.model.instance_hardness import kdn_score

warnings.filterwarnings('ignore')

def run_timing_experiment():
    """
    Compares the runtime of kDN and DDN on simulated datasets of
    increasing size.
    """
    # --- Experiment Configuration ---
    SAMPLE_SIZES = [100, 1000, 5000, 10000] # You can add 100000 for the final run
    N_FEATURES = 10
    K = 7  # A fixed k for a fair comparison
    N_REPEATS = 3  # Number of times to repeat each measurement for stability

    results = []

    print("Starting Runtime Comparison Experiment...")
    print(f"Sample Sizes: {SAMPLE_SIZES}")
    print(f"Number of Repeats per size: {N_REPEATS}\n")

    for n_samples in SAMPLE_SIZES:
        print(f"--- Processing n_samples = {n_samples} ---")
        
        times_kdn = []
        times_ddn = []

        for i in range(N_REPEATS):
            print(f"  Repeat {i+1}/{N_REPEATS}...")
            # 1. Generate synthetic data
            X, y = make_classification(
                n_samples=n_samples,
                n_features=N_FEATURES,
                n_informative=5,
                n_redundant=2,
                n_classes=2,
                flip_y=0.05,
                random_state=42 + i
            )

            # 2. Time kDN (using the corrected function)
            start_time_kdn = time.time()
            kdn_score(X, y, k=K) # Using your specific function
            end_time_kdn = time.time()
            times_kdn.append(end_time_kdn - start_time_kdn)

            # 3. Time DDN
            ddn = DDN(k=K)
            start_time_ddn = time.time()
            ddn.fit(X, y)
            end_time_ddn = time.time()
            times_ddn.append(end_time_ddn - start_time_ddn)

        # Calculate the average time for this sample size
        avg_time_kdn = np.mean(times_kdn)
        avg_time_ddn = np.mean(times_ddn)

        print(f"  Avg. time kDN: {avg_time_kdn:.4f}s")
        print(f"  Avg. time DDN: {avg_time_ddn:.4f}s")

        results.append({'method': 'kDN', 'n_samples': n_samples, 'time': avg_time_kdn})
        results.append({'method': 'DDN', 'n_samples': n_samples, 'time': avg_time_ddn})

    # 4. Process and save results
    df_results = pd.DataFrame(results)
    print("\n--- Experiment Finished ---")
    print("Results:")
    print(df_results)
    df_results.to_csv(f'results/timing_experiment_results_{N_FEATURES}_{K}.csv', index=False)
    print("\nResults saved to results/timing_experiment_results.csv")

    # 5. Plot the results
    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(12, 6))

    colors = sns.color_palette("pastel", 2)
    palette = {"DDN": colors[1], "kDN": colors[0]}
    
    # 5. Generate the line plot with updated styling
    ax = sns.lineplot(
        data=df_results,
        x='n_samples',
        y='time',
        hue='method',
        palette=palette,           # Use custom palette
        marker='o',
        markersize=12,
        linewidth=2.5,
        alpha=0.7,                 # Add transparency
        markeredgecolor='black',   # Add black edges to markers
        markeredgewidth=1.5
    )
    
    ax.set_xlabel('Number of Samples (n)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set(xscale="log", yscale="log")
    
    sns.despine(trim=False) # Ensure full box is drawn
    
    plt.tight_layout()
    plt.savefig('images/timing_experiment_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to images/timing_experiment_plot.png")
    plt.show()

if __name__ == "__main__":
    run_timing_experiment()