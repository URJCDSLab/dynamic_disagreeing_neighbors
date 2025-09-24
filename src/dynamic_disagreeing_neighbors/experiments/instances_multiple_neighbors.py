import os
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dynamic_disagreeing_neighbors.model.ddn import DDN

def calculate_m_greater_than_1_stats():
    """
    Calculate for each dataset the percentage of instances where 
    the final neighborhood size m > 1 when k=1
    """
    
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
    
    results = {}
    
    for experiment in experiments:
        print(f'Processing: {experiment}')
        
        # Load data
        data = pd.read_parquet(f'data/preprocessed/{experiment}.parquet')
        X = data.drop(columns=['y']).values
        y = data.y.values
        
        # Fit DDN with k=1
        ddn = DDN(k=1)
        ddn.fit(X, y)
        
        # Count instances where m > 1
        # self.neighbours contains the list of neighbors for each instance
        neighborhood_sizes = [len(neighbors) for neighbors in ddn.neighbours]
        
        # Calculate statistics
        n_instances = len(neighborhood_sizes)
        n_m_greater_1 = sum(1 for m in neighborhood_sizes if m > 1)
        percentage_m_greater_1 = (n_m_greater_1 / n_instances) * 100
        
        # Store results
        results[experiment] = {
            'n_instances': n_instances,
            'n_m_equals_1': n_instances - n_m_greater_1,
            'n_m_greater_1': n_m_greater_1,
            'percentage_m_equals_1': 100 - percentage_m_greater_1,
            'percentage_m_greater_1': percentage_m_greater_1,
            'mean_m': np.mean(neighborhood_sizes),
            'median_m': np.median(neighborhood_sizes),
            'max_m': max(neighborhood_sizes),
            'min_m': min(neighborhood_sizes)
        }
        
        print(f"  - {experiment}: {percentage_m_greater_1:.2f}% instances have m>1")
    
    # Calculate overall statistics
    total_instances = sum(r['n_instances'] for r in results.values())
    total_m_greater_1 = sum(r['n_m_greater_1'] for r in results.values())
    overall_percentage = (total_m_greater_1 / total_instances) * 100
    
    results['overall_statistics'] = {
        'total_instances': total_instances,
        'total_m_greater_1': total_m_greater_1,
        'overall_percentage_m_greater_1': overall_percentage,
        'n_datasets': len(experiments),
        'mean_percentage_across_datasets': np.mean([r['percentage_m_greater_1'] for r in results.values() if 'percentage_m_greater_1' in r]),
        'median_percentage_across_datasets': np.median([r['percentage_m_greater_1'] for r in results.values() if 'percentage_m_greater_1' in r]),
        'std_percentage_across_datasets': np.std([r['percentage_m_greater_1'] for r in results.values() if 'percentage_m_greater_1' in r])
    }
    
    # Save results
    os.makedirs('results/analysis', exist_ok=True)
    with open('results/analysis/m_greater_than_1_stats.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total datasets analyzed: {results['overall_statistics']['n_datasets']}")
    print(f"Total instances analyzed: {results['overall_statistics']['total_instances']:,}")
    print(f"\nInstances where m>1 when k=1:")
    print(f"  - Count: {results['overall_statistics']['total_m_greater_1']:,}")
    print(f"  - Overall percentage: {results['overall_statistics']['overall_percentage_m_greater_1']:.2f}%")
    print(f"\nAcross datasets:")
    print(f"  - Mean percentage: {results['overall_statistics']['mean_percentage_across_datasets']:.2f}%")
    print(f"  - Median percentage: {results['overall_statistics']['median_percentage_across_datasets']:.2f}%")
    print(f"  - Std deviation: {results['overall_statistics']['std_percentage_across_datasets']:.2f}%")
    
    return results

if __name__ == "__main__":
    results = calculate_m_greater_than_1_stats()