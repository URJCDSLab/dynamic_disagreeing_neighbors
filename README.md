# Dynamic Disagreeing Neighbors (DDN)

## Overview
Dynamic Disagreeing Neighbors (DDN) is a novel complexity measure introduced to estimate the classification difficulty of instances, classes, and datasets. This study explores instance-based complexity estimation and its relationship with classification performance. Among the existing measures, k-Disagreeing Neighbors (kDN) is widely used because of its simplicity, but it has notable limitations, including its dependence on a fixed number of neighbors *k* and lack of smoothness. Although previous studies recommend setting *k=5*, this choice has not been systematically analyzed, and the discrete nature of kDN values can limit its effectiveness in noise filtering and instance ranking. To address these issues, we introduce DDN, a complexity measure that extends kDN by incorporating dynamic neighborhoods that adapt to data density. Additionally, DDN weighs the contribution of each neighbor based on its distance, ensuring a smoother and more flexible complexity estimation. A systematic comparison between DDN and kDN was conducted across multiple binary classification datasets, analyzing the impact of *k*, global vs. partitioned complexity computation, and correlations with classification performance metrics. Furthermore, the relationship between complexity and expected performance was analyzed in the context of class imbalance, assessing the alignment of complexity measures with classification performance. The results provide insights into the variability and stability of complexity measures, offering a deeper understanding of how complexity estimations interact with classification performance across different scenarios. Beyond descriptive complexity estimation, our findings demonstrate that DDN enables a more robust and stable alignment with the classification performance. Furthermore, we explore a predictive approach that leverages complexity measures to estimate the classifier performance, showing that a simple decision tree model trained on DDN achieves precise performance predictions.

## Key Features
- **Dynamic Neighborhoods:** The size of the neighborhood adapts to data density, ensuring accurate complexity estimation in diverse regions.
- **Distance Weighting:** Neighbors closer to the instance have a higher impact on the complexity calculation.
- **Three-Level Analysis:** Complexity is computed at the instance, class, and dataset levels.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Vacek-Ace/dynamic_disagreeing_neighbors.git
   cd dynamic_disagreeing_neighbors
   ```
2. Install dependencies:
   ```bash
   pip install --editable .
   ```

## Project Structure
The repository is organized as follows:

```
dynamic_disagreeing_neighbors/
├── data/                          # Folder to store input datasets
│   ├── preprocessed/              # Preprocessed datasets in Parquet format
│   ├── raw/                       # Raw datasets in Parquet format, converted from raw_csv
│   ├── raw_csv/                   # Raw datasets in CSV format
│   └── results/                   # Results from experiments
│
├── images/                        # Folder to store project-related visualizations
│
├── notebooks/                     # Jupyter notebooks for experimentation and analysis
│
├── results/                       # Folder for output files such as logs and metrics
│
├── src/                           # Source code of the project
│   └── dynamic_disagreeing_neighbors/  # Main Python package (matches pyproject.toml name)
│       ├── data/                  # Data handling utilities and scripts
│       ├── experiments/           # Scripts for running experiments
│       ├── model/                 # Implementation of the main algorithm (DDN)
│       │   └── ddn.py             # Main implementation of the DDN algorithm
│       ├── results/               # Utilities for managing experiment results
│       ├── visualization/         # Scripts for generating visualizations and plots
│       └── utils.py               # Helper functions and utilities
│
├── LICENSE.txt                    # License file (GPL-3.0)
├── README.md                      # Project documentation
└── pyproject.toml                 # Installation script

```

## Usage
### Running the Algorithm
To compute DDN for a dataset:
```bash
python src/model/ddn.py --data path/to/dataset --k 3 --batch_size 1024 --tol 1e-4 --n_jobs -1
```

### Configuration
You can customize key parameters such as:
- `k`: Number of support neighbors.
- `batch_size`: Size of data batches processed in parallel.
- `tol`: Tolerance for neighborhood radius computation.
- `n_jobs`: Number of parallel jobs to run (-1 uses all processors).

Example command-line arguments:
```bash
python src/model/ddn.py --data data/example.csv --k 5 --batch_size 512 --tol 1e-3
```

### Example
To run with default settings:
```bash
python src/model/ddn.py --data data/example.csv
```

## Methodology
### Dynamic Neighborhoods
DDN employs the NCN classifier's approach to define neighborhoods:
- **Support Neighbors:** Calculate the centroid of the k nearest neighbors.
- **Dynamic Radius:** Expand neighborhoods in low-density areas and shrink them in high-density zones.

![Dynamic Neighborhoods](images/dynamic_neighbors.png)

### Complexity Calculation
The complexity of an instance is computed as:
```math
c(x_i) = \sum_{x_j \in N_k(x_i)} \frac{e^{-||x_j - x_i||_2}}{\sum_{x_j \in N_k(x_i)} e^{-||x_j - x_i||_2}} \cdot I_{[y_i \neq y_j]}
```
Where:
- N<sub>k</sub>(x<sub>i</sub>): Dynamic neighborhood of instance x<sub>i</sub>.
- I: Indicator function for class mismatch.


### Levels of Analysis
1. **Instance-Level:** Measure complexity for each instance.
2. **Class-Level:** Average complexity of all instances in a class.
3. **Dataset-Level:** Average complexity across all instances.

## Experiments

### Research Structure
- **Method Extension:** A novel method extending kDN was developed, incorporating dynamic neighborhoods to create a "continuous kDN."
- **Comparative Evaluation:** The method is consistently compared to kDN, which is computationally inexpensive and widely used for complexity analysis due to its simplicity.
- **Dataset Selection:** A total of 65 binary classification datasets with varying class imbalance levels were utilized.

#### Key Questions:
1. **Adaptive Neighborhood Definition** Introducing dynamic neighborhoods that adjust based on local data density allows for a more precise capture of instance-level complexity. This results in a finer and more continuous estimation than the discrete values offered by kDN.

2. **Stability in Complexity Estimation** Experiments have shown that DDN exhibits lower variability in complexity estimation at the instance, class, and dataset levels. This stability is especially important in scenarios with class imbalance, where robust estimates are crucial for making informed decisions regarding data preprocessing or sampling strategies.

3. **Sensitivity to the  *k*  Parameter** The choice of *k* has a significant impact on the correlation between the estimated complexity and classifier performance. In particular, DDN achieves its best correlation with low values of *k* (typically *k=1*), whereas kDN shows greater stability over a wider range (e.g., *k=3* to *k=8*. This suggests that the optimal *k* configuration may depend on both the complexity measure employed and performance metric of interest.

4. **Superior Robustness at *k=1*** Our experiments consistently show that for *k=1*, DDN achieves higher correlations with classifier performance than kDN. We hypothesize that this improvement is due to the dynamic neighborhood construction in DDN. Even when *k=1* is used, the neighborhood in DDN is defined by the distance to the first support neighbor, which in high-density regions, results in a neighborhood that naturally includes multiple nearby points. This richer local context provides a more robust and smooth estimation of complexity in contrast to kDN, where *k=1* considers only the single nearest neighbor and thus offers a less informative measure.

5. **Importance of Global Computation** Comparisons between global and partition-based complexity computations revealed that global complexity estimation tends to align more consistently with classifier performance. This finding simplifies the practical applicability that can be computed by computing the complexity of the full dataset without resorting to cross-validation, without sacrificing the quality of performance correlations.

6. **Relevance in Difficult Classes** Strongest correlations with performance metrics such as the F1-score, GPS, and MCC were observed for the minority class and the class exhibiting higher complexity. This underscores the usefulness of complexity measures in identifying and focusing on the instances and subgroups that present greater classification challenges.

7. **Effect of Class Proportion on Complexity-Performance Relationship:** Our experiments indicate that both kDN and DDN exhibit weaker alignment with classifier performance in highly imbalanced datasets, as evidenced by larger differences in metrics such as accuracy, F1-score, GPS, and MCC. This suggests that depending on the performance metric and degree of imbalance, a bias may be present in the complexity estimation. 

8. **Predicting Performance from Complexity:** The results show that DDN allows for an effective estimation of expected classification performance using a simple model. A decision tree trained with DDN as a complexity measure demonstrated lower residual variance and stronger alignment with the observed performance values than kDN. This suggests that DDN can serve not only as a complexity descriptor, but also as a reliable predictor of classification performance, opening new possibilities for its application in model selection and dataset evaluation.

## Contributions
This work was supported by:
- **Rey Juan Carlos University:** C1PREDOC2020.
- **Madrid Autonomous Community:** IND2019/TIC-17194.
- **Spanish Ministry of Science and Innovation:** XMIDAS (PID2021-122640OB-I00).
- **Basque Center for Applied Mathematics (BCAM):** Support and resources for algorithm development.

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE.txt](LICENSE.txt) file for details.

## Contact
For questions or collaboration opportunities, contact:
- **GitHub:** [Vacek-Ace](https://github.com/Vacek-Ace)
