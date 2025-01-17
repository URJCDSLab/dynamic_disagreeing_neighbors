# Dynamic Disagreeing Neighbors (DDN)

## Overview
Dynamic Disagreeing Neighbors (DDN) is a novel complexity measure introduced to estimate the classification difficulty of instances, classes, and datasets. Extending the widely used kDN metric, DDN incorporates dynamic neighborhoods and distance-weighted contributions, addressing the limitations of kDN and enhancing its effectiveness for Imbalanced Sampling (IS) problems.

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
   pip install -r requirements.txt
   ```

## Project Structure
The repository is organized as follows:

```
├── data/                 # Folder to store input datasets
│   ├── preprocessed/     # Preprocessed datasets in Parquet format
│   ├── raw/              # Raw datasets in Parquet format, converted from raw_csv
│   ├── raw_csv/          # Raw datasets in CSV format
│   └── results/          # Results from experiments
├── images/               # Folder to store project-related visualizations
├── notebooks/            # Jupyter notebooks for experimentation and analysis
├── results/              # Folder for output files such as logs and metrics
├── src/                  # Source code of the project
│   ├── data/             # Data handling utilities and scripts
│   ├── experiments/      # Scripts for running experiments
│   ├── model/            # Implementation of the main algorithm (DDN)
│   │   ├── ddn.py        # Main implementation of the DDN algorithm
│   ├── results/          # Utilities for managing experiment results
│   ├── visualization/    # Scripts for generating visualizations and plots
│   ├── utils.py          # Helper functions and utilities
├── LICENSE.txt           # License file (GPL-3.0)
├── README.md             # Project documentation
└── setup.py              # Installation script
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
1. **Behavior of Metrics (Partitioned vs. Global):**
   - Metrics were evaluated both by partitioning and globally.
   - **Finding:** On average, there are no significant differences; however, DDN exhibits lower variability.
2. **Optimal k Value:**
   - Correlation analysis was performed for various \(k\) values for both global and partitioned complexity measures.
   - Complexity was calculated for the entire dataset, the most/least complex class, and majority/minority classes.
   - **Finding:** \(k = 1\) consistently delivers the best results with higher overall correlations.
3. **Model-Specific Complexity Relationships:**
   - Analyzed complexity relationships across different ML models and performance metrics.
   - **Finding:** Minority class and most complex class performance show the strongest correlation with complexity, as expected.
4. **Impact of Class Imbalance:**
   - Investigated the relationship between complexity and performance at various imbalance levels.
   - Results were analyzed for all models collectively and for the best-performing models separately.
5. **Complexity Analysis:**
   - Explored deeper insights into complexity but encountered challenges in extracting definitive patterns.

#### Results Summary:
- **Stability:** DDN demonstrates reduced variability compared to kDN.
- **Correlation:** DDN shows stronger alignment with classification performance metrics.
- **Optimal Settings:** Best results achieved with \(k = 1\) under global calculation conditions.

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
