# README

## Introduction to Dynamic Disagreeing Neighbors (DDN) in Machine Learning

### Overview

This repository houses the research and results related to the "Dynamic Disagreeing Neighbors" (DDN) methodology, a novel complexity measure within the realm of Machine Learning (ML) and Instance Selection (IS). Our work focuses on enhancing classification tasks by effectively filtering noise and reducing redundant data while maximizing performance.

### Key Concepts

- **Machine Learning & Instance Selection**: In ML, IS is crucial for preprocessing data, which involves noise filtering and redundancy reduction. This process balances maximizing performance with minimizing the training dataset size.
- **Complexity Measures in ML**: These measures are vital for understanding the difficulty of classifying instances. They help in identifying challenging data elements like noisy or borderline points, making them crucial for IS.

### The Dynamic Disagreeing Neighbors (DDN) Measure

- **Definition and Levels**: DDN measures the complexity at three different levels - instance, class, and dataset. It's defined as the percentage of an instance's nearest neighbors belonging to different classes.
- **DDN vs. kDN**: Unlike the well-known k-Degree Neighbors (kDN), DDN is based on dynamically adjusted neighborhood computations (NCN), taking into account the data distribution and the distance of each neighbor.

### Methodology and Implementation

- **Instance-Level Complexity**: DDN assesses how much the surroundings of a point hinder its classification, providing a detailed complexity analysis at the instance level.
- **Dynamic Neighborhoods**: The size of neighborhoods varies depending on data density, allowing for more adaptable and accurate complexity measurement.
- **Weighted Complexity Calculation**: The complexity of an instance is calculated by weighting the distances of each neighbor, with closer neighbors having more influence.

### Application and Evaluation

- **Algorithmic Approach**: The repository includes a detailed algorithm for computing the DDN complexity measure, along with pseudo-code.
- **Experiments and Comparisons**: Our methodology has been rigorously tested against the kDN measure, demonstrating stability, a higher correlation with classification error, and improved performance in IS tasks.

### Future Directions

- **Enhancements and Applications**: Future work includes refining DDN through resampling methods, evaluating misclassified points, and developing regularization mechanisms for more accurate IS problem-solving.

### Keywords

Complexity measures, Data Complexity, Classification, Supervised problems, Instance Selection.

---
