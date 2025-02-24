# AI Privacy Toolkit: Enhanced Minimization with l‑Diversity and Privacy Monitoring using a Grid-Search

This repository extends the original data minimization solution by integrating two distinct security features that significantly enhance data protection:

1. **l‑diversity**  
   Data is partitioned into groups (cells) using a `DecisionTreeClassifier` based on fixed quasi-identifiers. Each group must contain at least *k* records and at least *l* distinct sensitive attribute values (e.g., "mean radius"). This ensures that every cell is diverse, reducing the risk of inference attacks.  
   - **k (Group Size):** Minimum number of records in each cell. Every group must have at least *k* members, ensuring anonymity.  
   - **l (Diversity Requirement):** Minimum number of distinct sensitive values in each cell. Note that l must be less than or equal to k.  
   This approach is inspired by the work of Machanavajjhala et al. ([2007](https://personal.utdallas.edu/~muratk/courses/privacy08f_files/ldiversity.pdf)) and uses existing anonymization modules as a guideline.

2. **Privacy Monitoring with Enhanced Minimization**  
   We monitor privacy by calculating the **deletion ratio** — the fraction of rows removed during anonymization — which serves as a simple measure of information loss. An incremental parameter search is then performed over pairs of parameters (k and l, with l ≤ k). For each configuration, if the deletion ratio is greater than 0 and the model’s accuracy on the minimized data remains above a target threshold, the configuration is stored.  
   Additionally, results are collected in a list so that the best configurations can be later reviewed and selected.  
   *Note:* Although our privacy metric is simple (deletion ratio), further work might incorporate more advanced measures such as the Normalized Certainty Penalty (NCP) ([Alahmadi et al., 2021](https://link.springer.com/article/10.1007/s43681-021-00095-8)).  
   
*Important:* This implementation works entirely on test data (no training data is needed) and is currently designed for classifier models. Future improvements could extend it to regression models and dynamically select quasi-identifiers and sensitive attributes through grid search over multiple permutations.

---

## Table of Contents

1. [Introduction and Motivation](#introduction-and-motivation)
2. [Features and Their Security Mechanisms](#features-and-their-security-mechanisms)
   - [l‑Anonymization](#l-anonymization)
   - [Privacy Monitoring and Enhanced Minimization](#privacy-monitoring-and-enhanced-minimization)
3. [Installation and Requirements](#installation-and-requirements)
4. [Conclusion](#conclusion)

---

## Introduction and Motivation

Traditional data minimization focuses on reducing data size while preserving model accuracy. However, these methods often lack robust privacy controls. Our project enhances minimization by adding a privacy layer that:

- **Enforces l‑Anonymization:** Every cell (group) must have at least *k* records and at least *l* distinct sensitive values.
- **Monitors Privacy:** Uses the deletion ratio (the fraction of rows removed) to measure information loss.
- **Optimizes Parameter Selection:** Iteratively searches over (k, l) pairs to find configurations that yield improved privacy without reducing model accuracy below a target threshold.

This work is performed solely on test data, making it applicable for scenarios where training data is not available.

---

## Features and Their Security Mechanisms

### Feature 1: l‑Anonymization

**Objective:**  
Enhance privacy by ensuring that every group of records has a high level of diversity in the sensitive attribute.

**Key Concepts:**

- **k (Group Size):**  
  The minimum number of records in each group. Every group must have at least *k* members. A larger *k* increases anonymity because an attacker cannot isolate an individual record.
  
- **l (Diversity Requirement):**  
  The minimum number of distinct sensitive attribute values that must appear in each group. Because a group of size *k* can have at most *k* distinct values, we enforce l ≤ k.

**How It Works:**  
- The `L_Diversity` class partitions the dataset into cells using a `DecisionTreeClassifier` based on fixed quasi-identifiers.
- For each cell, it counts the unique values of the sensitive attribute.
- Only cells with at least *l* unique values are retained.
- Quasi-identifier values in retained cells are generalized to a representative value (e.g., median or mode), which helps prevent re-identification.

### Feature 2: Privacy Monitoring of Enhanced Minimization

**Objective:** 

Determine how much the data is anonymized (information loss) and ensure that such anonymization does not degrade the classifier’s accuracy below a set threshold.

**How It Works:**  

1. Data Preparation

Test data is wrapped in an `ArrayDataset` with proper feature names to ensure that the dataset is correctly formatted for anonymization and subsequent processing.

2. Anonymization

The anonymization process is handled by the `L_Diversity.anonymize(dataset)` function which:
- Partitions the data using a decision tree classifier.
- Retains only those cells that meet the l‑diversity requirement.
- Records valid row indices in `self.valid_rows`.

3. Deletion Ratio Calculation

The deletion ratio is computed as the fraction of rows removed from the original dataset, providing a measure of the level of information loss due to anonymization.


$$
\text{Deletion Ratio} = \frac{\text{Original Rows} - \text{Anonymized Rows}}{\text{Original Rows}}
$$

A higher deletion ratio indicates more aggressive anonymization (greater information loss).


4. Enhanced Minimization

After anonymization, the data is split into two subsets:
- A subset for fitting the `GeneralizeToRepresentative` minimizer.
- A hold-out subset for evaluating the classifier’s accuracy.

The minimizer generalizes the quasi-identifiers further, and the classifier’s accuracy on the minimized hold-out data is calculated.

5. Incremental Search 

The `grid_search_privacy` function iterates over combinations of (k, l) values (with l from 1 to k) and performs the following steps:

- Applies the l‑anonymization module.
- Skips configurations where no rows are removed.
- Computes the deletion ratio.
- Splits the anonymized data for minimizer fitting and evaluation.
- Applies the `GeneralizeToRepresentative` minimizer.
- Evaluates the classifier’s accuracy on the minimized hold-out data.
- Stores configurations where accuracy remains above the target threshold.

Configurations that meet the criteria (deletion ratio > 0 and accuracy above the threshold) are stored in a results object for further analysis.

6. Results Display

A separate helper function is provided to filter, sort, and display the grid search results. This function allows for optional filtering based on a minimum accuracy threshold and a maximum deletion ratio, as well as sorting by accuracy (in descending order) or deletion ratio (in ascending order).

---

## Installation and Requirements

### Installation Steps

1.	Clone the Repository
 ``` bash
 git clone https://github.com/cmalvr/ai-privacy-toolkit.git bash
 ```
2.	Change into the Project Directory
```bash
cd ai-privacy-toolkit
 ```
3.	Install Required Packages
```bash
pip install -r requirements.txt
 ```

Note: The adversarial-robustness-toolbox (art) module has been removed from this file because it was not installing correctly. If needed, install it separately:
pip install adversarial-robustness-toolbox

Example Notebook

The notebook l-diversity-grid-diabetes.ipynb is provided as an example of how to run the code in the notebooks module. It was executed in a Google Colab environment using Python 3.11.11.

---
## Conclusion

This project enhances data minimization by integrating two key features:
	•	l‑Anonymization:
Ensures that each group (cell) contains at least k records and at least l distinct sensitive values (as detailed in Machanavajjhala et al., 2007). This reduces the risk of re-identification by ensuring high diversity in sensitive attributes.
	•	Privacy Monitoring with Enhanced Minimization:
Uses the deletion ratio to measure information loss and performs an incremental search over (k, l) pairs to identify configurations that maintain high classifier accuracy on minimized test data. Results are stored and can be reviewed with a helper function. Although our current metric is simple, future work may implement more advanced measures like the Normalized Certainty Penalty (NCP) as suggested in Alahmadi et al., 2021.

Our solution runs entirely on test data, making it suitable for real-world scenarios where training data is not available. Futher work could be done on training data. Currently, the implementation supports classifier models, with potential for future extension to regression models and dynamic selection of quasi-identifiers and sensitive attributes.
