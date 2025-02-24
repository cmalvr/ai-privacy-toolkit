import numpy as np
import pandas as pd
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

from apt.utils.datasets import ArrayDataset, DATA_PANDAS_NUMPY_TYPE
from typing import Union, Optional

class L_Diversity:
    """
    Performs model-guided anonymization while enforcing l-diversity on a sensitive attribute.
    
    This class partitions the data into equivalence classes (cells) using a decision tree
    based on the quasi-identifiers. It then filters out cells that do not have at least l 
    distinct values for the sensitive attribute. The indices of the remaining rows are stored 
    in `self.valid_rows` for proper label alignment.
    
    Parameters:
        k (int): Minimum group size for decision tree leaves (must be at least 2).
        l (int): l-diversity threshold; each equivalence class must have at least l distinct sensitive values.
        sensitive_attribute (str): Name of the sensitive attribute column.
        quasi_identifiers (list or np.ndarray): List of quasi-identifier column names.
        quasi_identifer_slices (list of lists, optional): Lists of feature names for handling one-hot encoded groups.
        categorical_features (list, optional): List of categorical feature names for preprocessing.
        train_only_QI (bool, optional): If True, the decision tree is trained using only quasi-identifiers.
    """
    def __init__(self, k: int,
                 l: int,
                 sensitive_attribute: str,
                 quasi_identifiers: Union[np.ndarray, list],
                 quasi_identifer_slices: Optional[list] = None,
                 categorical_features: Optional[list] = None,
                 train_only_QI: Optional[bool] = False):
        if k < 2:
            raise ValueError("Parameter k must be at least 2")
        if l < 1:
            raise ValueError("Parameter l must be at least 1")
        if not quasi_identifiers or len(quasi_identifiers) < 1:
            raise ValueError("Quasi-identifiers list cannot be empty")
        self.k = k
        self.l = l
        self.sensitive_attribute = sensitive_attribute
        self.quasi_identifiers = quasi_identifiers
        self.categorical_features = categorical_features
        self.train_only_QI = train_only_QI
        self.features_names = None
        self.features = None
        self.quasi_identifer_slices = quasi_identifer_slices
        self.valid_rows = None  # Indices of rows that pass l-diversity filtering

    def anonymize(self, dataset: ArrayDataset) -> DATA_PANDAS_NUMPY_TYPE:
        """
        Anonymize the dataset by enforcing l-diversity on the sensitive attribute.
        
        Steps:
          1. Validate input and set feature names.
          2. Convert quasi-identifiers and categorical feature names to indices.
          3. Determine the sensitive attribute index.
          4. Partition the data using a decision tree classifier.
          5. Replace quasi-identifier values with the representative of each cell.
          6. Filter out rows from cells that do not meet the l-diversity requirement and store valid indices.
        
        Parameters:
            dataset (ArrayDataset): The dataset containing samples and labels.
        
        Returns:
            The anonymized dataset (as a pandas DataFrame if the input was pandas).
        """
        if dataset.get_samples().shape[1] == 0:
            raise ValueError("No data provided")
        self.features = list(range(dataset.get_samples().shape[1]))
        self.features_names = dataset.features_names if dataset.features_names is not None else self.features

        # Validate that the sensitive attribute and quasi-identifiers are present.
        if self.sensitive_attribute not in self.features_names:
            raise ValueError("Sensitive attribute must be one of the features")
        if not set(self.quasi_identifiers).issubset(set(self.features_names)):
            raise ValueError("Quasi-identifiers must be a subset of the features")
        if self.categorical_features and not set(self.categorical_features).issubset(set(self.features_names)):
            raise ValueError("Categorical features must be a subset of the features")
        
        # Convert quasi-identifiers from names to indices.
        self.quasi_identifiers = [i for i, name in enumerate(self.features_names) if name in self.quasi_identifiers]
        
        # Process one-hot encoded slices, if provided.
        if self.quasi_identifer_slices:
            temp_slices = []
            for slice in self.quasi_identifer_slices:
                slice_indices = [i for i, name in enumerate(self.features_names) if name in slice]
                temp_slices.append(slice_indices)
            self.quasi_identifer_slices = temp_slices
        
        # Convert categorical feature names to indices.
        if self.categorical_features:
            self.categorical_features = [i for i, name in enumerate(self.features_names) if name in self.categorical_features]
        
        # Determine the sensitive attribute index.
        self.sensitive_index = self.features_names.index(self.sensitive_attribute)
        
        # Run the anonymization process.
        transformed = self._anonymize(dataset.get_samples().copy(), dataset.get_labels())
        if dataset.is_pandas:
            return pd.DataFrame(transformed, columns=self.features_names)
        else:
            return transformed

    def _anonymize(self, x, y):
        """
        Internal method that applies a decision tree to partition data and enforces l-diversity.
        
        Parameters:
            x (np.array): The feature matrix.
            y (np.array): The label vector.
        
        Returns:
            np.array: The anonymized feature matrix.
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in x and y must match")
        
        # Preprocess non-numeric data if necessary.
        if x.dtype.kind not in 'iufc':
            if not self.categorical_features:
                raise ValueError("For non-numeric data, specify categorical_features")
            x_prepared = self._modify_categorical_features(x)
        else:
            x_prepared = x
        
        # Use only quasi-identifiers if specified.
        x_anonymizer_train = x_prepared if not self.train_only_QI else x_prepared[:, self.quasi_identifiers]
        
        # Train a decision tree classifier to partition the data.
        self._anonymizer = DecisionTreeClassifier(random_state=10, min_samples_split=2, min_samples_leaf=self.k)
        self._anonymizer.fit(x_anonymizer_train, y)
        
        # Determine cells based on leaf nodes.
        cells_by_id = self._calculate_cells(x, x_anonymizer_train)
        node_ids = self._find_sample_nodes(x_anonymizer_train)
        
        # Build sensitive attribute sets for each cell.
        cell_sensitive = {}
        for idx, node_id in enumerate(node_ids):
            val = x[idx, self.sensitive_index]
            cell_sensitive.setdefault(node_id, set()).add(val)
        
        # Identify valid cells that meet l-diversity.
        valid_cells = {node_id for node_id, sens_set in cell_sensitive.items() if len(sens_set) >= self.l}
        
        # Mark rows that belong to valid cells.
        valid_rows = [i for i, node_id in enumerate(node_ids) if node_id in valid_cells]
        self.valid_rows = valid_rows  # Save valid row indices
        
        # Replace quasi-identifier values with the representative value for their cell.
        anonymized_x = self._anonymize_data(x, x_anonymizer_train, cells_by_id)
        # Return only the rows that passed the l-diversity test.
        final_x = anonymized_x[valid_rows, :]
        return final_x

    def _calculate_cells(self, x, x_anonymizer_train):
        """
        Identify the leaf nodes (cells) of the decision tree.
        """
        cells_by_id = {}
        leaves = []
        for node, feature in enumerate(self._anonymizer.tree_.feature):
            if feature == -2:  # Leaf node indicator in scikit-learn.
                leaves.append(node)
                hist = [int(i) for i in self._anonymizer.tree_.value[node][0]]
                cell = {'id': int(node), 'hist': hist}
                cells_by_id[cell['id']] = cell
        self._nodes = leaves
        self._find_representatives(x, x_anonymizer_train, cells_by_id.values())
        return cells_by_id

    def _find_representatives(self, x, x_anonymizer_train, cells):
        """
        For each cell, determine a representative value for each quasi-identifier.
        """
        node_ids = self._find_sample_nodes(x_anonymizer_train)
        all_one_hot_features = set()
        if self.quasi_identifer_slices:
            all_one_hot_features = set([feat for slice in self.quasi_identifer_slices for feat in slice])
        for cell in cells:
            cell['representative'] = {}
            indexes = [i for i, node_id in enumerate(node_ids) if node_id == cell['id']]
            rows = x[indexes]
            done = set()
            for feature in self.quasi_identifiers:
                if feature not in done:
                    # Handle one-hot encoded features as a group.
                    if feature in all_one_hot_features:
                        for slice in self.quasi_identifer_slices:
                            if feature in slice:
                                values = rows[:, slice]
                                unique_rows, counts = np.unique(values, axis=0, return_counts=True)
                                rep = unique_rows[np.argmax(counts)]
                                for i, feat in enumerate(slice):
                                    done.add(feat)
                                    cell['representative'][feat] = rep[i]
                    else:
                        values = rows[:, feature]
                        if self.categorical_features and feature in self.categorical_features:
                            cell['representative'][feature] = Counter(values).most_common(1)[0][0]
                        else:
                            median = np.median(values)
                            rep_val = min(values, key=lambda v: abs(v - median))
                            cell['representative'][feature] = rep_val

    def _find_sample_nodes(self, samples):
        """
        For each sample, determine its leaf node using the decision tree.
        """
        paths = self._anonymizer.decision_path(samples).toarray()
        node_set = set(self._nodes)
        return [list(set(np.where(p == 1)[0]) & node_set)[0] for p in paths]

    def _find_sample_cells(self, samples, cells_by_id):
        """
        Map each sample to its corresponding cell.
        """
        node_ids = self._find_sample_nodes(samples)
        return [cells_by_id[node_id] for node_id in node_ids]

    def _anonymize_data(self, x, x_anonymizer_train, cells_by_id):
        """
        Replace the quasi-identifier values in the data with the representative values of their cell.
        """
        cells = self._find_sample_cells(x_anonymizer_train, cells_by_id)
        for i, row in enumerate(x):
            cell = cells[i]
            for feature, rep_value in cell['representative'].items():
                row[feature] = rep_value
        return x

    def _modify_categorical_features(self, x):
        """
        Preprocess non-numeric data by applying one-hot encoding to categorical features.
        """
        used_features = self.features if not self.train_only_QI else self.quasi_identifiers
        numeric_features = [f for f in self.features if f in used_features and 
                            (not self.categorical_features or f not in self.categorical_features)]
        categorical_features = self.categorical_features if self.categorical_features else []
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))])
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])
        encoded = preprocessor.fit_transform(x)
        return encoded