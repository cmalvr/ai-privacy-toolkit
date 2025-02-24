import pandas as pd
from sklearn.model_selection import train_test_split
from l_diversity import L_Diversity
from apt.minimization.minimizer import GeneralizeToRepresentative

import warnings
warnings.filterwarnings("ignore")

def compute_privacy_metric(anonymized_df, sensitive_attribute, quasi_identifiers):
    """
    Compute the privacy metric by calculating the minimum number of distinct values
    of the sensitive attribute in each equivalence class (grouped by quasi-identifiers).

    Parameters:
        anonymized_df (pd.DataFrame): The anonymized dataset.
        sensitive_attribute (str): The sensitive attribute column name.
        quasi_identifiers (list): List of quasi-identifier column names.

    Returns:
        int: The minimum diversity (distinct sensitive values) across all groups.
    """
    groups = anonymized_df.groupby(quasi_identifiers)
    diversity_series = groups[sensitive_attribute].nunique()
    return diversity_series.min()

def grid_search_privacy(dataset, sensitive_attribute, quasi_identifiers, param_grid, privacy_threshold,
                        model, features, target_accuracy):
    """
    Perform a grid search over anonymization parameter configurations on the available test data.
    This function assumes that no training data is available—only test data is used.

    For each parameter configuration:
      1. The entire test dataset is anonymized using L_Diversity.
      2. A privacy metric is computed on the anonymized data.
      3. If the privacy metric meets the threshold, the anonymized data is split into:
           - A generalizer training set (to fit the minimizer).
           - A hold-out set for final evaluation.
      4. The minimizer (GeneralizeToRepresentative) is then fitted on the generalizer training set,
         and used to transform the hold-out set.
      5. The model’s performance is evaluated on the minimized hold-out set.
      6. The first configuration meeting the privacy threshold is returned.

    Parameters:
        dataset (ArrayDataset): The test dataset wrapped in an ArrayDataset.
        sensitive_attribute (str): The sensitive attribute column name.
        quasi_identifiers (list): List of quasi-identifier column names.
        param_grid (list of dict): List of parameter combinations (e.g., {'k': 3, 'l': 2}).
        privacy_threshold (int): Minimum acceptable diversity in any equivalence class.
        model: The pre-trained model (a classifier).
        features (list): List of feature names.
        target_accuracy (float): The target accuracy for the minimizer.

    Returns:
        tuple: (best_params, final_minimized_df, privacy_metric)
            best_params: The parameter combination that met the threshold.
            final_minimized_df: The minimized hold-out test dataset.
            privacy_metric: The computed privacy metric.
    """
    best_params = None
    final_minimized_df = None
    best_privacy = None
    # Loop over each parameter configuration.
    for params in param_grid:
        print(f"Testing parameters: k={params['k']}, l={params['l']}")
        anonymizer = L_Diversity(
            k=params['k'],
            l=params['l'],
            sensitive_attribute=sensitive_attribute,
            quasi_identifiers=quasi_identifiers,
            categorical_features=params.get('categorical_features', None),
            train_only_QI=params.get('train_only_QI', False)
        )
        # Anonymize the entire test dataset.
        anonymized_data = anonymizer.anonymize(dataset)
        if not isinstance(anonymized_data, pd.DataFrame):
            anonymized_data = pd.DataFrame(anonymized_data, columns=dataset.features_names)
        
        # Retrieve valid indices from the anonymizer.
        if not hasattr(anonymizer, "valid_rows"):
            raise ValueError("L_Diversity must set the 'valid_rows' attribute with the indices of valid rows.")
        valid_indices = anonymizer.valid_rows
        
        # Align labels with the anonymized data.
        y_anonymized = dataset.get_labels()[valid_indices]
        
        # Compute the privacy metric.
        metric = compute_privacy_metric(anonymized_data, sensitive_attribute, quasi_identifiers)
        print(f"Privacy metric: {metric}")
        
        if metric >= privacy_threshold:
            # Split the anonymized data (and labels) into a generalizer training set and a hold-out set.
            X_gen, X_holdout, y_gen, y_holdout = train_test_split(
                anonymized_data, y_anonymized, test_size=0.4, random_state=38
            )
            # Get predictions on the generalizer training set.
            train_preds = model.predict(X_gen)
            # Instantiate and fit the minimizer on the generalizer training set.
            minimizer_instance = GeneralizeToRepresentative(
                model, target_accuracy=target_accuracy, is_regression=False, features_to_minimize=quasi_identifiers
            )
            minimizer_instance.fit(X_gen, train_preds, features_names=features)
            # Transform the hold-out set.
            transformed = minimizer_instance.transform(X_holdout, features_names=features)
            transformed_df = pd.DataFrame(transformed, columns=features)
            acc = model.score(transformed_df, y_holdout)
            print("Accuracy on minimized data:", acc)
            print("Generalizations:", minimizer_instance.generalizations_)
            best_params = params
            best_privacy = metric
            final_minimized_df = transformed_df
            print("Configuration accepted.")
            break
        else:
            print("Configuration rejected: privacy level below threshold.")
    
    return best_params, final_minimized_df, best_privacy