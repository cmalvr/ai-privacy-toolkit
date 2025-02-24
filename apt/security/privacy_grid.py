# privacy_grid.py

import pandas as pd
from sklearn.model_selection import train_test_split
from diversity import L_Diversity
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
    Perform a grid search over anonymization parameter configurations to ensure that the data
    meets the desired privacy threshold and integrates with the minimizer.

    For each configuration:
      1. Anonymize the dataset using L_Diversity.
      2. Compute the privacy metric.
      3. If the metric meets the threshold, split the anonymized data into train and test sets.
      4. Apply the minimizer (GeneralizeToRepresentative) using the provided model,
         target accuracy, and features.
      5. Report the accuracy and generalizations on the minimized test data.

    Parameters:
        dataset (ArrayDataset): The dataset wrapped in an ArrayDataset.
        sensitive_attribute (str): The sensitive attribute column name.
        quasi_identifiers (list): List of quasi-identifier column names.
        param_grid (list of dict): List of parameter combinations (e.g., {'k': 3, 'l': 2}).
        privacy_threshold (int): Minimum acceptable diversity in any equivalence class.
        model: The pre-trained model to be used in the minimizer.
        features (list): List of feature names to minimize.
        target_accuracy (float): The target accuracy for the minimizer.

    Returns:
        tuple: (best_params, final_minimized_df, privacy_metric)
            best_params: The parameter combination that met the threshold.
            final_minimized_df: The resulting minimized dataset.
            privacy_metric: The computed privacy metric.
    """
    best_params = None
    final_minimized_df = None
    best_privacy = None

    for params in param_grid:
        print(f"Testing parameters: k={params['k']}, l={params['l']}")
        anonymizer = L_Diversity(
            k=params['k'],
            l=params['l'],
            sensitive_attribute=sensitive_attribute,
            quasi_identifiers=quasi_identifiers,
            categorical_features=params.get('categorical_features', None),
            is_regression=params.get('is_regression', True),
            train_only_QI=params.get('train_only_QI', False)
        )
        anonymized_data = anonymizer.anonymize(dataset)
        if not isinstance(anonymized_data, pd.DataFrame):
            anonymized_data = pd.DataFrame(anonymized_data, columns=dataset.features_names)

        # Compute the privacy metric.
        privacy_metric = compute_privacy_metric(anonymized_data, sensitive_attribute, quasi_identifiers)
        print(f"Privacy metric (min distinct sensitive values): {privacy_metric}")

        if privacy_metric >= privacy_threshold:
            # Split anonymized data into training and testing sets.
            X_gen_train, X_gen_test, y_gen_train, y_gen_test = train_test_split(
                anonymized_data, dataset.get_labels(), test_size=0.4, random_state=38
            )
            # Obtain predictions on the training set.
            train_preds = model.predict(X_gen_train)
            # Instantiate and fit the minimizer.
            minimizer_instance = GeneralizeToRepresentative(
                model, target_accuracy=target_accuracy, is_regression=True, features_to_minimize=features
            )
            minimizer_instance.fit(X_gen_train, train_preds, features_names=features)
            # Transform the test set.
            transformed = minimizer_instance.transform(X_gen_test, features_names=features)
            transformed_df = pd.DataFrame(transformed, columns=features)
            # Evaluate accuracy on minimized test data.
            acc = model.score(transformed_df, y_gen_test)
            print("Accuracy on minimized data:", acc)
            print("Generalizations:", minimizer_instance.generalizations)
            best_params = params
            best_privacy = privacy_metric
            final_minimized_df = transformed_df
            print("Configuration accepted.")
            break
        else:
            print("Configuration rejected: privacy level below threshold.")

    return best_params, final_minimized_df, best_privacy