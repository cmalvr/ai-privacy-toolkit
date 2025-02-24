import pandas as pd
from sklearn.model_selection import train_test_split
from apt.security.l_diversity import L_Diversity
from apt.minimization.minimizer import GeneralizeToRepresentative

import warnings
warnings.filterwarnings("ignore")

def grid_search_privacy(dataset, sensitive_attribute, quasi_identifiers, 
                        model, features, target_accuracy,
                        k_min=2, k_max=7, l_min=1):
    """
    Iterates over pairs of (k, l) values (with l from l_min to k) and collects configurations
    that result in a deletion ratio > 0 and minimized data accuracy above the target.

    For each (k, l) pair:
      1. Applies L_Diversity anonymization.
      2. Computes the deletion ratio:
             deletion_ratio = (original_rows - anonymized_rows) / original_rows.
         Skips configuration if no rows are removed.
      3. Splits the anonymized data into a generalizer training set and a hold-out set.
      4. Fits the minimizer and evaluates the model’s accuracy on the hold-out set.
      5. If the accuracy is at or above target_accuracy, the configuration is stored.

    Parameters:
        dataset: ArrayDataset containing test data.
        sensitive_attribute: The sensitive attribute column name.
        quasi_identifiers: List of quasi-identifier feature names.
        model: Pre-trained classifier.
        features: List of all feature names.
        target_accuracy: Minimum acceptable accuracy on minimized data.
        k_min, k_max: Range for k values.
        l_min: Minimum l value (default 1).

    Returns:
        List of dictionaries, where each dictionary has keys:
          - 'k': value of k
          - 'l': value of l
          - 'deletion_ratio': computed deletion ratio
          - 'accuracy': accuracy on minimized data
          - 'generalizations': summary of generalizations from the minimizer.
    """
    results = []

    # Ensure original data is a DataFrame
    original_samples = dataset.get_samples()
    if not isinstance(original_samples, pd.DataFrame):
        original_df = pd.DataFrame(original_samples, columns=dataset.features_names)
    else:
        original_df = original_samples

    for k in range(k_min, k_max + 1):
        for l in range(l_min, k + 1):
            anonymizer = L_Diversity(
                k=k,
                l=l,
                sensitive_attribute=sensitive_attribute,
                quasi_identifiers=quasi_identifiers,
                categorical_features=[sensitive_attribute],
                train_only_QI=False
            )
            anonymized_data = anonymizer.anonymize(dataset)
            if not isinstance(anonymized_data, pd.DataFrame):
                anonymized_data = pd.DataFrame(anonymized_data, columns=dataset.features_names)
            
            # Skip configuration if no rows were removed
            if anonymized_data.shape[0] == original_df.shape[0]:
                continue
            
            deletion_ratio = (original_df.shape[0] - anonymized_data.shape[0]) / original_df.shape[0]
            
            # Retrieve valid row indices
            if not hasattr(anonymizer, "valid_rows"):
                raise ValueError("L_Diversity must set 'valid_rows' with indices of valid rows.")
            valid_indices = anonymizer.valid_rows
            y_anonymized = dataset.get_labels()[valid_indices]
            
            X_gen, X_holdout, y_gen, y_holdout = train_test_split(
                anonymized_data, y_anonymized, test_size=0.4, random_state=38
            )
            
            train_preds = model.predict(X_gen)
            minimizer_instance = GeneralizeToRepresentative(
                model, target_accuracy=target_accuracy, is_regression=False,
                features_to_minimize=quasi_identifiers
            )
            minimizer_instance.fit(X_gen, train_preds, features_names=features)
            transformed = minimizer_instance.transform(X_holdout, features_names=features)
            transformed_df = pd.DataFrame(transformed, columns=features)
            acc = model.score(transformed_df, y_holdout)
            
            results.append({
                    'k': k,
                    'l': l,
                    'deletion_ratio': deletion_ratio,
                    'accuracy': acc,
                    'generalizations': minimizer_instance.generalizations
            })
    return results

#  Displays the grid search results 
def display_grid_search_results(results, min_accuracy=None, max_deletion_ratio=None, sort_by=None):
    """
    Displays grid search results based on optional filtering thresholds and sorting.
    
    Parameters:
      results (list): List of dictionaries, each containing 'k', 'l', 'deletion_ratio', 'accuracy',
                      and 'generalizations'.
      min_accuracy (float, optional): Only show configurations with accuracy >= min_accuracy.
      max_deletion_ratio (float, optional): Only show configurations with deletion_ratio <= max_deletion_ratio.
      sort_by (str, optional): Sort the results by "accuracy" (descending) or "deletion_ratio" (ascending).
      
    Example usages:
      - display_grid_search_results(results): Prints all results
      - display_grid_search_results(results, min_accuracy=0.9)
      - display_grid_search_results(results, max_deletion_ratio=0.5)
      - display_grid_search_results(results, min_accuracy=0.9, sort_by="deletion_ratio")
    """
    filtered = results
    if min_accuracy is not None:
        filtered = [r for r in filtered if r['accuracy'] >= min_accuracy]
    if max_deletion_ratio is not None:
        filtered = [r for r in filtered if r['deletion_ratio'] <= max_deletion_ratio]
    
    if sort_by == "accuracy":
        filtered.sort(key=lambda r: r['accuracy'], reverse=True)
    elif sort_by == "deletion_ratio":
        filtered.sort(key=lambda r: r['deletion_ratio'])
    
    if not filtered:
        print("No configurations match the specified thresholds.")
        return
    
    for r in filtered:
        print(f"Parameters: k={r['k']}, l={r['l']}")
        print(f"Deletion Ratio: {r['deletion_ratio']:.3f}")
        print(f"Accuracy on minimized data: {r['accuracy']:.3f}")
        print(f"Generalizations: {r['generalizations']}")
        print("-" * 40)