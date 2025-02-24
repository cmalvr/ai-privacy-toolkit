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
    Incrementally test pairs of (k, l) values where l runs from l_min to k.
    
    For each (k, l) pair:
      1. Apply L_Diversity anonymization.
      2. Compute the deletion ratio:
             deletion_ratio = (original_rows - anonymized_rows) / original_rows.
         Skip configuration if no rows are removed.
      3. Display the deletion ratio.
      4. Split the anonymized data into a generalizer training set and a hold-out set.
      5. Fit the minimizer on the training set and transform the hold-out set.
      6. Evaluate the model’s accuracy on the minimized hold-out set.
      7. Stop (and return the configuration) if accuracy drops below target_accuracy.
    
    Parameters:
      dataset: ArrayDataset containing your test data.
      sensitive_attribute: The sensitive attribute column name.
      quasi_identifiers: List of quasi-identifier feature names.
      model: Pre-trained classifier.
      features: List of all feature names.
      target_accuracy: Minimum acceptable accuracy on the minimized data.
      k_min, k_max: The range for k values to test.
      l_min: The minimum l value (defaults to 1).
    
    Returns:
      A tuple: (best_params, final_minimized_df, deletion_ratio)
      where best_params is a dictionary with the chosen k and l values,
      final_minimized_df is the minimized hold-out test dataset,
      and deletion_ratio is the computed privacy metric for that configuration.
    """
    original_rows = dataset.get_samples().shape[0]
    best_params = None
    final_minimized_df = None
    best_deletion_ratio = None

    # Get original data as a DataFrame
    original_df = dataset.get_samples() if isinstance(dataset.get_samples(), pd.DataFrame) \
        else pd.DataFrame(dataset.get_samples(), columns=dataset.features_names)

    # Iterate over k from k_min to k_max
    for k in range(k_min, k_max + 1):
        # For each k, iterate l from l_min up to k (ensuring l <= k)
        for l in range(l_min, k + 1):
            params = {'k': k, 'l': l, 'categorical_features': [sensitive_attribute]}
            print(f"Testing parameters: k={k}, l={l}")
            
            anonymizer = L_Diversity(
                k=k,
                l=l,
                sensitive_attribute=sensitive_attribute,
                quasi_identifiers=quasi_identifiers,
                categorical_features=[sensitive_attribute],
                train_only_QI=False
            )
            # Apply anonymization
            anonymized_data = anonymizer.anonymize(dataset)
            if not isinstance(anonymized_data, pd.DataFrame):
                anonymized_data = pd.DataFrame(anonymized_data, columns=dataset.features_names)
            
            # Check if any rows were removed; if not, skip this configuration
            if anonymized_data.shape[0] == original_rows:
                print("No rows removed during anonymization; skipping this configuration.\n")
                continue
            
            # Compute deletion ratio (privacy metric)
            deletion_ratio = (original_rows - anonymized_data.shape[0]) / original_rows
            print(f"Deletion ratio (privacy metric): {deletion_ratio:.3f}")
            
            # Retrieve valid indices from the anonymizer
            if not hasattr(anonymizer, "valid_rows"):
                raise ValueError("L_Diversity must set the 'valid_rows' attribute with the indices of valid rows.")
            valid_indices = anonymizer.valid_rows
            y_anonymized = dataset.get_labels()[valid_indices]
            
            # Split anonymized data for minimizer training.
            X_gen, X_holdout, y_gen, y_holdout = train_test_split(
                anonymized_data, y_anonymized, test_size=0.4, random_state=38
            )
            
            # Get predictions on the generalizer training set
            train_preds = model.predict(X_gen)
            # Fit the minimizer
            minimizer_instance = GeneralizeToRepresentative(
                model, target_accuracy=target_accuracy, is_regression=False,
                features_to_minimize=quasi_identifiers
            )
            minimizer_instance.fit(X_gen, train_preds, features_names=features)
            # Transform the hold-out set.
            transformed = minimizer_instance.transform(X_holdout, features_names=features)
            transformed_df = pd.DataFrame(transformed, columns=features)
            acc = model.score(transformed_df, y_holdout)
            print(f"Accuracy on minimized data: {acc:.3f}")
            print("Generalizations:", minimizer_instance.generalizations)
            
            # Check if accuracy falls below target
            if acc < target_accuracy:
                print("Accuracy dropped below target; stopping search.\n")
                best_params = params
                best_deletion_ratio = deletion_ratio
                final_minimized_df = transformed_df
                return best_params, final_minimized_df, best_deletion_ratio
            else:
                print("Configuration acceptable; continuing search...\n")
                
    return best_params, final_minimized_df, best_deletion_ratio
