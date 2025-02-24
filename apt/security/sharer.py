import pandas as pd
import numpy as np
from apt.security.shamir import Shamir

# --- Custom NCP Functions ---

def calc_ncp_numeric(original_series: pd.Series, generalized_series: pd.Series) -> float:
    """
    Compute the NCP for a numerical feature as the ratio of the generalized range to the original range.
    """
    orig_min, orig_max = original_series.min(), original_series.max()
    gen_min, gen_max = generalized_series.min(), generalized_series.max()
    total_range = orig_max - orig_min
    if total_range == 0:
        return 0.0
    gen_range = gen_max - gen_min
    return gen_range / total_range

def calc_ncp_categorical(original_series: pd.Series, generalized_series: pd.Series) -> float:
    """
    Compute the NCP for a categorical feature as one minus the relative frequency of the most common category.
    """
    counts = generalized_series.value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return 1 - counts.iloc[0]

def calculate_ncp_feature(original_df: pd.DataFrame, generalized_df: pd.DataFrame, feature: str) -> float:
    """
    Compute the NCP for a single feature by selecting the appropriate function based on the feature type.
    """
    if pd.api.types.is_numeric_dtype(original_df[feature]):
        return calc_ncp_numeric(original_df[feature], generalized_df[feature])
    else:
        return calc_ncp_categorical(original_df[feature], generalized_df[feature])

# --- Main Function to Evaluate and Select Best Secret-Sharing Candidate Feature ---

def select_best(minimized_df: pd.DataFrame, 
                                original_df: pd.DataFrame,
                                untouched_features: list, 
                                model, 
                                y_test,
                                threshold: int = 3, 
                                scale_factor: int = 100,
                                n_shares: int = 5,
                                min_acceptable_accuracy: float = None):


    # Initialize the Shamir wrapper.
    sss = Shamir(n_shares=n_shares, threshold=threshold,scale_factor=scale_factor)

    # Compute baseline accuracy on the minimized data.
    baseline_acc = model.score(minimized_df, y_test)
    print(f"[Debug] Baseline model accuracy on minimized data: {baseline_acc:.4f}")

    # Compute sensitivity scores (NCP) for each untouched feature.
    sensitivity_scores = {}
    for feature in untouched_features:
        if feature in original_df.columns and feature in minimized_df.columns:
            ncp_val = calculate_ncp_feature(original_df, minimized_df, feature)
            sensitivity_scores[feature] = ncp_val
            print(f"[Debug] NCP for feature '{feature}' = {ncp_val:.4f}")
        else:
            print(f"Warning: Feature '{feature}' not found in both DataFrames.")

    # Sort features by descending sensitivity (higher NCP means more sensitive).
    sorted_features = sorted(sensitivity_scores, key=sensitivity_scores.get, reverse=True)
    print(f"[Debug] Sorted untouched features by descending NCP: {sorted_features}")

    best_feature = None
    best_accuracy = -1
    best_reconstructed_df = None

    def reconstruct_column(shares_df, threshold=threshold):
        """Reconstruct a column from its shares DataFrame."""
        reconstructed = []
        for idx, row in shares_df.iterrows():
            share_values = row.tolist()
            # Re-create share tuples: assume x = 1, 2, ..., n_shares.
            share_tuples = [(i + 1, share_values[i]) for i in range(len(share_values))]
            recon_val = sss.reconstruct_value(share_tuples[:threshold])
            reconstructed.append(recon_val)
        return reconstructed

    # Iterate over candidate features in descending NCP order.
    for feature in sorted_features:
        current_ncp = sensitivity_scores[feature]
        print(f"\n[Debug] Trying secret sharing for feature '{feature}' (NCP={current_ncp:.4f})")
        shares_dict = sss.split_dataframe(minimized_df, [feature])
        reconstructed_feature = reconstruct_column(shares_dict[feature], threshold)
        
        # Create a new DataFrame with this feature replaced by its reconstructed values.
        rec_df = minimized_df.copy()
        rec_df[feature] = reconstructed_feature

        # Evaluate the model on the reconstructed dataset.
        acc = model.score(rec_df, y_test)
        print(f"[Debug] Model accuracy with feature '{feature}' reconstructed: {acc:.4f}")

        # Check if it meets or exceeds the best so far
        if acc > best_accuracy:
            print(f"[Debug] Feature '{feature}' yields new best accuracy: {acc:.4f} (old best was {best_accuracy:.4f})")
            best_accuracy = acc
            best_feature = feature
            best_reconstructed_df = rec_df.copy()

        # If a minimum acceptable accuracy is set, choose the first feature meeting it.
        if min_acceptable_accuracy is not None:
            if acc >= min_acceptable_accuracy:
                print(f"[Debug] Feature '{feature}' meets the minimum acceptable accuracy of {min_acceptable_accuracy:.4f}. Stopping.")
                break

    # Calculate relative accuracy change from baseline to the best found.
    rel_change = (best_accuracy - baseline_acc) / baseline_acc * 100 if baseline_acc != 0 else 0
    print(f"\n[Debug] Final selection -> Feature: {best_feature}, Accuracy: {best_accuracy:.4f}")
    print(f"[Debug] Relative accuracy change: {rel_change:.2f}% from baseline {baseline_acc:.4f}")

    return best_feature, best_accuracy, best_reconstructed_df