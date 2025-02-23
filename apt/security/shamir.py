import random

def mod_inverse(a, prime):
    """
    Compute the modular inverse of a modulo 'prime'.
    """
    return pow(a, -1, prime)

def polynom(x, coefficients, prime):
    """
    Evaluate a polynomial (coeff_0 + coeff_1*x + ...) at x, modulo 'prime'.
    """
    result = 0
    for coefficient in reversed(coefficients):
        result = (result * x + coefficient) % prime
    return result

def generate_shares(secret, n, k, prime=2**127 - 1):
    """
    Split 'secret' (an integer) into 'n' shares with threshold 'k'.
    Returns a list of (x, y) tuples.

    :param secret: The secret to be split (integer).
    :param n: The total number of shares to generate.
    :param k: The minimum number of shares needed to reconstruct the secret.
    :param prime: A large prime number > any secret value to use in modular arithmetic.
    """
    # Generate k-1 random coefficients for the polynomial.
    coefficients = [secret] + [random.randrange(0, prime) for _ in range(k - 1)]
    shares = []
    for i in range(1, n + 1):
        x = i
        y = polynom(x, coefficients, prime)
        shares.append((x, y))
    return shares

def reconstruct_secret(shares, prime=2**127 - 1):
    """
    Reconstruct the secret from a list of (x, y) shares using Lagrange interpolation.
    """
    secret = 0
    for j, (xj, yj) in enumerate(shares):
        numerator = 1
        denominator = 1
        for m, (xm, _) in enumerate(shares):
            if m != j:
                numerator = (numerator * (-xm)) % prime
                denominator = (denominator * (xj - xm)) % prime
        # Multiply the partial result by the modular inverse of 'denominator'
        lagrange_coeff = numerator * mod_inverse(denominator, prime)
        secret = (secret + yj * lagrange_coeff) % prime
    return secret

class ShamirSecretSharingWrapper:
    """
    A simple wrapper class that applies Shamir's Secret Sharing to pandas DataFrames.
    Splits integer (or integer-scaled float) values into shares, and reconstructs them.
    """
    def __init__(self, n_shares=5, threshold=3, prime=2**127 - 1):
        """
        :param n_shares: Total number of shares per secret.
        :param threshold: Minimum number of shares needed to reconstruct.
        :param prime: A large prime for modular arithmetic.
        """
        self.n_shares = n_shares
        self.threshold = threshold
        self.prime = prime

    def split_value(self, value):
        """
        Convert 'value' to int and split into shares.
        If you're dealing with floats, you should multiply by a scale factor 
        externally before calling this method.
        """
        secret = int(value)
        return generate_shares(secret, self.n_shares, self.threshold, self.prime)

    def split_dataframe(self, df, sensitive_columns):
        """
        For each column in 'sensitive_columns', split each cell's value into shares.
        Returns a dictionary mapping each column to a DataFrame of shares 
        (each row corresponds to a record, each column is one share).
        """
        import pandas as pd
        shares_dict = {}
        for col in sensitive_columns:
            col_shares = []
            for val in df[col]:
                # Generate shares for the value
                share_list = self.split_value(val)
                # Store only the y-part of each (x, y) share (assuming x=1..n_shares).
                col_shares.append([s[1] for s in share_list])
            # Create a DataFrame for these shares
            col_shares_df = pd.DataFrame(
                col_shares,
                columns=[f"{col}_share_{i}" for i in range(1, self.n_shares + 1)]
            )
            shares_dict[col] = col_shares_df
        return shares_dict

    def reconstruct_value(self, shares_subset):
        """
        Reconstruct a secret from a subset of shares (list of (x, y) tuples).
        The subset must meet or exceed the threshold.
        """
        return reconstruct_secret(shares_subset, self.prime)