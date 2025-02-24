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

def reconstruct_secret(shares, prime=2**127 - 1, scale_factor=1):
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

class Shamir:
    def __init__(self, n_shares=5, threshold=3, prime=2**127 - 1, scale_factor=1):
        self.n_shares = n_shares
        self.threshold = threshold
        self.prime = prime
        self.scale_factor = scale_factor  # <-- now we have scale_factor

    def split_value(self, value):
        # Multiply the float by scale_factor before splitting
        secret = int(round(value * self.scale_factor))
        return generate_shares(secret, self.n_shares, self.threshold, self.prime)

    def split_dataframe(self, df, sensitive_columns):
        import pandas as pd
        shares_dict = {}
        for col in sensitive_columns:
            col_shares = []
            for val in df[col]:
                shares = self.split_value(val)
                col_shares.append([s[1] for s in shares])
            col_shares_df = pd.DataFrame(
                col_shares,
                columns=[f"{col}_share_{i}" for i in range(1, self.n_shares + 1)]
            )
            shares_dict[col] = col_shares_df
        return shares_dict

    def reconstruct_value(self, shares_subset):
        # Reconstruct the scaled secret and then divide by scale_factor
        scaled_secret = reconstruct_secret(shares_subset, self.prime)
        return scaled_secret / self.scale_factor