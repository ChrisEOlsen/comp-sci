import numpy as np
from scipy import stats

def describe(data):
    """
    Prints the "5 Number Summary" + Mean/Std for a dataset.
    IMPORTANT: Uses Sample Variance (ddof=1), not Population.
    """
    arr = np.array(data)
    n = len(arr)
    
    print(f"\n--- Data Summary (n={n}) ---")
    print(f"Mean:   {np.mean(arr):.4f}")
    print(f"Median: {np.median(arr):.4f}")
    
    # Scipy Mode returns a mode object; we extract the value
    try:
        mode = stats.mode(arr, keepdims=True)[0][0]
        print(f"Mode:   {mode:.4f}")
    except:
        print("Mode:   (No unique mode)")

    # Variance & Standard Deviation (Sample, not Population)
    print(f"Var (S²): {np.var(arr, ddof=1):.4f}")
    print(f"Std (S):  {np.std(arr, ddof=1):.4f}")
    print("-" * 25)

def normal_prob(x, mean=0, std=1):
    """
    Replaces the Z-table.
    Calculates Z-score and probabilities for a Normal Distribution.
    """
    z_score = (x - mean) / std
    
    # CDF = Area to the LEFT (Percentile)
    prob_less = stats.norm.cdf(x, loc=mean, scale=std)
    
    # SF = Survival Function (Area to the RIGHT)
    prob_more = stats.norm.sf(x, loc=mean, scale=std)
    
    print(f"\n--- Normal Dist (Mean={mean}, Std={std}) ---")
    print(f"X value: {x}")
    print(f"Z-score: {z_score:.4f}")
    print(f"P(X < {x}): {prob_less:.4f}  (Left Tail)")
    print(f"P(X > {x}): {prob_more:.4f}  (Right Tail)")
    print("-" * 25)

def normal_cutoff(percentile, mean=0, std=1):
    """
    Inverse Normal. Finds the X score for a given percentile.
    Example: 0.95 finds the score separating the bottom 95% from top 5%.
    """
    # PPF = Percent Point Function (Inverse of CDF)
    cutoff = stats.norm.ppf(percentile, loc=mean, scale=std)
    print(f"\n--- Inverse Normal (Mean={mean}, Std={std}) ---")
    print(f"Percentile: {percentile}")
    print(f"Cutoff X:   {cutoff:.4f}")

def binom_prob(n, p, k):
    """
    Binomial Distribution helper.
    n = trials, p = probability of success, k = number of successes
    """
    exact = stats.binom.pmf(k, n, p)
    or_less = stats.binom.cdf(k, n, p)
    # Probability of k or more is (1 - cdf(k-1))
    or_more = stats.binom.sf(k-1, n, p) 
    
    print(f"\n--- Binomial (n={n}, p={p}) ---")
    print(f"P(X = {k}):  {exact:.4f}")
    print(f"P(X <= {k}): {or_less:.4f}")
    print(f"P(X >= {k}): {or_more:.4f}")
    print("-" * 25)
