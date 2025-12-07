import numpy as np
import plotext as plt
from scipy import stats
import math
from fractions import Fraction

def as_frac(numerator, denominator=None):
    """
    Converts a probability to a simplified fraction.
    Usage 1: as_frac(4, 52)        -> Prints 1/13
    Usage 2: as_frac(0.125)        -> Prints 1/8
    """
    if denominator:
        # If given two numbers (e.g., favorable, total)
        f = Fraction(int(numerator), int(denominator))
    else:
        # If given a single float (e.g., 0.125)
        # limit_denominator finds the closest nice fraction
        f = Fraction(numerator).limit_denominator()
        
    print(f"\nFraction: {f}")
    return f

def nCr(n, r):
    """
    Combinations: Ways to choose r items from n.
    Order does NOT matter. (e.g., Poker hands, Committees)
    """
    return math.comb(n, r)

def nPr(n, r):
    """
    Permutations: Ways to choose r items from n.
    Order DOES matter. (e.g., Race results, Passwords)
    """
    return math.perm(n, r)

def plot_normal(mean=0, std=1):
    """
    Plots a bell curve of the Normal Distribution in the terminal.
    """
    plt.clf() # Clear previous plots
    
    # Generate X points for +/- 4 standard deviations
    x = np.linspace(mean - 4*std, mean + 4*std, 200)
    y = stats.norm.pdf(x, mean, std)
    
    plt.plot(x, y, color="green")
    plt.title(f"Normal Distribution (Mean={mean}, Std={std})")
    plt.xlabel("X Score")
    plt.ylabel("Probability Density")
    plt.show()

def plot_hist(data, bins=10):
    """
    Plots a histogram of your data to check distribution shape.
    """
    plt.clf()
    
    plt.hist(data, bins, color="blue", fill=True)
    plt.title(f"Data Distribution (n={len(data)})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

def plot_binomial(n, p):
    """
    Plots the Probability Mass Function (PMF) of a Binomial Distribution.
    """
    plt.clf()
    
    # 1. Create the X-axis (0, 1, 2... up to n)
    x = list(range(n + 1))
    
    # 2. Calculate probabilities for each X
    y = [stats.binom.pmf(k, n, p) for k in x]
    
    # 3. Plot
    plt.bar(x, y, color="blue")
    plt.title(f"Binomial Distribution (n={n}, p={p})")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.show()

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
    print(f"Range: {max(data) - min(data)}")

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

def normal_between(lower, upper, mean=0, std=1):
    """
    Calculates the probability of being between two values (Area Between).
    Formula: CDF(upper) - CDF(lower)
    """
    prob_upper = stats.norm.cdf(upper, loc=mean, scale=std)
    prob_lower = stats.norm.cdf(lower, loc=mean, scale=std)
    prob = prob_upper - prob_lower
    
    print(f"\n--- Normal Between (Mean={mean}, Std={std}) ---")
    print(f"Range:       {lower} < X < {upper}")
    print(f"Probability: {prob:.4f}")
    print(f"Percentage:  {prob*100:.2f}%")
    return prob

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
