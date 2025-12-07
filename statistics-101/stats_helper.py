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

def plot_scatter(x, y, title="Scatter Plot"):
    """
    Plots a scatterplot of two variables and calculates Pearson's r.
    x: List of independent variable data
    y: List of dependent variable data
    """
    plt.clf() # Clear previous plots
    
    # 1. Plot the dots
    plt.scatter(x, y, color="blue")
    
    # 2. Calculate Correlation (r)
    # pearsonr returns (statistic, p-value)
    r, p_val = stats.pearsonr(x, y)
    
    # 3. Add Line of Best Fit (Optional but helpful)
    # Calculate simple linear regression (y = mx + b)
    slope, intercept = np.polyfit(x, y, 1)
    line_y = [slope * i + intercept for i in x]
    plt.plot(x, line_y, color="red")
    
    # 4. Display
    plt.title(f"{title} (r = {r:.4f})")
    plt.xlabel("X Variable")
    plt.ylabel("Y Variable")
    plt.show()
    
    print(f"\n--- Correlation Analysis ---")
    print(f"Pearson's r: {r:.4f}")
    print(f"R-squared:   {r**2:.4f} (Coefficient of Determination)")
    print(f"Equation:    y = {slope:.4f}x + {intercept:.4f}")

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


def standard_error(data, n_override=None):
    """
    Calculates SE and the Relative Standard Error (RSE).
    """
    s = np.std(data, ddof=1)
    
    if n_override:
        n = n_override
        print(f"Note: Using manual sample size n={n}")
    else:
        n = len(data)
        
    se = s / np.sqrt(n)
    mean_val = np.mean(data)
    
    # Calculate Relative Error (percentage)
    rse = (se / mean_val) * 100 if mean_val != 0 else 0
    
    print(f"\n--- Standard Error (SE) ---")
    print(f"Sample Mean:        {mean_val:.4f}")
    print(f"Standard Error:     {se:.4f}")
    print(f"Relative SE (RSE):  {rse:.2f}%")
    
    if rse > 25:
        print("⚠️  Warning: High RSE (>25%). Mean may be unreliable.")
    elif rse < 5:
        print("✅  Great: Low RSE (<5%). Mean is precise.")
        
    return se


def standard_error_stats(std, n):
    """
    Calculates Standard Error when you only have summary statistics.
    (No raw data provided).
    """
    se = std / np.sqrt(n)
    
    print(f"\n--- Standard Error (Summary Stats) ---")
    print(f"Std Dev (σ):    {std}")
    print(f"Sample Size (n): {n}")
    print(f"Formula:        {std} / √{n}")
    print(f"Standard Error:  {se:.4f}")
    return se


def correlation_details(x, y):
    """
    Calculates Pearson's r using the 'Summation Formula' method.
    Prints the intermediate sums (Sigma values) required for manual homework.
    """
    # Convert to numpy arrays for easier math
    arr_x = np.array(x)
    arr_y = np.array(y)
    n = len(arr_x)
    
    # 1. Calculate the 5 Essential Sums
    sum_x = np.sum(arr_x)
    sum_y = np.sum(arr_y)
    sum_xy = np.sum(arr_x * arr_y)      # Sum of products
    sum_x_sq = np.sum(arr_x ** 2)       # Sum of x squared
    sum_y_sq = np.sum(arr_y ** 2)       # Sum of y squared
    
    # 2. Calculate Numerator and Denominator components
    numerator = (n * sum_xy) - (sum_x * sum_y)
    denom_x = (n * sum_x_sq) - (sum_x ** 2)
    denom_y = (n * sum_y_sq) - (sum_y ** 2)
    denominator = np.sqrt(denom_x * denom_y)
    
    # 3. Calculate r
    if denominator == 0:
        r = 0
    else:
        r = numerator / denominator
        
    print(f"\n--- Pearson Correlation (Step-by-Step) ---")
    print(f"n:      {n}")
    print(f"Σx:     {sum_x}")
    print(f"Σy:     {sum_y}")
    print(f"Σxy:    {sum_xy}")
    print(f"Σx²:    {sum_x_sq}")
    print(f"Σy²:    {sum_y_sq}")
    print("-" * 30)
    print(f"Numerator:   {numerator}")
    print(f"Denominator: {denominator:.4f}")
    print(f"r:           {r:.4f}")
    
    # Interpretation based on your lesson
    if abs(r) > 0.8:
        strength = "Strong"
    elif abs(r) > 0.5:
        strength = "Moderate"
    else:
        strength = "Weak"
        
    direction = "Positive" if r > 0 else "Negative"
    print(f"Result:      {strength} {direction} Linear Correlation")
    
    return r

