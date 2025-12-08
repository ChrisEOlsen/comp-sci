import re
import inspect
import sys
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns
import numpy as np
from scipy import stats
import math
from fractions import Fraction
from sklearn.linear_model import LinearRegression

# Set the visual theme for all plots
sns.set_theme(style="darkgrid", palette="deep")

def list_tools():
    """
    Prints a menu of all available functions in this toolkit.
    """
    current_module = sys.modules[__name__]
    print(f"\n{'FUNCTION':<35} {'DESCRIPTION'}")
    print("=" * 90)
    
    functions = inspect.getmembers(current_module, inspect.isfunction)
    
    for name, func in functions:
        if func.__module__ == __name__:
            doc = inspect.getdoc(func)
            summary = doc.split('\n')[0] if doc else "No description available."
            sig = str(inspect.signature(func))
            print(f"{name + sig:<35} {summary}")
            
    print("=" * 90)

def as_frac(numerator, denominator=None):
    """
    Converts a probability to a simplified fraction.
    Usage 1: as_frac(4, 52)         -> Prints 1/13
    Usage 2: as_frac(0.125)         -> Prints 1/8
    """
    if denominator:
        f = Fraction(int(numerator), int(denominator))
    else:
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
    GUI: Plots a beautiful Normal Distribution curve with shading.
    """
    x = np.linspace(mean - 4*std, mean + 4*std, 500)
    y = stats.norm.pdf(x, mean, std)
    
    pyplot.figure(figsize=(10, 6))
    
    # Plot the line
    sns.lineplot(x=x, y=y, color="teal", linewidth=2)
    
    # Shade the area under the curve
    pyplot.fill_between(x, y, color="teal", alpha=0.3)
    
    pyplot.title(f"Normal Distribution (Mean={mean}, Std={std})", fontsize=14)
    pyplot.xlabel("X Score")
    pyplot.ylabel("Probability Density")
    pyplot.tight_layout()
    pyplot.show()

def plot_hist(data, bins=10):
    """
    GUI: Plots a Histogram with a Kernel Density Estimate (Smooth Curve).
    """
    pyplot.figure(figsize=(10, 6))
    
    # kde=True adds the smooth line overlay
    sns.histplot(data, bins=bins, kde=True, color="blue", element="step")
    
    # Mark Mean and Median
    mean_val = np.mean(data)
    median_val = np.median(data)
    pyplot.axvline(mean_val, color='red', linestyle='--', label=f'Mean ({mean_val:.2f})')
    pyplot.axvline(median_val, color='orange', linestyle='-', label=f'Median ({median_val:.2f})')
    
    pyplot.title(f"Data Distribution (n={len(data)})", fontsize=14)
    pyplot.xlabel("Value")
    pyplot.ylabel("Frequency")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()

def plot_scatter(x, y, title="Scatter Plot"):
    """
    GUI: Plots a Scatter plot with Regression Line and Confidence Interval.
    """
    pyplot.figure(figsize=(10, 6))
    
    # regplot adds the scatter, the line, and the shaded 95% confidence interval ribbon
    sns.regplot(x=x, y=y, color="purple", marker="o", scatter_kws={'s': 80})
    
    # Calculate stats for title
    r, p_val = stats.pearsonr(x, y)
    
    pyplot.title(f"{title} (r = {r:.4f})", fontsize=14)
    pyplot.xlabel("X Variable")
    pyplot.ylabel("Y Variable")
    pyplot.tight_layout()
    pyplot.show()
    
    # Print stats to terminal as well
    slope, intercept = np.polyfit(x, y, 1)
    print(f"\n--- Correlation Analysis ---")
    print(f"Pearson's r: {r:.4f}")
    print(f"R-squared:   {r**2:.4f}")
    print(f"Equation:    y = {slope:.4f}x + {intercept:.4f}")

def plot_binomial(n, p):
    """
    GUI: Plots the Probability Mass Function (PMF) of a Binomial Distribution.
    """
    x = list(range(n + 1))
    y = [stats.binom.pmf(k, n, p) for k in x]
    
    pyplot.figure(figsize=(10, 6))
    
    # Barplot is perfect for discrete distributions
    sns.barplot(x=x, y=y, color="cornflowerblue", edgecolor="black")
    
    pyplot.title(f"Binomial Distribution (n={n}, p={p})", fontsize=14)
    pyplot.xlabel("Number of Successes")
    pyplot.ylabel("Probability")
    
    # If n is huge, sparse the x-ticks so they don't overlap
    if n > 20:
        pyplot.xticks(ticks=range(0, n+1, 5))
        
    pyplot.tight_layout()
    pyplot.show()

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
    
    try:
        mode = stats.mode(arr, keepdims=True)[0][0]
        print(f"Mode:   {mode:.4f}")
    except:
        print("Mode:   (No unique mode)")
    print(f"Range: {max(data) - min(data)}")

    print(f"Var (S²): {np.var(arr, ddof=1):.4f}")
    print(f"Std (S):  {np.std(arr, ddof=1):.4f}")
    print("-" * 25)

def normal_prob(x, mean=0, std=1):
    """
    Replaces the Z-table.
    Calculates Z-score and probabilities for a Normal Distribution.
    """
    z_score = (x - mean) / std
    prob_less = stats.norm.cdf(x, loc=mean, scale=std)
    prob_more = stats.norm.sf(x, loc=mean, scale=std)
    
    print(f"\n--- Normal Dist (Mean={mean}, Std={std}) ---")
    print(f"X value: {x}")
    print(f"Z-score: {z_score:.4f}")
    print(f"P(X < {x}): {prob_less:.4f}  (Left Tail)")
    print(f"P(X > {x}): {prob_more:.4f}  (Right Tail)")
    print("-" * 25)

def normal_between(lower, upper, mean=0, std=1):
    """
    Calculates the probability of being between two values.
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
    """
    cutoff = stats.norm.ppf(percentile, loc=mean, scale=std)
    print(f"\n--- Inverse Normal (Mean={mean}, Std={std}) ---")
    print(f"Percentile: {percentile}")
    print(f"Cutoff X:   {cutoff:.4f}")

def binom_prob(n, p, k):
    """
    Binomial Distribution helper.
    """
    exact = stats.binom.pmf(k, n, p)
    or_less = stats.binom.cdf(k, n, p)
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
    """
    se = std / np.sqrt(n)
    print(f"\n--- Standard Error (Summary Stats) ---")
    print(f"Std Dev (σ):    {std}")
    print(f"Sample Size (n): {n}")
    print(f"Formula:        {std} / √{n}")
    print(f"Standard Error:  {se:.4f}")
    return se

def correlation_details(x=None, y=None, data_string=None):
    """
    Calculates Pearson's r using the 'Summation Formula' method.
    Accepts two lists OR a raw string of pairs.
    """
    if isinstance(x, str):
        data_string = x
        x = None
    if data_string:
        matches = re.findall(r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', data_string)
        if not matches:
            print("❌ Error: No valid (x, y) pairs found in string.")
            return 0
        x = [float(pair[0]) for pair in matches]
        y = [float(pair[1]) for pair in matches]
        print(f"✅ Parsed {len(x)} pairs from string.")

    arr_x = np.array(x)
    arr_y = np.array(y)
    n = len(arr_x)
    
    sum_x = np.sum(arr_x)
    sum_y = np.sum(arr_y)
    sum_xy = np.sum(arr_x * arr_y)
    sum_x_sq = np.sum(arr_x ** 2)
    sum_y_sq = np.sum(arr_y ** 2)
    
    numerator = (n * sum_xy) - (sum_x * sum_y)
    denom_x = (n * sum_x_sq) - (sum_x ** 2)
    denom_y = (n * sum_y_sq) - (sum_y ** 2)
    denominator = np.sqrt(denom_x * denom_y)
    
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
    
    if abs(r) > 0.8:
        strength = "Strong"
    elif abs(r) > 0.5:
        strength = "Moderate"
    else:
        strength = "Weak"
    direction = "Positive" if r > 0 else "Negative"
    print(f"Result:      {strength} {direction} Linear Correlation")
    
    return r

def regression_details(x=None, y=None, data_string=None):
    """
    Calculates Slope (m) and Intercept (b) for Linear Regression.
    """
    if isinstance(x, str):
        data_string = x
        x = None
    if data_string:
        matches = re.findall(r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', data_string)
        if not matches:
            print("❌ Error: No valid (x, y) pairs found in string.")
            return None, None
        x = [float(pair[0]) for pair in matches]
        y = [float(pair[1]) for pair in matches]
        print(f"✅ Parsed {len(x)} pairs from string.")

    arr_x = np.array(x)
    arr_y = np.array(y)
    n = len(arr_x)
    
    sum_x = np.sum(arr_x)
    sum_y = np.sum(arr_y)
    sum_xy = np.sum(arr_x * arr_y)
    sum_x_sq = np.sum(arr_x ** 2)
    
    numerator = (n * sum_xy) - (sum_x * sum_y)
    denominator = (n * sum_x_sq) - (sum_x ** 2)
    
    if denominator == 0:
        print("❌ Error: Vertical line (undefined slope)")
        return None, None

    m = numerator / denominator
    b = (sum_y - (m * sum_x)) / n
    
    print(f"\n--- Linear Regression (Least Squares) ---")
    print(f"n:              {n}")
    print(f"Slope (m):      {m:.4f}")
    print(f"Intercept (b):  {b:.4f}")
    print("-" * 30)
    print(f"Equation:       y = {m:.4f}x + {b:.4f}")
    print(f"\nTo predict a value, use: {m:.4f} * (your_x) + {b:.4f}")
    return m, b

def check_transformations(x=None, y=None, data_string=None):
    """
    GUI: Applies Power, Log, Square Root, and Reciprocal models.
    Identifies best fit and plots side-by-side comparison with regression lines.
    """
    if isinstance(x, str):
        data_string = x
        x = None
    if data_string:
        matches = re.findall(r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', data_string)
        if not matches:
            print("❌ Error: No pairs found.")
            return
        x = [float(pair[0]) for pair in matches]
        y = [float(pair[1]) for pair in matches]

    arr_x = np.array(x)
    arr_y = np.array(y)
    results = []
    
    # --- MODEL 1: Linear (Original) ---
    r, _ = stats.pearsonr(arr_x, arr_y)
    results.append({"name": "Linear (Original)", "r": r, "x": arr_x, "y": arr_y})
    
    # --- MODEL 2: Log X ---
    if np.min(arr_x) > 0: 
        log_x = np.log10(arr_x)
        r, _ = stats.pearsonr(log_x, arr_y)
        results.append({"name": "Logarithmic (Log X)", "r": r, "x": log_x, "y": arr_y})
    
    # --- MODEL 3: Log X, Log Y (Power) ---
    if np.min(arr_x) > 0 and np.min(arr_y) > 0:
        log_x = np.log10(arr_x)
        log_y = np.log10(arr_y)
        r, _ = stats.pearsonr(log_x, log_y)
        results.append({"name": "Power (Log X, Log Y)", "r": r, "x": log_x, "y": log_y})

    # --- MODEL 4: Sqrt Y ---
    if np.min(arr_y) >= 0:
        sqrt_y = np.sqrt(arr_y)
        r, _ = stats.pearsonr(arr_x, sqrt_y)
        results.append({"name": "Square Root (Sqrt Y)", "r": r, "x": arr_x, "y": sqrt_y})

    # --- MODEL 5: Reciprocal Y ---
    if 0 not in arr_y:
        recip_y = 1 / arr_y
        r, _ = stats.pearsonr(arr_x, recip_y)
        results.append({"name": "Reciprocal (1/Y)", "r": r, "x": arr_x, "y": recip_y})

    results.sort(key=lambda k: abs(k['r']), reverse=True)
    best = results[0]
    original = next(item for item in results if item["name"] == "Linear (Original)")
    
    print(f"\n{'MODEL':<25} | {'r (CORRELATION)':<15} | {'R-SQUARED':<10}")
    print("-" * 60)
    for res in results:
        print(f"{res['name']:<25} | {res['r']:.4f}          | {res['r']**2:.4f}")

    print(f"\n✅ Best Fit: {best['name']}")
    
    # --- GUI PLOT ---
    # Create two side-by-side plots
    fig, axes = pyplot.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Original Data
    sns.regplot(x=original['x'], y=original['y'], ax=axes[0], color="red", scatter_kws={'s': 60})
    axes[0].set_title(f"Original Data (r={original['r']:.4f})")
    
    # Plot 2: Transformed Data
    sns.regplot(x=best['x'], y=best['y'], ax=axes[1], color="green", scatter_kws={'s': 60})
    axes[1].set_title(f"Transformed: {best['name']} (r={best['r']:.4f})")

    pyplot.tight_layout()
    pyplot.show()

def output_transformations(x=None, y=None, data_string=None, model="linear"):
    """
    Applies a specific transformation to data and outputs it in (x, y) string format.
    
    Models:
    - "log_x"   (Logarithmic: x' = log(x))
    - "sqrt_y"  (Square Root: y' = √y)
    - "recip_y" (Reciprocal:  y' = 1/y)
    - "power"   (Power:       x' = log(x), y' = log(y))
    - "log_y"   (Exponential: y' = log(y))
    """
    # 1. Input Parsing (Reuse logic)
    if isinstance(x, str):
        data_string = x
        x = None
    if data_string:
        matches = re.findall(r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', data_string)
        if not matches:
            print("❌ Error: No pairs found.")
            return
        x = [float(pair[0]) for pair in matches]
        y = [float(pair[1]) for pair in matches]

    arr_x = np.array(x)
    arr_y = np.array(y)
    
    # 2. Apply Transformation
    model = model.lower()
    
    # Defaults (Original Data)
    new_x = arr_x
    new_y = arr_y
    name = "Linear (Original)"

    try:
        if "log_x" in model or "logarithmic" in model:
            if np.min(arr_x) <= 0: raise ValueError("Cannot log non-positive X")
            new_x = np.log10(arr_x)
            name = "Logarithmic (X' = log X)"
            
        elif "sqrt_y" in model or "square root" in model:
            if np.min(arr_y) < 0: raise ValueError("Cannot sqrt negative Y")
            new_y = np.sqrt(arr_y)
            name = "Square Root (Y' = √Y)"
            
        elif "recip_y" in model or "reciprocal" in model:
            if 0 in arr_y: raise ValueError("Cannot divide by zero Y")
            new_y = 1 / arr_y
            name = "Reciprocal (Y' = 1/Y)"
            
        elif "power" in model:
            if np.min(arr_x) <= 0 or np.min(arr_y) <= 0: raise ValueError("Cannot log non-positive values")
            new_x = np.log10(arr_x)
            new_y = np.log10(arr_y)
            name = "Power (X' = log X, Y' = log Y)"
            
        elif "log_y" in model or "exponential" in model:
            if np.min(arr_y) <= 0: raise ValueError("Cannot log non-positive Y")
            new_y = np.log10(arr_y)
            name = "Exponential (Y' = log Y)"

    except ValueError as e:
        print(f"❌ Transformation Error: {e}")
        return

    # 3. Format Output
    # We round to 4 decimal places for homework consistency
    pairs = []
    for i in range(len(new_x)):
        pairs.append(f"({new_x[i]:.4f}, {new_y[i]:.4f})")
    
    result_string = " ".join(pairs)
    
    print(f"\n--- Transformed Data: {name} ---")
    print(result_string)
    
    # Copy to clipboard hint
    print("-" * 30)
    
    return result_string

def calculate_r_squared(y_actual, y_predicted):
    """
    Calculates the R-squared (coefficient of determination) score manually.
    
    This function measures how well the regression line approximates the real data points.
    It compares the errors of our model against the errors of a simple baseline (the mean).

    Formula: R^2 = 1 - (Unexplained Variation / Total Variation)

    Parameters:
    -----------
    y_actual : array-like
        The true observed values (ground truth).
    y_predicted : array-like
        The values predicted by the linear regression model.

    Returns:
    --------
    float
        The R^2 score (typically between 0 and 1).
    """
    
    # 1. Calculate the mean of the actual data
    # This represents the baseline model (a horizontal line at the average).
    y_mean = np.mean(y_actual)
    
    # 2. Calculate Total Sum of Squares (TSS)
    # This captures the total variance inherent in the data.
    # Mathematically: sum((y - y_mean)^2)
    total_variance = np.sum((y_actual - y_mean) ** 2)
    
    # 3. Calculate Residual Sum of Squares (RSS)
    # This captures the variance the model failed to explain (the error).
    # Mathematically: sum((y - y_pred)^2)
    unexplained_variance = np.sum((y_actual - y_predicted) ** 2)
    
    # 4. Calculate R-squared
    # If unexplained_variance is 0, R^2 is 1 (Perfect fit).
    # If unexplained_variance equals total_variance, R^2 is 0 (Model is no better than the mean).
    r_squared = 1 - (unexplained_variance / total_variance)
    
    return r_squared


def calculate_pearson_r(x, y):
    """
    Calculates Pearson's Correlation Coefficient (r) manually
    using the 'Raw Score' algebraic formula.
    """
    n = len(x)
    
    # 1. Calculate the sums needed for the formula
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_sq = np.sum(x ** 2)
    sum_y_sq = np.sum(y ** 2)
    
    # 2. Calculate the Numerator (Top part)
    # "How much do X and Y move together?"
    numerator = (n * sum_xy) - (sum_x * sum_y)
    
    # 3. Calculate the Denominator (Bottom part)
    # "How much do X and Y move individually?"
    term_x = (n * sum_x_sq) - (sum_x ** 2)
    term_y = (n * sum_y_sq) - (sum_y ** 2)
    denominator = np.sqrt(term_x * term_y)
    
    # 4. Divide
    r = numerator / denominator
    return r

# --- Verification ---
X = np.array([1, 2, 3, 4, 5]) 
Y = np.array([50, 55, 65, 70, 72])

# Calculate r manually
r_manual = calculate_pearson_r(X, Y)

# Calculate R^2 manually (Method 1)
r_squared = r_manual ** 2

print(f"Manual Correlation (r): {r_manual:.4f}")
print(f"Manual R-squared (r^2): {r_squared:.4f}")
def show_sse_calculation(x, y):
    """
    Prints a detailed 'Accounting Table' comparing the errors of the Mean vs. Regression.
    
    Purpose:
    --------
    Visualizes exactly how R-squared is calculated by showing the "grunt work."
    It compares two models:
    1. The Dumb Guess (Mean Line): Assumes X has no effect on Y.
    2. The Smart Guess (Regression Line): Uses X to predict Y.
    
    Output:
    -------
    - A DataFrame showing the squared error for every single data point.
    - The final calculation of SSE_mean, SSE_regression, and R^2.
    """
    # 1. Create the Mean Line
    mean_y = np.mean(y)
    
    # 2. Create the Regression Line
    # Reshape x because sklearn expects a 2D array [[x1], [x2]...]
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    predicted_y = model.predict(x.reshape(-1, 1))
    
    # 3. Create a Dataframe to show the "Grunt Work"
    df = pd.DataFrame({
        'X': x,
        'Actual_Y': y,
        'Mean_Y': mean_y,
        'Mean_Error_Sq': (y - mean_y)**2,
        'Predicted_Y': predicted_y,
        'Reg_Error_Sq': (y - predicted_y)**2
    })
    
    print("--- The Calculation Table ---")
    print(df.round(2))
    
    print("\n--- The Totals ---")
    sse_mean = df['Mean_Error_Sq'].sum()
    sse_reg = df['Reg_Error_Sq'].sum()
    print(f"Total Variation (SSE Mean):     {sse_mean:.2f}")
    print(f"Unexplained Var (SSE Reg):      {sse_reg:.2f}")
    print(f"R-squared (1 - Reg/Mean):       {1 - (sse_reg/sse_mean):.4f}")