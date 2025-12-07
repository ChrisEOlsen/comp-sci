# add to ~/.ipython/profile_default/startup/00-stats.py

import numpy as np
import pandas as pd
from scipy import stats

# Try to import your helper.
try:
    from stats_helper import *
    print("✅ Stats Helper Loaded")
except ImportError:
    print("⚠️ stats_helper.py not found in current directory")

# Enable Autoreload so you can edit your helper file without restarting IPython
from IPython import get_ipython
ipython = get_ipython()

if ipython:
    # UPDATED SYNTAX HERE:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
