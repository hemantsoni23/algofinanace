"""
Configuration file for Regime Detection System
Adjust these parameters to tune the changepoint detection
"""

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Path to your OHLCV data file (CSV format)
DATA_PATH = "../Nifty_50.csv"

# Output directory for results
OUTPUT_DIR = "outputs"

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Rolling window for volatility calculation (in days)
# - Smaller windows (20) are more sensitive to short-term changes
# - Larger windows (60) capture sustained regime shifts
# - Default 30 is a good balanced choice
VOLATILITY_WINDOW = 30

# Alternative: Test multiple windows
# VOLATILITY_WINDOWS = [20, 30, 60]

# =============================================================================
# CHANGEPOINT DETECTION
# =============================================================================

# Detection method: 'n_breaks' or 'penalty'
# - 'n_breaks': Detect top N most significant changepoints
# - 'penalty': Use penalty parameter to control sensitivity
DETECTION_METHOD = 'n_breaks'

# Number of changepoints to detect (if method='n_breaks')
# Typical range: 10-20 for 15-20 years of data
N_BREAKS = 15

# Penalty parameter (if method='penalty')
# - Lower penalty (1-5): More changepoints, higher sensitivity
# - Medium penalty (5-15): Balanced, recommended
# - Higher penalty (15-50): Fewer changepoints, only major shifts
PENALTY = 10.0

# Run automatic penalty tuning
# This will test multiple penalty values and recommend optimal setting
TUNE_PENALTY = True

# Penalty range for tuning (if TUNE_PENALTY=True)
PENALTY_RANGE = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

# Expected range of changepoints (for tuning validation)
EXPECTED_CHANGEPOINT_RANGE = (5, 20)

# =============================================================================
# CHANGEPOINT ALGORITHM PARAMETERS
# =============================================================================

# Cost function model
# - "l2": Detects changes in mean (recommended for volatility)
# - "rbf": Detects changes using radial basis function
# - "normal": Detects changes in mean and variance
MODEL = "l2"

# Minimum segment size between changepoints (in days)
# Prevents detecting changepoints too close together
MIN_SIZE = 5

# Jump parameter (subsampling)
# - 1: No subsampling (recommended)
# - >1: Faster but less accurate
JUMP = 1

# =============================================================================
# VISUALIZATION
# =============================================================================

# Figure size (width, height in inches)
FIGURE_SIZE = (16, 10)

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Annotate major crises on plots
ANNOTATE_CRISES = True

# Show plots after generation
SHOW_PLOTS = True

# =============================================================================
# VALIDATION
# =============================================================================

# Major crises to validate detection against
# Format: {name: date}
KNOWN_CRISES = {
    '2008 Global Financial Crisis': '2008-10-01',
    '2020 COVID-19 Crash': '2020-03-01',
    '2011 European Debt Crisis': '2011-08-01',
    '2016 Brexit/Demonetization': '2016-11-01'
}

# Tolerance window for crisis detection (in days)
# A changepoint within this window is considered a valid detection
CRISIS_DETECTION_WINDOW = 60

# Critical crises that MUST be detected
CRITICAL_CRISES = [
    '2008 Global Financial Crisis',
    '2020 COVID-19 Crash'
]

# =============================================================================
# SIGNAL GENERATION
# =============================================================================

# Window around changepoints for detailed analysis (in days)
ANALYSIS_WINDOW_BEFORE = 30
ANALYSIS_WINDOW_AFTER = 30

# =============================================================================
# LOGGING
# =============================================================================

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# Save log to file
LOG_TO_FILE = True

# Log file name
LOG_FILE = 'outputs/regime_detection.log'

# =============================================================================
# PRESETS
# =============================================================================

# Uncomment one of these preset configurations to use:

# # PRESET 1: Conservative (fewer, high-confidence changepoints)
# VOLATILITY_WINDOW = 60
# DETECTION_METHOD = 'penalty'
# PENALTY = 20.0
# N_BREAKS = 10

# # PRESET 2: Balanced (recommended for most cases)
# VOLATILITY_WINDOW = 30
# DETECTION_METHOD = 'n_breaks'
# N_BREAKS = 15
# TUNE_PENALTY = True

# # PRESET 3: Sensitive (more changepoints, captures subtle shifts)
# VOLATILITY_WINDOW = 20
# DETECTION_METHOD = 'penalty'
# PENALTY = 5.0
# N_BREAKS = 20

# =============================================================================
# NOTES
# =============================================================================

"""
TUNING GUIDE:

If 2008/2020 crises are NOT detected:
1. Try LOWER penalty (5-8) or MORE n_breaks (20+)
2. Try SHORTER volatility_window (20 days)
3. Enable TUNE_PENALTY to find optimal settings

If TOO MANY changepoints detected:
1. Try HIGHER penalty (15-30) or FEWER n_breaks (10-12)
2. Try LONGER volatility_window (45-60 days)
3. Increase MIN_SIZE (10-15 days)

Best practice:
- Start with TUNE_PENALTY=True to explore
- Use 'n_breaks' method for consistent results
- Validate visually against known crises
"""
