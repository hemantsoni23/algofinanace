"""
Configuration file for Regime Detection System - MINUTE DATA VERSION
This configuration is optimized for minute-level OHLCV data
"""

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Path to your OHLCV data file (CSV format)
# NOTE: This is minute-level data
DATA_PATH = "../NIFTY 50_minute.csv"

# Output directory for results
OUTPUT_DIR = "outputs_minute"

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Rolling window for volatility calculation (in minutes)
# For minute data:
# - 1 trading day = ~375 minutes (6.25 hours)
# - 1 week = ~1,875 minutes
# - 1 month (20 days) = ~7,500 minutes
# 
# Recommended: 7500 minutes = ~20 trading days
VOLATILITY_WINDOW = 7500

# Alternative: Test multiple windows
# VOLATILITY_WINDOWS = [3750, 7500, 15000]  # ~10, 20, 40 trading days

# =============================================================================
# CHANGEPOINT DETECTION
# =============================================================================

# Detection method: 'n_breaks' or 'penalty'
# - 'n_breaks': Detect top N most significant changepoints
# - 'penalty': Use penalty parameter to control sensitivity
DETECTION_METHOD = 'penalty'

# Number of changepoints to detect (if method='n_breaks')
# With minute data, you have ~500k data points vs ~4k for daily
# So you might want slightly more changepoints
N_BREAKS = 20

# Penalty parameter (if method='penalty')
# For minute data, start with LOWER penalty because:
# - More data points means algorithm is more sensitive
# - Want to detect same major regimes as daily data
# 
# Recommended: Start with 1.0-3.0
PENALTY = 1.0

# Run automatic penalty tuning
# WARNING: This will be SLOW on minute data (10-30 minutes)
# Recommend: Set to False for minute data, manually tune instead
TUNE_PENALTY = False

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

# For minute data, you may want to downsample first for faster processing
# Options:
# - '5min': Resample to 5-minute bars
# - '15min': Resample to 15-minute bars
# - '1H': Resample to hourly bars
# - None: Use raw minute data (slowest but most accurate)
RESAMPLE_FREQUENCY = None  # Set to '15min' or '1H' for faster processing

# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================

# Expected number of changepoints (rough estimate)
# Used for validation
EXPECTED_MIN_CHANGEPOINTS = 5
EXPECTED_MAX_CHANGEPOINTS = 25

# =============================================================================
# NOTES FOR MINUTE DATA
# =============================================================================
"""
IMPORTANT DIFFERENCES FROM DAILY DATA:

1. DATA VOLUME:
   - Daily: ~4,400 rows (manageable)
   - Minute: ~500,000 rows (heavy computation)
   
2. PROCESSING TIME:
   - Daily: 30 seconds - 1 minute
   - Minute: 5-30 minutes depending on settings
   
3. VOLATILITY CALCULATION:
   - Daily: 30-day window = 30 data points
   - Minute: 30-day window = 11,250 data points (30 × 375 minutes/day)
   
4. CHANGEPOINT INTERPRETATION:
   - Daily: Changepoint = specific day
   - Minute: Changepoint = specific minute (more precise timing)
   
5. RECOMMENDED WORKFLOW:
   a. First run on daily data to understand regime structure
   b. Then run on minute data to get precise timing
   c. Use minute data for live monitoring (live_monitoring.py)

6. MEMORY USAGE:
   - Minute data requires ~2-4GB RAM
   - Daily data requires ~100MB RAM
   
7. OPTIMAL USE CASE:
   - Daily data: Historical regime analysis, strategy backtesting
   - Minute data: Live monitoring, precise entry/exit timing
"""
