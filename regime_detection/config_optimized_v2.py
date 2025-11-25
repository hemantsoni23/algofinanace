"""
Optimized Configuration for BCD-based Regime Detection
This config is tuned for high accuracy, precision, and recall with minimal false positives
"""

# =============================================================================
# BCD (Bayesian Changepoint Detection) PARAMETERS
# =============================================================================

# Cost function model for BCD
# - "l2": Detects changes in mean (best for volatility signals)
# - "rbf": Radial basis function (good for non-linear changes)
# - "normal": Detects changes in both mean and variance
# - "l1": L1 norm (robust to outliers)
BCD_MODEL = "l2"

# Minimum size between changepoints (prevents detecting too-close breakpoints)
# Hourly data: 24-48 hours (1-2 days)
# Daily data: 5-10 days minimum
BCD_MIN_SIZE = 48  # For hourly data (2 days)

# Jump parameter for computational efficiency
# - 1: No subsampling (most accurate, recommended)
# - >1: Subsample data (faster but less accurate)
BCD_JUMP = 1

# Primary signal for BCD
# This is the main feature used for changepoint detection
BCD_SIGNAL = "volatility_20d"

# =============================================================================
# HIERARCHICAL BCD CONFIGURATION (High Recall Mode)
# =============================================================================

# Number of major breakpoints to detect (high confidence regime changes)
# These are the most significant market regime shifts
N_BKPS_MAJOR = 15

# Number of minor breakpoints to detect (medium confidence changes)
# These capture more subtle regime transitions
N_BKPS_MINOR = 10

# Confidence thresholds
# Major breakpoints need higher confidence to reduce false positives
CONFIDENCE_THRESHOLD_MAJOR = 0.65  # 65% minimum confidence for major breaks

# Minor breakpoints can have lower confidence
CONFIDENCE_THRESHOLD_MINOR = 0.50  # 50% minimum confidence for minor breaks

# =============================================================================
# MULTI-SIGNAL BCD CONFIGURATION (High Precision Mode)
# =============================================================================

# Signals to use for multi-signal consensus
# Each signal votes independently, final decision requires minimum votes
MULTI_SIGNAL_LIST = [
    'volatility_20d',        # 20-day rolling volatility
    'returns_squared',       # Squared returns (proxy for variance)
    'vol_of_vol',           # Volatility of volatility
    'volume_volatility',     # Volume-based volatility
    'range_volatility'       # High-Low range volatility
]

# Number of breakpoints per signal (conservative to reduce false positives)
N_BKPS_PER_SIGNAL = 12

# Minimum number of signals that must agree for a breakpoint to be valid
# Higher = fewer false positives but might miss some true changes
MIN_SIGNAL_CONSENSUS = 2  # At least 2 out of 5 signals must agree

# Temporal clustering window (in periods)
# Breakpoints detected within this window by different signals are clustered
# For hourly data: 24-48 hours
TEMPORAL_CLUSTER_WINDOW = 48

# Minimum confidence for multi-signal mode
MIN_CONFIDENCE_MULTISIGNAL = 0.55  # 55% minimum

# =============================================================================
# FEATURE ENGINEERING PARAMETERS
# =============================================================================

# Volatility calculation windows (in periods)
# For hourly data: periods = hours
# For daily data: periods = days
VOLATILITY_WINDOW_SHORT = 20   # Short-term volatility
VOLATILITY_WINDOW_MEDIUM = 60  # Medium-term volatility
VOLATILITY_WINDOW_LONG = 120   # Long-term volatility

# Returns calculation
RETURNS_LAG = 1  # Period lag for returns calculation

# Volume analysis
VOLUME_WINDOW = 20  # Rolling window for volume analysis

# Range volatility
RANGE_WINDOW = 10  # Window for High-Low range analysis

# =============================================================================
# FILTERING AND POST-PROCESSING
# =============================================================================

# Minimum volatility change magnitude (percentage)
# Breakpoints must show at least this much change in volatility
MIN_VOLATILITY_CHANGE = 0.15  # 15% minimum change

# Minimum persistence (periods)
# Regime must persist for at least this many periods to be valid
MIN_PERSISTENCE = 24  # For hourly: 24 hours (1 day)

# Maximum breakpoints to return (safety limit)
MAX_BREAKPOINTS = 30

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Known major market events for validation
KNOWN_EVENTS = {
    '2008 Global Financial Crisis': '2008-09-15',
    '2008 Lehman Collapse': '2008-10-06',
    '2011 European Debt Crisis': '2011-08-08',
    '2016 Brexit': '2016-06-24',
    '2016 Demonetization': '2016-11-08',
    '2020 COVID-19 Crash': '2020-03-23',
    '2022 Russia-Ukraine': '2022-02-24'
}

# Tolerance window for event matching (in days)
EVENT_MATCH_WINDOW = 30  # ±30 days

# Critical events that MUST be detected (for validation)
CRITICAL_EVENTS = [
    '2008 Global Financial Crisis',
    '2020 COVID-19 Crash'
]

# =============================================================================
# PERFORMANCE TARGETS
# =============================================================================

# Target metrics for BCD performance
TARGET_PRECISION = 0.75    # 75% precision (minimize false positives)
TARGET_RECALL = 0.55       # 55% recall (catch majority of true changes)
TARGET_F1 = 0.60          # 60% F1 score (balanced performance)

# Acceptable false positive rate
MAX_FALSE_POSITIVE_RATE = 0.25  # Max 25% false positives

# Acceptable false negative rate
MAX_FALSE_NEGATIVE_RATE = 0.45  # Max 45% false negatives

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output directory
OUTPUT_DIR = "outputs_production"

# Save intermediate results
SAVE_INTERMEDIATE = True

# Verbosity level (0=quiet, 1=normal, 2=verbose)
VERBOSITY = 1

# =============================================================================
# COMPUTATIONAL PARAMETERS
# =============================================================================

# Number of CPU cores to use (-1 = all available)
N_JOBS = -1

# Random seed for reproducibility
RANDOM_SEED = 42

# Cache intermediate computations
USE_CACHE = True

# =============================================================================
# NOTES ON TUNING
# =============================================================================

"""
TUNING GUIDE FOR HIGH ACCURACY:

1. For HIGHER PRECISION (fewer false positives):
   - Increase MIN_SIGNAL_CONSENSUS (3-4)
   - Increase CONFIDENCE_THRESHOLD_MAJOR (0.70-0.75)
   - Increase MIN_VOLATILITY_CHANGE (0.20-0.25)
   - Decrease N_BKPS_MAJOR and N_BKPS_MINOR

2. For HIGHER RECALL (catch more true changes):
   - Decrease MIN_SIGNAL_CONSENSUS (2)
   - Decrease CONFIDENCE_THRESHOLD_MAJOR (0.55-0.60)
   - Decrease MIN_VOLATILITY_CHANGE (0.10-0.12)
   - Increase N_BKPS_MAJOR and N_BKPS_MINOR

3. For BALANCED PERFORMANCE (recommended):
   - Use default values
   - Run in hierarchical mode for Layer 1
   - Use multi-signal mode for standalone applications

4. For HOURLY vs DAILY data:
   - Hourly: BCD_MIN_SIZE = 24-48, TEMPORAL_CLUSTER_WINDOW = 48
   - Daily: BCD_MIN_SIZE = 5-10, TEMPORAL_CLUSTER_WINDOW = 5

5. Signal Selection:
   - Always include 'volatility_20d' (most reliable)
   - Add 'returns_squared' for variance changes
   - Add 'vol_of_vol' for second-order effects
   - Volume signals help with liquidity-driven regimes
   - Range volatility catches intraday patterns

PERFORMANCE EXPECTATIONS:
- Hierarchical Mode: ~55% recall, ~65% precision
- Multi-Signal Mode: ~75% precision, ~50% recall
- Combined (2-layer): Best overall performance
"""
