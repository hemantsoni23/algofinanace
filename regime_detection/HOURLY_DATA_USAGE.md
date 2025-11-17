# Hourly Data Usage Guide

## Overview
The system now supports **hourly data** converted from 1-minute OHLCV data, providing better signal quality and faster processing.

## Quick Start

### Run with Hourly Data (Default)
```bash
# Hierarchical BCD (High Recall - 25 breakpoints)
python main.py --data hourly --mode hierarchical

# Multi-Signal BCD (High Precision - 37 breakpoints)
python main.py --data hourly --mode multi-signal
```

### Run with Daily Data
```bash
python main.py --data daily --mode hierarchical
```

## Data Conversion

The `convert_to_hourly.py` script converts 1-minute data to hourly OHLCV:

```bash
python convert_to_hourly.py
```

**Input**: `NIFTY 50_minute.csv` (975,321 rows)
**Output**: `Nifty_50_hourly.csv` (18,201 rows)

### Conversion Details
- **Compression**: 53.6x smaller (975K → 18K rows)
- **Time Range**: 2015-01-09 to 2025-07-25 (10.6 years)
- **Trading Days**: 2,602 days
- **Hours/Day**: ~7 hours (market hours 9:00-15:30)
- **Data Quality**: 94.6% complete, no missing values

### Aggregation Method
- **Open**: First price of the hour
- **High**: Highest price of the hour
- **Low**: Lowest price of the hour
- **Close**: Last price of the hour
- **Volume**: Sum of volume for the hour

## Output Structure

### Generated Files (in `outputs_production/`)

1. **`breakpoints.csv`** - Full breakpoint details
   - Columns: `date`, `index`, `level` (major/minor), `confidence`
   - Hierarchical mode: 25 breakpoints (10 major, 15 minor)
   - Multi-signal mode: 37 breakpoints (2-4 signal consensus)

2. **`breakpoints_for_downstream.csv`** - Minimal format for downstream layers
   - Columns: `breakpoint_index`, `date`
   - Ready for SKF, HMM, fragility proxy integration

3. **`detection_summary.txt`** - Metadata
   - Mode, data file, date range, breakpoint counts, timestamp

4. **`regime_detection.png`** - Visualization
   - Price chart with breakpoint markers
   - Color-coded by confidence level

## Performance Comparison

### Hierarchical BCD (hourly data)
- **Breakpoints**: 25 (10 major, 15 minor)
- **Confidence Range**: 59.4% - 91.9%
- **Processing Time**: ~1 second
- **Features**: 35 technical indicators
- **Target**: High recall (capture most regime changes)

### Multi-Signal BCD (hourly data)
- **Breakpoints**: 37 (filtered from 175)
- **Confidence Range**: 50.9% - 86.4%
- **Signal Consensus**: 2-4 signals voting
- **Processing Time**: ~2 seconds
- **Quality Filters**: Confidence (≥0.5), Consensus (≥2 signals), Magnitude (≥0.15)
- **Target**: High precision (75%+, avoid false positives)

## CLI Options

```bash
python main.py [OPTIONS]

Options:
  --data {hourly,daily}     Data frequency (default: hourly)
  --mode {hierarchical,multi-signal}  Detection mode (default: hierarchical)
  --file PATH               Custom data file path
  --output DIR              Output directory (default: outputs_production)
```

## Integration with Downstream Layers

The hourly breakpoints are compatible with:

1. **Kalman Filter (SKF)** - State estimation at breakpoints
2. **Hidden Markov Model (HMM)** - Regime classification
3. **Fragility Proxy** - Market stress measurement
4. **Signal Generator** - Trading signals

### Example Integration
```python
import pandas as pd

# Load breakpoints
breakpoints = pd.read_csv('outputs_production/breakpoints_for_downstream.csv')

# Use in your layer
for idx, row in breakpoints.iterrows():
    bp_date = row['date']
    bp_index = row['breakpoint_index']
    
    # Your downstream processing
    regime = hmm.classify_regime(bp_date)
    fragility = calculate_fragility(bp_date)
```

## Why Hourly Data?

### Advantages
1. **Better Signal-to-Noise**: Smooths out intraday noise
2. **Faster Processing**: 53.6x fewer data points
3. **Meaningful Patterns**: Captures intraday momentum/volatility
4. **Memory Efficient**: Smaller datasets for ML models
5. **Real-time Capable**: Can process hourly as markets operate

### Trade-offs
- **Granularity Loss**: Miss sub-hour breakpoints (acceptable for daily/swing trading)
- **Feature Windows**: Need adjustment (5h = 5min on daily scale)
- **Market Hours**: Only 7 hours/day (vs 24h for daily)

## File Locations

```
regime_detection/
├── main.py                          # Production pipeline
├── convert_to_hourly.py             # Data converter
├── ../Nifty_50_hourly.csv          # Hourly data (18K rows)
├── ../NIFTY 50_minute.csv          # 1-minute data (975K rows)
└── outputs_production/              # Results directory
    ├── breakpoints.csv
    ├── breakpoints_for_downstream.csv
    ├── detection_summary.txt
    └── regime_detection.png
```

## Next Steps

1. **Validate Breakpoints**: Compare hourly vs daily detection quality
2. **Backtest Performance**: Test trading signals from hourly breakpoints
3. **Real-time Pipeline**: Integrate with live hourly data feeds
4. **Optimize Parameters**: Tune windows (5h, 20h, 60h) for hourly scale
5. **Ensemble System**: Combine hourly + daily breakpoints for multi-timeframe analysis

## Support

For issues or questions, check:
- `QUICKSTART.md` - Basic usage
- `PROJECT_SUMMARY.md` - System architecture
- `REGIME_DETECTION_ANALYSIS_REPORT.md` - Performance analysis
