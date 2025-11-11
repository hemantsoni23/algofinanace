# Quant Regime Identification System
## Layer 1: Fast Break Detection using Changepoint Analysis

A robust, modular system for detecting major structural breaks (regime changes) in financial time series using changepoint analysis.

---

## 🎯 Objective

Identify **statistically significant structural breaks** in market data (e.g., Nifty 50) that correspond to major regime changes such as:
- 2008 Global Financial Crisis
- 2020 COVID-19 Crash
- Other significant market dislocations

**Success Criteria**: The system must unambiguously identify the 2008 GFC and 2020 COVID crash as changepoints.

---

## 🏗️ System Architecture

### Modular Design
```
regime_detection/
├── main.py                      # Main orchestrator
├── modules/
│   ├── data_loader.py           # Data loading & validation
│   ├── feature_engineering.py   # Log returns & volatility calculation
│   ├── changepoint_detector.py  # Ruptures-based detection
│   ├── visualization.py         # Plotting & validation
│   └── signal_generator.py      # Binary regime signal generation
├── outputs/
│   ├── visualizations/          # Generated plots
│   ├── regime_signals.csv       # Binary time series signal
│   ├── regime_statistics.csv    # Statistics per regime
│   └── regime_detection.log     # Execution log
└── requirements.txt
```

---

## 🔬 Methodology

### Why Changepoint Detection?

Traditional approaches fail because:
- Raw prices are **non-stationary** (constant drift)
- Models detect price trends, not regime changes

Our approach:
1. **Calculate Log Returns**: Make the series stationary
2. **Calculate Rolling Volatility**: Volatility changes dramatically during regime shifts
3. **Apply Changepoint Detection**: Detect changes in volatility mean using ruptures

### Algorithm: PELT with L2-norm

- **PELT** (Pruned Exact Linear Time): Fast, exact algorithm
- **L2-norm**: Detects changes in mean (perfect for volatility)
- **Penalty Tuning**: Controls sensitivity (lower = more changepoints)

---

## 📦 Installation

### 1. Create Environment
```bash
cd regime_detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages:
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `ruptures>=1.1.9` - Changepoint detection
- `matplotlib>=3.7.0` - Visualization
- `scipy>=1.10.0` - Statistical functions

---

## 🚀 Usage

### Quick Start

```python
from main import RegimeDetectionPipeline

# Initialize pipeline
pipeline = RegimeDetectionPipeline(
    data_path="../Nifty_50.csv",
    output_dir="outputs"
)

# Run detection
results = pipeline.run(
    volatility_window=30,      # 30-day rolling volatility
    detection_method='n_breaks', # Detect top N changepoints
    n_breaks=15,               # Number of changepoints
    tune_penalty=True          # Auto-tune penalty parameter
)

# Get regime signal
regime_signal = pipeline.get_regime_signal()
```

### Command Line

```bash
cd regime_detection
python main.py
```

---

## ⚙️ Configuration

### Key Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `volatility_window` | Rolling window for volatility | 30 | 20-60 days |
| `detection_method` | 'n_breaks' or 'penalty' | 'n_breaks' | - |
| `n_breaks` | Number of changepoints | 15 | 10-20 |
| `penalty` | Penalty for new changepoints | 10.0 | 5-30 |
| `tune_penalty` | Auto-tune penalty | True | - |

### Volatility Window Guide
- **20 days**: Captures rapid regime shifts
- **30 days**: Standard, balanced approach ✅
- **60 days**: Captures sustained regime changes

### Penalty Tuning Guide
- **Low (0.5-2)**: Many changepoints, high sensitivity
- **Medium (3-10)**: Moderate, significant changes ✅
- **High (20+)**: Few changepoints, only major crises

---

## 📊 Outputs

### 1. Visualizations (`outputs/visualizations/`)

#### `changepoint_analysis.png`
Two-panel plot showing:
- **Top**: Price series with vertical lines at changepoints
- **Bottom**: Volatility signal with changepoints
- **Annotations**: Major crises (2008 GFC, 2020 COVID)

#### `changepoint_summary.png`
Timeline view with all detected changepoints labeled

#### `penalty_tuning.png` (if enabled)
Penalty vs. number of changepoints to help select optimal penalty

### 2. Data Files (`outputs/`)

#### `regime_signals.csv`
Binary time series signal:
```csv
date,regime_change_signal,regime_label
2007-01-01,0,0
2008-10-15,1,1
2008-10-16,0,1
...
```

#### `regime_statistics.csv`
Statistics for each regime period:
```csv
regime_id,start_date,end_date,duration_days,total_return_pct,annualized_volatility_pct,sharpe_ratio
0,2007-01-01,2008-10-15,653,45.2,18.5,1.2
1,2008-10-15,2009-03-01,137,-38.7,52.3,-0.8
...
```

#### `regime_detection.log`
Complete execution log with timestamps

---

## 🧪 Validation

### Automatic Crisis Detection

The system automatically validates against major historical crises:

✅ **2008 Global Financial Crisis** (October 2008)
✅ **2020 COVID-19 Crash** (March 2020)
⚠️ **2011 European Debt Crisis** (August 2011)
⚠️ **2016 Brexit/Demonetization** (November 2016)

### Success Criteria

The pipeline **MUST** detect:
1. A changepoint within 60 days of October 2008 (GFC)
2. A changepoint within 60 days of March 2020 (COVID)

If these are not detected, the model needs tuning.

---

## 🔧 Troubleshooting

### Problem: No changepoints detected or too many

**Solution**: Adjust penalty parameter
```python
# Too many changepoints → Increase penalty
results = pipeline.run(penalty=20.0)

# Too few changepoints → Decrease penalty
results = pipeline.run(penalty=5.0)

# Or use automatic tuning
results = pipeline.run(tune_penalty=True)
```

### Problem: 2008/2020 crises not detected

**Solution**: Try different volatility window
```python
# More sensitive (shorter window)
results = pipeline.run(volatility_window=20)

# Less sensitive (longer window)
results = pipeline.run(volatility_window=60)
```

### Problem: Data quality issues

The system automatically:
- Detects missing values
- Removes duplicates
- Validates OHLC consistency

Check `regime_detection.log` for details.

---

## 🧩 Module Details

### 1. DataLoader (`data_loader.py`)
- Loads CSV files with OHLCV data
- Validates data quality
- Handles missing values and duplicates
- Case-insensitive column detection

### 2. FeatureEngineer (`feature_engineering.py`)
- Calculates log returns (stationary transformation)
- Computes rolling volatility (annualized)
- Supports multiple volatility windows
- Prepares clean signals for detection

### 3. ChangepointDetector (`changepoint_detector.py`)
- Implements PELT algorithm with L2-norm
- Supports penalty-based and n_breaks methods
- Automatic penalty tuning
- Segment statistics calculation

### 4. Visualizer (`visualization.py`)
- Dual-panel changepoint plots
- Crisis annotations for validation
- Summary timelines
- Penalty tuning visualizations

### 5. RegimeSignalGenerator (`signal_generator.py`)
- Binary signal generation (0/1)
- Regime labeling (0, 1, 2, ...)
- Regime statistics calculation
- Signal export to CSV

---

## 📈 Example Results

### Typical Changepoints for Nifty 50 (2007-2025)

1. **October 2008** - Global Financial Crisis
2. **March 2009** - Market bottom/recovery
3. **August 2011** - European debt crisis
4. **August 2013** - Taper tantrum
5. **January 2016** - China slowdown fears
6. **November 2016** - Demonetization
7. **March 2020** - COVID-19 crash
8. **March 2020** - Rapid recovery
9. **February 2021** - Post-COVID normalization

---

## 🔮 Future Enhancements (Layer 2+)

This is **Layer 1** of the Quant Regime Identification System. Future layers:

- **Layer 2**: Hidden Markov Models for regime probability
- **Layer 3**: Machine Learning regime classification
- **Layer 4**: Multi-asset regime correlation
- **Layer 5**: Real-time regime monitoring

The modular design allows easy integration of additional layers.

---

## 📚 References

### Academic Papers
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). *Optimal detection of changepoints with a linear computational cost*. JASA.
- Truong, C., Oudre, L., & Vayatis, N. (2020). *Selective review of offline change point detection methods*. Signal Processing.

### Libraries
- [ruptures](https://github.com/deepcharles/ruptures) - Changepoint detection library
- [pandas](https://pandas.pydata.org/) - Data analysis
- [matplotlib](https://matplotlib.org/) - Visualization

---

## 📄 License

MIT License - Free for personal and commercial use

---

## 👤 Author

Quant Trading System
Layer 1: Fast Break Detection Module

---

## 🆘 Support

For issues or questions:
1. Check the log file: `outputs/regime_detection.log`
2. Review validation section in output
3. Try automatic penalty tuning: `tune_penalty=True`
4. Adjust volatility window (20-60 days)

---

## ✅ Verification Checklist

Before deployment, ensure:
- [ ] 2008 GFC detected within 60 days of October 2008
- [ ] 2020 COVID crash detected within 60 days of March 2020
- [ ] Visualizations clearly show changepoints at crises
- [ ] Regime signals exported successfully
- [ ] No data quality errors in log file
- [ ] Reasonable number of regimes (8-15 typical)

**If all criteria met → Layer 1 is working correctly! ✅**
