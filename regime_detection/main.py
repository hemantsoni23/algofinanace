"""
Main Script - Layer 1: Fast Break Detection System
Orchestrates the complete regime detection pipeline
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add modules to path
sys.path.append(str(Path(__file__).parent / 'modules'))

from modules.data_loader import DataLoader
from modules.feature_engineering import FeatureEngineer
from modules.changepoint_detector import ChangepointDetector
from modules.visualization import Visualizer
from modules.signal_generator import RegimeSignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/regime_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RegimeDetectionPipeline:
    """
    Complete pipeline for detecting market regime changes using changepoint analysis.
    """
    
    def __init__(self, data_path: str, output_dir: str = 'outputs'):
        """
        Initialize the pipeline.
        
        Args:
            data_path (str): Path to the OHLCV data file
            output_dir (str): Directory for outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.features = None
        self.changepoint_dates = None
        self.regime_signal = None
        
        logger.info("="*80)
        logger.info("REGIME DETECTION SYSTEM - LAYER 1: FAST BREAK DETECTION")
        logger.info("="*80)
    
    def run(self, 
            volatility_window: int = 30,
            detection_method: str = 'penalty',
            n_breaks: int = 10,
            penalty: float = 10.0,
            tune_penalty: bool = False) -> dict:
        """
        Run the complete regime detection pipeline.
        
        Args:
            volatility_window (int): Rolling window for volatility calculation
            detection_method (str): 'penalty' or 'n_breaks'
            n_breaks (int): Number of breaks (if method='n_breaks')
            penalty (float): Penalty parameter (if method='penalty')
            tune_penalty (bool): Whether to run penalty tuning
            
        Returns:
            dict: Results dictionary
        """
        results = {}
        
        # Step 1: Load Data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA LOADING AND VALIDATION")
        logger.info("="*80)
        
        loader = DataLoader(self.data_path)
        self.data = loader.load_data()
        is_valid, issues = loader.validate_data()
        
        if not is_valid:
            logger.warning("Data validation issues found. Attempting to clean...")
            self.data = loader.clean_data(method='drop')
            is_valid, issues = loader.validate_data()
        
        data_info = loader.get_data_info()
        results['data_info'] = data_info
        
        logger.info(f"✓ Loaded {data_info['total_rows']} rows")
        logger.info(f"✓ Date range: {data_info['start_date']} to {data_info['end_date']}")
        logger.info(f"✓ Years of data: {data_info['years_of_data']:.1f}")
        
        # Step 2: Feature Engineering
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*80)
        
        engineer = FeatureEngineer(self.data)
        
        # Calculate log returns
        log_returns = engineer.calculate_log_returns()
        
        # Calculate rolling volatility
        volatility = engineer.calculate_rolling_volatility(window=volatility_window)
        
        # Get combined data
        self.features = engineer.add_price_data()
        
        # Get clean signal for changepoint detection
        signal, dates = engineer.get_clean_signal(f'rolling_vol_{volatility_window}d')
        
        results['signal_stats'] = {
            'length': len(signal),
            'mean': float(np.mean(signal)),
            'std': float(np.std(signal)),
            'min': float(np.min(signal)),
            'max': float(np.max(signal))
        }
        
        logger.info(f"✓ Prepared signal: {len(signal)} data points")
        logger.info(f"✓ Volatility range: {results['signal_stats']['min']:.4f} to {results['signal_stats']['max']:.4f}")
        
        # Step 3: Changepoint Detection
        logger.info("\n" + "="*80)
        logger.info("STEP 3: CHANGEPOINT DETECTION")
        logger.info("="*80)
        
        detector = ChangepointDetector(model="l2", min_size=5)
        detector.fit(signal)
        
        # Optional: Tune penalty
        if tune_penalty:
            logger.info("\nTuning penalty parameter...")
            tuning_results = detector.tune_penalty(
                signal, 
                penalty_range=[0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50],
                expected_range=(5, 20)
            )
            results['tuning_results'] = tuning_results
            
            # Save tuning results
            tuning_results.to_csv(self.output_dir / 'penalty_tuning_results.csv', index=False)
            
            # Create tuning plot
            visualizer = Visualizer()
            visualizer.plot_penalty_tuning(
                tuning_results,
                save_path=self.output_dir / 'visualizations' / 'penalty_tuning.png'
            )
        
        # Detect changepoints
        if detection_method == 'n_breaks':
            logger.info(f"\nDetecting top {n_breaks} changepoints...")
            breakpoints = detector.predict_with_n_breaks(n_bkps=n_breaks)
        else:
            logger.info(f"\nDetecting changepoints with penalty={penalty}...")
            breakpoints = detector.predict_with_penalty(pen=penalty)
        
        # Map to dates
        self.changepoint_dates = detector.map_breakpoints_to_dates(dates)
        
        results['changepoints'] = {
            'count': len(self.changepoint_dates),
            'dates': [str(d.date()) for d in self.changepoint_dates]
        }
        
        logger.info(f"\n✓ DETECTED {len(self.changepoint_dates)} CHANGEPOINTS:")
        for i, date in enumerate(self.changepoint_dates, 1):
            logger.info(f"   {i:2d}. {date.date()}")
        
        # Check for key historical events
        self._validate_crisis_detection()
        
        # Step 4: Generate Regime Signals
        logger.info("\n" + "="*80)
        logger.info("STEP 4: REGIME SIGNAL GENERATION")
        logger.info("="*80)
        
        signal_gen = RegimeSignalGenerator(self.features.index)
        
        # Binary signal
        self.regime_signal = signal_gen.generate_binary_signal(self.changepoint_dates)
        
        # Regime labels
        regime_labels = signal_gen.generate_regime_labels(self.changepoint_dates)
        
        # Calculate regime statistics
        regime_stats = signal_gen.get_regime_statistics(self.features, price_column='Close')
        results['regime_stats'] = regime_stats
        
        logger.info(f"✓ Generated regime signals")
        logger.info(f"✓ Identified {regime_labels.nunique()} distinct regimes")
        
        # Save regime statistics
        regime_stats.to_csv(self.output_dir / 'regime_statistics.csv', index=False)
        logger.info(f"✓ Saved regime statistics")
        
        # Export signals
        signal_gen.export_signals(self.output_dir / 'regime_signals.csv')
        logger.info(f"✓ Exported regime signals")
        
        # Step 5: Visualization
        logger.info("\n" + "="*80)
        logger.info("STEP 5: VISUALIZATION")
        logger.info("="*80)
        
        visualizer = Visualizer(figsize=(16, 10))
        
        # Main changepoint plot
        fig1 = visualizer.plot_changepoints(
            data=self.features,
            signal_column=f'rolling_vol_{volatility_window}d',
            changepoint_dates=self.changepoint_dates,
            price_column='Close',
            title=f'Regime Detection - {volatility_window}-Day Volatility Analysis',
            save_path=self.output_dir / 'visualizations' / 'changepoint_analysis.png',
            crisis_annotations=True
        )
        
        # Summary plot
        fig2 = visualizer.plot_changepoint_summary(
            changepoint_dates=self.changepoint_dates,
            data=self.features,
            price_column='Close',
            save_path=self.output_dir / 'visualizations' / 'changepoint_summary.png'
        )
        
        logger.info(f"✓ Created visualizations")
        
        # Final Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("="*80)
        logger.info(f"Data Period: {data_info['start_date']} to {data_info['end_date']}")
        logger.info(f"Total Changepoints Detected: {len(self.changepoint_dates)}")
        logger.info(f"Regime Signal Generated: {self.regime_signal.sum()} regime changes")
        logger.info(f"Output Directory: {self.output_dir.absolute()}")
        logger.info("="*80)
        
        return results
    
    def _validate_crisis_detection(self):
        """
        Validate that major historical crises are detected as changepoints.
        """
        logger.info("\n🔍 VALIDATING CRISIS DETECTION:")
        
        # Define major historical crises
        crises = {
            '2008 Global Financial Crisis': pd.Timestamp('2008-10-01'),
            '2020 COVID-19 Crash': pd.Timestamp('2020-03-01'),
            '2011 European Debt Crisis': pd.Timestamp('2011-08-01'),
            '2016 Brexit/Demonetization': pd.Timestamp('2016-11-01')
        }
        
        detected_crises = []
        missed_crises = []
        
        for crisis_name, crisis_date in crises.items():
            # Check if crisis date is in data range
            if self.data.index[0] <= crisis_date <= self.data.index[-1]:
                # Check if a changepoint was detected within 60 days
                time_diffs = [abs((cp - crisis_date).days) for cp in self.changepoint_dates]
                
                if time_diffs and min(time_diffs) <= 60:
                    closest_cp = self.changepoint_dates[np.argmin(time_diffs)]
                    days_diff = min(time_diffs)
                    logger.info(f"   ✓ {crisis_name}: DETECTED")
                    logger.info(f"      Expected: {crisis_date.date()}, Detected: {closest_cp.date()} ({days_diff} days diff)")
                    detected_crises.append(crisis_name)
                else:
                    logger.warning(f"   ✗ {crisis_name}: NOT DETECTED")
                    logger.warning(f"      Expected around: {crisis_date.date()}")
                    missed_crises.append(crisis_name)
        
        # Success criteria
        critical_crises = ['2008 Global Financial Crisis', '2020 COVID-19 Crash']
        critical_detected = [c for c in critical_crises if c in detected_crises]
        
        if len(critical_detected) == len(critical_crises):
            logger.info("\n   ✅ SUCCESS: All critical crises (2008 GFC, 2020 COVID) detected!")
        else:
            logger.error("\n   ❌ FAILURE: Critical crises missing. Model needs tuning.")
    
    def get_regime_signal(self) -> pd.Series:
        """
        Get the final regime change signal.
        
        Returns:
            pd.Series: Binary signal (0/1)
        """
        if self.regime_signal is None:
            raise ValueError("Pipeline not run yet. Call run() first.")
        
        return self.regime_signal


def main():
    """
    Main entry point for the regime detection system.
    """
    # Import configuration
    try:
        import config
        DATA_PATH = config.DATA_PATH
        OUTPUT_DIR = config.OUTPUT_DIR
        VOLATILITY_WINDOW = config.VOLATILITY_WINDOW
        DETECTION_METHOD = config.DETECTION_METHOD
        N_BREAKS = config.N_BREAKS
        PENALTY = config.PENALTY
        TUNE_PENALTY = config.TUNE_PENALTY
        logger.info("✓ Loaded configuration from config.py")
    except ImportError:
        # Fallback to defaults
        DATA_PATH = "../Nifty_50.csv"
        OUTPUT_DIR = "outputs"
        VOLATILITY_WINDOW = 30
        DETECTION_METHOD = 'n_breaks'
        N_BREAKS = 15
        PENALTY = 10.0
        TUNE_PENALTY = True
        logger.info("Using default configuration")
    
    # Create and run pipeline
    pipeline = RegimeDetectionPipeline(DATA_PATH, OUTPUT_DIR)
    
    try:
        results = pipeline.run(
            volatility_window=VOLATILITY_WINDOW,
            detection_method=DETECTION_METHOD,
            n_breaks=N_BREAKS,
            penalty=PENALTY,
            tune_penalty=TUNE_PENALTY
        )
        
        # Get the regime signal
        regime_signal = pipeline.get_regime_signal()
        
        logger.info("\n✅ Pipeline executed successfully!")
        logger.info(f"Check outputs in: {OUTPUT_DIR}/")
        
        # Show plots
        plt.show()
        
        return results
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
