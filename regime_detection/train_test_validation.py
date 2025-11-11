import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

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
        logging.FileHandler('outputs/train_test_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainTestValidator:
    """
    Validates regime detection using train-test split methodology
    """
    
    def __init__(self, data_path: str, output_dir: str = 'outputs'):
        """
        Initialize the validator.
        
        Args:
            data_path (str): Path to the OHLCV data file
            output_dir (str): Directory for outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.train_data = None
        self.test_data = None
        self.split_date = None
        
        logger.info("="*80)
        logger.info("REGIME DETECTION - TRAIN/TEST VALIDATION")
        logger.info("="*80)
    
    def split_data(self, train_ratio: float = 0.8, split_date: str = None):
        """
        Split data into train and test sets.
        
        Args:
            train_ratio (float): Ratio of data for training (0-1)
            split_date (str): Optional explicit split date (YYYY-MM-DD)
        """
        logger.info("\n" + "="*80)
        logger.info("DATA SPLITTING")
        logger.info("="*80)
        
        # Load full data
        loader = DataLoader(self.data_path)
        self.data = loader.load_data()
        
        if split_date:
            self.split_date = pd.Timestamp(split_date)
            logger.info(f"Using explicit split date: {self.split_date.date()}")
        else:
            # Calculate split point
            n_samples = len(self.data)
            split_idx = int(n_samples * train_ratio)
            self.split_date = self.data.index[split_idx]
            logger.info(f"Using ratio split ({train_ratio:.0%}): {self.split_date.date()}")
        
        # Split the data
        self.train_data = self.data[self.data.index < self.split_date]
        self.test_data = self.data[self.data.index >= self.split_date]
        
        logger.info(f"\n✓ TRAIN SET:")
        logger.info(f"  - Samples: {len(self.train_data)}")
        logger.info(f"  - Period: {self.train_data.index[0].date()} to {self.train_data.index[-1].date()}")
        logger.info(f"  - Years: {(self.train_data.index[-1] - self.train_data.index[0]).days / 365.25:.1f}")
        
        logger.info(f"\n✓ TEST SET:")
        logger.info(f"  - Samples: {len(self.test_data)}")
        logger.info(f"  - Period: {self.test_data.index[0].date()} to {self.test_data.index[-1].date()}")
        logger.info(f"  - Years: {(self.test_data.index[-1] - self.test_data.index[0]).days / 365.25:.1f}")
        
        return self.train_data, self.test_data
    
    def detect_changepoints(self, data, volatility_window=30, penalty=1.0):
        """
        Detect changepoints in a dataset.
        
        Args:
            data (pd.DataFrame): Data to process
            volatility_window (int): Rolling window for volatility
            penalty (float): Penalty parameter
            
        Returns:
            dict: Results including changepoint dates
        """
        # Feature Engineering
        engineer = FeatureEngineer(data)
        log_returns = engineer.calculate_log_returns()
        volatility = engineer.calculate_rolling_volatility(window=volatility_window)
        features = engineer.add_price_data()
        
        # Get clean signal
        signal, dates = engineer.get_clean_signal(f'rolling_vol_{volatility_window}d')
        
        # Changepoint Detection
        detector = ChangepointDetector(model="l2", min_size=5)
        detector.fit(signal)
        breakpoints = detector.predict_with_penalty(pen=penalty)
        
        # Handle case with no changepoints
        if len(breakpoints) == 0:
            changepoint_dates = []
        else:
            changepoint_dates = detector.map_breakpoints_to_dates(dates)
        
        return {
            'features': features,
            'signal': signal,
            'dates': dates,
            'changepoint_dates': changepoint_dates,
            'detector': detector
        }
    
    def validate(self, volatility_window=30, penalty=1.0):
        """
        Run complete train-test validation.
        
        Args:
            volatility_window (int): Rolling window for volatility
            penalty (float): Penalty parameter
            
        Returns:
            dict: Validation results
        """
        results = {
            'train': {},
            'test': {},
            'validation': {},
            'full_dataset': {}
        }
        
        # TRAIN SET DETECTION
        logger.info("\n" + "="*80)
        logger.info("TRAIN SET - CHANGEPOINT DETECTION")
        logger.info("="*80)
        
        train_results = self.detect_changepoints(
            self.train_data, 
            volatility_window=volatility_window,
            penalty=penalty
        )
        
        logger.info(f"✓ Detected {len(train_results['changepoint_dates'])} changepoints in TRAIN set")
        for i, date in enumerate(train_results['changepoint_dates'], 1):
            logger.info(f"   {i:2d}. {date.date()}")
        
        results['train'] = {
            'changepoint_dates': [str(d.date()) for d in train_results['changepoint_dates']],
            'count': len(train_results['changepoint_dates'])
        }
        
        # TEST SET DETECTION
        logger.info("\n" + "="*80)
        logger.info("TEST SET - CHANGEPOINT DETECTION")
        logger.info("="*80)
        
        test_results = self.detect_changepoints(
            self.test_data,
            volatility_window=volatility_window,
            penalty=penalty
        )
        
        logger.info(f"✓ Detected {len(test_results['changepoint_dates'])} changepoints in TEST set")
        for i, date in enumerate(test_results['changepoint_dates'], 1):
            logger.info(f"   {i:2d}. {date.date()}")
        
        results['test'] = {
            'changepoint_dates': [str(d.date()) for d in test_results['changepoint_dates']],
            'count': len(test_results['changepoint_dates'])
        }
        
        # FULL DATASET DETECTION (for comparison)
        logger.info("\n" + "="*80)
        logger.info("FULL DATASET - CHANGEPOINT DETECTION")
        logger.info("="*80)
        
        full_results = self.detect_changepoints(
            self.data,
            volatility_window=volatility_window,
            penalty=penalty
        )
        
        logger.info(f"✓ Detected {len(full_results['changepoint_dates'])} changepoints in FULL dataset")
        for i, date in enumerate(full_results['changepoint_dates'], 1):
            logger.info(f"   {i:2d}. {date.date()}")
        
        results['full_dataset'] = {
            'changepoint_dates': [str(d.date()) for d in full_results['changepoint_dates']],
            'count': len(full_results['changepoint_dates'])
        }
        
        # VALIDATION METRICS
        logger.info("\n" + "="*80)
        logger.info("VALIDATION ANALYSIS")
        logger.info("="*80)
        
        # Check how many train changepoints are detected in full dataset
        train_in_full = sum(
            1 for train_cp in train_results['changepoint_dates']
            if any(abs((train_cp - full_cp).days) <= 30 
                   for full_cp in full_results['changepoint_dates'])
        )
        
        consistency_rate = train_in_full / len(train_results['changepoint_dates']) * 100 if len(train_results['changepoint_dates']) > 0 else 0
        
        logger.info(f"✓ Consistency: {train_in_full}/{len(train_results['changepoint_dates'])} train changepoints found in full dataset ({consistency_rate:.1f}%)")
        
        # Check if test changepoints were anticipated
        test_anticipated = sum(
            1 for test_cp in test_results['changepoint_dates']
            if any(abs((test_cp - full_cp).days) <= 30 
                   for full_cp in full_results['changepoint_dates'])
        )
        
        anticipation_rate = test_anticipated / len(test_results['changepoint_dates']) * 100 if len(test_results['changepoint_dates']) > 0 else 0
        
        logger.info(f"✓ Anticipation: {test_anticipated}/{len(test_results['changepoint_dates'])} test changepoints found in full dataset ({anticipation_rate:.1f}%)")
        
        results['validation'] = {
            'split_date': str(self.split_date.date()),
            'train_changepoints_in_full': train_in_full,
            'consistency_rate': consistency_rate,
            'test_changepoints_anticipated': test_anticipated,
            'anticipation_rate': anticipation_rate,
            'total_full_changepoints': len(full_results['changepoint_dates'])
        }
        
        # Generate visualizations
        self._create_validation_plots(train_results, test_results, full_results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _create_validation_plots(self, train_results, test_results, full_results):
        """Create comprehensive validation visualizations."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*80)
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Plot 1: Train-Test Split Comparison
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Train set
        ax = axes[0]
        train_vol = train_results['features']['rolling_vol_30d'].dropna()
        ax.plot(train_vol.index, train_vol.values, color='blue', alpha=0.6, linewidth=1)
        for cp in train_results['changepoint_dates']:
            ax.axvline(cp, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_ylabel('Volatility', fontsize=12, fontweight='bold')
        ax.set_title('TRAIN SET - Changepoint Detection', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(['Volatility', 'Changepoints'], loc='upper left')
        
        # Test set
        ax = axes[1]
        test_vol = test_results['features']['rolling_vol_30d'].dropna()
        ax.plot(test_vol.index, test_vol.values, color='green', alpha=0.6, linewidth=1)
        for cp in test_results['changepoint_dates']:
            ax.axvline(cp, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_ylabel('Volatility', fontsize=12, fontweight='bold')
        ax.set_title('TEST SET - Changepoint Detection', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(['Volatility', 'Changepoints'], loc='upper left')
        
        # Full dataset
        ax = axes[2]
        full_vol = full_results['features']['rolling_vol_30d'].dropna()
        ax.plot(full_vol.index, full_vol.values, color='purple', alpha=0.6, linewidth=1)
        ax.axvline(self.split_date, color='black', linestyle='-', alpha=0.8, linewidth=3, label='Split Date')
        for cp in full_results['changepoint_dates']:
            ax.axvline(cp, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility', fontsize=12, fontweight='bold')
        ax.set_title('FULL DATASET - Changepoint Detection', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(['Volatility', 'Split Date', 'Changepoints'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'train_test_validation.png', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved train-test validation plot")
        plt.close()
        
        # Plot 2: Price with Changepoints
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot price
        ax.plot(self.data.index, self.data['Close'], color='black', linewidth=1.5, label='Price')
        
        # Mark split
        ax.axvline(self.split_date, color='black', linestyle='-', alpha=0.8, linewidth=3, label='Train/Test Split')
        
        # Mark changepoints from full dataset
        for cp in full_results['changepoint_dates']:
            if cp < self.split_date:
                ax.axvline(cp, color='blue', linestyle='--', alpha=0.6, linewidth=1.5)
            else:
                ax.axvline(cp, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Add dummy lines for legend
        ax.plot([], [], color='blue', linestyle='--', linewidth=1.5, label='Train Changepoints')
        ax.plot([], [], color='green', linestyle='--', linewidth=1.5, label='Test Changepoints')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.set_title('Price with Train-Test Split and Changepoints', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'price_with_changepoints.png', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved price with changepoints plot")
        plt.close()
    
    def _save_results(self, results):
        """Save results to JSON file."""
        results_file = self.output_dir / 'train_test_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Saved results to {results_file}")


def main():
    """Main execution function."""
    
    # Configuration
    DATA_PATH = "../Nifty_50.csv"
    OUTPUT_DIR = "outputs"
    TRAIN_RATIO = 0.8  # 80% train, 20% test
    VOLATILITY_WINDOW = 30
    PENALTY = 1.0  # Using recommended penalty from tuning
    
    # Alternative: Use explicit split date
    # SPLIT_DATE = "2022-01-01"
    SPLIT_DATE = None
    
    # Create validator
    validator = TrainTestValidator(DATA_PATH, OUTPUT_DIR)
    
    # Split data
    validator.split_data(train_ratio=TRAIN_RATIO, split_date=SPLIT_DATE)
    
    # Run validation
    results = validator.validate(
        volatility_window=VOLATILITY_WINDOW,
        penalty=PENALTY
    )
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)
    logger.info(f"✓ Train set detected {results['train']['count']} changepoints")
    logger.info(f"✓ Test set detected {results['test']['count']} changepoints")
    logger.info(f"✓ Full dataset detected {results['full_dataset']['count']} changepoints")
    logger.info(f"✓ Consistency rate: {results['validation']['consistency_rate']:.1f}%")
    logger.info(f"✓ Results saved to: {OUTPUT_DIR}/train_test_results.json")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    main()
