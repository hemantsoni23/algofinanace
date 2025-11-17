"""
Main Script - Nifty 50 Regime Detection with BCD
Uses hourly data for high-frequency regime detection
Supports both hierarchical (high recall) and multi-signal (high precision) BCD
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add modules to path
sys.path.append(str(Path(__file__).parent / 'modules'))

from modules.data_loader import DataLoader
from modules.feature_engineering_v2 import EnhancedFeatureEngineer
from modules.hierarchical_changepoint_detector import HierarchicalChangepointDetector
from modules.multi_signal_bcd import MultiSignalBCD
from modules.visualization import Visualizer

# Import configs
import config_optimized_v2 as config


class RegimeDetectionPipeline:
    """
    Complete pipeline for detecting market regime changes using BCD
    """
    
    def __init__(self, 
                 data_file: str = "../Nifty_50_hourly.csv",
                 mode: str = "hierarchical",
                 output_dir: str = "outputs_production"):
        """
        Initialize the pipeline
        
        Args:
            data_file: Path to OHLCV data (daily or hourly)
            mode: "hierarchical" (high recall) or "multi-signal" (high precision)
            output_dir: Directory for outputs
        """
        self.data_file = data_file
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.data = None
        self.features = None
        self.breakpoints = None
        self.detector = None
        
    def run(self):
        """Execute full pipeline"""
        
        print("="*80)
        print("NIFTY 50 REGIME DETECTION PIPELINE")
        print(f"Mode: {self.mode.upper()}")
        print(f"Data: {self.data_file}")
        print("="*80)
        
        # Step 1: Load data
        self._load_data()
        
        # Step 2: Engineer features
        self._engineer_features()
        
        # Step 3: Detect breakpoints
        self._detect_breakpoints()
        
        # Step 4: Export results
        self._export_results()
        
        # Step 5: Visualize
        self._visualize()
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE")
        print("="*80)
        
        return {
            'breakpoints': self.breakpoints,
            'data': self.data,
            'features': self.features
        }
    
    def _load_data(self):
        """Load OHLCV data"""
        print("\n📂 STEP 1: Loading Data")
        print("-"*80)
        
        loader = DataLoader(self.data_file)
        self.data = loader.load_data()
        
        print(f"✓ Loaded {len(self.data)} rows")
        print(f"  Date range: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"  Columns: {', '.join(self.data.columns)}")
        
        # Detect frequency
        time_diff = (self.data.index[1] - self.data.index[0]).total_seconds() / 3600
        if time_diff <= 1:
            freq = "Hourly"
        elif time_diff <= 24:
            freq = "Daily"
        else:
            freq = "Unknown"
        print(f"  Frequency: {freq}")
    
    def _engineer_features(self):
        """Create technical features"""
        print("\n🔧 STEP 2: Engineering Features")
        print("-"*80)
        
        engineer = EnhancedFeatureEngineer()
        self.features = engineer.engineer_features(self.data)
        
        print(f"✓ Created {self.features.shape[1]} features")
        
        # Show sample features
        key_features = ['volatility_20d', 'returns_squared', 'vol_of_vol', 
                       'volume_volatility', 'range_volatility']
        available = [f for f in key_features if f in self.features.columns]
        print(f"  Key signals: {', '.join(available)}")
    
    def _detect_breakpoints(self):
        """Detect regime change breakpoints"""
        print("\n🎯 STEP 3: Detecting Breakpoints")
        print("-"*80)
        
        if self.mode == "hierarchical":
            self._detect_hierarchical()
        elif self.mode == "multi-signal":
            self._detect_multisignal()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _detect_hierarchical(self):
        """Hierarchical BCD - high recall (55.6%), suitable for Layer 1"""
        print("Using: Hierarchical BCD (High Recall)")
        
        # Initialize detector with config parameters
        self.detector = HierarchicalChangepointDetector(
            model=config.BCD_MODEL,
            min_size=config.BCD_MIN_SIZE,
            jump=config.BCD_JUMP,
            confidence_threshold=config.CONFIDENCE_THRESHOLD_MAJOR
        )
        
        # Get signal
        signal = self.features[config.BCD_SIGNAL].values
        
        # Fit (hierarchical detector only takes signal)
        self.detector.fit(signal)
        
        # Detect
        result = self.detector.detect_hierarchical(
            n_bkps_major=config.N_BKPS_MAJOR,
            n_bkps_minor=config.N_BKPS_MINOR,
            dates=self.data.index
        )
        
        # Combine major and minor (breakpoints are integers, confidence is separate)
        major_bps = result['major_breakpoints']
        major_confs = result['major_confidence']
        minor_bps = result['minor_breakpoints']
        minor_confs = result['minor_confidence']
        
        breakpoints_list = []
        
        # Add major breakpoints
        for bp_idx, conf in zip(major_bps, major_confs):
            breakpoints_list.append({
                'date': self.data.index[bp_idx],
                'index': bp_idx,
                'level': 'major',
                'confidence': conf
            })
        
        # Add minor breakpoints
        for bp_idx, conf in zip(minor_bps, minor_confs):
            breakpoints_list.append({
                'date': self.data.index[bp_idx],
                'index': bp_idx,
                'level': 'minor',
                'confidence': conf
            })
        
        self.breakpoints = pd.DataFrame(breakpoints_list).sort_values('index')
        
        print(f"✓ Detected {len(self.breakpoints)} breakpoints")
        print(f"  Major: {len(major_bps)} (high confidence)")
        print(f"  Minor: {len(minor_bps)} (medium confidence)")
        if len(self.breakpoints) > 0:
            print(f"  Confidence range: {self.breakpoints['confidence'].min():.1%} - {self.breakpoints['confidence'].max():.1%}")
    
    def _detect_multisignal(self):
        """Multi-Signal BCD - high precision (75%), suitable for standalone"""
        print("Using: Multi-Signal BCD (High Precision)")
        
        # Initialize detector
        self.detector = MultiSignalBCD(config)
        
        # Fit on multiple signals
        self.detector.fit(self.data, self.features)
        
        print(f"  Signals used: {len(self.detector.signal_breakpoints)}")
        for signal_name, bps in self.detector.signal_breakpoints.items():
            print(f"    - {signal_name}: {len(bps)} breakpoints")
        
        # Detect with consensus (min 2 signals must agree)
        self.breakpoints = self.detector.detect_with_consensus(min_votes=2)
        
        # Apply advanced filters
        self.breakpoints = self._apply_filters(self.breakpoints)
        
        print(f"\n✓ Detected {len(self.breakpoints)} high-quality breakpoints")
        if 'n_signals_voting' in self.breakpoints.columns:
            print(f"  Signal consensus: {self.breakpoints['n_signals_voting'].min():.0f}-{self.breakpoints['n_signals_voting'].max():.0f} signals")
        
        confidence_col = 'final_confidence' if 'final_confidence' in self.breakpoints.columns else 'confidence'
        if confidence_col in self.breakpoints.columns:
            print(f"  Confidence range: {self.breakpoints[confidence_col].min():.1%} - {self.breakpoints[confidence_col].max():.1%}")
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply post-detection filters"""
        
        if len(df) == 0:
            return df
        
        initial_count = len(df)
        
        # Filter by confidence
        confidence_col = 'final_confidence' if 'final_confidence' in df.columns else 'confidence'
        min_confidence = 0.50
        df = df[df[confidence_col] >= min_confidence]
        print(f"  After confidence filter (>={min_confidence}): {len(df)}")
        
        # Filter by signal consensus
        if 'n_signals_voting' in df.columns:
            df = df[df['n_signals_voting'] >= 2]
            print(f"  After consensus filter (>=2 signals): {len(df)}")
        
        # Filter by magnitude
        if 'vol_change' in df.columns:
            min_vol_change = 0.15
            df = df[df['vol_change'] >= min_vol_change]
            print(f"  After magnitude filter (>={min_vol_change}): {len(df)}")
        
        print(f"  Removed {initial_count - len(df)} low-quality breakpoints")
        
        return df
    
    def _export_results(self):
        """Export breakpoints and metadata"""
        print("\n💾 STEP 4: Exporting Results")
        print("-"*80)
        
        # Export breakpoints
        bp_file = self.output_dir / "breakpoints.csv"
        self.breakpoints.to_csv(bp_file, index=False)
        print(f"✓ Saved breakpoints: {bp_file}")
        
        # Export for downstream layers (SKF, HMM, etc.)
        if self.mode == "hierarchical":
            # Export major breakpoints for downstream
            major = self.breakpoints[self.breakpoints['level'] == 'major']
            downstream_file = self.output_dir / "breakpoints_for_downstream.csv"
            major[['date', 'index', 'confidence']].to_csv(downstream_file, index=False)
            print(f"✓ Saved for downstream: {downstream_file}")
        elif self.mode == "multi-signal":
            # All breakpoints are high quality
            downstream_file = self.output_dir / "breakpoints_for_downstream.csv"
            
            confidence_col = 'final_confidence' if 'final_confidence' in self.breakpoints.columns else 'confidence'
            export_cols = ['date', 'index' if 'index' in self.breakpoints.columns else 'breakpoint_index', confidence_col]
            
            # Rename for consistency
            export_df = self.breakpoints[export_cols].copy()
            export_df.columns = ['date', 'index', 'confidence']
            
            export_df.to_csv(downstream_file, index=False)
            print(f"✓ Saved for downstream: {downstream_file}")
        
        # Export summary statistics
        stats = {
            'mode': self.mode,
            'data_file': str(self.data_file),
            'total_rows': len(self.data),
            'date_range': f"{self.data.index[0]} to {self.data.index[-1]}",
            'total_breakpoints': len(self.breakpoints),
            'timestamp': datetime.now().isoformat()
        }
        
        if self.mode == "hierarchical":
            stats['major_breakpoints'] = len(self.breakpoints[self.breakpoints['level'] == 'major'])
            stats['minor_breakpoints'] = len(self.breakpoints[self.breakpoints['level'] == 'minor'])
        
        stats_file = self.output_dir / "detection_summary.txt"
        with open(stats_file, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        print(f"✓ Saved summary: {stats_file}")
    
    def _visualize(self):
        """Create visualizations"""
        print("\n📈 STEP 5: Creating Visualizations")
        print("-"*80)
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        
        # Create 3-panel visualization
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Panel 1: Price with breakpoints
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=1, alpha=0.7)
        
        for _, bp in self.breakpoints.iterrows():
            bp_date = pd.to_datetime(bp['date'])
            
            # Color by confidence or level
            if 'level' in bp:
                color = 'red' if bp['level'] == 'major' else 'orange'
            else:
                confidence = bp.get('final_confidence', bp.get('confidence', 0.5))
                color = 'red' if confidence >= 0.7 else 'orange'
            
            ax1.axvline(bp_date, color=color, alpha=0.6, linestyle='--', linewidth=2)
        
        ax1.set_title(f'Nifty 50: Price with Detected Breakpoints ({self.mode.title()} BCD)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (INR)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Volatility with breakpoints
        ax2 = axes[1]
        ax2.plot(self.features.index, self.features['volatility_20d'], 
                label='20d Volatility', linewidth=1, color='blue', alpha=0.7)
        
        for _, bp in self.breakpoints.iterrows():
            bp_date = pd.to_datetime(bp['date'])
            
            if 'level' in bp:
                color = 'red' if bp['level'] == 'major' else 'orange'
            else:
                confidence = bp.get('final_confidence', bp.get('confidence', 0.5))
                color = 'red' if confidence >= 0.7 else 'orange'
            
            ax2.axvline(bp_date, color=color, alpha=0.6, linestyle='--', linewidth=2)
        
        ax2.set_title('Volatility Signal with Breakpoints', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Confidence scores
        ax3 = axes[2]
        
        bp_dates = pd.to_datetime(self.breakpoints['date'])
        
        if 'level' in self.breakpoints.columns:
            # Hierarchical mode
            for level, color in [('major', 'red'), ('minor', 'orange')]:
                level_bps = self.breakpoints[self.breakpoints['level'] == level]
                ax3.scatter(pd.to_datetime(level_bps['date']), 
                          level_bps['confidence'],
                          c=color, s=100, alpha=0.7, edgecolors='black', label=level.title())
        else:
            # Multi-signal mode
            confidence_col = 'final_confidence' if 'final_confidence' in self.breakpoints.columns else 'confidence'
            confidences = self.breakpoints[confidence_col]
            colors = ['red' if c >= 0.7 else 'orange' if c >= 0.5 else 'yellow' for c in confidences]
            ax3.scatter(bp_dates, confidences, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        ax3.axhline(0.5, color='green', linestyle='--', label='Min Threshold', alpha=0.5)
        ax3.set_title('Breakpoint Confidence Scores', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Confidence')
        ax3.set_xlabel('Date')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        viz_file = self.output_dir / "regime_detection.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: {viz_file}")
        plt.close()


def main():
    """Main entry point with CLI"""
    
    parser = argparse.ArgumentParser(
        description="Nifty 50 Regime Detection with BCD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on hourly data with hierarchical BCD (high recall)
  python main.py --data hourly --mode hierarchical
  
  # Run on daily data with multi-signal BCD (high precision)
  python main.py --data daily --mode multi-signal
  
  # Custom data file
  python main.py --file ../my_data.csv --mode hierarchical
        """
    )
    
    parser.add_argument(
        '--data',
        choices=['hourly', 'daily'],
        default='hourly',
        help='Data frequency (default: hourly)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['hierarchical', 'multi-signal'],
        default='hierarchical',
        help='BCD mode: hierarchical (high recall) or multi-signal (high precision)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Custom data file path (overrides --data)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs_production',
        help='Output directory (default: outputs_production)'
    )
    
    args = parser.parse_args()
    
    # Determine data file
    if args.file:
        data_file = args.file
    elif args.data == 'hourly':
        data_file = "../Nifty_50_hourly.csv"
    else:
        data_file = "../Nifty_50.csv"
    
    # Run pipeline
    pipeline = RegimeDetectionPipeline(
        data_file=data_file,
        mode=args.mode,
        output_dir=args.output
    )
    
    results = pipeline.run()
    
    # Print summary
    print(f"\n📊 SUMMARY")
    print(f"   Breakpoints detected: {len(results['breakpoints'])}")
    print(f"   Data rows analyzed: {len(results['data'])}")
    print(f"   Output directory: {args.output}/")
    print(f"\n✅ Results ready for downstream layers (SKF, HMM, etc.)")


if __name__ == "__main__":
    main()
