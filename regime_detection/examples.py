"""
Example Script: Using the Regime Detection System
Demonstrates how to use the regime detection output in practice
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the main pipeline
from main import RegimeDetectionPipeline


def example_1_basic_usage():
    """
    Example 1: Basic usage - Run the complete pipeline
    """
    print("="*80)
    print("EXAMPLE 1: Basic Pipeline Execution")
    print("="*80)
    
    # Initialize the pipeline
    pipeline = RegimeDetectionPipeline(
        data_path="../Nifty_50.csv",
        output_dir="outputs"
    )
    
    # Run with default settings
    results = pipeline.run()
    
    # Get the regime signal
    regime_signal = pipeline.get_regime_signal()
    
    print(f"\nGenerated regime signal with {regime_signal.sum()} regime changes")
    print(f"Output saved to: outputs/")
    
    return results, regime_signal


def example_2_load_and_analyze():
    """
    Example 2: Load generated outputs and analyze
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Load and Analyze Regime Data")
    print("="*80)
    
    # Load regime signals
    signals = pd.read_csv('outputs/regime_signals.csv', parse_dates=['date'])
    signals.set_index('date', inplace=True)
    
    print(f"\nLoaded {len(signals)} days of regime signals")
    print(f"Regime changes detected: {signals['regime_change_signal'].sum()}")
    print(f"Unique regimes: {signals['regime_label'].nunique()}")
    
    # Load regime statistics
    stats = pd.read_csv('outputs/regime_statistics.csv')
    
    print("\n--- Regime Statistics Summary ---")
    print(f"Average regime duration: {stats['duration_days'].mean():.0f} days")
    print(f"Average return per regime: {stats['total_return_pct'].mean():.2f}%")
    print(f"Average volatility: {stats['annualized_volatility_pct'].mean():.2f}%")
    
    # Find best and worst regimes
    best_regime = stats.loc[stats['total_return_pct'].idxmax()]
    worst_regime = stats.loc[stats['total_return_pct'].idxmin()]
    
    print(f"\nBest regime: {best_regime['start_date']} to {best_regime['end_date']}")
    print(f"  Return: {best_regime['total_return_pct']:.2f}%, Vol: {best_regime['annualized_volatility_pct']:.2f}%")
    
    print(f"\nWorst regime: {worst_regime['start_date']} to {worst_regime['end_date']}")
    print(f"  Return: {worst_regime['total_return_pct']:.2f}%, Vol: {worst_regime['annualized_volatility_pct']:.2f}%")
    
    return signals, stats


def example_3_regime_based_strategy():
    """
    Example 3: Simple regime-based trading strategy
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Regime-Based Trading Strategy")
    print("="*80)
    
    # Load data
    signals = pd.read_csv('outputs/regime_signals.csv', parse_dates=['date'])
    signals.set_index('date', inplace=True)
    
    stats = pd.read_csv('outputs/regime_statistics.csv')
    
    # Identify high and low volatility regimes
    median_vol = stats['annualized_volatility_pct'].median()
    high_vol_regimes = stats[stats['annualized_volatility_pct'] > median_vol]['regime_id'].values
    low_vol_regimes = stats[stats['annualized_volatility_pct'] <= median_vol]['regime_id'].values
    
    print(f"\nHigh volatility regimes: {high_vol_regimes}")
    print(f"Low volatility regimes: {low_vol_regimes}")
    
    # Create strategy: Reduce position in high volatility regimes
    signals['strategy_position'] = 1.0  # Default: fully invested
    signals.loc[signals['regime_label'].isin(high_vol_regimes), 'strategy_position'] = 0.5  # Half position
    
    print(f"\nStrategy Rules:")
    print(f"  - Full position (100%) in low volatility regimes")
    print(f"  - Half position (50%) in high volatility regimes")
    
    # Calculate time in each position
    full_position_days = (signals['strategy_position'] == 1.0).sum()
    half_position_days = (signals['strategy_position'] == 0.5).sum()
    
    print(f"\nStrategy Statistics:")
    print(f"  Full position: {full_position_days} days ({full_position_days/len(signals)*100:.1f}%)")
    print(f"  Half position: {half_position_days} days ({half_position_days/len(signals)*100:.1f}%)")
    
    return signals


def example_4_crisis_detection():
    """
    Example 4: Use regime changes as crisis warnings
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Crisis Detection System")
    print("="*80)
    
    # Load data
    signals = pd.read_csv('outputs/regime_signals.csv', parse_dates=['date'])
    signals.set_index('date', inplace=True)
    
    stats = pd.read_csv('outputs/regime_statistics.csv')
    
    # Detect crisis regimes (high volatility + negative returns)
    crisis_regimes = stats[
        (stats['annualized_volatility_pct'] > 40) & 
        (stats['total_return_pct'] < 0)
    ]
    
    print(f"\nDetected {len(crisis_regimes)} crisis periods:")
    for _, regime in crisis_regimes.iterrows():
        print(f"\n  Crisis Period {regime['regime_id']}:")
        print(f"    Date: {regime['start_date']} to {regime['end_date']}")
        print(f"    Duration: {regime['duration_days']} days")
        print(f"    Return: {regime['total_return_pct']:.2f}%")
        print(f"    Volatility: {regime['annualized_volatility_pct']:.2f}%")
    
    # Create crisis warning signal
    signals['crisis_warning'] = 0
    for _, regime in crisis_regimes.iterrows():
        regime_id = regime['regime_id']
        signals.loc[signals['regime_label'] == regime_id, 'crisis_warning'] = 1
    
    crisis_days = signals['crisis_warning'].sum()
    print(f"\nTotal days in crisis mode: {crisis_days} ({crisis_days/len(signals)*100:.1f}%)")
    
    return signals, crisis_regimes


def example_5_custom_detection():
    """
    Example 5: Run detection with custom parameters
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Parameter Detection")
    print("="*80)
    
    # Initialize pipeline
    pipeline = RegimeDetectionPipeline(
        data_path="../Nifty_50.csv",
        output_dir="outputs_custom"
    )
    
    # Run with custom parameters
    print("\nRunning with custom settings:")
    print("  - Volatility window: 20 days (more sensitive)")
    print("  - Detection method: penalty")
    print("  - Penalty: 2.0 (moderate sensitivity)")
    
    results = pipeline.run(
        volatility_window=20,      # Shorter window = more sensitive
        detection_method='penalty', # Use penalty-based method
        penalty=2.0,               # Lower penalty = more breaks
        tune_penalty=False         # Skip tuning for speed
    )
    
    print(f"\nDetected {results['changepoints']['count']} changepoints")
    print("\nChangepoint dates:")
    for date in results['changepoints']['dates']:
        print(f"  - {date}")
    
    return results


def example_6_visualize_regimes():
    """
    Example 6: Create custom regime visualization
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Custom Regime Visualization")
    print("="*80)
    
    # Load data
    signals = pd.read_csv('outputs/regime_signals.csv', parse_dates=['date'])
    signals.set_index('date', inplace=True)
    
    stats = pd.read_csv('outputs/regime_statistics.csv')
    
    # Create custom plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot regime volatility over time
    for _, regime in stats.iterrows():
        start = pd.to_datetime(regime['start_date'])
        end = pd.to_datetime(regime['end_date'])
        vol = regime['annualized_volatility_pct']
        
        # Color based on volatility level
        if vol > 40:
            color = 'red'
            alpha = 0.3
        elif vol > 25:
            color = 'orange'
            alpha = 0.2
        else:
            color = 'green'
            alpha = 0.1
        
        ax.axvspan(start, end, color=color, alpha=alpha)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regime Color (Red=High Vol, Green=Low Vol)', fontsize=12, fontweight='bold')
    ax.set_title('Regime Volatility Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/custom_regime_plot.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved custom visualization to outputs/custom_regime_plot.png")
    
    return fig


def example_7_regime_transitions():
    """
    Example 7: Analyze regime transitions
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Regime Transition Analysis")
    print("="*80)
    
    # Load data
    signals = pd.read_csv('outputs/regime_signals.csv', parse_dates=['date'])
    stats = pd.read_csv('outputs/regime_statistics.csv')
    
    # Identify transitions
    changepoint_dates = signals[signals['regime_change_signal'] == 1]['date'].tolist()
    
    print(f"\nAnalyzing {len(changepoint_dates)} regime transitions:\n")
    
    for i, date in enumerate(changepoint_dates, 1):
        # Find regime before and after
        before_regime = stats[stats['end_date'] == str(date.date())]
        after_regime = stats[stats['start_date'] == str(date.date())]
        
        if len(before_regime) > 0 and len(after_regime) > 0:
            before = before_regime.iloc[0]
            after = after_regime.iloc[0]
            
            vol_change = after['annualized_volatility_pct'] - before['annualized_volatility_pct']
            
            print(f"Transition {i}: {date.date()}")
            print(f"  Volatility: {before['annualized_volatility_pct']:.1f}% → {after['annualized_volatility_pct']:.1f}% ({vol_change:+.1f}%)")
            print(f"  Type: {'📈 Volatility Spike' if vol_change > 10 else '📉 Volatility Drop' if vol_change < -10 else '➡️ Stable Transition'}")
            print()
    
    return changepoint_dates


if __name__ == "__main__":
    """
    Run all examples
    """
    print("\n" + "🚀 "*20)
    print("REGIME DETECTION SYSTEM - USAGE EXAMPLES")
    print("🚀 "*20)
    
    # Example 1: Basic usage (commented out to avoid re-running)
    # results, regime_signal = example_1_basic_usage()
    
    # Example 2: Load and analyze existing results
    signals, stats = example_2_load_and_analyze()
    
    # Example 3: Regime-based strategy
    strategy_signals = example_3_regime_based_strategy()
    
    # Example 4: Crisis detection
    crisis_signals, crisis_regimes = example_4_crisis_detection()
    
    # Example 5: Custom detection (commented out to avoid re-running)
    # custom_results = example_5_custom_detection()
    
    # Example 6: Custom visualization
    fig = example_6_visualize_regimes()
    
    # Example 7: Regime transitions
    transitions = example_7_regime_transitions()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nCheck the outputs/ directory for generated files.")
    print("\nYou can now use these patterns in your own trading systems!")
