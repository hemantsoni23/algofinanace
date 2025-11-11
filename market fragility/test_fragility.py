"""
Quick Test Script: Fragility Analysis on Your Nifty 50 Data

Run this to see the fragility proxy in action on your actual data.
"""

import pandas as pd
import sys
sys.path.append('/Users/hemantsoni/Documents/AlgoFinance')

from market_fragility_proxy import (
    calculate_fragility,
    plot_fragility_index,
    plot_fragility_components,
    generate_fragility_report
)

def main():
    print("\n" + "="*70)
    print("NIFTY 50 FRAGILITY ANALYSIS - QUICK TEST")
    print("="*70 + "\n")
    
    # Load your minute data
    print("Loading Nifty 50 minute data...")
    df = pd.read_csv('NIFTY 50_minute.csv')
    
    # Parse date column
    date_col = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            column_mapping[col] = 'Open'
        elif 'high' in col_lower:
            column_mapping[col] = 'High'
        elif 'low' in col_lower:
            column_mapping[col] = 'Low'
        elif 'close' in col_lower:
            column_mapping[col] = 'Close'
        elif 'volume' in col_lower or 'vol' in col_lower:
            column_mapping[col] = 'Volume'
    
    df.rename(columns=column_mapping, inplace=True)
    
    print(f"✓ Loaded {len(df)} bars")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Keep last 3000 bars for faster processing
    df = df.tail(3000)
    print(f"  Using last 3000 bars for analysis")
    
    # Calculate fragility
    print("\nCalculating fragility metrics...")
    df_fragility = calculate_fragility(
        df,
        fast_window=60,     # 1 hour
        slow_window=240,    # 4 hours
        z_window=240
    )
    
    # Generate report
    print("\nGenerating fragility report...")
    report = generate_fragility_report(df_fragility)
    
    print("\n" + "="*70)
    print("CURRENT MARKET STATUS")
    print("="*70)
    print(f"Current Fragility Index: {report['current_fragility']:.3f}")
    print(f"Risk Level: {report['risk_level']}")
    print(f"Interpretation: {report['interpretation']}")
    print(f"Percentile Rank: {report['percentile_rank']:.1f}%")
    
    if 'component_values' in report:
        print("\nComponent Breakdown:")
        print(f"  Volatility of Volatility: {report['component_values']['vol_of_vol']:>7.3f}")
        print(f"  Price Impact (Illiquidity): {report['component_values']['impact']:>7.3f}")
        print(f"  Tail Risk (Kurtosis):      {report['component_values']['tail_risk']:>7.3f}")
        print(f"\nDominant Driver: {report['dominant_driver']}")
    
    # Trading recommendations
    print("\n" + "="*70)
    print("TRADING RECOMMENDATIONS")
    print("="*70)
    
    fragility = report['current_fragility']
    
    if fragility > 2.5:
        print("⛔ DO NOT TRADE - Extreme fragility detected")
        print("   Recommendation: Close all positions, wait for stability")
        position_size = "0% (Flat)"
    elif fragility > 2.0:
        print("⚠️  CRITICAL - Severely fragile market")
        print("   Recommendation: Minimal exposure, very wide stops")
        position_size = "25% of normal"
    elif fragility > 1.0:
        print("⚠️  CAUTION - Elevated fragility")
        print("   Recommendation: Reduced exposure, wider stops")
        position_size = "50% of normal"
    elif fragility > 0:
        print("ℹ️  MODERATE - Above-average fragility")
        print("   Recommendation: Slightly reduced exposure")
        position_size = "75% of normal"
    else:
        print("✓  NORMAL - Market relatively stable")
        print("   Recommendation: Normal trading conditions")
        position_size = "100% (Full size)"
    
    print(f"\nSuggested Position Size: {position_size}")
    
    # Show last 10 fragility readings
    print("\n" + "="*70)
    print("RECENT FRAGILITY HISTORY (Last 10 bars)")
    print("="*70)
    recent = df_fragility[['Close', 'Fragility_Index']].tail(10)
    print(recent.to_string())
    
    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\nPlot 1: Price & Fragility Index...")
    plot_fragility_index(df_fragility)
    
    print("Plot 2: Component Dashboard...")
    plot_fragility_components(df_fragility)
    
    # Save results
    import os
    os.makedirs('outputs', exist_ok=True)
    output_file = 'outputs/fragility_analysis.csv'
    df_fragility.to_csv(output_file)
    print(f"\n✓ Full results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the visualizations to understand fragility patterns")
    print("2. Integrate fragility into your regime detection (see FRAGILITY_INTEGRATION_GUIDE.md)")
    print("3. Backtest your strategies with fragility-based position sizing")
    print("4. Set up real-time monitoring using calculate_fragility_live()")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
