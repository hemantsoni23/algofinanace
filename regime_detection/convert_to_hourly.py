"""
Convert 1-minute Nifty 50 data to hourly OHLCV data
Handles market hours (9:15 AM - 3:30 PM IST)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def convert_minute_to_hourly(
    input_file: str = "../NIFTY 50_minute.csv",
    output_file: str = "../Nifty_50_hourly.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Convert 1-minute OHLCV data to hourly
    
    Args:
        input_file: Path to 1-minute CSV
        output_file: Path to save hourly CSV
        verbose: Print progress
    
    Returns:
        DataFrame with hourly OHLCV data
    """
    
    if verbose:
        print("="*80)
        print("CONVERTING 1-MINUTE DATA TO HOURLY")
        print("="*80)
    
    # Load 1-minute data
    if verbose:
        print(f"\n📂 Loading 1-minute data from: {input_file}")
    
    df = pd.read_csv(input_file)
    
    # Parse datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    if verbose:
        print(f"   Loaded {len(df):,} rows")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Columns: {', '.join(df.columns)}")
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # Set datetime as index for resampling
    df = df.set_index('date')
    
    # Resample to hourly (H = hour end)
    if verbose:
        print("\n🔄 Resampling to hourly frequency...")
    
    hourly = df.resample('H').agg({
        'Open': 'first',      # First price in the hour
        'High': 'max',        # Highest price in the hour
        'Low': 'min',         # Lowest price in the hour
        'Close': 'last',      # Last price in the hour
        'Volume': 'sum'       # Total volume in the hour
    })
    
    # Remove rows with no data (non-trading hours)
    hourly = hourly.dropna()
    
    # Filter only market hours (9:15 AM - 3:30 PM IST)
    # Keep hours from 9:00 to 15:00 (15:30 falls in 15:00 hour)
    hourly = hourly[hourly.index.hour.isin([9, 10, 11, 12, 13, 14, 15])]
    
    if verbose:
        print(f"   Hourly rows: {len(hourly):,}")
        print(f"   Date range: {hourly.index.min()} to {hourly.index.max()}")
        print(f"   Trading hours: {sorted(hourly.index.hour.unique())}")
    
    # Reset index to have date as column
    hourly = hourly.reset_index()
    hourly = hourly.rename(columns={'date': 'Date'})
    
    # Validate data quality
    if verbose:
        print("\n✅ Data Quality Checks:")
        print(f"   Missing values: {hourly.isnull().sum().sum()}")
        print(f"   Zero prices: {(hourly['Close'] == 0).sum()}")
        print(f"   Negative volumes: {(hourly['Volume'] < 0).sum()}")
    
    # Calculate basic statistics
    if verbose:
        print("\n📊 Hourly Data Statistics:")
        print(f"   Close price range: {hourly['Close'].min():.2f} - {hourly['Close'].max():.2f}")
        print(f"   Average daily hours: {len(hourly) / hourly['Date'].dt.date.nunique():.1f}")
        print(f"   Total trading days: {hourly['Date'].dt.date.nunique():,}")
    
    # Save to CSV
    if output_file:
        hourly.to_csv(output_file, index=False)
        if verbose:
            print(f"\n💾 Saved hourly data to: {output_file}")
    
    if verbose:
        print("\n" + "="*80)
        print("✅ CONVERSION COMPLETE")
        print("="*80)
    
    return hourly


def validate_hourly_data(df: pd.DataFrame) -> dict:
    """
    Validate hourly data quality
    
    Returns:
        dict with validation results
    """
    
    results = {
        'total_rows': len(df),
        'date_range': (df['Date'].min(), df['Date'].max()),
        'trading_days': df['Date'].dt.date.nunique(),
        'hours_per_day': len(df) / df['Date'].dt.date.nunique(),
        'missing_values': df.isnull().sum().sum(),
        'zero_prices': (df['Close'] == 0).sum(),
        'negative_volumes': (df['Volume'] < 0).sum(),
        'price_range': (df['Close'].min(), df['Close'].max()),
        'avg_volume': df['Volume'].mean(),
    }
    
    # Check for data gaps
    dates = pd.to_datetime(df['Date']).dt.date.unique()
    date_range = pd.date_range(start=dates.min(), end=dates.max(), freq='D')
    trading_days_expected = len([d for d in date_range if d.weekday() < 5])  # Mon-Fri
    results['data_completeness'] = len(dates) / trading_days_expected * 100
    
    return results


def compare_minute_vs_hourly():
    """Compare 1-minute vs hourly data characteristics"""
    
    print("\n" + "="*80)
    print("COMPARING 1-MINUTE vs HOURLY DATA")
    print("="*80)
    
    # Load minute data
    minute = pd.read_csv("../NIFTY 50_minute.csv")
    minute_count = len(minute)
    
    # Load hourly data
    hourly = pd.read_csv("../Nifty_50_hourly.csv")
    hourly_count = len(hourly)
    
    print(f"\n📊 Row Counts:")
    print(f"   1-minute data: {minute_count:,} rows")
    print(f"   Hourly data:   {hourly_count:,} rows")
    print(f"   Compression:   {minute_count/hourly_count:.1f}x reduction")
    
    # Calculate trading days
    minute_days = pd.to_datetime(minute['date']).dt.date.nunique()
    hourly_days = pd.to_datetime(hourly['Date']).dt.date.nunique()
    
    print(f"\n📅 Trading Days:")
    print(f"   1-minute: {minute_days:,} days")
    print(f"   Hourly:   {hourly_days:,} days")
    
    # Calculate average bars per day
    print(f"\n📈 Bars per Day:")
    print(f"   1-minute: {minute_count/minute_days:.1f} bars/day")
    print(f"   Hourly:   {hourly_count/hourly_days:.1f} bars/day")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Convert minute to hourly
    hourly_df = convert_minute_to_hourly(
        input_file="../NIFTY 50_minute.csv",
        output_file="../Nifty_50_hourly.csv",
        verbose=True
    )
    
    # Validate
    print("\n📋 Validation Results:")
    validation = validate_hourly_data(hourly_df)
    for key, value in validation.items():
        print(f"   {key}: {value}")
    
    # Compare
    compare_minute_vs_hourly()
    
    print("\n✅ Ready for BCD analysis!")
    print("   Next: python main.py --data hourly")
