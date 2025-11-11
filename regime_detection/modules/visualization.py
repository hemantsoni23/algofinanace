"""
Visualization Module
Creates plots for changepoint detection validation
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Visualizer:
    """
    Creates visualizations for regime detection and changepoint validation.
    """
    
    def __init__(self, figsize: tuple = (16, 10), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the Visualizer.
        
        Args:
            figsize (tuple): Figure size (width, height)
            style (str): Matplotlib style
        """
        self.figsize = figsize
        
        # Try to set style, fallback to default if not available
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                logger.warning("Could not set plot style, using default")
        
    def plot_changepoints(self,
                         data: pd.DataFrame,
                         signal_column: str,
                         changepoint_dates: pd.DatetimeIndex,
                         price_column: str = 'Close',
                         title: str = "Regime Detection - Changepoint Analysis",
                         save_path: Optional[str] = None,
                         crisis_annotations: bool = True) -> plt.Figure:
        """
        Create a dual-panel plot showing price and volatility with changepoints.
        
        This is the KEY VALIDATION PLOT. If the model is working correctly,
        you should see vertical red lines at major market crises.
        
        Args:
            data (pd.DataFrame): DataFrame with price and signal data
            signal_column (str): Column name for the signal (e.g., volatility)
            changepoint_dates (pd.DatetimeIndex): Dates where changepoints detected
            price_column (str): Column name for price
            title (str): Plot title
            save_path (str): Path to save the figure
            crisis_annotations (bool): Whether to annotate known crises
            
        Returns:
            plt.Figure: The created figure
        """
        logger.info("Creating changepoint visualization")
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        # Find the correct column names (case insensitive)
        price_col = None
        signal_col = None
        
        for col in data.columns:
            if col.lower() == price_column.lower():
                price_col = col
            if col == signal_column:
                signal_col = col
        
        if price_col is None:
            raise ValueError(f"Price column '{price_column}' not found")
        if signal_col is None:
            raise ValueError(f"Signal column '{signal_column}' not found")
        
        # Top panel: Price with changepoints
        ax1 = axes[0]
        ax1.plot(data.index, data[price_col], color='#2E86C1', linewidth=1.5, label='Price')
        ax1.set_ylabel('Nifty 50 Price', fontsize=12, fontweight='bold')
        ax1.set_title('Price Series with Detected Regime Changes', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Add changepoint lines to price plot
        for date in changepoint_dates:
            ax1.axvline(date, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Bottom panel: Volatility signal with changepoints
        ax2 = axes[1]
        signal_data = data[signal_col].dropna()
        ax2.plot(signal_data.index, signal_data, color='#E67E22', linewidth=1.5, label='Rolling Volatility')
        ax2.set_ylabel('Annualized Volatility', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_title('Volatility Signal (Input to Changepoint Detector)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Add changepoint lines to volatility plot
        for date in changepoint_dates:
            ax2.axvline(date, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Annotate known crises for validation
        if crisis_annotations:
            # 2008 Global Financial Crisis (September-October 2008)
            crisis_2008 = pd.Timestamp('2008-10-01')
            if data.index[0] <= crisis_2008 <= data.index[-1]:
                ax1.annotate('2008 GFC', xy=(crisis_2008, data[price_col].loc[crisis_2008:].iloc[0]),
                           xytext=(10, 20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                           fontsize=9, fontweight='bold')
            
            # 2020 COVID-19 Crash (March 2020)
            crisis_2020 = pd.Timestamp('2020-03-01')
            if data.index[0] <= crisis_2020 <= data.index[-1]:
                ax1.annotate('COVID-19 Crash', xy=(crisis_2020, data[price_col].loc[crisis_2020:].iloc[0]),
                           xytext=(10, 20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                           fontsize=9, fontweight='bold')
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved plot to {save_path}")
        
        return fig
    
    def plot_changepoint_summary(self,
                                changepoint_dates: pd.DatetimeIndex,
                                data: pd.DataFrame,
                                price_column: str = 'Close',
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a summary plot showing changepoint timeline and price movements.
        
        Args:
            changepoint_dates (pd.DatetimeIndex): Dates of changepoints
            data (pd.DataFrame): Price data
            price_column (str): Column name for price
            save_path (str): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        logger.info("Creating changepoint summary plot")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Find price column
        price_col = None
        for col in data.columns:
            if col.lower() == price_column.lower():
                price_col = col
                break
        
        if price_col is None:
            raise ValueError(f"Price column '{price_column}' not found")
        
        # Plot price
        ax.plot(data.index, data[price_col], color='#34495E', linewidth=2, label='Nifty 50')
        
        # Highlight changepoint regions
        for i, date in enumerate(changepoint_dates):
            ax.axvline(date, color='red', linestyle='--', linewidth=2, alpha=0.6)
            
            # Add changepoint labels
            price_at_cp = data[price_col].asof(date)
            ax.scatter(date, price_at_cp, color='red', s=100, zorder=5)
            ax.text(date, price_at_cp, f'  CP{i+1}', 
                   fontsize=8, verticalalignment='bottom', fontweight='bold')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.set_title(f'Detected {len(changepoint_dates)} Regime Changes', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved summary plot to {save_path}")
        
        return fig
    
    def plot_penalty_tuning(self, 
                          tuning_results: pd.DataFrame,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the results of penalty parameter tuning.
        
        Args:
            tuning_results (pd.DataFrame): Results from tune_penalty()
            save_path (str): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        logger.info("Creating penalty tuning plot")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter out None values
        valid_results = tuning_results[tuning_results['n_changepoints'].notna()]
        
        # Plot
        ax.plot(valid_results['penalty'], valid_results['n_changepoints'], 
               marker='o', linewidth=2, markersize=8, color='#2E86C1')
        
        # Highlight points in expected range
        in_range = valid_results[valid_results['in_expected_range'] == True]
        if len(in_range) > 0:
            ax.scatter(in_range['penalty'], in_range['n_changepoints'],
                      color='green', s=150, marker='*', zorder=5, 
                      label='In Expected Range')
        
        ax.set_xlabel('Penalty Parameter', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Changepoints', fontsize=12, fontweight='bold')
        ax.set_title('Penalty Parameter Tuning Results', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved tuning plot to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Test visualization
    print("Visualization module loaded successfully")
