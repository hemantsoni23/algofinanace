"""
Live Data Monitoring System
Real-time regime monitoring + periodic batch changepoint detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Tuple, Dict
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveRegimeMonitor:
    """
    Real-time monitoring system that tracks current market state
    and alerts when potential regime changes are detected.
    
    This does NOT replace PELT analysis - it's a fast monitoring layer
    that runs in real-time and triggers full analysis when needed.
    """
    
    def __init__(self, historical_data_path: str, regime_stats_path: str = None):
        """
        Initialize the live monitor.
        
        Args:
            historical_data_path: Path to historical data CSV
            regime_stats_path: Path to regime statistics from PELT analysis
        """
        self.historical_path = historical_data_path
        self.regime_stats_path = regime_stats_path or "outputs/regime_statistics.csv"
        
        # Load historical context
        self.historical_data = self._load_historical_data()
        self.regime_stats = self._load_regime_stats()
        
        # Current state
        self.current_regime = self._get_current_regime()
        self.alert_history = []
        
        logger.info("LiveRegimeMonitor initialized")
        logger.info(f"Current regime: {self.current_regime['regime_id']}")
        logger.info(f"Current regime volatility: {self.current_regime['annualized_volatility_pct']:.2f}%")
    
    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical market data."""
        df = pd.read_csv(self.historical_path, parse_dates=['Date'], index_col='Date')
        df.sort_index(inplace=True)
        return df
    
    def _load_regime_stats(self) -> pd.DataFrame:
        """Load regime statistics from previous PELT analysis."""
        try:
            df = pd.read_csv(self.regime_stats_path)
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['end_date'] = pd.to_datetime(df['end_date'])
            return df
        except FileNotFoundError:
            logger.warning("No regime statistics found. Run main.py first.")
            return None
    
    def _get_current_regime(self) -> Dict:
        """Get statistics of the current regime."""
        if self.regime_stats is None:
            return {'regime_id': 'Unknown', 'annualized_volatility_pct': 15.0}
        
        # Find the regime that includes today
        today = pd.Timestamp.now()
        current = self.regime_stats[
            (self.regime_stats['start_date'] <= today) & 
            (self.regime_stats['end_date'] >= today)
        ]
        
        if len(current) == 0:
            # Use the most recent regime
            current = self.regime_stats.iloc[-1]
            return current.to_dict()
        
        return current.iloc[0].to_dict()
    
    def calculate_live_volatility(self, 
                                   prices: pd.Series = None, 
                                   window: int = 30) -> float:
        """
        Calculate current volatility from recent prices.
        
        Args:
            prices: Price series (if None, uses last window days from historical)
            window: Rolling window size in days
            
        Returns:
            Annualized volatility (as percentage)
        """
        if prices is None:
            prices = self.historical_data['Close'].iloc[-window:]
        
        # Calculate log returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252) * 100
        
        return volatility
    
    def check_volatility_alert(self, current_volatility: float) -> Tuple[str, str]:
        """
        Check if current volatility signals a potential regime change.
        
        Args:
            current_volatility: Current annualized volatility (percentage)
            
        Returns:
            Tuple of (alert_level, message)
            alert_level: "OK", "WARNING", "ALERT", "CRITICAL"
        """
        regime_vol = self.current_regime['annualized_volatility_pct']
        
        # Calculate volatility ratio
        vol_ratio = current_volatility / regime_vol
        
        # Absolute thresholds
        if current_volatility > 40.0:
            return "CRITICAL", (
                f"⚠️ CRITICAL: Extreme volatility {current_volatility:.1f}% "
                f"(regime avg: {regime_vol:.1f}%). Likely crisis/crash scenario!"
            )
        
        if current_volatility > 30.0:
            return "ALERT", (
                f"🔴 ALERT: Very high volatility {current_volatility:.1f}% "
                f"(regime avg: {regime_vol:.1f}%). Strong regime change signal!"
            )
        
        # Relative thresholds (vs current regime)
        if vol_ratio > 1.8:
            return "ALERT", (
                f"🔴 ALERT: Volatility {current_volatility:.1f}% is {vol_ratio:.1f}x "
                f"regime average ({regime_vol:.1f}%). Possible regime shift!"
            )
        
        if vol_ratio > 1.5:
            return "WARNING", (
                f"⚠️ WARNING: Volatility {current_volatility:.1f}% is {vol_ratio:.1f}x "
                f"regime average ({regime_vol:.1f}%). Monitor closely."
            )
        
        if vol_ratio > 1.3:
            return "INFO", (
                f"ℹ️ INFO: Volatility {current_volatility:.1f}% is elevated "
                f"({vol_ratio:.1f}x regime avg). Watch for further increases."
            )
        
        return "OK", (
            f"✅ OK: Volatility {current_volatility:.1f}% is within normal range "
            f"(regime avg: {regime_vol:.1f}%)"
        )
    
    def check_price_movement(self, window: int = 5) -> Tuple[str, str]:
        """
        Check recent price movements for unusual patterns.
        
        Args:
            window: Number of days to check
            
        Returns:
            Tuple of (alert_level, message)
        """
        recent_prices = self.historical_data['Close'].iloc[-window:]
        returns = recent_prices.pct_change().dropna()
        
        # Check for large single-day moves
        max_daily_move = returns.abs().max() * 100
        
        # Check for sustained directional moves
        cumulative_move = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1) * 100
        
        if max_daily_move > 5.0:
            return "ALERT", (
                f"🔴 ALERT: Large single-day move detected ({max_daily_move:.1f}%). "
                f"Possible regime change event!"
            )
        
        if abs(cumulative_move) > 10.0:
            direction = "gain" if cumulative_move > 0 else "drop"
            return "WARNING", (
                f"⚠️ WARNING: Significant {window}-day {direction} of {abs(cumulative_move):.1f}%. "
                f"Possible regime transition."
            )
        
        return "OK", f"✅ OK: Price movements within normal range"
    
    def monitor_current_state(self, new_data: pd.DataFrame = None) -> Dict:
        """
        Main monitoring function - check current market state.
        
        Args:
            new_data: New market data (if None, uses historical data)
            
        Returns:
            Dictionary with monitoring results
        """
        timestamp = pd.Timestamp.now()
        
        # Calculate current metrics
        current_vol = self.calculate_live_volatility()
        
        # Check alerts
        vol_level, vol_message = self.check_volatility_alert(current_vol)
        price_level, price_message = self.check_price_movement()
        
        # Determine overall alert level
        alert_levels = {"OK": 0, "INFO": 1, "WARNING": 2, "ALERT": 3, "CRITICAL": 4}
        overall_level = max(vol_level, price_level, key=lambda x: alert_levels[x])
        
        # Create result
        result = {
            'timestamp': str(timestamp),
            'current_volatility': current_vol,
            'regime_volatility': self.current_regime['annualized_volatility_pct'],
            'volatility_ratio': current_vol / self.current_regime['annualized_volatility_pct'],
            'alert_level': overall_level,
            'volatility_alert': vol_message,
            'price_alert': price_message,
            'recommendation': self._get_recommendation(overall_level)
        }
        
        # Log results
        logger.info("="*80)
        logger.info(f"LIVE MONITORING REPORT - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        logger.info(vol_message)
        logger.info(price_message)
        logger.info(f"\n{result['recommendation']}")
        logger.info("="*80)
        
        # Store alert if significant
        if alert_levels[overall_level] >= 2:  # WARNING or higher
            self.alert_history.append(result)
        
        return result
    
    def _get_recommendation(self, alert_level: str) -> str:
        """Get trading recommendation based on alert level."""
        recommendations = {
            "OK": (
                "📊 RECOMMENDATION: Normal market conditions.\n"
                "   - Continue standard position sizing\n"
                "   - No immediate action required\n"
                "   - Monitor daily for changes"
            ),
            "INFO": (
                "📊 RECOMMENDATION: Slightly elevated volatility.\n"
                "   - Monitor more frequently (hourly/daily)\n"
                "   - Consider tightening stop losses slightly\n"
                "   - Prepare to reduce leverage if escalates"
            ),
            "WARNING": (
                "⚠️ RECOMMENDATION: Significant volatility increase.\n"
                "   - Reduce position sizes by 25-50%\n"
                "   - Tighten stop losses significantly\n"
                "   - Avoid new leveraged positions\n"
                "   - Run expedited PELT analysis on recent data\n"
                "   - Monitor intraday price action"
            ),
            "ALERT": (
                "🔴 RECOMMENDATION: High probability of regime change.\n"
                "   - Reduce positions by 50-75%\n"
                "   - Exit all leveraged positions\n"
                "   - Set tight protective stops on remaining positions\n"
                "   - RUN FULL PELT ANALYSIS IMMEDIATELY\n"
                "   - Prepare for extended high volatility period\n"
                "   - Consider hedging strategies"
            ),
            "CRITICAL": (
                "🚨 RECOMMENDATION: CRISIS-LEVEL VOLATILITY!\n"
                "   - REDUCE TO MINIMAL POSITIONS (10-25% normal size)\n"
                "   - EXIT ALL LEVERAGE IMMEDIATELY\n"
                "   - Implement protective hedges (options, inverse ETFs)\n"
                "   - RUN EMERGENCY PELT ANALYSIS\n"
                "   - Wait for regime stabilization before re-entry\n"
                "   - Expected recovery: 30-90 days based on historical regimes"
            )
        }
        
        return recommendations[alert_level]
    
    def should_run_full_analysis(self) -> bool:
        """
        Determine if full PELT analysis should be triggered.
        
        Returns:
            True if full analysis recommended
        """
        # Check recent alerts
        if len(self.alert_history) == 0:
            return False
        
        recent_alerts = [a for a in self.alert_history 
                        if (pd.Timestamp.now() - pd.Timestamp(a['timestamp'])).days <= 7]
        
        # Trigger if multiple warnings in past week
        if len(recent_alerts) >= 3:
            logger.info("⚠️ Multiple alerts in past 7 days - FULL ANALYSIS RECOMMENDED")
            return True
        
        # Trigger if any critical alert
        critical_alerts = [a for a in recent_alerts if a['alert_level'] == 'CRITICAL']
        if len(critical_alerts) > 0:
            logger.info("🚨 Critical alert detected - FULL ANALYSIS REQUIRED")
            return True
        
        return False
    
    def export_report(self, filepath: str = "outputs/live_monitoring_report.json"):
        """Export monitoring report to JSON."""
        # Convert current_regime to JSON-serializable format
        current_regime_json = {}
        for key, value in self.current_regime.items():
            if isinstance(value, pd.Timestamp):
                current_regime_json[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                current_regime_json[key] = float(value)
            else:
                current_regime_json[key] = value
        
        report = {
            'generated_at': str(pd.Timestamp.now()),
            'current_regime': current_regime_json,
            'recent_alerts': self.alert_history[-10:],  # Last 10 alerts
            'full_analysis_recommended': self.should_run_full_analysis()
        }
        
        Path(filepath).parent.mkdir(exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report exported to {filepath}")


def continuous_monitoring(check_interval_minutes: int = 60):
    """
    Run continuous monitoring loop.
    
    Args:
        check_interval_minutes: How often to check (default: hourly)
    """
    import time
    
    monitor = LiveRegimeMonitor("../Nifty_50.csv")
    
    logger.info(f"Starting continuous monitoring (checking every {check_interval_minutes} minutes)")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            # Monitor current state
            result = monitor.monitor_current_state()
            
            # Check if full analysis needed
            if monitor.should_run_full_analysis():
                logger.info("\n🔔 TRIGGERING FULL PELT ANALYSIS...")
                # Here you would trigger main.py or run analysis
                # For now, just log
                logger.info("   Run: python main.py")
                logger.info("   Or: python train_test_validation.py")
            
            # Export report
            monitor.export_report()
            
            # Wait for next check
            logger.info(f"\nNext check in {check_interval_minutes} minutes...")
            time.sleep(check_interval_minutes * 60)
            
    except KeyboardInterrupt:
        logger.info("\n\nMonitoring stopped by user")


def single_check():
    """
    Perform a single monitoring check (useful for cron jobs).
    """
    monitor = LiveRegimeMonitor("../Nifty_50.csv")
    result = monitor.monitor_current_state()
    
    if monitor.should_run_full_analysis():
        logger.info("\n🔔 ALERT: Full PELT analysis recommended!")
        logger.info("   Run: python main.py")
    
    monitor.export_report()
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        # Run continuous monitoring
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        continuous_monitoring(check_interval_minutes=interval)
    else:
        # Single check (default)
        print("\n" + "="*80)
        print("LIVE REGIME MONITORING - SINGLE CHECK")
        print("="*80)
        single_check()
        print("\n" + "="*80)
        print("For continuous monitoring, run:")
        print("  python live_monitoring.py --continuous 60  # Check every 60 minutes")
        print("="*80)
