"""
Enhanced Changepoint Detector - Layer 1 Optimized
Hierarchical detection with confidence scoring for multi-layer regime system
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from typing import Tuple, List, Dict, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class HierarchicalChangepointDetector:
    """
    Enhanced changepoint detector with hierarchical detection and confidence scoring.
    Optimized for Layer 1 in a multi-layer regime detection system.
    """
    
    def __init__(self, 
                 model: str = "l2",
                 min_size: int = 20,
                 jump: int = 5,
                 confidence_threshold: float = 0.6):
        """
        Initialize hierarchical detector.
        
        Args:
            model: Cost function ('l2', 'rbf', 'linear', 'normal', 'ar')
            min_size: Minimum segment size (increased for stability)
            jump: Jump parameter for optimization speed
            confidence_threshold: Minimum confidence for breakpoint acceptance
        """
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.confidence_threshold = confidence_threshold
        self.fitted_algo = None
        self.signal = None
        self.dates = None
        
        logger.info(f"Initialized HierarchicalChangepointDetector with model='{model}', "
                   f"min_size={min_size}, confidence_threshold={confidence_threshold}")
    
    def fit(self, signal: np.ndarray):
        """Fit the changepoint detection model."""
        self.signal = signal
        # Use Binary Segmentation - more robust than PELT for financial data
        self.fitted_algo = rpt.Binseg(model=self.model, min_size=self.min_size, jump=self.jump)
        self.fitted_algo.fit(signal)
        logger.info(f"✓ Model fitted on signal of length {len(signal)} using BinarySegmentation")
        return self
    
    def _calculate_signal_strength(self, signal: np.ndarray, breakpoints: List[int]) -> np.ndarray:
        """
        Calculate signal strength at each breakpoint based on statistical change.
        
        Returns:
            Array of signal strengths (0-1) for each breakpoint
        """
        if len(breakpoints) == 0:
            return np.array([])
        
        strengths = []
        
        for i, bp in enumerate(breakpoints):
            # Get segments before and after breakpoint
            start_idx = 0 if i == 0 else breakpoints[i-1]
            end_idx = len(signal) if bp == breakpoints[-1] else breakpoints[i+1]
            
            # Segments (flatten if 2D)
            before = signal[start_idx:bp].flatten()
            after = signal[bp:end_idx].flatten()
            signal_flat = signal.flatten()
            
            if len(before) < 2 or len(after) < 2:
                strengths.append(0.0)
                continue
            
            # Statistical tests
            # 1. Mean difference (normalized)
            mean_diff = abs(np.mean(after) - np.mean(before)) / (np.std(signal_flat) + 1e-8)
            
            # 2. Variance ratio
            var_ratio = max(np.var(after), np.var(before)) / (min(np.var(after), np.var(before)) + 1e-8)
            var_score = min(var_ratio / 5.0, 1.0)  # Cap at 5x variance change
            
            # 3. T-test p-value (smaller = more significant)
            t_stat, p_value = stats.ttest_ind(before, after, equal_var=False)
            p_score = 1.0 - min(p_value * 10, 1.0)  # Convert to 0-1 score
            
            # 4. Kolmogorov-Smirnov test (distribution change)
            ks_stat, ks_p = stats.ks_2samp(before, after)
            ks_score = min(ks_stat * 2, 1.0)
            
            # Combined strength score
            strength = (mean_diff * 0.4 + var_score * 0.2 + p_score * 0.2 + ks_score * 0.2)
            strength = min(strength, 1.0)
            
            strengths.append(strength)
        
        return np.array(strengths)
    
    def _calculate_persistence(self, signal: np.ndarray, breakpoints: List[int], 
                               window: int = 10) -> np.ndarray:
        """
        Calculate persistence score - how long the regime change lasts.
        """
        if len(breakpoints) == 0:
            return np.array([])
        
        persistence_scores = []
        
        for i, bp in enumerate(breakpoints):
            # Look ahead to see if change persists
            end_idx = min(bp + window, len(signal))
            
            if bp >= len(signal) - 1:
                persistence_scores.append(0.0)
                continue
            
            # Calculate stability after breakpoint
            after_segment = signal[bp:end_idx].flatten()
            
            if len(after_segment) < 2:
                persistence_scores.append(0.0)
                continue
            
            # Lower CV (coefficient of variation) = more persistent
            cv = np.std(after_segment) / (abs(np.mean(after_segment)) + 1e-8)
            persistence = 1.0 / (1.0 + cv)  # Convert to 0-1 score
            
            persistence_scores.append(persistence)
        
        return np.array(persistence_scores)
    
    def _multi_timeframe_validation(self, signal: np.ndarray, breakpoints: List[int],
                                    dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Validate breakpoints across multiple timeframes.
        Higher score if breakpoint visible in both short and long windows.
        """
        if len(breakpoints) == 0 or dates is None:
            return np.ones(len(breakpoints))
        
        validation_scores = []
        
        for bp in breakpoints:
            if bp >= len(signal) - 1 or bp == 0:
                validation_scores.append(0.5)
                continue
            
            # Check if breakpoint is significant in both short and long windows
            short_window = 5
            long_window = 20
            
            # Short-term change
            short_start = max(0, bp - short_window)
            short_end = min(len(signal), bp + short_window)
            short_before = signal[short_start:bp].flatten()
            short_after = signal[bp:short_end].flatten()
            
            # Long-term change
            long_start = max(0, bp - long_window)
            long_end = min(len(signal), bp + long_window)
            long_before = signal[long_start:bp].flatten()
            long_after = signal[bp:long_end].flatten()
            
            # Calculate change in both windows
            short_change = 0.0
            long_change = 0.0
            signal_flat = signal.flatten()
            
            if len(short_before) > 0 and len(short_after) > 0:
                short_change = abs(np.mean(short_after) - np.mean(short_before)) / (np.std(signal_flat) + 1e-8)
            
            if len(long_before) > 0 and len(long_after) > 0:
                long_change = abs(np.mean(long_after) - np.mean(long_before)) / (np.std(signal_flat) + 1e-8)
            
            # Score based on consistency
            if short_change > 0 and long_change > 0:
                # Both show change - good
                consistency = 1.0 - abs(short_change - long_change) / max(short_change, long_change, 1e-8)
                validation_scores.append(consistency)
            else:
                validation_scores.append(0.3)  # Low score if not consistent
        
        return np.array(validation_scores)
    
    def detect_hierarchical(self, n_bkps_major: int = 10, 
                           n_bkps_minor: int = 25,
                           dates: Optional[pd.DatetimeIndex] = None) -> Dict:
        """
        Detect changepoints hierarchically with confidence scores.
        
        Args:
            n_bkps_major: Number of major breakpoints to detect (fewer = higher confidence)
            n_bkps_minor: Number of total breakpoints including minor (more = capture more changes)
            dates: DatetimeIndex for temporal validation
            
        Returns:
            Dictionary with major/minor breakpoints and confidence scores
        """
        if self.fitted_algo is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.dates = dates
        
        logger.info(f"\n{'='*80}")
        logger.info("HIERARCHICAL CHANGEPOINT DETECTION")
        logger.info(f"{'='*80}")
        
        # Detect major breakpoints (fewer = most significant)
        logger.info(f"\n1. Detecting MAJOR breakpoints (n_bkps={n_bkps_major})...")
        major_breaks = self.fitted_algo.predict(n_bkps=n_bkps_major)
        major_breaks = [bp for bp in major_breaks if bp < len(self.signal)]
        logger.info(f"   Found {len(major_breaks)} major breakpoints")
        
        # Detect all breakpoints including minor
        logger.info(f"\n2. Detecting ALL breakpoints (n_bkps={n_bkps_minor})...")
        all_breaks = self.fitted_algo.predict(n_bkps=n_bkps_minor)
        all_breaks = [bp for bp in all_breaks if bp < len(self.signal)]
        # Minor = all breaks not in major list
        minor_breaks = [bp for bp in all_breaks if bp not in major_breaks]
        logger.info(f"   Found {len(minor_breaks)} minor breakpoints (total {len(all_breaks)})")
        
        # Calculate confidence scores for major breakpoints
        logger.info(f"\n3. Calculating confidence scores...")
        
        major_confidences = self._calculate_comprehensive_confidence(
            self.signal, major_breaks, dates, level='major'
        )
        
        minor_confidences = self._calculate_comprehensive_confidence(
            self.signal, minor_breaks, dates, level='minor'
        )
        
        # Filter by confidence threshold
        major_filtered = [(bp, conf) for bp, conf in zip(major_breaks, major_confidences) 
                         if conf >= self.confidence_threshold]
        minor_filtered = [(bp, conf) for bp, conf in zip(minor_breaks, minor_confidences) 
                         if conf >= self.confidence_threshold * 0.7]  # Lower threshold for minor
        
        logger.info(f"\n4. Filtered by confidence threshold ({self.confidence_threshold}):")
        logger.info(f"   Major: {len(major_breaks)} → {len(major_filtered)} (after filtering)")
        logger.info(f"   Minor: {len(minor_breaks)} → {len(minor_filtered)} (after filtering)")
        
        # Log confidence distribution
        if len(major_filtered) > 0:
            major_confs = [c for _, c in major_filtered]
            logger.info(f"\n   Major confidence: min={min(major_confs):.2f}, "
                       f"max={max(major_confs):.2f}, mean={np.mean(major_confs):.2f}")
        
        if len(minor_filtered) > 0:
            minor_confs = [c for _, c in minor_filtered]
            logger.info(f"   Minor confidence: min={min(minor_confs):.2f}, "
                       f"max={max(minor_confs):.2f}, mean={np.mean(minor_confs):.2f}")
        
        return {
            'major_breakpoints': [bp for bp, _ in major_filtered],
            'major_confidence': [conf for _, conf in major_filtered],
            'minor_breakpoints': [bp for bp, _ in minor_filtered],
            'minor_confidence': [conf for _, conf in minor_filtered],
            'all_breakpoints': [bp for bp, _ in major_filtered] + [bp for bp, _ in minor_filtered],
            'all_confidence': [conf for _, conf in major_filtered] + [conf for _, conf in minor_filtered]
        }
    
    def _calculate_comprehensive_confidence(self, signal: np.ndarray, 
                                           breakpoints: List[int],
                                           dates: Optional[pd.DatetimeIndex],
                                           level: str = 'major') -> np.ndarray:
        """
        Calculate comprehensive confidence score combining multiple factors.
        """
        if len(breakpoints) == 0:
            return np.array([])
        
        # 1. Signal strength (how big is the change?)
        strength = self._calculate_signal_strength(signal, breakpoints)
        
        # 2. Persistence (does the change last?)
        persistence = self._calculate_persistence(signal, breakpoints)
        
        # 3. Multi-timeframe validation
        validation = self._multi_timeframe_validation(signal, breakpoints, dates)
        
        # 4. Isolation score (not too close to other breakpoints)
        isolation = self._calculate_isolation_score(breakpoints, min_distance=self.min_size)
        
        # Weighted combination
        if level == 'major':
            # Major breakpoints: emphasize strength and persistence
            confidence = (strength * 0.4 + persistence * 0.3 + 
                         validation * 0.2 + isolation * 0.1)
        else:
            # Minor breakpoints: more balanced
            confidence = (strength * 0.3 + persistence * 0.3 + 
                         validation * 0.25 + isolation * 0.15)
        
        return confidence
    
    def _calculate_isolation_score(self, breakpoints: List[int], 
                                   min_distance: int) -> np.ndarray:
        """
        Calculate isolation score - breakpoints too close together are less reliable.
        """
        if len(breakpoints) <= 1:
            return np.ones(len(breakpoints))
        
        scores = []
        
        for i, bp in enumerate(breakpoints):
            # Distance to previous breakpoint
            if i == 0:
                dist_prev = bp
            else:
                dist_prev = bp - breakpoints[i-1]
            
            # Distance to next breakpoint
            if i == len(breakpoints) - 1:
                dist_next = len(self.signal) - bp
            else:
                dist_next = breakpoints[i+1] - bp
            
            # Minimum distance
            min_dist = min(dist_prev, dist_next)
            
            # Score: 1.0 if well-separated, lower if too close
            isolation = min(min_dist / (min_distance * 2), 1.0)
            scores.append(isolation)
        
        return np.array(scores)
    
    def map_breakpoints_to_dates(self, breakpoints: List[int], 
                                 dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Map breakpoint indices to dates."""
        return [dates[bp] for bp in breakpoints if bp < len(dates)]
    
    def export_for_downstream(self, result: Dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Export breakpoints in format suitable for downstream layers (SKF, HMM, etc).
        
        Returns:
            DataFrame with columns: date, level, confidence, signal_strength
        """
        records = []
        
        # Major breakpoints
        for bp, conf in zip(result['major_breakpoints'], result['major_confidence']):
            if bp < len(dates):
                records.append({
                    'date': dates[bp],
                    'breakpoint_index': bp,
                    'level': 'major',
                    'confidence': conf,
                    'for_downstream': True  # Flag for use in other layers
                })
        
        # Minor breakpoints
        for bp, conf in zip(result['minor_breakpoints'], result['minor_confidence']):
            if bp < len(dates):
                records.append({
                    'date': dates[bp],
                    'breakpoint_index': bp,
                    'level': 'minor',
                    'confidence': conf,
                    'for_downstream': True
                })
        
        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
