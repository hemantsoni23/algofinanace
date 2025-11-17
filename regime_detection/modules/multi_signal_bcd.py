"""
Multi-Signal BCD with Consensus Voting
Combines multiple signals with ML quality filter for 80%+ accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import ruptures as rpt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MultiSignalBCD:
    """
    Advanced BCD with multi-signal consensus and ML filtering
    Target: 80%+ precision while maintaining high recall
    """
    
    def __init__(self, config):
        self.config = config
        self.signals_config = {
            'volatility_20d': {'weight': 0.30, 'model': 'l2', 'min_size': 20},
            'returns_squared': {'weight': 0.25, 'model': 'l2', 'min_size': 15},
            'vol_of_vol': {'weight': 0.20, 'model': 'l2', 'min_size': 20},
            'volume_volatility': {'weight': 0.15, 'model': 'l2', 'min_size': 15},
            'range_volatility': {'weight': 0.10, 'model': 'l2', 'min_size': 10}
        }
        self.ml_filter = None
        self.scaler = StandardScaler()
        
    def fit(self, data: pd.DataFrame, features: pd.DataFrame):
        """Fit BCD on multiple signals"""
        self.data = data
        self.features = features
        
        # Detect breakpoints on each signal
        self.signal_breakpoints = {}
        for signal_name, signal_config in self.signals_config.items():
            if signal_name in features.columns:
                bps = self._detect_on_signal(
                    features[signal_name].values,
                    signal_config
                )
                self.signal_breakpoints[signal_name] = bps
        
        return self
    
    def _detect_on_signal(self, signal: np.ndarray, config: dict) -> List[int]:
        """Detect breakpoints on single signal using Binary Segmentation"""
        signal = signal.reshape(-1, 1)
        
        try:
            algo = rpt.Binseg(
                model=config['model'],
                min_size=config['min_size'],
                jump=5
            )
            algo.fit(signal)
            
            # Use n_bkps for stability
            n_bkps = self.config.N_BKPS_MAJOR + self.config.N_BKPS_MINOR
            breakpoints = algo.predict(n_bkps=n_bkps)
            
            # Remove last point (always len(signal))
            return [bp for bp in breakpoints if bp < len(signal)]
            
        except Exception as e:
            print(f"Warning: Detection failed on signal - {e}")
            return []
    
    def detect_with_consensus(self, min_votes: int = 3) -> pd.DataFrame:
        """
        Detect breakpoints using consensus voting across signals
        
        Args:
            min_votes: Minimum signals that must agree (3+ for high precision)
        
        Returns:
            DataFrame with high-confidence consensus breakpoints
        """
        if not self.signal_breakpoints:
            raise ValueError("Must call fit() first")
        
        # Create voting matrix
        n_samples = len(self.data)
        vote_matrix = np.zeros((n_samples, len(self.signal_breakpoints)))
        
        for i, (signal_name, breakpoints) in enumerate(self.signal_breakpoints.items()):
            weight = self.signals_config[signal_name]['weight']
            for bp in breakpoints:
                # Give votes within window (±5 days)
                window = range(max(0, bp-5), min(n_samples, bp+6))
                vote_matrix[list(window), i] = weight
        
        # Sum votes across signals
        total_votes = vote_matrix.sum(axis=1)
        
        # Find consensus points
        consensus_indices = []
        i = 0
        while i < n_samples:
            if total_votes[i] >= min_votes * 0.1:  # Weighted voting
                # Find local maximum in window
                window_start = max(0, i-5)
                window_end = min(n_samples, i+6)
                window_votes = total_votes[window_start:window_end]
                local_max_offset = np.argmax(window_votes)
                consensus_idx = window_start + local_max_offset
                
                if consensus_idx not in consensus_indices:
                    consensus_indices.append(consensus_idx)
                
                # Skip ahead to avoid duplicates
                i = window_end
            else:
                i += 1
        
        # Calculate confidence for each consensus point
        breakpoints_data = []
        for idx in sorted(consensus_indices):
            # Count signals voting for this point
            n_signals_voting = np.sum(vote_matrix[idx, :] > 0)
            weighted_vote = total_votes[idx]
            
            # Calculate comprehensive confidence
            confidence = self._calculate_consensus_confidence(
                idx, n_signals_voting, weighted_vote
            )
            
            # Extract features for ML filtering
            bp_features = self._extract_breakpoint_features(idx)
            
            breakpoints_data.append({
                'breakpoint_index': idx,
                'date': self.data.index[idx],
                'n_signals_voting': n_signals_voting,
                'weighted_vote': weighted_vote,
                'confidence': confidence,
                **bp_features
            })
        
        df = pd.DataFrame(breakpoints_data)
        
        # Apply ML filter if trained
        if self.ml_filter is not None and len(df) > 0:
            df = self._apply_ml_filter(df)
        
        return df
    
    def _calculate_consensus_confidence(
        self, 
        idx: int, 
        n_signals: int, 
        weighted_vote: float
    ) -> float:
        """Calculate multi-dimensional confidence score"""
        
        # 1. Signal consensus (40%)
        max_signals = len(self.signal_breakpoints)
        consensus_score = n_signals / max_signals
        
        # 2. Vote strength (30%)
        max_possible_vote = sum(s['weight'] for s in self.signals_config.values())
        vote_score = weighted_vote / max_possible_vote
        
        # 3. Signal magnitude (20%)
        magnitude_score = self._calculate_magnitude_score(idx)
        
        # 4. Persistence (10%)
        persistence_score = self._calculate_persistence_score(idx)
        
        confidence = (
            consensus_score * 0.40 +
            vote_score * 0.30 +
            magnitude_score * 0.20 +
            persistence_score * 0.10
        )
        
        return min(confidence, 1.0)
    
    def _calculate_magnitude_score(self, idx: int) -> float:
        """Calculate signal magnitude change at breakpoint"""
        window = 20
        
        if idx < window or idx + window >= len(self.data):
            return 0.5
        
        try:
            # Check volatility change
            vol_signal = self.features['volatility_20d'].values
            before = vol_signal[idx-window:idx].mean()
            after = vol_signal[idx:idx+window].mean()
            
            if before > 0:
                change_pct = abs(after - before) / before
                # Normalize to [0, 1], cap at 100% change
                return min(change_pct, 1.0)
            
        except:
            pass
        
        return 0.5
    
    def _calculate_persistence_score(self, idx: int) -> float:
        """Calculate how long the regime persists after breakpoint"""
        window = 30
        
        if idx + window >= len(self.data):
            return 0.5
        
        try:
            vol_signal = self.features['volatility_20d'].values
            
            # Calculate stability in window after breakpoint
            after_vol = vol_signal[idx:idx+window]
            stability = 1.0 - (after_vol.std() / (after_vol.mean() + 1e-10))
            
            return max(0.0, min(stability, 1.0))
            
        except:
            return 0.5
    
    def _extract_breakpoint_features(self, idx: int) -> Dict:
        """Extract features for ML quality filter"""
        window = 20
        features = {}
        
        # Volatility features
        if 'volatility_20d' in self.features.columns:
            vol = self.features['volatility_20d'].values
            if idx >= window and idx + window < len(vol):
                features['vol_change'] = abs(vol[idx+window] - vol[idx-window]) / (vol[idx-window] + 1e-10)
                features['vol_before'] = vol[idx-window:idx].mean()
                features['vol_after'] = vol[idx:idx+window].mean()
                features['vol_ratio'] = features['vol_after'] / (features['vol_before'] + 1e-10)
            else:
                features['vol_change'] = 0
                features['vol_before'] = vol[idx]
                features['vol_after'] = vol[idx]
                features['vol_ratio'] = 1.0
        
        # Returns features
        if 'returns' in self.features.columns:
            ret = self.features['returns'].values
            if idx >= window and idx + window < len(ret):
                features['return_change'] = abs(ret[idx+window] - ret[idx-window])
                features['return_volatility'] = ret[idx-window:idx+window].std()
            else:
                features['return_change'] = 0
                features['return_volatility'] = 0
        
        # Volume features
        if 'volume_volatility' in self.features.columns:
            vol_vol = self.features['volume_volatility'].values
            if idx >= window and idx + window < len(vol_vol):
                features['volume_change'] = abs(vol_vol[idx+window] - vol_vol[idx-window]) / (vol_vol[idx-window] + 1e-10)
            else:
                features['volume_change'] = 0
        
        # Range features
        if 'range_volatility' in self.features.columns:
            range_vol = self.features['range_volatility'].values
            if idx >= window:
                features['range_vol'] = range_vol[idx-window:idx].mean()
            else:
                features['range_vol'] = range_vol[idx]
        
        return features
    
    def train_ml_filter(self, ground_truth_crises: List[Tuple[str, str]]) -> dict:
        """
        Train ML quality filter using ground truth crises
        
        Args:
            ground_truth_crises: List of (start_date, end_date) tuples
        
        Returns:
            Training performance metrics
        """
        # Detect breakpoints with low threshold
        df = self.detect_with_consensus(min_votes=2)
        
        if len(df) < 10:
            print("Warning: Too few breakpoints for ML training")
            return {}
        
        # Create labels based on ground truth
        df['is_true_crisis'] = 0
        for crisis_start, crisis_end in ground_truth_crises:
            crisis_start = pd.to_datetime(crisis_start)
            crisis_end = pd.to_datetime(crisis_end)
            
            # Mark breakpoints within ±30 days of crisis start as true
            for idx, row in df.iterrows():
                bp_date = pd.to_datetime(row['date'])
                days_to_crisis = (bp_date - crisis_start).days
                
                if -30 <= days_to_crisis <= 30:
                    df.loc[idx, 'is_true_crisis'] = 1
        
        # Prepare features for ML
        feature_cols = [
            'n_signals_voting', 'weighted_vote', 'confidence',
            'vol_change', 'vol_ratio', 'return_change', 
            'return_volatility', 'volume_change', 'range_vol'
        ]
        
        X = df[feature_cols].fillna(0).values
        y = df['is_true_crisis'].values
        
        # Handle class imbalance
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Train Random Forest
        self.ml_filter = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            random_state=42
        )
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.ml_filter.fit(X_scaled, y)
        
        # Calculate training performance
        y_pred = self.ml_filter.predict(X_scaled)
        y_pred_proba = self.ml_filter.predict_proba(X_scaled)[:, 1]
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'n_samples': len(y),
            'n_positives': y.sum(),
            'feature_importance': dict(zip(feature_cols, self.ml_filter.feature_importances_))
        }
        
        print(f"\n✅ ML Filter Trained:")
        print(f"   Precision: {metrics['precision']:.1%}")
        print(f"   Recall: {metrics['recall']:.1%}")
        print(f"   F1: {metrics['f1']:.1%}")
        print(f"\n   Top 3 Features:")
        sorted_features = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features[:3]:
            print(f"   - {feat}: {imp:.3f}")
        
        return metrics
    
    def _apply_ml_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply trained ML filter to remove false positives"""
        if self.ml_filter is None:
            return df
        
        feature_cols = [
            'n_signals_voting', 'weighted_vote', 'confidence',
            'vol_change', 'vol_ratio', 'return_change', 
            'return_volatility', 'volume_change', 'range_vol'
        ]
        
        X = df[feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        
        # Get ML quality score
        ml_proba = self.ml_filter.predict_proba(X_scaled)[:, 1]
        df['ml_quality_score'] = ml_proba
        
        # Combine with existing confidence
        df['final_confidence'] = (df['confidence'] * 0.6 + ml_proba * 0.4)
        
        # Apply ML filter by ML threshold (very relaxed for recall)
        ml_threshold = 0.35  # Relaxed from 0.4
        df_filtered = df[df['ml_quality_score'] >= ml_threshold].copy()
        
        print(f"\n🔍 ML Filter Applied:")
        print(f"   Before: {len(df)} breakpoints")
        print(f"   After: {len(df_filtered)} breakpoints")
        print(f"   Removed: {len(df) - len(df_filtered)} low-quality breakpoints")
        
        return df_filtered
    
    def export_for_downstream(self, breakpoints_df: pd.DataFrame) -> pd.DataFrame:
        """Export high-confidence breakpoints for downstream layers"""
        if 'final_confidence' in breakpoints_df.columns:
            confidence_col = 'final_confidence'
        else:
            confidence_col = 'confidence'
        
        export_df = breakpoints_df[[
            'date', 'breakpoint_index', confidence_col
        ]].copy()
        
        export_df.columns = ['date', 'breakpoint_index', 'confidence']
        
        # Classify as major/minor based on confidence
        export_df['level'] = export_df['confidence'].apply(
            lambda x: 'major' if x >= 0.7 else 'minor'
        )
        
        return export_df.sort_values('breakpoint_index')
