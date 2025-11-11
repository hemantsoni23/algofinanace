"""
Data Loader Module
Handles loading and validation of OHLCV time series data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and validates OHLCV time series data for regime detection.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader.
        
        Args:
            file_path (str): Path to the CSV file containing OHLCV data
        """
        self.file_path = Path(file_path)
        self.data = None
        
    def load_data(self, date_column: str = 'Date', parse_dates: bool = True) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.
        
        Args:
            date_column (str): Name of the date column
            parse_dates (bool): Whether to parse dates
            
        Returns:
            pd.DataFrame: Loaded data with datetime index
        """
        logger.info(f"Loading data from {self.file_path}")
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        # Load the CSV file
        self.data = pd.read_csv(self.file_path)
        
        # Detect date column (case insensitive)
        date_col = None
        for col in self.data.columns:
            if col.lower() in ['date', 'datetime', 'timestamp', 'time']:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("No date column found in the data. Expected column names: 'Date', 'DateTime', 'Timestamp'")
        
        # Parse dates and set as index
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data.set_index(date_col, inplace=True)
        self.data.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(self.data)} rows from {self.data.index[0]} to {self.data.index[-1]}")
        
        return self.data
    
    def validate_data(self, required_columns: list = None) -> Tuple[bool, list]:
        """
        Validate the loaded data for completeness and quality.
        
        Args:
            required_columns (list): List of required column names
            
        Returns:
            Tuple[bool, list]: (is_valid, list_of_issues)
        """
        if self.data is None:
            return False, ["Data not loaded. Call load_data() first."]
        
        issues = []
        
        # Default required columns for OHLCV data
        if required_columns is None:
            required_columns = ['Open', 'High', 'Low', 'Close']
        
        # Check for required columns (case insensitive)
        data_columns_lower = [col.lower() for col in self.data.columns]
        for col in required_columns:
            if col.lower() not in data_columns_lower:
                issues.append(f"Missing required column: {col}")
        
        # Check for missing values
        missing_counts = self.data.isnull().sum()
        if missing_counts.any():
            for col, count in missing_counts.items():
                if count > 0:
                    issues.append(f"Column '{col}' has {count} missing values ({count/len(self.data)*100:.2f}%)")
        
        # Check for duplicate dates
        if self.data.index.duplicated().any():
            dup_count = self.data.index.duplicated().sum()
            issues.append(f"Found {dup_count} duplicate dates")
        
        # Check for data anomalies (e.g., High < Low)
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            anomalies = self.data[self.data['High'] < self.data['Low']]
            if len(anomalies) > 0:
                issues.append(f"Found {len(anomalies)} rows where High < Low")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Data validation passed")
        else:
            logger.warning(f"✗ Data validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def clean_data(self, method: str = 'drop') -> pd.DataFrame:
        """
        Clean the data by handling missing values and anomalies.
        
        Args:
            method (str): Method for handling missing values ('drop' or 'ffill')
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info(f"Cleaning data using method: {method}")
        
        original_length = len(self.data)
        
        # Remove duplicate dates
        if self.data.index.duplicated().any():
            self.data = self.data[~self.data.index.duplicated(keep='first')]
            logger.info(f"Removed {original_length - len(self.data)} duplicate dates")
        
        # Handle missing values
        if method == 'drop':
            self.data = self.data.dropna()
            logger.info(f"Dropped rows with missing values. Remaining: {len(self.data)}")
        elif method == 'ffill':
            self.data = self.data.fillna(method='ffill')
            logger.info("Forward-filled missing values")
        
        return self.data
    
    def get_data_info(self) -> dict:
        """
        Get summary information about the loaded data.
        
        Returns:
            dict: Dictionary containing data statistics
        """
        if self.data is None:
            return {"status": "No data loaded"}
        
        info = {
            "total_rows": len(self.data),
            "start_date": str(self.data.index[0].date()),
            "end_date": str(self.data.index[-1].date()),
            "years_of_data": (self.data.index[-1] - self.data.index[0]).days / 365.25,
            "columns": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict()
        }
        
        return info


if __name__ == "__main__":
    # Test the module
    loader = DataLoader("../../Nifty_50.csv")
    data = loader.load_data()
    is_valid, issues = loader.validate_data()
    info = loader.get_data_info()
    
    print("\nData Info:")
    for key, value in info.items():
        print(f"{key}: {value}")
