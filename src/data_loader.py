# src/data_loader.py - Advanced Data Loader for Yahoo Finance
"""
Advanced Data Loader สำหรับดาวน์โหลดข้อมูล cryptocurrency จาก Yahoo Finance

คุณสมบัติหลัก:
1. ดาวน์โหลดข้อมูลจาก Yahoo Finance API
2. รองรับ multiple timeframes (1d, 1h, 5m, etc.)
3. บันทึกไฟล์ในรูปแบบ {symbol}-{timeframe}-{start}-{end}.csv
4. จัดการ data validation และ error handling
5. รองรับ batch downloads และ parallel processing
6. Cache management และ data integrity checks

Usage:
    loader = YahooDataLoader()
    
    # Download single symbol
    loader.download_symbol('BTC-USD', '2023-01-01', '2024-01-01', '1d')
    
    # Download multiple symbols
    symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD']
    loader.download_batch(symbols, '2023-01-01', '2024-01-01', '1d')
    
    # Load existing data
    data = loader.load_symbol('BTC-USD', '2023-01-01', '2024-01-01', '1d')
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import time
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

class YahooDataLoader:
    """
    Advanced Yahoo Finance Data Loader
    
    รองรับการดาวน์โหลดข้อมูล cryptocurrency และ stock data
    พร้อมระบบ cache, validation, และ parallel processing
    """
    
    def __init__(self, base_dir: str = None, max_workers: int = 4):
        """
        Initialize Data Loader
        
        Args:
            base_dir: Base directory สำหรับเก็บข้อมูล (default: DATA_DIR from config)
            max_workers: จำนวน threads สำหรับ parallel downloads
        """
        self.base_dir = base_dir or DATA_DIR
        self.raw_dir = os.path.join(self.base_dir, 'raw')
        self.max_workers = max_workers
        
        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Supported timeframes
        self.valid_intervals = [
            '1m', '2m', '5m', '15m', '30m', '60m', '90m',
            '1h', '1d', '5d', '1wk', '1mo', '3mo'
        ]
        
        # Rate limiting
        self.request_delay = 0.1  # seconds between requests
        self.last_request_time = 0
        
        self.logger.info(f"YahooDataLoader initialized - Raw data directory: {self.raw_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('YahooDataLoader')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _validate_inputs(self, symbol: str, start_date: str, end_date: str, interval: str) -> Tuple[str, str, str, str]:
        """
        Validate และ normalize input parameters
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            interval: Time interval
            
        Returns:
            Tuple of validated (symbol, start_date, end_date, interval)
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.upper().strip()
        
        # Validate interval
        if interval not in self.valid_intervals:
            raise ValueError(f"Invalid interval '{interval}'. Valid intervals: {self.valid_intervals}")
        
        # Validate and parse dates
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD format. Error: {e}")
        
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")
        
        if end_dt > pd.Timestamp.now():
            self.logger.warning(f"End date {end_date} is in the future. Using current date.")
            end_dt = pd.Timestamp.now()
        
        # Convert back to string format
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
        
        return symbol, start_date, end_date, interval
    
    def _generate_filename(self, symbol: str, start_date: str, end_date: str, interval: str) -> str:
        """
        Generate filename ในรูปแบบ {symbol}-{timeframe}-{start}-{end}.csv
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval
            
        Returns:
            Filename string
        """
        # Clean symbol (remove special characters)
        clean_symbol = symbol.replace('/', '_').replace('-', '_')
        
        # Format dates (remove hyphens)
        start_clean = start_date.replace('-', '')
        end_clean = end_date.replace('-', '')
        
        filename = f"{clean_symbol}-{interval}-{start_clean}-{end_clean}.csv"
        return filename
    
    def _get_file_path(self, symbol: str, start_date: str, end_date: str, interval: str) -> str:
        """Get full file path for data file"""
        filename = self._generate_filename(symbol, start_date, end_date, interval)
        return os.path.join(self.raw_dir, filename)
    
    def _download_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Download data จาก Yahoo Finance
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Time interval
            
        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit()
        
        try:
            self.logger.info(f"Downloading {symbol} data from {start_date} to {end_date} ({interval})")
            
            # Create yfinance ticker
            ticker = yf.Ticker(symbol)
            
            # Download data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                raise ValueError(f"No data found for {symbol} in the specified date range")
            
            # Reset index to make datetime a column
            data.reset_index(inplace=True)
            
            # Rename columns to standard format
            column_mapping = {
                'Date': 'timestamp',
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            data.rename(columns=column_mapping, inplace=True)
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Ensure we have required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Sort by timestamp
            data.sort_values('timestamp', inplace=True)
            data.reset_index(drop=True, inplace=True)
            
            # Add metadata columns
            data['interval'] = interval
            data['start_date'] = start_date
            data['end_date'] = end_date
            data['download_time'] = pd.Timestamp.now()
            
            self.logger.info(f"Successfully downloaded {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to download {symbol}: {str(e)}")
            raise
    
    def _save_data(self, data: pd.DataFrame, file_path: str, symbol: str) -> bool:
        """
        Save data to CSV file
        
        Args:
            data: DataFrame to save
            file_path: File path to save to
            symbol: Symbol name for logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to CSV
            data.to_csv(file_path, index=False)
            
            # Verify file was created and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                file_size = os.path.getsize(file_path) / 1024  # KB
                self.logger.info(f"Saved {symbol} data to {file_path} ({file_size:.1f} KB)")
                return True
            else:
                self.logger.error(f"Failed to save {symbol} data - file not created or empty")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving {symbol} data to {file_path}: {str(e)}")
            return False
    
    def _load_existing_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load existing data from file
        
        Args:
            file_path: Path to data file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            data = pd.read_csv(file_path)
            
            # Validate data
            if data.empty:
                self.logger.warning(f"Existing file {file_path} is empty")
                return None
            
            # Convert timestamp column
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Loaded existing data from {file_path} ({len(data)} rows)")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading existing data from {file_path}: {str(e)}")
            return None
    
    def _check_data_integrity(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Check data integrity และ quality
        
        Args:
            data: DataFrame to check
            symbol: Symbol name for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check if data is empty
            if data.empty:
                self.logger.error(f"Data for {symbol} is empty")
                return False
            
            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing columns for {symbol}: {missing_columns}")
                return False
            
            # Check for null values in critical columns
            critical_columns = ['open', 'high', 'low', 'close']
            null_counts = data[critical_columns].isnull().sum()
            
            if null_counts.any():
                self.logger.warning(f"Null values found in {symbol} data: {null_counts.to_dict()}")
            
            # Check price data validity
            price_issues = (
                (data['high'] < data['low']) |
                (data['open'] < 0) |
                (data['close'] < 0) |
                (data['high'] < 0) |
                (data['low'] < 0)
            ).sum()
            
            if price_issues > 0:
                self.logger.warning(f"Found {price_issues} price data issues for {symbol}")
            
            # Check timestamp ordering
            if not data['timestamp'].is_monotonic_increasing:
                self.logger.warning(f"Timestamps for {symbol} are not in order")
            
            self.logger.info(f"Data integrity check passed for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking data integrity for {symbol}: {str(e)}")
            return False
    
    def download_symbol(self, symbol: str, start_date: str, end_date: str, 
                       interval: str = '1d', force_download: bool = False) -> pd.DataFrame:
        """
        Download data สำหรับ symbol เดียว
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            interval: Time interval (default: '1d')
            force_download: Force download even if file exists
            
        Returns:
            DataFrame with downloaded data
        """
        # Validate inputs
        symbol, start_date, end_date, interval = self._validate_inputs(symbol, start_date, end_date, interval)
        
        # Get file path
        file_path = self._get_file_path(symbol, start_date, end_date, interval)
        
        # Check if file exists and force_download is False
        if not force_download and os.path.exists(file_path):
            self.logger.info(f"File already exists for {symbol}: {file_path}")
            existing_data = self._load_existing_data(file_path)
            
            if existing_data is not None and self._check_data_integrity(existing_data, symbol):
                return existing_data
            else:
                self.logger.warning(f"Existing file corrupted, re-downloading {symbol}")
        
        # Download data
        data = self._download_data(symbol, start_date, end_date, interval)
        
        # Check data integrity
        if not self._check_data_integrity(data, symbol):
            raise ValueError(f"Downloaded data for {symbol} failed integrity check")
        
        # Save data
        if self._save_data(data, file_path, symbol):
            return data
        else:
            raise RuntimeError(f"Failed to save data for {symbol}")
    
    def download_batch(self, symbols: List[str], start_date: str, end_date: str, 
                      interval: str = '1d', force_download: bool = False,
                      parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Download data สำหรับหลาย symbols
        
        Args:
            symbols: List of trading symbols
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            interval: Time interval (default: '1d')
            force_download: Force download even if files exist
            parallel: Use parallel processing
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        self.logger.info(f"Starting batch download for {len(symbols)} symbols")
        
        results = {}
        failed_symbols = []
        
        if parallel and len(symbols) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all download tasks
                future_to_symbol = {
                    executor.submit(self.download_symbol, symbol, start_date, end_date, interval, force_download): symbol
                    for symbol in symbols
                }
                
                # Collect results
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        results[symbol] = data
                        self.logger.info(f"✅ Successfully downloaded {symbol}")
                    except Exception as e:
                        failed_symbols.append(symbol)
                        self.logger.error(f"❌ Failed to download {symbol}: {str(e)}")
        else:
            # Sequential processing
            for symbol in symbols:
                try:
                    data = self.download_symbol(symbol, start_date, end_date, interval, force_download)
                    results[symbol] = data
                    self.logger.info(f"✅ Successfully downloaded {symbol}")
                except Exception as e:
                    failed_symbols.append(symbol)
                    self.logger.error(f"❌ Failed to download {symbol}: {str(e)}")
        
        # Summary
        self.logger.info(f"Batch download completed: {len(results)} successful, {len(failed_symbols)} failed")
        if failed_symbols:
            self.logger.warning(f"Failed symbols: {failed_symbols}")
        
        return results
    
    def load_symbol(self, symbol: str, start_date: str, end_date: str, 
                   interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Load existing data สำหรับ symbol
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Time interval
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        # Validate inputs
        symbol, start_date, end_date, interval = self._validate_inputs(symbol, start_date, end_date, interval)
        
        # Get file path
        file_path = self._get_file_path(symbol, start_date, end_date, interval)
        
        return self._load_existing_data(file_path)
    
    def list_available_data(self) -> List[Dict[str, str]]:
        """
        List all available data files
        
        Returns:
            List of dictionaries with file information
        """
        available_files = []
        
        if not os.path.exists(self.raw_dir):
            return available_files
        
        for filename in os.listdir(self.raw_dir):
            if filename.endswith('.csv'):
                try:
                    # Parse filename: {symbol}-{interval}-{start}-{end}.csv
                    name_parts = filename[:-4].split('-')  # Remove .csv extension
                    
                    if len(name_parts) >= 4:
                        symbol = name_parts[0]
                        interval = name_parts[1]
                        start_date = name_parts[2]
                        end_date = name_parts[3]
                        
                        # Format dates
                        start_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
                        end_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
                        
                        file_path = os.path.join(self.raw_dir, filename)
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        available_files.append({
                            'symbol': symbol,
                            'interval': interval,
                            'start_date': start_formatted,
                            'end_date': end_formatted,
                            'filename': filename,
                            'file_path': file_path,
                            'file_size_kb': round(file_size, 1),
                            'modified_time': mod_time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Could not parse filename {filename}: {e}")
        
        # Sort by symbol and modification time
        available_files.sort(key=lambda x: (x['symbol'], x['modified_time']))
        
        return available_files
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Get summary of available data
        
        Returns:
            Dictionary with data summary
        """
        files = self.list_available_data()
        
        if not files:
            return {'total_files': 0, 'symbols': [], 'intervals': [], 'total_size_mb': 0}
        
        symbols = list(set(f['symbol'] for f in files))
        intervals = list(set(f['interval'] for f in files))
        total_size_mb = sum(f['file_size_kb'] for f in files) / 1024
        
        return {
            'total_files': len(files),
            'symbols': sorted(symbols),
            'intervals': sorted(intervals),
            'total_size_mb': round(total_size_mb, 2),
            'raw_directory': self.raw_dir
        }
    
    def clean_cache(self, older_than_days: int = 30) -> int:
        """
        Clean old cache files
        
        Args:
            older_than_days: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        if not os.path.exists(self.raw_dir):
            return 0
        
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        removed_count = 0
        
        for filename in os.listdir(self.raw_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.raw_dir, filename)
                
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        self.logger.info(f"Removed old file: {filename}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove {filename}: {e}")
        
        self.logger.info(f"Cache cleanup completed: {removed_count} files removed")
        return removed_count

# Convenience functions
def download_crypto_data(symbols: Union[str, List[str]], start_date: str, end_date: str,
                        interval: str = '1d', force_download: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Convenience function สำหรับดาวน์โหลดข้อมูล crypto
    
    Args:
        symbols: Symbol หรือ list of symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Time interval
        force_download: Force download
        
    Returns:
        DataFrame (single symbol) หรือ Dict[str, DataFrame] (multiple symbols)
    """
    loader = YahooDataLoader()
    
    if isinstance(symbols, str):
        return loader.download_symbol(symbols, start_date, end_date, interval, force_download)
    else:
        return loader.download_batch(symbols, start_date, end_date, interval, force_download)

def load_crypto_data(symbol: str, start_date: str, end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    Convenience function สำหรับโหลดข้อมูล crypto ที่มีอยู่
    
    Args:
        symbol: Trading symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Time interval
        
    Returns:
        DataFrame if available, None otherwise
    """
    loader = YahooDataLoader()
    return loader.load_symbol(symbol, start_date, end_date, interval)

def list_available_crypto_data() -> List[Dict[str, str]]:
    """
    Convenience function สำหรับแสดงรายการข้อมูลที่มีอยู่
    
    Returns:
        List of available data files
    """
    loader = YahooDataLoader()
    return loader.list_available_data()

def get_crypto_data_summary() -> Dict[str, any]:
    """
    Convenience function สำหรับแสดงสรุปข้อมูลที่มีอยู่
    
    Returns:
        Data summary dictionary
    """
    loader = YahooDataLoader()
    return loader.get_data_summary()

# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = YahooDataLoader()
    
    print("🚀 Yahoo Finance Data Loader")
    print("=" * 50)
    
    # Example 1: Download single symbol
    print("\n📊 Example 1: Download BTC-USD data")
    try:
        btc_data = loader.download_symbol(
            symbol='BTC-USD',
            start_date='2024-01-01',
            end_date='2024-02-01',
            interval='1d'
        )
        print(f"✅ Downloaded BTC data: {len(btc_data)} rows")
        print(btc_data.head())
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Download multiple symbols
    print("\n📊 Example 2: Download multiple crypto symbols")
    try:
        crypto_symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD']
        batch_data = loader.download_batch(
            symbols=crypto_symbols,
            start_date='2024-01-01',
            end_date='2024-01-15',
            interval='1d'
        )
        print(f"✅ Downloaded {len(batch_data)} symbols")
        for symbol, data in batch_data.items():
            print(f"   {symbol}: {len(data)} rows")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: List available data
    print("\n📋 Example 3: Available data files")
    try:
        available = loader.list_available_data()
        print(f"📁 Found {len(available)} data files:")
        for file_info in available[:5]:  # Show first 5
            print(f"   {file_info['symbol']} ({file_info['interval']}) - {file_info['file_size_kb']} KB")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Data summary
    print("\n📈 Example 4: Data summary")
    try:
        summary = loader.get_data_summary()
        print(f"📊 Summary:")
        print(f"   Total files: {summary['total_files']}")
        print(f"   Symbols: {summary['symbols']}")
        print(f"   Intervals: {summary['intervals']}")
        print(f"   Total size: {summary['total_size_mb']} MB")
    except Exception as e:
        print(f"❌ Error: {e}")