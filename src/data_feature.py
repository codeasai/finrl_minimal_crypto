# src/data_feature.py - Advanced Technical Indicators Feature Engineering
"""
Advanced Feature Engineering สำหรับ cryptocurrency trading data

คุณสมบัติหลัก:
1. เพิ่ม technical indicators ครบถ้วน (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
2. Volume indicators และ price action features
3. Market sentiment indicators
4. Risk management features
5. Multi-timeframe analysis
6. Feature scaling และ normalization
7. Data validation และ quality checks
8. บันทึกไฟล์ในรูปแบบ {symbol}-{timeframe}-{start}-{end}-features.csv

Usage:
    processor = CryptoFeatureProcessor()
    
    # Process single file
    features_data = processor.process_raw_data('data/raw/BTC_USD-1d-20240101-20240201.csv')
    
    # Process all raw data files
    processor.process_all_raw_data()
    
    # Load feature data
    data = processor.load_feature_data('BTC-USD', '2024-01-01', '2024-02-01', '1d')
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import logging
import glob
from pathlib import Path

# Technical Analysis Libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️  TA-Lib not available. Using pandas-based calculations.")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

class CryptoFeatureProcessor:
    """
    Advanced Feature Engineering สำหรับ cryptocurrency trading data
    
    รองรับการเพิ่ม technical indicators, volume analysis, 
    market sentiment features, และ risk management indicators
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize Feature Processor
        
        Args:
            base_dir: Base directory สำหรับเก็บข้อมูล (default: DATA_DIR from config)
        """
        self.base_dir = base_dir or DATA_DIR
        self.raw_dir = os.path.join(self.base_dir, 'raw')
        self.feature_dir = os.path.join(self.base_dir, 'feature')
        
        # Create directories
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Feature configuration
        self.feature_config = {
            # Moving Averages
            'sma_periods': [5, 10, 20, 50, 100, 200],
            'ema_periods': [5, 10, 20, 50, 100, 200],
            
            # Momentum Indicators
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # MACD
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2,
            
            # Stochastic
            'stoch_k': 14,
            'stoch_d': 3,
            
            # Volume
            'volume_sma_periods': [10, 20, 50],
            
            # ATR (Average True Range)
            'atr_period': 14,
            
            # Williams %R
            'williams_period': 14,
            
            # CCI (Commodity Channel Index)
            'cci_period': 20,
            
            # ROC (Rate of Change)
            'roc_period': 10,
            
            # Price Action
            'price_change_periods': [1, 3, 5, 10, 20],
            
            # Volatility
            'volatility_periods': [10, 20, 30]
        }
        
        self.logger.info(f"CryptoFeatureProcessor initialized - Feature directory: {self.feature_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('CryptoFeatureProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _calculate_sma(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Simple Moving Averages"""
        for period in periods:
            if TALIB_AVAILABLE:
                data[f'sma_{period}'] = talib.SMA(data['close'].values, timeperiod=period)
            else:
                data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        return data
    
    def _calculate_ema(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages"""
        for period in periods:
            if TALIB_AVAILABLE:
                data[f'ema_{period}'] = talib.EMA(data['close'].values, timeperiod=period)
            else:
                data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        return data
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        if TALIB_AVAILABLE:
            data['rsi'] = talib.RSI(data['close'].values, timeperiod=period)
        else:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        data['rsi_overbought'] = (data['rsi'] > self.feature_config['rsi_overbought']).astype(int)
        data['rsi_oversold'] = (data['rsi'] < self.feature_config['rsi_oversold']).astype(int)
        
        return data
    
    def _calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if TALIB_AVAILABLE:
            macd, macdsignal, macdhist = talib.MACD(data['close'].values, 
                                                   fastperiod=fast, 
                                                   slowperiod=slow, 
                                                   signalperiod=signal)
            data['macd'] = macd
            data['macd_signal'] = macdsignal
            data['macd_histogram'] = macdhist
        else:
            ema_fast = data['close'].ewm(span=fast).mean()
            ema_slow = data['close'].ewm(span=slow).mean()
            data['macd'] = ema_fast - ema_slow
            data['macd_signal'] = data['macd'].ewm(span=signal).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # MACD signals
        data['macd_bullish'] = (data['macd'] > data['macd_signal']).astype(int)
        data['macd_bearish'] = (data['macd'] < data['macd_signal']).astype(int)
        
        return data
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        if TALIB_AVAILABLE:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values, 
                                                        timeperiod=period, 
                                                        nbdevup=std, 
                                                        nbdevdn=std)
            data['bb_upper'] = bb_upper
            data['bb_middle'] = bb_middle
            data['bb_lower'] = bb_lower
        else:
            rolling_mean = data['close'].rolling(window=period).mean()
            rolling_std = data['close'].rolling(window=period).std()
            data['bb_upper'] = rolling_mean + (rolling_std * std)
            data['bb_middle'] = rolling_mean
            data['bb_lower'] = rolling_mean - (rolling_std * std)
        
        # Bollinger Band indicators
        data['bb_width'] = data['bb_upper'] - data['bb_lower']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        data['bb_squeeze'] = (data['bb_width'] < data['bb_width'].rolling(20).mean()).astype(int)
        
        return data
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(data['high'].values, 
                                      data['low'].values, 
                                      data['close'].values,
                                      fastk_period=k_period,
                                      slowk_period=d_period,
                                      slowd_period=d_period)
            data['stoch_k'] = slowk
            data['stoch_d'] = slowd
        else:
            lowest_low = data['low'].rolling(window=k_period).min()
            highest_high = data['high'].rolling(window=k_period).max()
            data['stoch_k'] = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
            data['stoch_d'] = data['stoch_k'].rolling(window=d_period).mean()
        
        return data
    
    def _calculate_volume_indicators(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Volume Indicators"""
        # Volume moving averages
        for period in periods:
            data[f'volume_sma_{period}'] = data['volume'].rolling(window=period).mean()
        
        # Volume ratio
        data['volume_ratio'] = data['volume'] / data['volume_sma_20']
        
        # On-Balance Volume (OBV)
        if TALIB_AVAILABLE:
            data['obv'] = talib.OBV(data['close'].values, data['volume'].values)
        else:
            obv = [0]
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv.append(obv[-1] + data['volume'].iloc[i])
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv.append(obv[-1] - data['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            data['obv'] = obv
        
        # Volume Price Trend (VPT)
        price_change = data['close'].pct_change()
        data['vpt'] = (price_change * data['volume']).cumsum()
        
        return data
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        if TALIB_AVAILABLE:
            data['atr'] = talib.ATR(data['high'].values, 
                                   data['low'].values, 
                                   data['close'].values, 
                                   timeperiod=period)
        else:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            data['atr'] = true_range.rolling(window=period).mean()
        
        # ATR-based indicators
        data['atr_percent'] = data['atr'] / data['close'] * 100
        
        return data
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R"""
        if TALIB_AVAILABLE:
            data['williams_r'] = talib.WILLR(data['high'].values, 
                                           data['low'].values, 
                                           data['close'].values, 
                                           timeperiod=period)
        else:
            highest_high = data['high'].rolling(window=period).max()
            lowest_low = data['low'].rolling(window=period).min()
            data['williams_r'] = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
        
        return data
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index"""
        if TALIB_AVAILABLE:
            data['cci'] = talib.CCI(data['high'].values, 
                                   data['low'].values, 
                                   data['close'].values, 
                                   timeperiod=period)
        else:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            data['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return data
    
    def _calculate_roc(self, data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Calculate Rate of Change"""
        if TALIB_AVAILABLE:
            data['roc'] = talib.ROC(data['close'].values, timeperiod=period)
        else:
            data['roc'] = data['close'].pct_change(periods=period) * 100
        
        return data
    
    def _calculate_price_action_features(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Price Action Features"""
        # Price changes
        for period in periods:
            data[f'price_change_{period}'] = data['close'].pct_change(periods=period)
            data[f'high_change_{period}'] = data['high'].pct_change(periods=period)
            data[f'low_change_{period}'] = data['low'].pct_change(periods=period)
        
        # Candlestick patterns (basic)
        data['body_size'] = np.abs(data['close'] - data['open'])
        data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        data['total_range'] = data['high'] - data['low']
        
        # Doji pattern
        data['is_doji'] = (data['body_size'] / data['total_range'] < 0.1).astype(int)
        
        # Hammer pattern
        data['is_hammer'] = (
            (data['lower_shadow'] > 2 * data['body_size']) & 
            (data['upper_shadow'] < data['body_size'])
        ).astype(int)
        
        # Gap analysis
        data['gap_up'] = (data['open'] > data['close'].shift()).astype(int)
        data['gap_down'] = (data['open'] < data['close'].shift()).astype(int)
        
        return data
    
    def _calculate_volatility_features(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Volatility Features"""
        for period in periods:
            # Price volatility (standard deviation of returns)
            returns = data['close'].pct_change()
            data[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
            
            # High-Low volatility
            data[f'hl_volatility_{period}'] = (
                (data['high'] / data['low'] - 1).rolling(window=period).mean()
            )
            
            # True Range volatility
            if 'atr' in data.columns:
                data[f'atr_volatility_{period}'] = data['atr'].rolling(window=period).mean()
        
        return data
    
    def _calculate_market_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Market Sentiment Features"""
        # Fear & Greed indicators
        data['fear_greed_rsi'] = np.where(data['rsi'] < 30, 1,  # Fear
                                         np.where(data['rsi'] > 70, -1, 0))  # Greed
        
        # Momentum strength
        data['momentum_strength'] = (
            data['rsi'] / 100 * 
            np.sign(data['macd']) * 
            (data['volume_ratio'] if 'volume_ratio' in data.columns else 1)
        )
        
        # Trend strength
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            data['trend_strength'] = (data['sma_20'] - data['sma_50']) / data['sma_50']
        
        # Support/Resistance levels (simplified)
        data['support_level'] = data['low'].rolling(window=20).min()
        data['resistance_level'] = data['high'].rolling(window=20).max()
        data['support_distance'] = (data['close'] - data['support_level']) / data['close']
        data['resistance_distance'] = (data['resistance_level'] - data['close']) / data['close']
        
        return data
    
    def _calculate_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Risk Management Features"""
        # Drawdown calculation
        rolling_max = data['close'].expanding().max()
        data['drawdown'] = (data['close'] - rolling_max) / rolling_max
        data['max_drawdown_20'] = data['drawdown'].rolling(window=20).min()
        
        # Sharpe ratio (simplified)
        if 'price_change_1' in data.columns:
            returns = data['price_change_1']
            data['sharpe_ratio_20'] = (
                returns.rolling(window=20).mean() / 
                returns.rolling(window=20).std()
            )
        
        # Risk-adjusted returns
        if 'volatility_20' in data.columns:
            data['risk_adjusted_return'] = data['price_change_1'] / data['volatility_20']
        
        return data
    
    def _add_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Add lagged features"""
        key_features = ['close', 'volume', 'rsi', 'macd', 'bb_position']
        
        for feature in key_features:
            if feature in data.columns:
                for lag in lags:
                    data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
        
        return data
    
    def _add_rolling_statistics(self, data: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling statistics"""
        key_features = ['close', 'volume', 'rsi']
        
        for feature in key_features:
            if feature in data.columns:
                for window in windows:
                    data[f'{feature}_mean_{window}'] = data[feature].rolling(window=window).mean()
                    data[f'{feature}_std_{window}'] = data[feature].rolling(window=window).std()
                    data[f'{feature}_min_{window}'] = data[feature].rolling(window=window).min()
                    data[f'{feature}_max_{window}'] = data[feature].rolling(window=window).max()
        
        return data
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features (optional)"""
        # List of features to normalize
        normalize_features = [col for col in data.columns if any(x in col for x in [
            'sma_', 'ema_', 'volume_', 'atr', 'volatility_', 'price_change_'
        ])]
        
        for feature in normalize_features:
            if feature in data.columns:
                # Z-score normalization
                mean_val = data[feature].mean()
                std_val = data[feature].std()
                if std_val > 0:
                    data[f'{feature}_normalized'] = (data[feature] - mean_val) / std_val
        
        return data
    
    def _validate_feature_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate feature data quality"""
        try:
            # Check if data is empty
            if data.empty:
                self.logger.error(f"Feature data for {symbol} is empty")
                return False
            
            # Check for required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns for {symbol}: {missing_columns}")
                return False
            
            # Check for excessive null values
            null_percentage = data.isnull().sum() / len(data) * 100
            high_null_features = null_percentage[null_percentage > 50].index.tolist()
            
            if high_null_features:
                self.logger.warning(f"High null percentage in features for {symbol}: {high_null_features}")
            
            # Check for infinite values
            inf_columns = data.columns[data.isin([np.inf, -np.inf]).any()].tolist()
            if inf_columns:
                self.logger.warning(f"Infinite values found in {symbol}: {inf_columns}")
                # Replace infinite values with NaN
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            self.logger.info(f"Feature validation passed for {symbol} - {len(data)} rows, {len(data.columns)} features")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating feature data for {symbol}: {str(e)}")
            return False
    
    def process_raw_data(self, raw_file_path: str, save_features: bool = True) -> pd.DataFrame:
        """
        Process raw data file และเพิ่ม technical indicators
        
        Args:
            raw_file_path: Path to raw data file
            save_features: Whether to save feature data to file
            
        Returns:
            DataFrame with features added
        """
        try:
            self.logger.info(f"Processing raw data: {raw_file_path}")
            
            # Load raw data
            if not os.path.exists(raw_file_path):
                raise FileNotFoundError(f"Raw data file not found: {raw_file_path}")
            
            data = pd.read_csv(raw_file_path)
            
            # Convert timestamp
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.sort_values('timestamp', inplace=True)
            data.reset_index(drop=True, inplace=True)
            
            # Get symbol info from filename or data
            symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'UNKNOWN'
            
            self.logger.info(f"Loaded {len(data)} rows for {symbol}")
            
            # Add all technical indicators
            self.logger.info("Adding technical indicators...")
            
            # Moving Averages
            data = self._calculate_sma(data, self.feature_config['sma_periods'])
            data = self._calculate_ema(data, self.feature_config['ema_periods'])
            
            # Momentum Indicators
            data = self._calculate_rsi(data, self.feature_config['rsi_period'])
            data = self._calculate_macd(data, 
                                      self.feature_config['macd_fast'],
                                      self.feature_config['macd_slow'],
                                      self.feature_config['macd_signal'])
            
            # Volatility Indicators
            data = self._calculate_bollinger_bands(data, 
                                                  self.feature_config['bb_period'],
                                                  self.feature_config['bb_std'])
            data = self._calculate_atr(data, self.feature_config['atr_period'])
            
            # Oscillators
            data = self._calculate_stochastic(data, 
                                            self.feature_config['stoch_k'],
                                            self.feature_config['stoch_d'])
            data = self._calculate_williams_r(data, self.feature_config['williams_period'])
            data = self._calculate_cci(data, self.feature_config['cci_period'])
            data = self._calculate_roc(data, self.feature_config['roc_period'])
            
            # Volume Indicators
            data = self._calculate_volume_indicators(data, self.feature_config['volume_sma_periods'])
            
            # Price Action Features
            data = self._calculate_price_action_features(data, self.feature_config['price_change_periods'])
            
            # Volatility Features
            data = self._calculate_volatility_features(data, self.feature_config['volatility_periods'])
            
            # Market Sentiment Features
            data = self._calculate_market_sentiment_features(data)
            
            # Risk Features
            data = self._calculate_risk_features(data)
            
            # Advanced Features
            data = self._add_lag_features(data)
            data = self._add_rolling_statistics(data)
            
            # Optional normalization
            # data = self._normalize_features(data)
            
            # Validate feature data
            if not self._validate_feature_data(data, symbol):
                raise ValueError(f"Feature validation failed for {symbol}")
            
            # Save feature data
            if save_features:
                feature_file_path = self._get_feature_file_path(raw_file_path)
                self._save_feature_data(data, feature_file_path, symbol)
            
            self.logger.info(f"✅ Successfully processed {symbol} - {len(data)} rows, {len(data.columns)} features")
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing raw data {raw_file_path}: {str(e)}")
            raise
    
    def _get_feature_file_path(self, raw_file_path: str) -> str:
        """Generate feature file path from raw file path"""
        raw_filename = os.path.basename(raw_file_path)
        # Replace .csv with -features.csv
        feature_filename = raw_filename.replace('.csv', '-features.csv')
        return os.path.join(self.feature_dir, feature_filename)
    
    def _save_feature_data(self, data: pd.DataFrame, file_path: str, symbol: str) -> bool:
        """Save feature data to CSV file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to CSV
            data.to_csv(file_path, index=False)
            
            # Verify file was created
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                file_size = os.path.getsize(file_path) / 1024  # KB
                self.logger.info(f"Saved {symbol} feature data to {file_path} ({file_size:.1f} KB)")
                return True
            else:
                self.logger.error(f"Failed to save {symbol} feature data")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving {symbol} feature data: {str(e)}")
            return False
    
    def process_all_raw_data(self, force_reprocess: bool = False) -> Dict[str, bool]:
        """
        Process all raw data files in the raw directory
        
        Args:
            force_reprocess: Force reprocessing even if feature file exists
            
        Returns:
            Dictionary with file processing results
        """
        self.logger.info("Processing all raw data files...")
        
        results = {}
        
        # Find all raw data files
        raw_files = glob.glob(os.path.join(self.raw_dir, "*.csv"))
        
        if not raw_files:
            self.logger.warning(f"No raw data files found in {self.raw_dir}")
            return results
        
        self.logger.info(f"Found {len(raw_files)} raw data files")
        
        for raw_file in raw_files:
            try:
                filename = os.path.basename(raw_file)
                feature_file = self._get_feature_file_path(raw_file)
                
                # Check if feature file already exists
                if not force_reprocess and os.path.exists(feature_file):
                    self.logger.info(f"⏭️  Skipping {filename} - feature file already exists")
                    results[filename] = True
                    continue
                
                # Process the file
                self.process_raw_data(raw_file, save_features=True)
                results[filename] = True
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process {filename}: {str(e)}")
                results[filename] = False
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        self.logger.info(f"Processing completed: {successful}/{total} files successful")
        
        return results
    
    def load_feature_data(self, symbol: str, start_date: str, end_date: str, 
                         interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Load feature data for specific symbol and date range
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval
            
        Returns:
            DataFrame with feature data if available
        """
        try:
            # Generate expected filename
            clean_symbol = symbol.replace('/', '_').replace('-', '_')
            start_clean = start_date.replace('-', '')
            end_clean = end_date.replace('-', '')
            
            feature_filename = f"{clean_symbol}-{interval}-{start_clean}-{end_clean}-features.csv"
            feature_file_path = os.path.join(self.feature_dir, feature_filename)
            
            if not os.path.exists(feature_file_path):
                self.logger.warning(f"Feature file not found: {feature_file_path}")
                return None
            
            # Load feature data
            data = pd.read_csv(feature_file_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Loaded feature data: {len(data)} rows, {len(data.columns)} features")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading feature data: {str(e)}")
            return None
    
    def list_available_feature_data(self) -> List[Dict[str, str]]:
        """List all available feature data files"""
        available_files = []
        
        if not os.path.exists(self.feature_dir):
            return available_files
        
        for filename in os.listdir(self.feature_dir):
            if filename.endswith('-features.csv'):
                try:
                    # Parse filename: {symbol}-{interval}-{start}-{end}-features.csv
                    name_parts = filename[:-13].split('-')  # Remove -features.csv
                    
                    if len(name_parts) >= 4:
                        symbol = name_parts[0]
                        interval = name_parts[1]
                        start_date = name_parts[2]
                        end_date = name_parts[3]
                        
                        # Format dates
                        start_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
                        end_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
                        
                        file_path = os.path.join(self.feature_dir, filename)
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        # Get feature count
                        try:
                            sample_data = pd.read_csv(file_path, nrows=1)
                            feature_count = len(sample_data.columns)
                        except:
                            feature_count = 0
                        
                        available_files.append({
                            'symbol': symbol,
                            'interval': interval,
                            'start_date': start_formatted,
                            'end_date': end_formatted,
                            'filename': filename,
                            'file_path': file_path,
                            'file_size_kb': round(file_size, 1),
                            'feature_count': feature_count,
                            'modified_time': mod_time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Could not parse filename {filename}: {e}")
        
        # Sort by symbol and modification time
        available_files.sort(key=lambda x: (x['symbol'], x['modified_time']))
        
        return available_files
    
    def get_feature_summary(self) -> Dict[str, any]:
        """Get summary of available feature data"""
        files = self.list_available_feature_data()
        
        if not files:
            return {'total_files': 0, 'symbols': [], 'intervals': [], 'total_size_mb': 0}
        
        symbols = list(set(f['symbol'] for f in files))
        intervals = list(set(f['interval'] for f in files))
        total_size_mb = sum(f['file_size_kb'] for f in files) / 1024
        avg_features = np.mean([f['feature_count'] for f in files if f['feature_count'] > 0])
        
        return {
            'total_files': len(files),
            'symbols': sorted(symbols),
            'intervals': sorted(intervals),
            'total_size_mb': round(total_size_mb, 2),
            'average_features': round(avg_features, 0) if avg_features else 0,
            'feature_directory': self.feature_dir
        }

# Convenience functions
def process_crypto_features(raw_file_path: str) -> pd.DataFrame:
    """
    Convenience function สำหรับ process raw data file
    
    Args:
        raw_file_path: Path to raw data file
        
    Returns:
        DataFrame with features
    """
    processor = CryptoFeatureProcessor()
    return processor.process_raw_data(raw_file_path)

def process_all_crypto_features(force_reprocess: bool = False) -> Dict[str, bool]:
    """
    Convenience function สำหรับ process all raw data files
    
    Args:
        force_reprocess: Force reprocessing
        
    Returns:
        Processing results
    """
    processor = CryptoFeatureProcessor()
    return processor.process_all_raw_data(force_reprocess)

def load_crypto_features(symbol: str, start_date: str, end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    Convenience function สำหรับ load feature data
    
    Args:
        symbol: Trading symbol
        start_date: Start date
        end_date: End date
        interval: Time interval
        
    Returns:
        Feature DataFrame
    """
    processor = CryptoFeatureProcessor()
    return processor.load_feature_data(symbol, start_date, end_date, interval)

def get_crypto_feature_summary() -> Dict[str, any]:
    """
    Convenience function สำหรับ get feature summary
    
    Returns:
        Feature summary
    """
    processor = CryptoFeatureProcessor()
    return processor.get_feature_summary()

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = CryptoFeatureProcessor()
    
    print("🔧 Crypto Feature Engineering System")
    print("=" * 50)
    
    # Example 1: Process all raw data
    print("\n📊 Example 1: Process all raw data files")
    try:
        results = processor.process_all_raw_data()
        print(f"✅ Processing results:")
        for filename, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {filename}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: List available feature data
    print("\n📋 Example 2: Available feature data")
    try:
        available = processor.list_available_feature_data()
        print(f"📁 Found {len(available)} feature files:")
        for file_info in available[:5]:  # Show first 5
            print(f"   {file_info['symbol']} ({file_info['interval']}) - {file_info['feature_count']} features - {file_info['file_size_kb']} KB")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: Feature summary
    print("\n📈 Example 3: Feature summary")
    try:
        summary = processor.get_feature_summary()
        print(f"📊 Summary:")
        print(f"   Total files: {summary['total_files']}")
        print(f"   Symbols: {summary['symbols']}")
        print(f"   Average features: {summary['average_features']}")
        print(f"   Total size: {summary['total_size_mb']} MB")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Load specific feature data
    print("\n📥 Example 4: Load specific feature data")
    try:
        # Try to load BTC feature data
        btc_features = processor.load_feature_data('BTC-USD', '2024-01-01', '2024-02-01', '1d')
        
        if btc_features is not None:
            print(f"✅ Loaded BTC feature data: {len(btc_features)} rows, {len(btc_features.columns)} features")
            print(f"Feature columns: {list(btc_features.columns[:10])}...")  # Show first 10 columns
        else:
            print("ℹ️  No BTC feature data found for this date range")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Feature Engineering System Testing Completed!")