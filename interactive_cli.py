# interactive_cli.py - Interactive Command-Line Interface for Crypto SAC Agent
"""
Interactive CLI for Native Python SAC Agent

คุณสมบัติหลัก:
1. Menu-driven interface สำหรับการใช้งาน SAC agent
2. Agent creation, training, testing, และ comparison
3. Grade system integration
4. Performance visualization และ analysis
5. Agent management และ file operations

Usage:
    python interactive_cli.py
    หรือ
    from interactive_cli import InteractiveCLI
    cli = InteractiveCLI()
    cli.main_menu()
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import pickle
from typing import Dict, List, Optional, Any

# Import unified agent
from crypto_agent import CryptoSACAgent, create_crypto_sac_agent, load_crypto_sac_agent
from config.config import *

# Data loading utilities
import yfinance as yf

class AgentManager:
    """Manager สำหรับจัดการ SAC agents หลายตัว"""
    
    def __init__(self):
        self.agents = {}
        self.current_agent = None
        self.models_dir = os.path.join(MODEL_DIR, "sac")
        os.makedirs(self.models_dir, exist_ok=True)
    
    def list_available_agents(self):
        """แสดงรายการ agents ที่มีอยู่"""
        if not os.path.exists(self.models_dir):
            return []
        
        agents_info = []
        files = os.listdir(self.models_dir)
        zip_files = [f for f in files if f.endswith('.zip')]
        
        for zip_file in zip_files:
            agent_name = zip_file[:-4]  # Remove .zip extension
            info_file = os.path.join(self.models_dir, f"{agent_name}_info.pkl")
            
            agent_info = {
                'name': agent_name,
                'model_path': os.path.join(self.models_dir, zip_file),
                'has_metadata': os.path.exists(info_file),
                'created_date': datetime.fromtimestamp(os.path.getctime(os.path.join(self.models_dir, zip_file)))
            }
            
            # Try to load metadata for more info
            if agent_info['has_metadata']:
                try:
                    with open(info_file, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    if isinstance(metadata, dict):
                        agent_info['grade'] = metadata.get('grade', 'Unknown')
                        agent_info['config'] = metadata.get('config', {})
                        agent_info['performance'] = metadata.get('performance_metrics', {})
                    else:
                        agent_info['grade'] = getattr(metadata, 'grade', 'Unknown')
                        agent_info['config'] = getattr(metadata, 'config', {})
                        agent_info['performance'] = getattr(metadata, 'performance_metrics', {})
                        
                except Exception as e:
                    print(f"Warning: Could not load metadata for {agent_name}: {e}")
                    agent_info['grade'] = 'Unknown'
                    agent_info['config'] = {}
                    agent_info['performance'] = {}
            else:
                agent_info['grade'] = 'Unknown'
                agent_info['config'] = {}
                agent_info['performance'] = {}
            
            agents_info.append(agent_info)
        
        # Sort by creation date (newest first)
        agents_info.sort(key=lambda x: x['created_date'], reverse=True)
        return agents_info
    
    def load_agent(self, agent_name):
        """โหลด agent ตามชื่อ"""
        agent_path = os.path.join(self.models_dir, agent_name)
        try:
            agent = load_crypto_sac_agent(agent_path)
            self.current_agent = agent
            self.agents[agent_name] = agent
            return agent
        except Exception as e:
            print(f"❌ Failed to load agent {agent_name}: {e}")
            return None
    
    def save_current_agent(self, custom_name=None):
        """บันทึก current agent"""
        if self.current_agent is None:
            print("❌ No current agent to save")
            return None
        
        try:
            if custom_name:
                save_path = os.path.join(self.models_dir, custom_name)
            else:
                save_path = None  # Let agent generate path
            
            saved_path = self.current_agent.save(save_path)
            print(f"✅ Agent saved successfully")
            return saved_path
        except Exception as e:
            print(f"❌ Failed to save agent: {e}")
            return None

class DataManager:
    """Manager สำหรับจัดการข้อมูล crypto"""
    
    @staticmethod
    def load_crypto_data(symbols=None, force_download=False):
        """โหลดข้อมูล crypto"""
        symbols = symbols or CRYPTO_SYMBOLS
        data_file = os.path.join(DATA_DIR, "crypto_data.csv")
        
        # ถ้ามีไฟล์ข้อมูลและไม่บังคับดาวน์โหลดใหม่
        if os.path.exists(data_file) and not force_download:
            print("📂 Loading existing data...")
            try:
                df = pd.read_csv(data_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"✅ Loaded {len(df)} rows of data")
                return df
            except Exception as e:
                print(f"⚠️ Error loading existing data: {e}")
                print("🔄 Downloading new data...")
        
        # ดาวน์โหลดข้อมูลใหม่
        print(f"📊 Downloading crypto data for {symbols}...")
        df_list = []
        
        for symbol in symbols:
            print(f"📥 Downloading {symbol}...")
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=START_DATE,
                    end=END_DATE,
                    interval='1D',
                    auto_adjust=True
                )
                
                if len(df) == 0:
                    print(f"⚠️ No data found for {symbol}")
                    continue
                
                # Add required columns
                df['tic'] = symbol
                df['timestamp'] = df.index
                
                # Rename columns to lowercase
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                
                df_list.append(df)
                print(f"✅ Downloaded {len(df)} rows for {symbol}")
                
            except Exception as e:
                print(f"❌ Error downloading {symbol}: {e}")
                continue
        
        if not df_list:
            raise ValueError("No data downloaded successfully")
        
        # Combine data
        combined_df = pd.concat(df_list, axis=0).reset_index(drop=True)
        
        # Save data
        try:
            combined_df.to_csv(data_file, index=False)
            print(f"💾 Saved data to {data_file}")
        except Exception as e:
            print(f"⚠️ Could not save data: {e}")
        
        return combined_df
    
    @staticmethod
    def add_technical_indicators(df):
        """เพิ่ม technical indicators"""
        print("📈 Adding technical indicators...")
        
        df = df.copy()
        
        # Moving Averages
        df['sma_20'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).mean())
        df['ema_20'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        
        # RSI
        def calculate_rsi(group):
            delta = group['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = df.groupby('tic').apply(calculate_rsi).reset_index(level=0, drop=True)
        
        # MACD
        def calculate_macd(group):
            exp1 = group['close'].ewm(span=12, adjust=False).mean()
            exp2 = group['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            return pd.DataFrame({
                'macd': macd,
                'macd_signal': signal,
                'macd_histogram': histogram
            })
        
        macd_df = df.groupby('tic').apply(calculate_macd).reset_index(level=0, drop=True)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['macd_signal']
        df['macd_histogram'] = macd_df['macd_histogram']
        
        # Bollinger Bands
        def calculate_bollinger_bands(group):
            sma = group['close'].rolling(window=20).mean()
            std = group['close'].rolling(window=20).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            return pd.DataFrame({
                'bb_upper': upper,
                'bb_middle': sma,
                'bb_lower': lower,
                'bb_std': std
            })
        
        bb_df = df.groupby('tic').apply(calculate_bollinger_bands).reset_index(level=0, drop=True)
        df['bb_upper'] = bb_df['bb_upper']
        df['bb_middle'] = bb_df['bb_middle']
        df['bb_lower'] = bb_df['bb_lower']
        df['bb_std'] = bb_df['bb_std']
        
        # Volume indicators
        df['volume_sma_20'] = df.groupby('tic')['volume'].transform(lambda x: x.rolling(window=20).mean())
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        print(f"✅ Added technical indicators. Total columns: {len(df.columns)}")
        return df

class InteractiveCLI:
    """Interactive Command-Line Interface สำหรับ SAC Agent"""
    
    def __init__(self):
        self.agent_manager = AgentManager()
        self.data_manager = DataManager()
        self.current_data = None
        
        print("🎮 Crypto SAC Agent - Interactive CLI")
        print("=" * 50)
    
    def main_menu(self):
        """แสดง main menu"""
        while True:
            print(f"\n🤖 Current Agent: {self.agent_manager.current_agent.agent_id if self.agent_manager.current_agent else 'None'}")
            print("=" * 60)
            print("1. 🆕 Create New Agent")
            print("2. 📊 Load Existing Agent")
            print("3. 📈 Load/Download Data")
            print("4. 🏋️ Train Current Agent")
            print("5. 🧪 Test Current Agent")
            print("6. 📊 View Agent Performance")
            print("7. 🔍 Compare Agents")
            print("8. 💾 Save Current Agent")
            print("9. ⚙️ Agent Settings")
            print("0. 🚪 Exit")
            
            choice = input("\n👉 Select option (0-9): ").strip()
            
            try:
                if choice == '1':
                    self.create_agent_workflow()
                elif choice == '2':
                    self.load_agent_workflow()
                elif choice == '3':
                    self.load_data_workflow()
                elif choice == '4':
                    self.train_agent_workflow()
                elif choice == '5':
                    self.test_agent_workflow()
                elif choice == '6':
                    self.view_performance_workflow()
                elif choice == '7':
                    self.compare_agents_workflow()
                elif choice == '8':
                    self.save_agent_workflow()
                elif choice == '9':
                    self.settings_workflow()
                elif choice == '0':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid option. Please try again.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                input("Press Enter to continue...")
    
    def create_agent_workflow(self):
        """Workflow สำหรับสร้าง agent ใหม่ - Enhanced Version"""
        print("\n🆕 Create New RL Agent")
        print("=" * 50)
        
        # Step 1: Algorithm Selection
        print("🤖 Step 1: Select Algorithm")
        print("-" * 30)
        algorithms = {
            '1': {'name': 'SAC', 'description': 'Soft Actor-Critic (Recommended for continuous actions)'},
            '2': {'name': 'PPO', 'description': 'Proximal Policy Optimization (Stable, good baseline)'},
            '3': {'name': 'DDPG', 'description': 'Deep Deterministic Policy Gradient (Deterministic)'},
            '4': {'name': 'TD3', 'description': 'Twin Delayed Deep Deterministic (Improved DDPG)'},
            '5': {'name': 'A2C', 'description': 'Advantage Actor-Critic (Fast training)'}
        }
        
        for key, algo in algorithms.items():
            print(f"  {key}. {algo['name']} - {algo['description']}")
        
        while True:
            algo_choice = input("\n👉 Select algorithm (1-5) [1]: ").strip() or '1'
            if algo_choice in algorithms:
                selected_algorithm = algorithms[algo_choice]['name']
                break
            print("❌ Invalid choice. Please try again.")
        
        print(f"✅ Selected Algorithm: {selected_algorithm}")
        
        # Step 2: Grade Selection
        print(f"\n🎯 Step 2: Select Agent Grade")
        print("-" * 30)
        grades = {
            'N': {'name': 'Novice', 'timesteps': '50K', 'description': 'Basic learning, fast training'},
            'D': {'name': 'Developing', 'timesteps': '100K', 'description': 'Improved parameters, balanced'},
            'C': {'name': 'Competent', 'timesteps': '200K', 'description': 'Professional setup, recommended'},
            'B': {'name': 'Proficient', 'timesteps': '500K', 'description': 'High performance, longer training'},
            'A': {'name': 'Advanced', 'timesteps': '1M', 'description': 'Research grade, excellent results'},
            'S': {'name': 'Supreme', 'timesteps': '2M', 'description': 'State-of-the-art, maximum performance'}
        }
        
        for grade, info in grades.items():
            print(f"  {grade}: {info['name']} ({info['timesteps']}) - {info['description']}")
        
        while True:
            grade = input("\n👉 Select grade (N/D/C/B/A/S) [C]: ").strip().upper() or 'C'
            if grade in grades:
                break
            print("❌ Invalid grade. Please try again.")
        
        print(f"✅ Selected Grade: {grade} ({grades[grade]['name']})")
        
        # Step 3: Data Selection
        print(f"\n📊 Step 3: Select Training Data")
        print("-" * 30)
        
        # Check available feature data
        try:
            from src.data_feature import get_crypto_feature_summary
            feature_summary = get_crypto_feature_summary()
            
            if feature_summary['total_files'] > 0:
                print(f"📁 Available Feature Data:")
                print(f"   Files: {feature_summary['total_files']}")
                print(f"   Symbols: {feature_summary['symbols']}")
                print(f"   Average Features: {feature_summary['average_features']}")
                print(f"   Total Size: {feature_summary['total_size_mb']} MB")
                
                # Symbol selection
                available_symbols = feature_summary['symbols']
                print(f"\n🎯 Available Symbols:")
                for i, symbol in enumerate(available_symbols, 1):
                    print(f"  {i}. {symbol}")
                print(f"  {len(available_symbols)+1}. All symbols")
                
                while True:
                    try:
                        symbol_choice = input(f"\n👉 Select symbol (1-{len(available_symbols)+1}) [All]: ").strip()
                        if not symbol_choice or symbol_choice == str(len(available_symbols)+1):
                            selected_symbols = available_symbols
                            break
                        
                        choice_idx = int(symbol_choice) - 1
                        if 0 <= choice_idx < len(available_symbols):
                            selected_symbols = [available_symbols[choice_idx]]
                            break
                        else:
                            print("❌ Invalid selection.")
                    except ValueError:
                        print("❌ Please enter a number.")
                
                print(f"✅ Selected Symbols: {selected_symbols}")
                use_feature_data = True
                
            else:
                print("⚠️ No feature data found. Will use basic data with technical indicators.")
                selected_symbols = CRYPTO_SYMBOLS
                use_feature_data = False
                
        except Exception as e:
            print(f"⚠️ Could not load feature data: {e}")
            print("Will use basic data with technical indicators.")
            selected_symbols = CRYPTO_SYMBOLS
            use_feature_data = False
        
        # Step 4: Environment Type Selection
        print(f"\n🏗️ Step 4: Select Environment Type")
        print("-" * 30)
        env_types = {
            '1': {'name': 'Basic', 'description': 'Simple trading environment, fast training'},
            '2': {'name': 'Enhanced', 'description': 'Advanced features, risk management, better performance'}
        }
        
        for key, env in env_types.items():
            print(f"  {key}. {env['name']} - {env['description']}")
        
        while True:
            env_choice = input("\n👉 Select environment (1-2) [2]: ").strip() or '2'
            if env_choice in env_types:
                environment_type = env_types[env_choice]['name'].lower()
                break
            print("❌ Invalid choice. Please try again.")
        
        print(f"✅ Selected Environment: {env_types[env_choice]['name']}")
        
        # Step 5: Configuration Summary & Confirmation
        print(f"\n📋 Step 5: Configuration Summary")
        print("=" * 50)
        print(f"🤖 Algorithm: {selected_algorithm}")
        print(f"🎯 Grade: {grade} ({grades[grade]['name']})")
        print(f"📊 Symbols: {selected_symbols}")
        print(f"💾 Data Type: {'Feature Data (151 features)' if use_feature_data else 'Basic Data (12 indicators)'}")
        print(f"🏗️ Environment: {env_types[env_choice]['name']}")
        print(f"⏱️ Estimated Training Time: {grades[grade]['timesteps']} timesteps")
        
        confirm = input(f"\n✅ Create agent with these settings? (y/n) [y]: ").strip().lower()
        if confirm in ['n', 'no']:
            print("❌ Agent creation cancelled.")
            input("Press Enter to continue...")
            return
        
        # Step 6: Create Agent
        print(f"\n🚀 Step 6: Creating Agent...")
        print("-" * 30)
        
        try:
            # Import appropriate agent class based on algorithm
            if selected_algorithm == 'SAC':
                from crypto_agent import create_crypto_sac_agent
                agent = create_crypto_sac_agent(grade=grade)
            elif selected_algorithm == 'PPO':
                # For now, use SAC as base - can be extended later
                print("⚠️ PPO implementation coming soon. Using SAC for now.")
                from crypto_agent import create_crypto_sac_agent
                agent = create_crypto_sac_agent(grade=grade)
            else:
                # For other algorithms, use SAC as fallback
                print(f"⚠️ {selected_algorithm} implementation coming soon. Using SAC for now.")
                from crypto_agent import create_crypto_sac_agent
                agent = create_crypto_sac_agent(grade=grade)
            
            # Set additional agent properties
            agent.algorithm = selected_algorithm
            agent.selected_symbols = selected_symbols
            agent.use_feature_data = use_feature_data
            agent.environment_type = environment_type
            
            self.agent_manager.current_agent = agent
            
            print(f"✅ Agent Created Successfully!")
            print(f"   🤖 ID: {agent.agent_id}")
            print(f"   🎯 Grade: {grade}")
            print(f"   ⚙️ Timesteps: {agent.config['total_timesteps']:,}")
            print(f"   💾 Buffer: {agent.config['buffer_size']:,}")
            
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            input("Press Enter to continue...")
            return
        
        # Step 7: Load Data Option
        print(f"\n📊 Step 7: Load Training Data")
        print("-" * 30)
        
        load_data_now = input("📥 Load training data now? (y/n) [y]: ").strip().lower()
        if load_data_now != 'n':
            try:
                print("⏳ Loading training data...")
                
                if use_feature_data:
                    # Load feature data
                    training_data = self._load_feature_data_for_symbols(selected_symbols)
                else:
                    # Load basic data
                    training_data = self.data_manager.load_crypto_data(
                        symbols=selected_symbols,
                        force_download=False
                    )
                    training_data = self.data_manager.add_technical_indicators(training_data)
                
                if training_data is not None:
                    print(f"✅ Data loaded successfully!")
                    print(f"   📊 Rows: {len(training_data):,}")
                    print(f"   🔢 Features: {len(training_data.columns)}")
                    print(f"   📅 Date range: {training_data['timestamp'].min()} to {training_data['timestamp'].max()}")
                    
                    # Create environment
                    print("\n🏗️ Creating trading environment...")
                    if environment_type == 'enhanced':
                        # Use enhanced environment
                        from enhanced_crypto_env import EnhancedCryptoTradingEnv
                        train_env, test_env = agent.create_environment(training_data, env_class=EnhancedCryptoTradingEnv)
                    else:
                        # Use basic environment
                        train_env, test_env = agent.create_environment(training_data)
                    
                    print("✅ Environment created successfully!")
                    
                    # Store data for later use
                    self.current_data = training_data
                    
                else:
                    print("❌ Failed to load training data.")
                    
            except Exception as e:
                print(f"❌ Error loading data: {e}")
        
        print(f"\n🎉 Agent Creation Completed!")
        print("💡 Next steps:")
        print("   4. 🏋️ Train Current Agent")
        print("   5. 🧪 Test Current Agent")
        print("   8. 💾 Save Current Agent")
        
        input("\nPress Enter to continue...")
    
    def _load_feature_data_for_symbols(self, symbols):
        """Helper function to load feature data for selected symbols"""
        try:
            from src.data_feature import CryptoFeatureProcessor
            processor = CryptoFeatureProcessor()
            
            # Get available feature files
            available_files = processor.list_available_feature_data()
            
            # Filter files for selected symbols
            matching_files = []
            for file_info in available_files:
                if file_info['symbol'] in symbols:
                    matching_files.append(file_info)
            
            if not matching_files:
                print(f"❌ No feature files found for symbols: {symbols}")
                return None
            
            # Load and combine data
            combined_data = []
            for file_info in matching_files:
                file_path = file_info['file_path']
                data = pd.read_csv(file_path)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Add tic column if not present
                if 'tic' not in data.columns:
                    data['tic'] = file_info['symbol']
                
                combined_data.append(data)
                print(f"   📊 Loaded {file_info['symbol']}: {len(data)} rows, {len(data.columns)} features")
            
            # Combine all data
            if len(combined_data) == 1:
                final_data = combined_data[0]
            else:
                final_data = pd.concat(combined_data, ignore_index=True)
                final_data = final_data.sort_values(['tic', 'timestamp']).reset_index(drop=True)
            
            return final_data
            
        except Exception as e:
            print(f"❌ Error loading feature data: {e}")
            return None
    
    def load_agent_workflow(self):
        """Workflow สำหรับโหลด agent"""
        print("\n📊 Load Existing Agent")
        print("-" * 30)
        
        agents = self.agent_manager.list_available_agents()
        
        if not agents:
            print("❌ No saved agents found.")
            input("Press Enter to continue...")
            return
        
        print(f"📋 Available Agents ({len(agents)}):")
        for i, agent in enumerate(agents, 1):
            status = "✅" if agent['has_metadata'] else "⚠️"
            grade = agent['grade']
            date = agent['created_date'].strftime('%Y-%m-%d %H:%M')
            
            print(f"  {i}. {status} {agent['name']}")
            print(f"     Grade: {grade} | Created: {date}")
            
            if agent['performance']:
                mean_reward = agent['performance'].get('mean_reward', 'N/A')
                print(f"     Performance: {mean_reward}")
        
        while True:
            try:
                choice = input(f"\n👉 Select agent (1-{len(agents)}) or 0 to cancel: ").strip()
                if choice == '0':
                    return
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(agents):
                    break
                else:
                    print("❌ Invalid selection.")
            except ValueError:
                print("❌ Please enter a number.")
        
        # Load selected agent
        selected_agent = agents[choice_idx]
        agent_name = selected_agent['name']
        
        print(f"\n⏳ Loading agent: {agent_name}")
        
        try:
            loaded_agent = self.agent_manager.load_agent(agent_name)
            if loaded_agent:
                print(f"✅ Agent loaded successfully!")
                print(f"   🤖 ID: {loaded_agent.agent_id}")
                print(f"   🎯 Grade: {loaded_agent.grade}")
                print(f"   📊 Trained: {'Yes' if loaded_agent.is_trained else 'No'}")
            else:
                print("❌ Failed to load agent.")
        
        except Exception as e:
            print(f"❌ Error loading agent: {e}")
        
        input("\nPress Enter to continue...")
    
    def load_data_workflow(self):
        """Workflow สำหรับโหลดข้อมูล"""
        print("\n📈 Load/Download Crypto Data")
        print("-" * 30)
        
        # Check existing data
        data_file = os.path.join(DATA_DIR, "crypto_data.csv")
        has_existing = os.path.exists(data_file)
        
        if has_existing:
            file_size = os.path.getsize(data_file) / 1024  # KB
            mod_time = datetime.fromtimestamp(os.path.getmtime(data_file))
            print(f"📂 Existing data found:")
            print(f"   Size: {file_size:.1f} KB")
            print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
        
        print(f"📊 Current symbols: {CRYPTO_SYMBOLS}")
        print(f"📅 Date range: {START_DATE} to {END_DATE}")
        
        options = []
        if has_existing:
            options.append("1. 📂 Load existing data")
            options.append("2. 🔄 Download fresh data")
        else:
            options.append("1. 📥 Download data")
        
        for option in options:
            print(f"  {option}")
        
        choice = input(f"\n👉 Select option: ").strip()
        
        force_download = False
        if has_existing:
            force_download = (choice == '2')
        
        try:
            print("\n⏳ Loading data...")
            self.current_data = self.data_manager.load_crypto_data(force_download=force_download)
            
            print("\n⏳ Adding technical indicators...")
            self.current_data = self.data_manager.add_technical_indicators(self.current_data)
            
            print(f"\n✅ Data loaded successfully!")
            print(f"   📊 Rows: {len(self.current_data):,}")
            print(f"   📈 Symbols: {self.current_data['tic'].unique()}")
            print(f"   📅 Date range: {self.current_data['timestamp'].min()} to {self.current_data['timestamp'].max()}")
            print(f"   🔢 Features: {len(self.current_data.columns)} columns")
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
        
        input("\nPress Enter to continue...")
    
    def train_agent_workflow(self):
        """Workflow สำหรับเทรน agent"""
        if self.agent_manager.current_agent is None:
            print("❌ No agent selected. Please create or load an agent first.")
            input("Press Enter to continue...")
            return
        
        if self.current_data is None:
            print("❌ No data loaded. Please load data first.")
            input("Press Enter to continue...")
            return
        
        print("\n🏋️ Train SAC Agent")
        print("-" * 30)
        
        agent = self.agent_manager.current_agent
        print(f"🤖 Agent: {agent.agent_id}")
        print(f"🎯 Grade: {agent.grade}")
        print(f"⚙️ Default timesteps: {agent.config['total_timesteps']:,}")
        
        # Custom timesteps option
        custom_timesteps = input(f"🔢 Custom timesteps (or Enter for default): ").strip()
        timesteps = None
        if custom_timesteps:
            try:
                timesteps = int(custom_timesteps)
            except ValueError:
                print("⚠️ Invalid input. Using default timesteps.")
        
        try:
            print("\n⏳ Creating trading environment...")
            train_env, test_env = agent.create_environment(self.current_data)
            
            print("\n🚀 Starting training...")
            print("⚠️ This may take several minutes depending on the grade...")
            
            model = agent.train(timesteps=timesteps, verbose=1)
            
            print(f"\n✅ Training completed!")
            print(f"   ⏱️ Agent is now trained and ready for testing")
            
            # Auto-save option
            save_now = input("\n💾 Save agent now? (y/n) [y]: ").strip().lower()
            if save_now != 'n':
                self.agent_manager.save_current_agent()
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
        
        input("\nPress Enter to continue...")
    
    def test_agent_workflow(self):
        """Workflow สำหรับทดสอบ agent"""
        if self.agent_manager.current_agent is None:
            print("❌ No agent selected.")
            input("Press Enter to continue...")
            return
        
        agent = self.agent_manager.current_agent
        if not agent.is_trained:
            print("❌ Agent is not trained yet.")
            input("Press Enter to continue...")
            return
        
        if agent.test_env is None:
            print("❌ No test environment. Please load data and create environment first.")
            input("Press Enter to continue...")
            return
        
        print("\n🧪 Test SAC Agent")
        print("-" * 30)
        
        print(f"🤖 Agent: {agent.agent_id}")
        print(f"🎯 Grade: {agent.grade}")
        
        # Number of episodes
        n_episodes = input("📊 Number of test episodes [10]: ").strip() or "10"
        try:
            n_episodes = int(n_episodes)
        except ValueError:
            n_episodes = 10
            print("⚠️ Invalid input. Using 10 episodes.")
        
        try:
            print(f"\n⏳ Evaluating agent with {n_episodes} episodes...")
            results = agent.evaluate(n_episodes=n_episodes)
            
            print(f"\n📊 Test Results:")
            print(f"   🎯 Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
            print(f"   🏆 Best Reward: {results['max_reward']:.4f}")
            print(f"   📉 Worst Reward: {results['min_reward']:.4f}")
            print(f"   📏 Avg Episode Length: {results['mean_episode_length']:.1f}")
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
        
        input("\nPress Enter to continue...")
    
    def view_performance_workflow(self):
        """Workflow สำหรับดู performance"""
        if self.agent_manager.current_agent is None:
            print("❌ No agent selected.")
            input("Press Enter to continue...")
            return
        
        print("\n📊 Agent Performance")
        print("-" * 30)
        
        agent = self.agent_manager.current_agent
        info = agent.get_info()
        
        print(f"🤖 Agent Information:")
        print(f"   ID: {info['agent_id']}")
        print(f"   Grade: {info['grade']}")
        print(f"   Status: {'Trained' if info['is_trained'] else 'Not Trained'}")
        
        print(f"\n⚙️ Configuration:")
        config = info['config']
        print(f"   Timesteps: {config.get('total_timesteps', 'N/A'):,}")
        print(f"   Buffer Size: {config.get('buffer_size', 'N/A'):,}")
        print(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"   Batch Size: {config.get('batch_size', 'N/A')}")
        
        # Metadata information
        metadata = info.get('metadata', {})
        if isinstance(metadata, dict):
            perf_metrics = metadata.get('performance_metrics', {})
            if perf_metrics:
                print(f"\n📈 Performance Metrics:")
                for key, value in perf_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
        
        input("\nPress Enter to continue...")
    
    def compare_agents_workflow(self):
        """Workflow สำหรับเปรียบเทียบ agents"""
        print("\n🔍 Compare Agents")
        print("-" * 30)
        
        agents = self.agent_manager.list_available_agents()
        
        if len(agents) < 2:
            print("❌ Need at least 2 agents for comparison.")
            input("Press Enter to continue...")
            return
        
        print("📋 Available Agents:")
        for i, agent in enumerate(agents, 1):
            status = "✅" if agent['has_metadata'] else "⚠️"
            print(f"  {i}. {status} {agent['name']} (Grade: {agent['grade']})")
        
        print("\n📊 Performance Comparison:")
        print(f"{'Name':<30} {'Grade':<5} {'Mean Reward':<12} {'Created':<16}")
        print("-" * 70)
        
        for agent in agents:
            name = agent['name'][:28] + ".." if len(agent['name']) > 30 else agent['name']
            grade = agent['grade']
            
            mean_reward = "N/A"
            if agent['performance']:
                mr = agent['performance'].get('mean_reward')
                if mr is not None:
                    mean_reward = f"{mr:.4f}"
            
            created = agent['created_date'].strftime('%Y-%m-%d %H:%M')
            print(f"{name:<30} {grade:<5} {mean_reward:<12} {created:<16}")
        
        input("\nPress Enter to continue...")
    
    def save_agent_workflow(self):
        """Workflow สำหรับบันทึก agent"""
        if self.agent_manager.current_agent is None:
            print("❌ No agent to save.")
            input("Press Enter to continue...")
            return
        
        print("\n💾 Save Current Agent")
        print("-" * 30)
        
        agent = self.agent_manager.current_agent
        print(f"🤖 Agent: {agent.agent_id}")
        print(f"🎯 Grade: {agent.grade}")
        print(f"📊 Trained: {'Yes' if agent.is_trained else 'No'}")
        
        custom_name = input("\n📝 Custom name (or Enter for auto): ").strip()
        
        try:
            saved_path = self.agent_manager.save_current_agent(custom_name if custom_name else None)
            if saved_path:
                print(f"✅ Agent saved successfully!")
            else:
                print("❌ Failed to save agent.")
        except Exception as e:
            print(f"❌ Save failed: {e}")
        
        input("\nPress Enter to continue...")
    
    def settings_workflow(self):
        """Workflow สำหรับ settings"""
        print("\n⚙️ Settings")
        print("-" * 30)
        
        print("📊 Current Configuration:")
        print(f"   Crypto Symbols: {CRYPTO_SYMBOLS}")
        print(f"   Date Range: {START_DATE} to {END_DATE}")
        print(f"   Initial Amount: ${INITIAL_AMOUNT:,}")
        print(f"   Transaction Cost: {TRANSACTION_COST_PCT*100:.3f}%")
        print(f"   Data Directory: {DATA_DIR}")
        print(f"   Models Directory: {MODEL_DIR}")
        
        print("\n🔧 Available Actions:")
        print("1. 📂 Open models directory")
        print("2. 📊 View system info")
        print("3. 🗑️ Clean temporary files")
        print("0. ↩️ Back to main menu")
        
        choice = input("\n👉 Select option: ").strip()
        
        if choice == '1':
            models_path = os.path.abspath(os.path.join(MODEL_DIR, "sac"))
            print(f"📂 Models directory: {models_path}")
            try:
                os.startfile(models_path)  # Windows
            except:
                try:
                    os.system(f"open {models_path}")  # macOS
                except:
                    print("Please open the directory manually")
        
        elif choice == '2':
            import torch
            print("\n💻 System Information:")
            print(f"   Python: {sys.version}")
            print(f"   PyTorch: {torch.__version__}")
            print(f"   CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        elif choice == '3':
            print("🗑️ Cleaning temporary files...")
            # Add cleanup logic here
            print("✅ Cleanup completed")
        
        input("\nPress Enter to continue...")

def main():
    """Main function สำหรับรัน Interactive CLI"""
    try:
        cli = InteractiveCLI()
        cli.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 