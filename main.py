# main_refactored.py - Unified Entry Point for Native Python SAC Implementation
"""
Unified Main Entry Point สำหรับ Crypto SAC Agent

คุณสมบัติหลัก:
1. Multiple operation modes (interactive, train, test, compare, legacy)
2. Command-line argument parsing
3. Integration กับ unified SAC agent และ interactive CLI
4. Grade system support
5. Legacy FinRL compatibility
6. Performance analysis และ plotting
7. Comprehensive error handling และ logging

Usage Examples:
    python main.py                              # Interactive mode
    python main.py --mode train --grade B       # Train Grade B agent
    python main.py --mode test --agent-id ABC123  # Test specific agent
    python main.py --mode legacy                # Legacy FinRL mode
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import argparse
import os
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import unified components
from crypto_agent import CryptoSACAgent, create_crypto_sac_agent, load_crypto_sac_agent
from interactive_cli import InteractiveCLI, AgentManager, DataManager
from config.config import *

# Legacy imports (optional)
try:
    import yfinance as yf
    import torch
    from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    print("⚠️ Legacy FinRL components not available. Legacy mode disabled.")

def setup_argument_parser():
    """Setup command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='Crypto SAC Agent - Unified Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s --mode train --grade B             # Train Grade B agent
  %(prog)s --mode test --agent-id ABC123      # Test specific agent
  %(prog)s --mode compare                     # Compare all agents
  %(prog)s --mode legacy                      # Legacy FinRL mode
  %(prog)s --mode train --grade A --timesteps 500000  # Custom training

Modes:
  interactive: Interactive CLI menu
  train: Train new agent
  test: Test existing agent
  compare: Compare all agents
  info: System information
  legacy: Legacy FinRL implementation
  
Grades:
  N: Novice (50K timesteps)      D: Developing (100K timesteps)
  C: Competent (200K timesteps)  B: Proficient (500K timesteps)  
  A: Advanced (1M timesteps)     S: Supreme (2M timesteps)
        """
    )
    
    # Main operation mode
    parser.add_argument(
        '--mode', 
        choices=['interactive', 'train', 'test', 'compare', 'info', 'legacy'], 
        default='interactive',
        help='Operation mode (default: interactive)'
    )
    
    # Agent configuration
    parser.add_argument(
        '--grade', 
        choices=['N', 'D', 'C', 'B', 'A', 'S'], 
        default='C',
        help='Agent grade for training (default: C)'
    )
    
    parser.add_argument(
        '--algorithm', 
        choices=['SAC', 'PPO', 'DDPG', 'TD3', 'A2C'], 
        default='SAC',
        help='RL algorithm to use (default: SAC)'
    )
    
    parser.add_argument(
        '--environment', 
        choices=['basic', 'enhanced'], 
        default='enhanced',
        help='Environment type (default: enhanced)'
    )
    
    parser.add_argument(
        '--agent-id',
        help='Agent ID for loading/testing (required for test mode)'
    )
    
    # Training parameters
    parser.add_argument(
        '--timesteps', 
        type=int,
        help='Training timesteps (overrides grade default)'
    )
    
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=10,
        help='Number of test episodes (default: 10)'
    )
    
    # Data options
    parser.add_argument(
        '--force-download', 
        action='store_true',
        help='Force download fresh data'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Crypto symbols to use (default: BTC-USD)'
    )
    
    # Output options
    parser.add_argument(
        '--save-path',
        help='Custom save path for trained agent'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count', 
        default=1,
        help='Increase verbosity (-v, -vv, -vvv)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output'
    )
    
    # Legacy options
    parser.add_argument(
        '--legacy-model',
        choices=['ppo', 'sac', 'ddpg'],
        default='ppo',
        help='Legacy model type (default: ppo)'
    )
    
    parser.add_argument(
        '--plot-results',
        action='store_true',
        help='Generate performance plots'
    )
    
    return parser

def print_banner():
    """Print application banner"""
    print("🚀 Crypto SAC Agent - Unified Implementation")
    print("=" * 60)
    print("📊 Deep Reinforcement Learning for Cryptocurrency Trading")
    print("🎯 Grade System: N, D, C, B, A, S")
    print("🤖 Algorithms: SAC, PPO, DDPG, TD3, A2C")
    print("🔄 Legacy FinRL Support Available" if LEGACY_AVAILABLE else "⚠️ Legacy FinRL Not Available")
    print("=" * 60)

def legacy_mode(model_type='ppo', symbols=None, force_download=False, plot_results=False, verbose=1):
    """Legacy FinRL mode for backward compatibility"""
    if verbose >= 1:
        print(f"\n🔄 Starting Legacy Mode - {model_type.upper()}")
    
    if not LEGACY_AVAILABLE:
        if verbose >= 1:
            print("❌ Legacy FinRL components not available")
            print("💡 Please install FinRL: pip install finrl")
        return False
    
    try:
        # Setup device
        device = setup_device()
        
        # Download data using legacy method
        df = download_crypto_data_legacy(
            force_download=force_download,
            symbols=symbols,
            verbose=verbose
        )
        
        if df is None:
            if verbose >= 1:
                print("❌ Failed to download data")
            return False
        
        # Add technical indicators (simplified)
        if verbose >= 1:
            print("📈 Adding technical indicators...")
        
        # Basic indicators
        df = df.sort_values(['tic', 'timestamp']).reset_index(drop=True)
        for symbol in df['tic'].unique():
            mask = df['tic'] == symbol
            df.loc[mask, 'sma_20'] = df.loc[mask, 'close'].rolling(20).mean()
            df.loc[mask, 'ema_20'] = df.loc[mask, 'close'].ewm(span=20).mean()
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Create FinRL environment
        if verbose >= 1:
            print("🏗️ Creating FinRL environment...")
        
        # Split data
        split_date = pd.to_datetime(START_DATE) + pd.Timedelta(days=int((pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days * 0.8))
        train_df = df[df['timestamp'] < split_date].reset_index(drop=True)
        test_df = df[df['timestamp'] >= split_date].reset_index(drop=True)
        
        if verbose >= 1:
            print(f"📊 Train data: {len(train_df)} rows")
            print(f"📊 Test data: {len(test_df)} rows")
        
        # Prepare data for FinRL
        stock_dimension = len(df['tic'].unique())
        state_space = 1 + 2 * stock_dimension + len(['sma_20', 'ema_20']) * stock_dimension
        
        env_kwargs = {
            "hmax": HMAX,
            "initial_amount": INITIAL_AMOUNT,
            "transaction_cost_pct": TRANSACTION_COST_PCT,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": ['sma_20', 'ema_20'],
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }
        
        # Create environments
        train_env = StockTradingEnv(df=train_df, **env_kwargs)
        test_env = StockTradingEnv(df=test_df, **env_kwargs)
        
        # Train model
        if verbose >= 1:
            print(f"🧠 Training {model_type.upper()} model...")
        
        agent = DRLAgent(env=train_env)
        
        if model_type.lower() == 'ppo':
            model_kwargs = {
                'learning_rate': 1e-4,
                'batch_size': 128,
                'n_steps': 1024,
                'gamma': 0.99,
                'device': device
            }
        elif model_type.lower() == 'sac':
            model_kwargs = {
                'learning_rate': 1e-4,
                'buffer_size': 100000,
                'batch_size': 256,
                'gamma': 0.99,
                'device': device
            }
        else:  # ddpg
            model_kwargs = {
                'learning_rate': 1e-4,
                'buffer_size': 100000,
                'batch_size': 128,
                'gamma': 0.99,
                'device': device
            }
        
        model = agent.get_model(model_type.lower(), model_kwargs=model_kwargs)
        
        trained_model = agent.train_model(
            model=model,
            tb_log_name=f"legacy_{model_type}",
            total_timesteps=50000
        )
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"legacy_{model_type}")
        trained_model.save(model_path)
        
        if verbose >= 1:
            print(f"💾 Model saved to {model_path}")
        
        # Test model
        if verbose >= 1:
            print("📊 Testing model...")
        
        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=trained_model,
            environment=test_env
        )
        
        # Simple analysis
        initial_value = INITIAL_AMOUNT
        final_value = df_account_value['account_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        if verbose >= 1:
            print(f"\n🎉 Legacy {model_type.upper()} completed successfully!")
            print(f"🏆 Agent achieved {total_return:.2f}% return")
            print(f"💾 Final portfolio value: ${final_value:,.2f}")
        
        # Generate plots if requested
        if plot_results:
            try:
                results = analyze_and_plot_results(
                    df_account_value, 
                    test_df, 
                    save_path=os.path.join(MODEL_DIR, f'legacy_{model_type}_performance.png'),
                    verbose=verbose
                )
            except Exception as e:
                if verbose >= 1:
                    print(f"⚠️ Plotting failed: {e}")
        
        return True
        
    except Exception as e:
        if verbose >= 1:
            print(f"❌ Legacy mode failed: {e}")
        return False

def load_and_prepare_data(symbols=None, force_download=False, verbose=1):
    """Load และเตรียมข้อมูล"""
    data_manager = DataManager()
    
    if verbose >= 1:
        print("\n📊 Loading cryptocurrency data...")
    
    try:
        # Load data
        data = data_manager.load_crypto_data(
            symbols=symbols or CRYPTO_SYMBOLS,
            force_download=force_download
        )
        
        # Add technical indicators
        data = data_manager.add_technical_indicators(data)
        
        if verbose >= 1:
            print(f"✅ Data prepared successfully!")
            print(f"   📈 Symbols: {data['tic'].unique()}")
            print(f"   📊 Rows: {len(data):,}")
            print(f"   📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            print(f"   🔢 Features: {len(data.columns)} columns")
        
        return data
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def interactive_mode(verbose=1):
    """Interactive CLI mode"""
    if verbose >= 1:
        print("\n🎮 Starting Interactive Mode...")
    
    try:
        cli = InteractiveCLI()
        cli.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting interactive mode...")
    except Exception as e:
        print(f"❌ Interactive mode error: {e}")

def train_mode(grade='C', algorithm='SAC', environment='enhanced', timesteps=None, 
               symbols=None, force_download=False, save_path=None, verbose=1):
    """Direct training mode"""
    if verbose >= 1:
        print(f"\n🏋️ Starting Training Mode")
        print(f"   🤖 Algorithm: {algorithm}")
        print(f"   🎯 Grade: {grade}")
        print(f"   🏗️ Environment: {environment}")
    
    try:
        # Load data
        data = load_and_prepare_data(symbols, force_download, verbose)
        if data is None:
            return False
        
        # Create agent based on algorithm
        if verbose >= 1:
            print(f"\n🤖 Creating {algorithm} Agent (Grade {grade})...")
        
        if algorithm == 'SAC':
            agent = create_crypto_sac_agent(grade=grade)
        elif algorithm == 'PPO':
            # For now, use SAC as base - can be extended later
            if verbose >= 1:
                print("⚠️ PPO implementation coming soon. Using SAC for now.")
            agent = create_crypto_sac_agent(grade=grade)
        else:
            # For other algorithms, use SAC as fallback
            if verbose >= 1:
                print(f"⚠️ {algorithm} implementation coming soon. Using SAC for now.")
            agent = create_crypto_sac_agent(grade=grade)
        
        # Set additional agent properties
        agent.algorithm = algorithm
        agent.environment_type = environment
        
        if verbose >= 1:
            print(f"   ID: {agent.agent_id}")
            print(f"   Timesteps: {agent.config['total_timesteps']:,}")
            print(f"   Buffer Size: {agent.config['buffer_size']:,}")
        
        # Create environment based on type
        if verbose >= 1:
            print(f"\n🏗️ Creating {environment} trading environment...")
        
        if environment == 'enhanced':
            # Use enhanced environment
            try:
                from enhanced_crypto_env import EnhancedCryptoTradingEnv
                train_env, test_env = agent.create_environment(data, env_class=EnhancedCryptoTradingEnv)
            except ImportError:
                if verbose >= 1:
                    print("⚠️ Enhanced environment not available. Using basic environment.")
                train_env, test_env = agent.create_environment(data)
        else:
            # Use basic environment
            train_env, test_env = agent.create_environment(data)
        
        # Train agent
        if verbose >= 1:
            print("\n🚀 Starting training...")
            print("⚠️ This may take several minutes...")
        
        model = agent.train(timesteps=timesteps, verbose=verbose)
        
        # Save agent
        if verbose >= 1:
            print("\n💾 Saving trained agent...")
        
        saved_path = agent.save(save_path)
        
        if verbose >= 1:
            print(f"✅ Training completed successfully!")
            print(f"   📄 Saved to: {saved_path}")
            print(f"   🤖 Agent ID: {agent.agent_id}")
            print(f"   🎯 Algorithm: {algorithm}")
            print(f"   🏗️ Environment: {environment}")
        
        # Quick evaluation
        if verbose >= 1:
            print("\n🧪 Quick evaluation...")
        
        try:
            results = agent.evaluate(n_episodes=5, verbose=0)
            if verbose >= 1:
                print(f"   🎯 Mean Reward: {results['mean_reward']:.4f}")
                print(f"   🏆 Best Reward: {results['max_reward']:.4f}")
        except Exception as e:
            if verbose >= 1:
                print(f"   ⚠️ Evaluation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def test_mode(agent_id, episodes=10, verbose=1):
    """Testing mode"""
    if verbose >= 1:
        print(f"\n🧪 Starting Test Mode...")
        print(f"   🤖 Agent ID: {agent_id}")
        print(f"   📊 Episodes: {episodes}")
    
    try:
        # Find agent file
        agent_manager = AgentManager()
        agents = agent_manager.list_available_agents()
        
        # Find matching agent
        matching_agent = None
        for agent_info in agents:
            if agent_id in agent_info['name'] or agent_info['name'] == agent_id:
                matching_agent = agent_info
                break
        
        if matching_agent is None:
            print(f"❌ Agent not found: {agent_id}")
            print("📋 Available agents:")
            for agent_info in agents:
                print(f"   - {agent_info['name']}")
            return False
        
        # Load agent
        if verbose >= 1:
            print(f"\n📊 Loading agent: {matching_agent['name']}")
        
        agent = agent_manager.load_agent(matching_agent['name'])
        if agent is None:
            return False
        
        # Check if agent is trained
        if not agent.is_trained:
            print("❌ Agent is not trained")
            return False
        
        # Load data for testing
        data = load_and_prepare_data(verbose=verbose)
        if data is None:
            return False
        
        # Create environment if needed
        if agent.test_env is None:
            if verbose >= 1:
                print("🏗️ Creating test environment...")
            agent.create_environment(data)
        
        # Evaluate agent
        if verbose >= 1:
            print(f"\n🔍 Evaluating agent...")
        
        results = agent.evaluate(n_episodes=episodes, verbose=verbose)
        
        if verbose >= 1:
            print(f"\n📊 Test Results:")
            print(f"   🎯 Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
            print(f"   🏆 Best Reward: {results['max_reward']:.4f}")
            print(f"   📉 Worst Reward: {results['min_reward']:.4f}")
            print(f"   📏 Avg Episode Length: {results['mean_episode_length']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

def compare_mode(verbose=1):
    """Comparison mode"""
    if verbose >= 1:
        print("\n🔍 Starting Comparison Mode...")
    
    try:
        agent_manager = AgentManager()
        agents = agent_manager.list_available_agents()
        
        if len(agents) == 0:
            print("❌ No agents found for comparison")
            return False
        
        if verbose >= 1:
            print(f"📋 Found {len(agents)} agents")
        
        # Display comparison table
        print(f"\n📊 Agent Comparison:")
        print(f"{'Name':<35} {'Grade':<5} {'Mean Reward':<12} {'Created':<16}")
        print("-" * 75)
        
        for agent in agents:
            name = agent['name'][:33] + ".." if len(agent['name']) > 35 else agent['name']
            grade = agent['grade']
            
            mean_reward = "N/A"
            if agent['performance']:
                mr = agent['performance'].get('mean_reward')
                if mr is not None:
                    mean_reward = f"{mr:.4f}"
            
            created = agent['created_date'].strftime('%Y-%m-%d %H:%M')
            print(f"{name:<35} {grade:<5} {mean_reward:<12} {created:<16}")
        
        # Summary by grade
        if verbose >= 1:
            print(f"\n📈 Summary by Grade:")
            grade_counts = {}
            for agent in agents:
                grade = agent['grade']
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
            
            for grade in ['N', 'D', 'C', 'B', 'A', 'S']:
                count = grade_counts.get(grade, 0)
                if count > 0:
                    print(f"   Grade {grade}: {count} agents")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

def info_mode(verbose=1):
    """Information mode"""
    if verbose >= 1:
        print("\n📋 System Information")
        print("-" * 30)
    
    try:
        import torch
        import pandas as pd
        import numpy as np
        
        print(f"🐍 Python: {sys.version}")
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"📊 Pandas: {pd.__version__}")
        print(f"🔢 NumPy: {np.__version__}")
        print(f"🎯 CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Project info
        print(f"\n📁 Project Directories:")
        print(f"   Data: {DATA_DIR}")
        print(f"   Models: {MODEL_DIR}")
        print(f"   SAC Models: {os.path.join(MODEL_DIR, 'sac')}")
        
        # Agent count
        agent_manager = AgentManager()
        agents = agent_manager.list_available_agents()
        print(f"\n🤖 Saved Agents: {len(agents)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Info failed: {e}")
        return False

def main():
    """Main function"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set verbosity
    if args.quiet:
        verbose = 0
    else:
        verbose = args.verbose
    
    # Print banner
    if verbose >= 1:
        print_banner()
        print(f"🎮 Mode: {args.mode}")
        if args.mode != 'interactive':
            print(f"⚙️ Arguments: {vars(args)}")
    
    # Execute based on mode
    success = False
    
    try:
        if args.mode == 'interactive':
            interactive_mode(verbose)
            success = True
            
        elif args.mode == 'train':
            success = train_mode(
                grade=args.grade,
                algorithm=args.algorithm,
                environment=args.environment,
                timesteps=args.timesteps,
                symbols=args.symbols,
                force_download=args.force_download,
                save_path=args.save_path,
                verbose=verbose
            )
            
        elif args.mode == 'test':
            if not args.agent_id:
                print("❌ Agent ID required for test mode. Use --agent-id")
                parser.print_help()
                sys.exit(1)
            
            success = test_mode(
                agent_id=args.agent_id,
                episodes=args.episodes,
                verbose=verbose
            )
            
        elif args.mode == 'compare':
            success = compare_mode(verbose)
            
        elif args.mode == 'info':
            success = info_mode(verbose)
        
        elif args.mode == 'legacy':
            success = legacy_mode(
                model_type=args.legacy_model,
                symbols=args.symbols,
                force_download=args.force_download,
                plot_results=args.plot_results,
                verbose=verbose
            )
        
        # Exit with appropriate code
        if success:
            if verbose >= 1:
                print(f"\n🎉 {args.mode.title()} mode completed successfully!")
            sys.exit(0)
        else:
            if verbose >= 1:
                print(f"\n❌ {args.mode.title()} mode failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        if verbose >= 1:
            print("\n\n👋 Operation cancelled by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if verbose >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 