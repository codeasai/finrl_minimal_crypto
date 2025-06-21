# main_refactored.py - Unified Entry Point for Native Python SAC Implementation
"""
Refactored Main Entry Point สำหรับ Crypto SAC Agent

คุณสมบัติหลัก:
1. Multiple operation modes (interactive, train, test, compare)
2. Command-line argument parsing
3. Integration กับ unified SAC agent และ interactive CLI
4. Grade system support
5. Comprehensive error handling และ logging

Usage Examples:
    python main_refactored.py                    # Interactive mode
    python main_refactored.py --mode train --grade B
    python main_refactored.py --mode test --agent-id ABC123
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import argparse
import os
from datetime import datetime
from typing import Optional, Dict, Any

# Import unified components
from crypto_agent import CryptoSACAgent, create_crypto_sac_agent, load_crypto_sac_agent
from interactive_cli import InteractiveCLI, AgentManager, DataManager
from config import *

def setup_argument_parser():
    """Setup command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='Crypto SAC Agent - Native Python Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s --mode train --grade B             # Train Grade B agent
  %(prog)s --mode test --agent-id ABC123      # Test specific agent
  %(prog)s --mode compare                     # Compare all agents
  %(prog)s --mode train --grade A --timesteps 500000  # Custom training

Grades:
  N: Novice (50K timesteps)      D: Developing (100K timesteps)
  C: Competent (200K timesteps)  B: Proficient (500K timesteps)  
  A: Advanced (1M timesteps)     S: Supreme (2M timesteps)
        """
    )
    
    # Main operation mode
    parser.add_argument(
        '--mode', 
        choices=['interactive', 'train', 'test', 'compare', 'info'], 
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
    
    return parser

def print_banner():
    """Print application banner"""
    print("🚀 Crypto SAC Agent - Native Python Implementation")
    print("=" * 60)
    print("📊 Deep Reinforcement Learning for Cryptocurrency Trading")
    print("🎯 Grade System: N, D, C, B, A, S")
    print("🤖 Algorithm: SAC (Soft Actor-Critic)")
    print("=" * 60)

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

def train_mode(grade='C', timesteps=None, symbols=None, force_download=False, 
               save_path=None, verbose=1):
    """Direct training mode"""
    if verbose >= 1:
        print(f"\n🏋️ Starting Training Mode (Grade {grade})...")
    
    try:
        # Load data
        data = load_and_prepare_data(symbols, force_download, verbose)
        if data is None:
            return False
        
        # Create agent
        if verbose >= 1:
            print(f"\n🤖 Creating SAC Agent (Grade {grade})...")
        
        agent = create_crypto_sac_agent(grade=grade)
        
        if verbose >= 1:
            print(f"   ID: {agent.agent_id}")
            print(f"   Timesteps: {agent.config['total_timesteps']:,}")
            print(f"   Buffer Size: {agent.config['buffer_size']:,}")
        
        # Create environment
        if verbose >= 1:
            print("\n🏗️ Creating trading environment...")
        
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