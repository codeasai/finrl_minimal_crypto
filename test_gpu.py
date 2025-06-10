import torch
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
root_path = Path(__file__).parent
sys.path.append(str(root_path))

# Import FinRL modules
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def setup_cuda():
    """ตั้งค่า CUDA และตรวจสอบการใช้งาน GPU"""
    try:
        if torch.cuda.is_available():
            # ตั้งค่า device เป็น CUDA
            device = torch.device("cuda")
            # ล้างหน่วยความจำ GPU
            torch.cuda.empty_cache()
            # ตั้งค่า default tensor type เป็น float32 บน CUDA
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            return device
        else:
            return torch.device("cpu")
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการตั้งค่า CUDA: {str(e)}")
        return torch.device("cpu")

def calculate_technical_indicators(df):
    """คำนวณ technical indicators"""
    try:
        # คำนวณ SMA
        df['sma_20'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).mean())
        
        # คำนวณ EMA
        df['ema_20'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        
        # คำนวณ RSI
        def calculate_rsi(data, periods=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = df.groupby('tic')['close'].transform(calculate_rsi)
        
        # เพิ่ม features อื่นๆ
        df['macd'] = df.groupby('tic')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean())
        df['macd_signal'] = df.groupby('tic')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).mean())
        df['bb_std'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).std())
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Momentum
        df['momentum'] = df.groupby('tic')['close'].transform(lambda x: x.pct_change(periods=10))
        
        # Volatility
        df['volatility'] = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=20).std())
        
        # Volume indicators
        df['volume_sma'] = df.groupby('tic')['volume'].transform(lambda x: x.rolling(window=20).mean())
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการคำนวณ technical indicators: {str(e)}")
        return df

def normalize_data(df):
    """Normalize ข้อมูลและจัดการกับค่า NaN"""
    try:
        # เลือกคอลัมน์ที่ต้องการ normalize
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume']
        tech_cols = [col for col in df.columns if col not in ['date', 'tic'] + price_cols + volume_cols]
        
        # Normalize ราคา
        for col in price_cols:
            df[col] = df.groupby('tic')[col].transform(lambda x: (x - x.mean()) / x.std())
        
        # Normalize volume
        for col in volume_cols:
            df[col] = df.groupby('tic')[col].transform(lambda x: (x - x.mean()) / x.std())
        
        # Normalize technical indicators
        for col in tech_cols:
            df[col] = df.groupby('tic')[col].transform(lambda x: (x - x.mean()) / x.std())
        
        # แทนที่ค่า NaN ด้วย 0
        df = df.fillna(0)
        
        return df
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการ normalize ข้อมูล: {str(e)}")
        return df

def test_gpu_availability():
    """ทดสอบการใช้งาน GPU"""
    print("\n=== 🎮 GPU Availability Test ===")
    
    try:
        # ตรวจสอบ CUDA
        cuda_available = torch.cuda.is_available()
        print(f"\n1. CUDA Available: {'✅' if cuda_available else '❌'}")
        
        if cuda_available:
            # จำนวน GPU
            device_count = torch.cuda.device_count()
            print(f"\n2. Number of GPUs: {device_count}")
            
            # ข้อมูลแต่ละ GPU
            print("\n3. GPU Information:")
            for i in range(device_count):
                print(f"\n   GPU {i+1}:")
                print(f"   - Name: {torch.cuda.get_device_name(i)}")
                print(f"   - Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
                print(f"   - Memory Reserved: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
            
            # CUDA Version
            print(f"\n4. CUDA Version: {torch.version.cuda}")
            print(f"5. PyTorch Version: {torch.__version__}")
            
            # ตั้งค่า CUDA
            device = setup_cuda()
            print(f"\n6. Using device: {device}")
        else:
            print("\n❌ ไม่พบ GPU ที่สามารถใช้งานได้")
            print("   - ตรวจสอบการติดตั้ง CUDA")
            print("   - ตรวจสอบการติดตั้ง GPU drivers")
            print("   - ตรวจสอบการเชื่อมต่อ GPU")
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการตรวจสอบ GPU: {str(e)}")

def test_tensor_operations():
    """ทดสอบการทำงานกับ Tensor บน GPU"""
    print("\n=== 🔢 Tensor Operations Test ===")
    
    try:
        if not torch.cuda.is_available():
            print("❌ ข้ามการทดสอบ Tensor Operations เนื่องจากไม่พบ GPU")
            return
        
        # ตั้งค่า CUDA
        device = setup_cuda()
        
        # สร้าง tensor ขนาดใหญ่
        size = 1000
        print(f"\n1. สร้าง Tensor ขนาด {size}x{size}")
        
        # สร้างบน CPU
        start_time = datetime.now()
        cpu_tensor = torch.randn(size, size)
        cpu_time = (datetime.now() - start_time).total_seconds()
        print(f"   - CPU Time: {cpu_time:.4f} seconds")
        
        # สร้างบน GPU
        start_time = datetime.now()
        gpu_tensor = torch.randn(size, size, device=device)
        gpu_time = (datetime.now() - start_time).total_seconds()
        print(f"   - GPU Time: {gpu_time:.4f} seconds")
        
        # ทดสอบการคำนวณ
        print("\n2. ทดสอบการคำนวณ Matrix Multiplication")
        
        # บน CPU
        start_time = datetime.now()
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = (datetime.now() - start_time).total_seconds()
        print(f"   - CPU Time: {cpu_time:.4f} seconds")
        
        # บน GPU
        start_time = datetime.now()
        gpu_result = torch.matmul(gpu_tensor, gpu_tensor)
        gpu_time = (datetime.now() - start_time).total_seconds()
        print(f"   - GPU Time: {gpu_time:.4f} seconds")
        
        # เปรียบเทียบความเร็ว
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\n3. GPU Speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการทดสอบ Tensor Operations: {str(e)}")

def test_stable_baselines3():
    """ทดสอบการทำงานกับ Stable-Baselines3"""
    print("\n=== 🤖 Stable-Baselines3 Test ===")
    
    try:
        if not torch.cuda.is_available():
            print("❌ ข้ามการทดสอบ Stable-Baselines3 เนื่องจากไม่พบ GPU")
            return
        
        # ตั้งค่า CUDA
        device = setup_cuda()
        
        # สร้างข้อมูลจำลอง
        print("\n1. สร้างข้อมูลจำลอง")
        df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100),
            'tic': ['BTC'] * 100,
            'open': np.random.randn(100) + 100,  # เพิ่มค่าเฉลี่ย
            'high': np.random.randn(100) + 100,
            'low': np.random.randn(100) + 100,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randn(100) + 1000  # เพิ่มค่าเฉลี่ย
        })
        
        # คำนวณ technical indicators
        df = calculate_technical_indicators(df)
        
        # Normalize ข้อมูล
        df = normalize_data(df)
        
        # กำหนด technical indicators ที่จะใช้
        tech_indicator_list = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'momentum', 'volatility',
            'volume_ratio'
        ]
        
        # สร้าง environment จำลอง
        print("\n2. สร้าง Environment จำลอง")
        env = DummyVecEnv([lambda: StockTradingEnv(
            df=df,
            hmax=100,
            initial_amount=10000,
            num_stock_shares=[0],
            buy_cost_pct=[0.001],
            sell_cost_pct=[0.001],
            state_space=15,  # 5 (OHLCV) + 10 (technical indicators)
            stock_dim=1,
            tech_indicator_list=tech_indicator_list,
            action_space=1,
            reward_scaling=1e-4
        )])
        
        # สร้าง agent
        print("\n3. สร้าง PPO Agent")
        agent = DRLAgent(env=env)
        
        # กำหนดพารามิเตอร์
        PPO_PARAMS = {
            'learning_rate': 5e-5,  # ลด learning rate ลงอีก
            'n_steps': 2048,
            'batch_size': 128,  # ลด batch size
            'n_epochs': 20,  # เพิ่มจำนวน epochs
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.1,  # ลด clip range
            'max_grad_norm': 0.3,  # ลด max gradient norm
            'ent_coef': 0.005,  # ลด entropy coefficient
            'vf_coef': 0.5,
            'target_kl': 0.01,  # ลด target KL divergence
            'device': device  # ระบุ device ที่ใช้
        }
        
        # สร้างโมเดล
        print("\n4. สร้างโมเดล PPO")
        model = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        
        # ทดสอบการเทรน
        print("\n5. ทดสอบการเทรน")
        start_time = datetime.now()
        model.learn(total_timesteps=1000)
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"   - เวลาที่ใช้ในการเทรน: {training_time:.2f} seconds")
        
        # ตรวจสอบ device ที่ใช้
        print(f"\n6. Device ที่ใช้: {model.device}")
        
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการทดสอบ Stable-Baselines3: {str(e)}")

def test_finrl():
    """ทดสอบการทำงานกับ FinRL"""
    print("\n=== 📈 FinRL Test ===")
    
    try:
        if not torch.cuda.is_available():
            print("❌ ข้ามการทดสอบ FinRL เนื่องจากไม่พบ GPU")
            return
        
        # ตั้งค่า CUDA
        device = setup_cuda()
        
        # สร้างข้อมูลจำลอง
        print("\n1. สร้างข้อมูลจำลอง")
        df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100),
            'tic': ['BTC'] * 100,
            'open': np.random.randn(100) + 100,  # เพิ่มค่าเฉลี่ย
            'high': np.random.randn(100) + 100,
            'low': np.random.randn(100) + 100,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randn(100) + 1000  # เพิ่มค่าเฉลี่ย
        })
        
        # คำนวณ technical indicators
        print("\n2. คำนวณ Technical Indicators")
        df = calculate_technical_indicators(df)
        
        # Normalize ข้อมูล
        df = normalize_data(df)
        
        # กำหนด technical indicators ที่จะใช้
        tech_indicator_list = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'momentum', 'volatility',
            'volume_ratio'
        ]
        
        # สร้าง environment
        print("\n3. สร้าง Trading Environment")
        env = StockTradingEnv(
            df=df,
            hmax=100,
            initial_amount=10000,
            num_stock_shares=[0],
            buy_cost_pct=[0.001],
            sell_cost_pct=[0.001],
            state_space=15,  # 5 (OHLCV) + 10 (technical indicators)
            stock_dim=1,
            tech_indicator_list=tech_indicator_list,
            action_space=1,
            reward_scaling=1e-4
        )
        
        # สร้าง agent
        print("\n4. สร้าง DRL Agent")
        agent = DRLAgent(env=env)
        
        # ทดสอบการเทรน
        print("\n5. ทดสอบการเทรน")
        start_time = datetime.now()
        model = agent.get_model("ppo", model_kwargs={
            'learning_rate': 5e-5,  # ลด learning rate ลงอีก
            'n_steps': 2048,
            'batch_size': 128,  # ลด batch size
            'n_epochs': 20,  # เพิ่มจำนวน epochs
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.1,  # ลด clip range
            'max_grad_norm': 0.3,  # ลด max gradient norm
            'ent_coef': 0.005,  # ลด entropy coefficient
            'vf_coef': 0.5,
            'target_kl': 0.01,  # ลด target KL divergence
            'device': device  # ระบุ device ที่ใช้
        })
        model.learn(total_timesteps=1000)
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"   - เวลาที่ใช้ในการเทรน: {training_time:.2f} seconds")
        
        # ทดสอบการทำนาย
        print("\n6. ทดสอบการทำนาย")
        df_account_value, df_actions = agent.DRL_prediction(model=model, environment=env)
        print(f"   - จำนวนการทำนาย: {len(df_account_value)}")
        
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการทดสอบ FinRL: {str(e)}")

def test_memory_management():
    """ทดสอบการจัดการหน่วยความจำ GPU"""
    print("\n=== 💾 GPU Memory Management Test ===")
    
    try:
        if not torch.cuda.is_available():
            print("❌ ข้ามการทดสอบ Memory Management เนื่องจากไม่พบ GPU")
            return
        
        # ตั้งค่า CUDA
        device = setup_cuda()
        
        # ตรวจสอบหน่วยความจำเริ่มต้น
        initial_memory = torch.cuda.memory_allocated()
        print(f"\n1. หน่วยความจำเริ่มต้น: {initial_memory/1024**2:.2f} MB")
        
        # สร้าง tensor ขนาดใหญ่
        print("\n2. สร้าง Tensor ขนาดใหญ่")
        large_tensor = torch.randn(1000, 1000, device=device)
        memory_after_alloc = torch.cuda.memory_allocated()
        print(f"   - หน่วยความจำหลังสร้าง Tensor: {memory_after_alloc/1024**2:.2f} MB")
        
        # ลบ tensor
        print("\n3. ลบ Tensor")
        del large_tensor
        torch.cuda.empty_cache()
        memory_after_delete = torch.cuda.memory_allocated()
        print(f"   - หน่วยความจำหลังลบ Tensor: {memory_after_delete/1024**2:.2f} MB")
        
        # ตรวจสอบการคืนหน่วยความจำ
        memory_recovered = memory_after_alloc - memory_after_delete
        print(f"\n4. หน่วยความจำที่คืนมา: {memory_recovered/1024**2:.2f} MB")
        
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการทดสอบ Memory Management: {str(e)}")

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    try:
        print("=== 🚀 Starting GPU Test Suite ===")
        
        # ตั้งค่า CUDA
        device = setup_cuda()
        print(f"\nUsing device: {device}")
        
        # ทดสอบการใช้งาน GPU
        test_gpu_availability()
        
        # ทดสอบการทำงานกับ Tensor
        test_tensor_operations()
        
        # ทดสอบการทำงานกับ Stable-Baselines3
        test_stable_baselines3()
        
        # ทดสอบการทำงานกับ FinRL
        test_finrl()
        
        # ทดสอบการจัดการหน่วยความจำ
        test_memory_management()
        
        print("\n=== ✅ GPU Test Suite Completed ===")
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการทดสอบ: {str(e)}")

if __name__ == "__main__":
    main() 