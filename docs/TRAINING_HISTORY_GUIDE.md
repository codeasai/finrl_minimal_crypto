# คู่มือการดูประวัติการ Train และประสิทธิภาพ Model

> คู่มือครบถ้วนสำหรับการตรวจสอบ Training History และ Performance ของ Models ในโปรเจค finrl_minimal_crypto

## 📋 สารบัญ

1. [ระบบ Tracking ที่มีอยู่](#ระบบ-tracking-ที่มีอยู่)
2. [การดู Training Logs](#การดู-training-logs)
3. [การดู Model Performance](#การดู-model-performance)
4. [การใช้ Tensorboard](#การใช้-tensorboard)
5. [การดู Saved Models](#การดู-saved-models)
6. [การ Compare Models](#การ-compare-models)
7. [การ Export และ Analysis](#การ-export-และ-analysis)
8. [Advanced Monitoring](#advanced-monitoring)

---

## 🎯 ระบบ Tracking ที่มีอยู่

### โครงสร้างไฟล์ระบบติดตาม

```
finrl_minimal_crypto/
├── logs/                           # Training logs
│   ├── sac_enhanced_env/          # Enhanced environment logs
│   ├── sac_original_env/          # Original environment logs  
│   ├── sac_eval_grade_A/          # Grade A evaluation logs
│   └── sac_graded/               # Graded training logs
├── models/                        # Saved models
│   ├── sac/                      # SAC models
│   │   ├── metadata/             # Model metadata
│   │   ├── best_enhanced_env/    # Best enhanced models
│   │   ├── best_original_env/    # Best original models
│   │   ├── *.zip                 # Model files
│   │   ├── *_info.pkl           # Model information
│   │   └── *.png                # Performance charts
│   └── *.zip                     # Other models (PPO, etc.)
└── ui/                           # Streamlit UI
    └── pages/6_Manage_Agents.py  # Model management
```

### ระบบ Metadata Management

โปรเจคใช้ **SAC Metadata Manager** สำหรับติดตาม:
- ✅ Configuration และ hyperparameters
- ✅ Training history และ performance metrics
- ✅ Evaluation results และ backtest data
- ✅ Grade-based organization (N, D, C, B, A, S)
- ✅ Model versioning และ timestamps

---

## 📊 การดู Training Logs

### 1. วิธีการเร็ว - ใช้ test_sac_results.py

```bash
python test_sac_results.py
```

**ผลลัพธ์ที่ได้:**
```
🔍 ตรวจสอบผลลัพธ์ SAC Agent
==================================================
📁 โหลดข้อมูลจาก: sac_agent_20250619_151128_XXBF8G_info.pkl

📊 ข้อมูล SAC Agent:
------------------------------
🔤 ชื่อ Model: sac_agent_20250619_151128_XXBF8G
🤖 Algorithm: SAC
📅 วันที่สร้าง: 2025-06-19 15:11:28
💰 เงินเริ่มต้น: $100,000
💸 ค่าธรรมเนียม: 0.1%
📈 สัญลักษณ์: ['BTC-USD']
```

### 2. ดูโฟลเดอร์ logs ทั้งหมด

```bash
# ดู logs directories
ls -la logs/

# ดูรายละเอียด logs
ls -la logs/sac_enhanced_env/
ls -la logs/sac_original_env/
```

### 3. สร้างสคริปต์ดู logs แบบละเอียด

```python
# view_training_logs.py
import os
import json
from datetime import datetime

def view_training_logs():
    """ดู training logs ทั้งหมด"""
    
    print("📊 Training Logs Overview")
    print("=" * 50)
    
    logs_dir = "logs"
    
    for subdir in os.listdir(logs_dir):
        subdir_path = os.path.join(logs_dir, subdir)
        if os.path.isdir(subdir_path):
            print(f"\n📁 {subdir}/")
            
            # นับไฟล์และแสดงขนาด
            total_files = 0
            total_size = 0
            
            for root, dirs, files in os.walk(subdir_path):
                total_files += len(files)
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            
            print(f"  📄 Files: {total_files}")
            print(f"  💾 Total Size: {total_size / (1024*1024):.1f} MB")
            
            # แสดงไฟล์ล่าสุด
            latest_files = []
            for root, dirs, files in os.walk(subdir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    mod_time = os.path.getmtime(file_path)
                    latest_files.append((file, mod_time))
            
            if latest_files:
                latest_files.sort(key=lambda x: x[1], reverse=True)
                latest_file, latest_time = latest_files[0]
                latest_datetime = datetime.fromtimestamp(latest_time)
                print(f"  🕐 Latest: {latest_file} ({latest_datetime.strftime('%Y-%m-%d %H:%M')})")

if __name__ == "__main__":
    view_training_logs()
```

---

## 🤖 การดู Model Performance

### 1. ใช้ SAC Metadata Manager

```python
# view_model_performance.py
from models.sac.sac_metadata_manager import SAC_MetadataManager

def view_all_models():
    """ดูประสิทธิภาพ models ทั้งหมด"""
    
    manager = SAC_MetadataManager()
    manager.load_all_metadata()
    
    agents = manager.list_agents()
    
    print("🤖 SAC Models Overview")
    print("=" * 50)
    print(f"Total Models: {len(agents)}")
    
    if not agents:
        print("❌ ไม่พบ SAC models")
        return
    
    # แสดงสถิติตาม grade
    grade_stats = manager.get_grade_statistics()
    
    print(f"\n📊 Statistics by Grade:")
    print("-" * 30)
    
    for grade, stats in grade_stats.items():
        print(f"Grade {grade}: {stats['count']} models")
        print(f"  📈 Mean Reward: {stats['mean_reward']:.4f}")
        print(f"  🏆 Best Reward: {stats['best_reward']:.4f}")
        print(f"  ✅ Success Rate: {stats['success_rate']:.2%}")
        print(f"  ⏱️  Avg Duration: {stats['avg_training_duration']:.0f}s")
        print()
    
    # แสดง models ที่ดีที่สุด
    print(f"🏅 Best Models:")
    print("-" * 30)
    
    best_agents = manager.get_best_agents_by_grade(top_n=2)
    
    for grade, agents_list in best_agents.items():
        if agents_list:
            print(f"Grade {grade}:")
            for i, agent in enumerate(agents_list, 1):
                reward = agent.performance_metrics.get('mean_reward', 0)
                duration = agent.get_training_duration_formatted()
                print(f"  {i}. {agent.agent_id[:20]}...")
                print(f"     Reward: {reward:.4f}, Duration: {duration}")
            print()

if __name__ == "__main__":
    view_all_models()
```

### 2. ดู Performance Charts

```python
# view_charts.py
import os
import subprocess
from datetime import datetime

def view_performance_charts():
    """แสดง performance charts"""
    
    charts_dir = "models/sac"
    chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
    
    print("📊 Available Performance Charts")
    print("=" * 40)
    
    if not chart_files:
        print("❌ ไม่พบ performance charts")
        return
    
    for i, chart_file in enumerate(chart_files, 1):
        file_path = os.path.join(charts_dir, chart_file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        print(f"{i:2d}. {chart_file}")
        print(f"     Size: {file_size:.1f} KB")
        print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # เปิดดู chart ล่าสุด
    if chart_files:
        latest_chart = max(chart_files, 
                          key=lambda x: os.path.getmtime(os.path.join(charts_dir, x)))
        
        print(f"\n🖼️  Opening latest chart: {latest_chart}")
        chart_path = os.path.join(charts_dir, latest_chart)
        
        # เปิดด้วย default image viewer
        try:
            subprocess.run(['start', chart_path], shell=True)
        except:
            print(f"📁 Chart location: {chart_path}")

if __name__ == "__main__":
    view_performance_charts()
```

---

## 📈 การใช้ Tensorboard

### 1. เริ่ม Tensorboard

```bash
# รัน tensorboard (พื้นฐาน)
tensorboard --logdir=logs

# รันแบบระบุ port
tensorboard --logdir=logs --port=6006

# เปิดใน browser
start http://localhost:6006
```

### 2. ดู Metrics ใน Tensorboard

**Metrics ที่สามารถดูได้:**
- 📊 **rollout/ep_rew_mean**: Average episode reward
- 📈 **train/learning_rate**: Learning rate over time
- 🧠 **train/policy_loss**: Policy network loss
- 🔍 **train/value_loss**: Value network loss
- 📉 **train/entropy_loss**: Entropy loss

### 3. Compare หลาย Experiments

```python
# organize_tensorboard.py
import os

def organize_tensorboard_runs():
    """จัดระเบียบ tensorboard logs สำหรับ comparison"""
    
    base_dir = "logs"
    comparison_dir = "logs/comparison"
    
    # สร้าง comparison directory
    os.makedirs(comparison_dir, exist_ok=True)
    
    # สร้าง symbolic links
    experiments = {
        "Enhanced_SAC": "sac_enhanced_env",
        "Original_SAC": "sac_original_env",
        "Grade_A_SAC": "sac_eval_grade_A"
    }
    
    for exp_name, source_dir in experiments.items():
        source_path = os.path.join(base_dir, source_dir)
        target_path = os.path.join(comparison_dir, exp_name)
        
        if os.path.exists(source_path) and not os.path.exists(target_path):
            try:
                os.symlink(os.path.abspath(source_path), target_path)
                print(f"✅ Linked {exp_name}")
            except:
                print(f"❌ Failed to link {exp_name}")
    
    print(f"\n🚀 Run comparison with:")
    print(f"tensorboard --logdir={comparison_dir}")

if __name__ == "__main__":
    organize_tensorboard_runs()
```

---

## 💾 การดู Saved Models

### 1. แสดงรายการ Models

```python
# list_models.py
import os
import pickle
from datetime import datetime

def list_all_models():
    """แสดงรายการ models ทั้งหมด"""
    
    print("🤖 All Saved Models")
    print("=" * 60)
    
    # SAC Models
    sac_dir = "models/sac"
    if os.path.exists(sac_dir):
        print("\n🧠 SAC Models:")
        print("-" * 30)
        
        sac_files = [f for f in os.listdir(sac_dir) if f.endswith('.zip')]
        sac_files.sort(key=lambda x: os.path.getmtime(os.path.join(sac_dir, x)), reverse=True)
        
        for i, model_file in enumerate(sac_files, 1):
            model_path = os.path.join(sac_dir, model_file)
            file_size = os.path.getsize(model_path) / (1024*1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            print(f"{i:2d}. {model_file}")
            print(f"     Size: {file_size:.1f} MB")
            print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # ดู metadata ถ้ามี
            info_file = model_file.replace('.zip', '_info.pkl')
            info_path = os.path.join(sac_dir, info_file)
            
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'rb') as f:
                        info = pickle.load(f)
                    print(f"     Created: {info.get('created_date', 'N/A')[:19]}")
                    print(f"     Symbols: {info.get('crypto_symbols', 'N/A')}")
                except:
                    pass
            print()
    
    # PPO Models
    models_dir = "models"
    ppo_files = [f for f in os.listdir(models_dir) 
                 if f.endswith('.zip') and ('ppo' in f.lower() or 'PPO' in f)]
    
    if ppo_files:
        print("🏃 PPO Models:")
        print("-" * 30)
        
        for i, model_file in enumerate(ppo_files, 1):
            model_path = os.path.join(models_dir, model_file)
            file_size = os.path.getsize(model_path) / (1024*1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            print(f"{i:2d}. {model_file}")
            print(f"     Size: {file_size:.1f} MB")
            print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()

if __name__ == "__main__":
    list_all_models()
```

### 2. โหลดและทดสอบ Model

```python
# test_model.py
from stable_baselines3 import SAC
import os
import pickle

def load_and_test_model(model_name=None):
    """โหลดและทดสอบ model"""
    
    sac_dir = "models/sac"
    
    if model_name is None:
        # หา model ล่าสุด
        sac_files = [f for f in os.listdir(sac_dir) if f.endswith('.zip')]
        if not sac_files:
            print("❌ ไม่พบ SAC models")
            return None
        
        latest_file = max(sac_files, 
                         key=lambda x: os.path.getmtime(os.path.join(sac_dir, x)))
        model_name = latest_file.replace('.zip', '')
    
    model_path = os.path.join(sac_dir, f"{model_name}.zip")
    info_path = os.path.join(sac_dir, f"{model_name}_info.pkl")
    
    print(f"🤖 Loading Model: {model_name}")
    print("=" * 50)
    
    # ดู model info
    if os.path.exists(info_path):
        try:
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
            
            print("📊 Model Information:")
            print(f"  Algorithm: {info.get('algorithm', 'N/A')}")
            print(f"  Created: {info.get('created_date', 'N/A')}")
            print(f"  Initial Amount: ${info.get('initial_amount', 0):,}")
            print(f"  Crypto Symbols: {info.get('crypto_symbols', 'N/A')}")
            print(f"  Train Data: {info.get('train_data_shape', 'N/A')}")
            print(f"  Test Data: {info.get('test_data_shape', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️  Cannot read model info: {e}")
    
    # โหลด model
    if os.path.exists(model_path):
        try:
            model = SAC.load(model_path)
            print(f"\n✅ Model loaded successfully!")
            
            print(f"\nModel Details:")
            print(f"  Policy: {model.policy.__class__.__name__}")
            print(f"  Learning Rate: {model.learning_rate}")
            print(f"  Buffer Size: {model.buffer_size:,}")
            print(f"  Batch Size: {model.batch_size}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    else:
        print(f"❌ Model file not found: {model_path}")
        return None

if __name__ == "__main__":
    # ใช้ model ล่าสุด
    model = load_and_test_model()
    
    # หรือระบุ model name
    # model = load_and_test_model("sac_agent_20250619_151128_XXBF8G")
```

---

## 🆚 การ Compare Models

### 1. ใช้ SAC Metadata Manager

```python
# compare_models.py
from models.sac.sac_metadata_manager import SAC_MetadataManager
import pandas as pd
import matplotlib.pyplot as plt

def compare_models():
    """เปรียบเทียบ models"""
    
    manager = SAC_MetadataManager()
    manager.load_all_metadata()
    
    agents = manager.list_agents()
    
    if len(agents) < 2:
        print("❌ ต้องมีอย่างน้อย 2 models เพื่อเปรียบเทียบ")
        return
    
    print("🆚 Model Comparison")
    print("=" * 60)
    
    # สร้าง comparison table
    comparison_df = manager.get_performance_comparison()
    
    if comparison_df.empty:
        print("❌ ไม่มีข้อมูลสำหรับเปรียบเทียบ")
        return
    
    print("\n📊 Performance Comparison:")
    print(comparison_df.to_string())
    
    # หา top performers
    if 'mean_reward' in comparison_df.columns:
        top_performers = comparison_df.nlargest(3, 'mean_reward')
        
        print(f"\n🏆 Top 3 Performers:")
        print("-" * 30)
        for i, (agent_id, row) in enumerate(top_performers.iterrows(), 1):
            print(f"{i}. {agent_id}")
            print(f"   Mean Reward: {row['mean_reward']:.4f}")
            print(f"   Best Reward: {row.get('best_reward', 'N/A')}")
            print(f"   Grade: {row.get('grade', 'N/A')}")
            print()
    
    # สร้างกราฟเปรียบเทียบ
    if len(comparison_df) > 1:
        create_comparison_chart(comparison_df)
    
    return comparison_df

def create_comparison_chart(df):
    """สร้างกราฟเปรียบเทียบ"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Mean Reward
    if 'mean_reward' in df.columns:
        axes[0,0].bar(range(len(df)), df['mean_reward'])
        axes[0,0].set_title('Mean Reward')
        axes[0,0].set_xticks(range(len(df)))
        axes[0,0].set_xticklabels([idx[:10] + '...' for idx in df.index], rotation=45)
    
    # Best Reward
    if 'best_reward' in df.columns:
        axes[0,1].bar(range(len(df)), df['best_reward'])
        axes[0,1].set_title('Best Reward')
        axes[0,1].set_xticks(range(len(df)))
        axes[0,1].set_xticklabels([idx[:10] + '...' for idx in df.index], rotation=45)
    
    # Stability Score
    if 'stability_score' in df.columns:
        axes[1,0].bar(range(len(df)), df['stability_score'])
        axes[1,0].set_title('Stability Score')
        axes[1,0].set_xticks(range(len(df)))
        axes[1,0].set_xticklabels([idx[:10] + '...' for idx in df.index], rotation=45)
    
    # Training Duration
    if 'training_duration' in df.columns:
        axes[1,1].bar(range(len(df)), df['training_duration'])
        axes[1,1].set_title('Training Duration (seconds)')
        axes[1,1].set_xticks(range(len(df)))
        axes[1,1].set_xticklabels([idx[:10] + '...' for idx in df.index], rotation=45)
    
    plt.tight_layout()
    
    # บันทึกกราฟ
    output_path = "models/sac/comparison_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison chart saved: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    comparison_df = compare_models()
```

### 2. ใช้ Streamlit UI

```bash
# เปิด Streamlit UI
cd ui
streamlit run app.py
```

จากนั้นไปที่หน้า **"6_Manage_Agents"** เพื่อ:
- ✅ ดูรายการ models ทั้งหมด
- ✅ เปรียบเทียบประสิทธิภาพ  
- ✅ ดู training history
- ✅ จัดการไฟล์ models

---

## 📤 การ Export และ Analysis

### 1. Export เป็น CSV

```python
# export_data.py
from models.sac.sac_metadata_manager import SAC_MetadataManager
from datetime import datetime

def export_model_data():
    """Export ข้อมูล models เป็น CSV"""
    
    manager = SAC_MetadataManager()
    manager.load_all_metadata()
    
    # Export performance data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"model_performance_{timestamp}.csv"
    
    manager.export_to_csv(output_file)
    
    print(f"✅ Model data exported to: {output_file}")
    
    # อ่านและแสดงตัวอย่าง
    import pandas as pd
    df = pd.read_csv(output_file)
    
    print(f"\n📊 Export Summary:")
    print(f"  Total Models: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    return output_file

if __name__ == "__main__":
    export_file = export_model_data()
```

### 2. สร้าง Performance Report

```python
# generate_report.py
import os
import matplotlib.pyplot as plt
from datetime import datetime
from models.sac.sac_metadata_manager import SAC_MetadataManager

def generate_performance_report():
    """สร้าง performance report"""
    
    manager = SAC_MetadataManager()
    manager.load_all_metadata()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = f"performance_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"📊 Generating Performance Report...")
    print(f"📁 Output: {report_dir}")
    
    # 1. รวบรวมข้อมูล
    agents = manager.list_agents()
    grade_stats = manager.get_grade_statistics()
    comparison_df = manager.get_performance_comparison()
    
    # 2. สร้างกราฟ
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SAC Models Performance Report', fontsize=16)
    
    # Grade distribution
    grades = list(grade_stats.keys())
    counts = [grade_stats[g]['count'] for g in grades]
    
    if grades:
        axes[0,0].pie(counts, labels=grades, autopct='%1.1f%%')
        axes[0,0].set_title('Models by Grade')
        
        # Mean reward by grade
        mean_rewards = [grade_stats[g]['mean_reward'] for g in grades]
        axes[0,1].bar(grades, mean_rewards)
        axes[0,1].set_title('Mean Reward by Grade')
        
        # Success rate by grade
        success_rates = [grade_stats[g]['success_rate'] * 100 for g in grades]
        axes[1,0].bar(grades, success_rates)
        axes[1,0].set_title('Success Rate by Grade (%)')
        
        # Training duration
        durations = [grade_stats[g]['avg_training_duration'] / 60 for g in grades]  # minutes
        axes[1,1].bar(grades, durations)
        axes[1,1].set_title('Avg Training Duration (minutes)')
    
    plt.tight_layout()
    
    # บันทึกกราฟ
    chart_path = os.path.join(report_dir, 'performance_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Chart saved: {chart_path}")
    
    # 3. Export data
    csv_path = os.path.join(report_dir, 'model_data.csv')
    manager.export_to_csv(csv_path)
    print(f"📋 Data exported: {csv_path}")
    
    # 4. สร้าง text report
    report_path = os.path.join(report_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SAC Models Performance Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total Models: {len(agents)}\n")
        f.write(f"Grades Available: {', '.join(grades)}\n\n")
        
        f.write("Grade Statistics:\n")
        f.write("-" * 30 + "\n")
        for grade, stats in grade_stats.items():
            f.write(f"Grade {grade}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Mean Reward: {stats['mean_reward']:.4f}\n")
            f.write(f"  Best Reward: {stats['best_reward']:.4f}\n")
            f.write(f"  Success Rate: {stats['success_rate']:.2%}\n\n")
        
        # Best models
        best_agents = manager.get_best_agents_by_grade(top_n=2)
        f.write("Best Models by Grade:\n")
        f.write("-" * 30 + "\n")
        for grade, agents_list in best_agents.items():
            f.write(f"Grade {grade}:\n")
            for i, agent in enumerate(agents_list, 1):
                f.write(f"  {i}. {agent.agent_id}\n")
                f.write(f"     Mean Reward: {agent.performance_metrics.get('mean_reward', 0):.4f}\n")
                f.write(f"     Duration: {agent.get_training_duration_formatted()}\n")
            f.write("\n")
    
    print(f"📝 Report saved: {report_path}")
    print(f"\n✅ Performance report completed!")
    
    return report_dir

if __name__ == "__main__":
    report_dir = generate_performance_report()
```

---

## 🎯 สรุป - วิธีดูประวัติและประสิทธิภาพแบบเร็ว

### 🚀 คำสั่งพื้นฐาน

```bash
# 1. ดู model ล่าสุด (เร็วที่สุด)
python test_sac_results.py

# 2. ดูรายการ models ทั้งหมด
python -c "
import os
sac_files = [f for f in os.listdir('models/sac') if f.endswith('.zip')]
print(f'Found {len(sac_files)} SAC models')
for f in sorted(sac_files, reverse=True)[:5]:
    print(f'  {f}')
"

# 3. เริ่ม tensorboard
tensorboard --logdir=logs --port=6006

# 4. เปิด Streamlit UI
cd ui && streamlit run app.py
```

### 📊 Python Scripts ที่มีประโยชน์

```python
# quick_model_check.py - ดูข้อมูลรวดเร็ว
from models.sac.sac_metadata_manager import SAC_MetadataManager

manager = SAC_MetadataManager()
manager.load_all_metadata()

agents = manager.list_agents()
print(f"Total models: {len(agents)}")

if agents:
    latest = max(agents, key=lambda x: x.updated_at)
    print(f"Latest: {latest.agent_id}")
    print(f"Grade: {latest.grade}")
    print(f"Mean reward: {latest.performance_metrics.get('mean_reward', 0):.4f}")
```

### 🎨 การดู Visualizations

1. **Performance Charts**: `models/sac/*.png`
2. **Tensorboard**: `http://localhost:6006` หลังรัน `tensorboard --logdir=logs`
3. **Streamlit UI**: `http://localhost:8501` หลังรัน `streamlit run app.py`

### 📁 โครงสร้างไฟล์สำคัญ

```
logs/                    # Training logs (tensorboard)
models/sac/             # SAC models และ metadata
  ├── *.zip             # Model files
  ├── *_info.pkl        # Model information
  ├── *.png             # Performance charts
  └── metadata/         # Detailed metadata
ui/pages/6_Manage_Agents.py  # Streamlit model management
```

---

*คู่มือนี้ครอบคลุมทุกวิธีการดูประวัติและประสิทธิภาพ model ตั้งแต่แบบเร็วไปจนถึงการวิเคราะห์เชิงลึก! เริ่มจาก `python test_sac_results.py` แล้วค่อยใช้เครื่องมืออื่นๆ ตามความต้องการ* 🎉 