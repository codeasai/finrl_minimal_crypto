# FinRL Models และ Algorithms Guide

> คู่มือสำหรับ Deep Reinforcement Learning Models ที่รองรับใน FinRL Framework

## 📋 สารบัญ

1. [ภาพรวม](#ภาพรวม)
2. [On-Policy Algorithms](#on-policy-algorithms)
3. [Off-Policy Algorithms](#off-policy-algorithms)
4. [High-Throughput Algorithms](#high-throughput-algorithms)
5. [Model-based RL](#model-based-rl)
6. [Offline RL และ Imitation Learning](#offline-rl-และ-imitation-learning)
7. [Algorithm Extensions](#algorithm-extensions)
8. [การเลือก Algorithm](#การเลือก-algorithm)
9. [การใช้งานใน finrl_minimal_crypto](#การใช้งานใน-finrl_minimal_crypto)
10. [ตัวอย่างการใช้งาน](#ตัวอย่างการใช้งาน)

---

## 🎯 ภาพรวม

FinRL รองรับ Deep Reinforcement Learning algorithms หลากหลาย ซึ่งสามารถแบ่งได้เป็น 6 หมวดหลัก:

| หมวด | จำนวน Algorithms | เหมาะสำหรับ |
|------|------------------|-------------|
| **On-Policy** | 1 | Stable training, good sample efficiency |
| **Off-Policy** | 3 | Sample efficiency, continuous actions |
| **High-Throughput** | 2 | Distributed training, scalability |
| **Model-based** | 1 | Sample efficiency, complex environments |
| **Offline RL** | 3 | Historical data, imitation learning |
| **Extensions** | 1+ | Exploration, curiosity-driven learning |

---

## 🚀 On-Policy Algorithms

### PPO (Proximal Policy Optimization) ⭐ **แนะนำ**

**คำอธิบาย:**
- Policy gradient algorithm ที่ใช้ clipping เพื่อควบคุม policy updates
- สมดุลระหว่าง sample efficiency และ stability
- เป็น default algorithm ใน finrl_minimal_crypto

**จุดเด่น:**
- ✅ Stable training
- ✅ รองรับทั้ง continuous และ discrete actions
- ✅ Good performance across various tasks
- ✅ Easy to tune hyperparameters

**จุดด้อย:**
- ❌ Sample efficiency ต่ำกว่า off-policy methods
- ❌ Requires on-policy data collection

**เหมาะสำหรับ:**
- การเทรด cryptocurrency
- Portfolio management
- Beginners ใน RL

**Hyperparameters สำคัญ:**
```python
PPO_PARAMS = {
    'learning_rate': 1e-4,      # Learning rate
    'n_steps': 1024,            # Steps per rollout
    'batch_size': 128,          # Mini-batch size
    'n_epochs': 4,              # Training epochs per update
    'gamma': 0.99,              # Discount factor
    'gae_lambda': 0.95,         # GAE parameter
    'clip_range': 0.2,          # PPO clipping parameter
    'ent_coef': 0.01,           # Entropy coefficient
}
```

---

## 🎭 Off-Policy Algorithms

### DQN (Deep Q Networks)

**คำอธิบาย:**
- Value-based method ที่ใช้ neural network ประมาณ Q-function
- รองรับเฉพาะ discrete action spaces
- มี Rainbow extensions (Dueling, Double-Q, Distributional DQN)

**จุดเด่น:**
- ✅ Sample efficient
- ✅ Stable with replay buffer
- ✅ Good for discrete actions

**จุดด้อย:**
- ❌ เฉพาะ discrete actions
- ❌ อาจ overestimate Q-values

**เหมาะสำหรับ:**
- Trading decisions (buy/sell/hold)
- Portfolio rebalancing
- Rule-based strategies

### SAC (Soft Actor-Critic) ⭐ **แนะนำสำหรับ Continuous Actions**

**คำอธิบาย:**
- Actor-critic method ที่ใช้ maximum entropy principle
- เพิ่ม exploration ผ่าน entropy regularization
- รองรับ continuous action spaces

**จุดเด่น:**
- ✅ เยี่ยมสำหรับ continuous actions
- ✅ Good exploration
- ✅ Sample efficient
- ✅ Robust performance

**จุดด้อย:**
- ❌ Complex hyperparameter tuning
- ❌ Higher computational cost

**เหมาะสำหรับ:**
- Position sizing
- Portfolio allocation weights
- Continuous trading strategies

**Hyperparameters สำคัญ:**
```python
SAC_PARAMS = {
    'learning_rate': 3e-4,      # Learning rate
    'buffer_size': 100000,      # Replay buffer size
    'batch_size': 256,          # Batch size
    'tau': 0.005,               # Target network update rate
    'gamma': 0.99,              # Discount factor
    'train_freq': 1,            # Training frequency
    'ent_coef': 'auto',         # Automatic entropy tuning
}
```

### DDPG (Deep Deterministic Policy Gradient)

**คำอธิบาย:**
- Deterministic policy gradient สำหรับ continuous actions
- ใช้ actor-critic architecture
- Predecessor ของ SAC

**จุดเด่น:**
- ✅ Deterministic policies
- ✅ Continuous actions
- ✅ Relatively simple

**จุดด้อย:**
- ❌ Sensitive to hyperparameters
- ❌ ปัญหา exploration
- ❌ Less stable than SAC

**เหมาะสำหรับ:**
- Simple continuous control
- When deterministic policies preferred

---

## ⚡ High-Throughput Algorithms

### APPO (Asynchronous PPO)

**คำอธิบาย:**
- Asynchronous version ของ PPO
- ใช้ V-trace สำหรับ off-policy correction
- เหมาะสำหรับ distributed training

**จุดเด่น:**
- ✅ Scalable to many workers
- ✅ High throughput
- ✅ Handles off-policy data

**จุดด้อย:**
- ❌ Complex implementation
- ❌ Requires distributed setup

**เหมาะสำหรับ:**
- Large-scale trading systems
- High-frequency trading
- When speed is critical

### IMPALA (Importance Weighted Actor-Learner Architecture)

**คำอธิบาย:**
- Highly scalable distributed RL
- ใช้ V-trace สำหรับ off-policy correction
- รองรับหลาย environments พร้อมกัน

**จุดเด่น:**
- ✅ Extremely scalable
- ✅ High sample throughput
- ✅ Stable training

**จุดด้อย:**
- ❌ เฉพาะ discrete actions หลัก
- ❌ Complex infrastructure requirements

**เหมาะสำหรับ:**
- Massive parallel trading
- Multiple market environments
- Research applications

---

## 🔮 Model-based RL

### DreamerV3

**คำอธิบาย:**
- เรียนรู้ world model ของ environment
- ใช้ model ที่เรียนรู้ได้เพื่อ plan และ train policy
- Sample efficient มาก

**จุดเด่น:**
- ✅ Very sample efficient
- ✅ รองรับทั้ง continuous และ discrete
- ✅ Can learn from complex observations

**จุดด้อย:**
- ❌ Complex implementation
- ❌ Model bias ปัญหา
- ❌ Higher computational cost

**เหมาะสำหรับ:**
- Complex market dynamics
- When sample efficiency critical
- Research applications

---

## 📚 Offline RL และ Imitation Learning

### BC (Behavior Cloning)

**คำอธิบาย:**
- เรียนรู้โดยการ imitate expert behavior
- Supervised learning approach
- ไม่ต้องการ reward signal

**จุดเด่น:**
- ✅ Simple implementation
- ✅ Stable training
- ✅ No environment interaction needed

**จุดด้อย:**
- ❌ Limited to expert performance
- ❌ Distribution shift problems
- ❌ No improvement beyond data

**เหมาะสำหรับ:**
- Imitating successful traders
- Learning from historical strategies
- Initial policy training

### MARWIL (Monotonic Advantage Re-Weighted Imitation Learning)

**คำอธิบาย:**
- ผสม imitation learning กับ policy gradient
- ใช้ advantage weighting
- เรียนรู้จาก offline data พร้อม rewards

**จุดเด่น:**
- ✅ Better than pure BC
- ✅ Uses reward information
- ✅ Monotonic improvement

**จุดด้อย:**
- ❌ Still limited by data quality
- ❌ More complex than BC

**เหมาะสำหรับ:**
- Learning from profitable trades
- Improving existing strategies
- Risk-aware trading

### CQL (Conservative Q-Learning)

**คำอธิบาย:**
- Offline RL ที่ลด overestimation ของ Q-values
- เพิ่ม conservative penalty
- เหมาะสำหรับ offline datasets

**จุดเด่น:**
- ✅ Handles distribution shift
- ✅ Conservative estimates
- ✅ Works with suboptimal data

**จุดด้อย:**
- ❌ อาจ too conservative
- ❌ Complex hyperparameter tuning

**เหมาะสำหรับ:**
- Learning from historical market data
- Risk-averse strategies
- Backtesting improvements

---

## 🔧 Algorithm Extensions

### ICM (Intrinsic Curiosity Module)

**คำอธิบาย:**
- เพิ่ม curiosity-driven exploration
- เรียนรู้ world model เพื่อสร้าง intrinsic rewards
- สามารถใช้กับ algorithm อื่นได้

**จุดเด่น:**
- ✅ Better exploration
- ✅ Works with any base algorithm
- ✅ Handles sparse rewards

**จุดด้อย:**
- ❌ Additional computational cost
- ❌ May distract from main objective

**เหมาะสำหรับ:**
- Markets with sparse signals
- Discovering new trading patterns
- Research applications

---

## 🎯 การเลือก Algorithm

### ตาม Action Space:

| Action Type | แนะนำ Algorithms |
|-------------|------------------|
| **Discrete** (Buy/Sell/Hold) | PPO, DQN, IMPALA |
| **Continuous** (Position sizes) | SAC, DDPG, PPO |
| **Mixed** | PPO, DreamerV3 |

### ตาม Data Availability:

| Data Type | แนะนำ Algorithms |
|-----------|------------------|
| **Online Learning** | PPO, SAC, DQN |
| **Offline Only** | BC, MARWIL, CQL |
| **Mixed** | MARWIL, CQL |

### ตาม Performance Requirements:

| Priority | แนะนำ Algorithms |
|----------|------------------|
| **Sample Efficiency** | SAC, DreamerV3, CQL |
| **Training Speed** | PPO, APPO |
| **Scalability** | IMPALA, APPO |
| **Stability** | PPO, BC |

### ตาม Experience Level:

| Level | แนะนำ Algorithms |
|-------|------------------|
| **Beginner** | PPO, BC |
| **Intermediate** | SAC, DQN, MARWIL |
| **Advanced** | DreamerV3, APPO, IMPALA |

---

## 🏗️ การใช้งานใน finrl_minimal_crypto

### Current Implementation:

```python
# main.py - ใช้ PPO เป็นหลัก
model_name = "ppo"
model = agent.get_model(model_name, model_kwargs=PPO_PARAMS)
```

### Notebooks Configuration:

```python
# notebooks/config.py - รองรับ 4 algorithms
def get_model_params(model_name):
    params_map = {
        'PPO': PPO_PARAMS,
        'A2C': A2C_PARAMS, 
        'DDPG': DDPG_PARAMS,
        'SAC': SAC_PARAMS
    }
    return params_map.get(model_name.upper(), PPO_PARAMS)
```

### Streamlit UI:

- รองรับ algorithm หลากหลาย
- Grade system: N, D, C, B, A, S
- Interactive parameter tuning

---

## 💡 ตัวอย่างการใช้งาน

### 1. การเปลี่ยน Algorithm ใน main.py:

```python
# แทนที่ PPO ด้วย SAC
model_name = "sac"
SAC_PARAMS = {
    'learning_rate': 3e-4,
    'buffer_size': 50000,
    'batch_size': 256,
    'tau': 0.005,
}
model = agent.get_model(model_name, model_kwargs=SAC_PARAMS)
```

### 2. การใช้ Multiple Algorithms:

```python
# เปรียบเทียบ algorithms หลายตัว
algorithms = ['PPO', 'SAC', 'A2C']
results = {}

for algo in algorithms:
    model_params = get_model_params(algo)
    model = agent.get_model(algo.lower(), model_kwargs=model_params)
    # Train and evaluate
    results[algo] = evaluate_model(model)
```

### 3. การ Fine-tune Hyperparameters:

```python
# Grid search สำหรับ PPO
from ray import tune

ppo_config = {
    'learning_rate': tune.grid_search([1e-5, 1e-4, 1e-3]),
    'n_steps': tune.grid_search([512, 1024, 2048]),
    'batch_size': tune.grid_search([64, 128, 256]),
}
```

---

## 📈 Performance Comparison

| Algorithm | Sample Efficiency | Training Speed | Stability | Continuous Actions |
|-----------|------------------|----------------|-----------|-------------------|
| **PPO** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **SAC** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **DQN** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ |
| **A2C** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **DDPG** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

---

## 🔗 ข้อมูลเพิ่มเติม

### Documentation:
- [FinRL Official Documentation](https://finrl.readthedocs.io/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/)

### Research Papers:
- PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- SAC: [Soft Actor-Critic Algorithms](https://arxiv.org/abs/1812.05905)
- DQN: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

### การติดตั้งเพิ่มเติม:
```bash
# สำหรับ advanced algorithms
pip install ray[rllib]
pip install tensorboard

# สำหรับ hyperparameter tuning
pip install optuna
pip install wandb
```

---

## 📝 สรุป

การเลือก algorithm ที่เหมาะสมขึ้นอยู่กับ:

1. **ประเภทของ action space** (discrete vs continuous)
2. **ข้อมูลที่มี** (online vs offline)
3. **เป้าหมายการใช้งาน** (sample efficiency vs speed)
4. **ความซับซ้อนของ environment**
5. **ประสบการณ์ของผู้ใช้**

สำหรับ **cryptocurrency trading** แนะนำให้เริ่มต้นด้วย **PPO** เนื่องจากมีความเสถียรสูงและใช้งานง่าย จากนั้นค่อยทดลองกับ **SAC** สำหรับ continuous actions หรือ **DQN** สำหรับ discrete actions ตามความต้องการ

---

*เอกสารนี้อัปเดตล่าสุด: มกราคม 2025*
*สำหรับโปรเจค: finrl_minimal_crypto* 