# SAC Agent Optimization Guide

> คู่มือการปรับปรุงประสิทธิภาพ Soft Actor-Critic (SAC) Agent สำหรับการเทรด Cryptocurrency

## 📋 สารบัญ

1. [ภาพรวม SAC Algorithm](#ภาพรวม-sac-algorithm)
2. [วิเคราะห์ Implementation ปัจจุบัน](#วิเคราะห์-implementation-ปัจจุบัน)
3. [การปรับปรุง Hyperparameters](#การปรับปรุง-hyperparameters)
4. [การปรับปรุง Environment](#การปรับปรุง-environment)
5. [การปรับปรุง Reward Function](#การปรับปรุง-reward-function)
6. [การปรับปรุง Feature Engineering](#การปรับปรุง-feature-engineering)
7. [Advanced Optimization Techniques](#advanced-optimization-techniques)
8. [การปรับปรุง Training Process](#การปรับปรุง-training-process)
9. [Evaluation และ Benchmarking](#evaluation-และ-benchmarking)
10. [ตัวอย่างการใช้งาน](#ตัวอย่างการใช้งาน)

---

## 🎯 ภาพรวม SAC Algorithm

### SAC คืออะไร?

**Soft Actor-Critic (SAC)** เป็น off-policy algorithm ที่:
- ใช้ **maximum entropy principle** เพื่อส่งเสริม exploration
- รองรับ **continuous action spaces** เหมาะสำหรับ position sizing
- มี **sample efficiency** สูง เนื่องจากเป็น off-policy
- **Robust** และ stable ในหลายๆ environment

### จุดเด่นสำหรับ Crypto Trading

✅ **Continuous Actions**: เหมาะสำหรับการกำหนด position size แบบต่อเนื่อง  
✅ **Good Exploration**: Maximum entropy ช่วยค้นหา strategy ใหม่ๆ  
✅ **Sample Efficient**: ใช้ replay buffer ทำให้เรียนรู้จากข้อมูลเก่าได้  
✅ **Stable Training**: มี automatic entropy tuning  

---

## 🔍 วิเคราะห์ Implementation ปัจจุบัน

### Current SAC Configuration

```python
# จาก sac.py (lines 334-351)
model = SAC(
    "MlpPolicy",
    vec_env,
    learning_rate=0.0003,      # ค่อนข้างสูง
    buffer_size=100000,        # ขนาดเล็ก
    learning_starts=10000,     # เริ่มเรียนรู้ช้า
    batch_size=256,            # ขนาดมาตรฐาน
    tau=0.005,                 # Target network update ช้า
    gamma=0.99,                # Discount factor มาตรฐาน
    train_freq=1,              # เทรนทุก step
    gradient_steps=1,          # Gradient steps น้อย
    target_update_interval=1,  # อัพเดทบ่อย
    verbose=1,
    seed=312,
    device=device
)
```

### จุดที่ควรปรับปรุง

❌ **Buffer size เล็กเกินไป** (100K) สำหรับ crypto data  
❌ **Learning starts สูงเกินไป** (10K) ทำให้เริ่มช้า  
❌ **Gradient steps น้อย** (1) ทำให้ learning ช้า  
❌ **ขาด entropy tuning** อาจทำให้ exploration ไม่เหมาะสม  
❌ **Training timesteps น้อย** (50K) อาจไม่พอสำหรับ convergence  

---

## ⚙️ การปรับปรุง Hyperparameters

### 1. Optimized SAC Parameters

```python
# SAC Parameters ที่ปรับปรุงแล้ว
OPTIMIZED_SAC_PARAMS = {
    'policy': 'MlpPolicy',
    'learning_rate': 1e-4,          # ลดลงเพื่อ stability
    'buffer_size': 1000000,         # เพิ่มเป็น 1M สำหรับ crypto
    'learning_starts': 5000,        # ลดลงเพื่อเรียนรู้เร็วขึ้น
    'batch_size': 512,              # เพิ่มขึ้นสำหรับ stability
    'tau': 0.01,                    # เพิ่ม target network update
    'gamma': 0.995,                 # เพิ่ม long-term thinking
    'train_freq': 4,                # ลด training frequency
    'gradient_steps': 4,            # เพิ่ม gradient steps
    'target_update_interval': 1,
    'ent_coef': 'auto',            # Automatic entropy tuning
    'target_entropy': 'auto',      # Automatic target entropy
    'use_sde': True,               # State-dependent exploration
    'sde_sample_freq': 64,         # SDE sampling frequency
    'verbose': 1,
    'seed': 42,
    'device': 'auto'
}
```

### 2. Adaptive Learning Rate

```python
# Linear schedule สำหรับ learning rate
from stable_baselines3.common.utils import linear_schedule

ADAPTIVE_SAC_PARAMS = {
    'learning_rate': linear_schedule(3e-4, 1e-5),  # ลดลงตามเวลา
    'clip_range': linear_schedule(0.2, 0.05),      # Clipping ลดลง
}
```

### 3. Environment-Specific Parameters

```python
# Parameters เฉพาะสำหรับ Crypto Trading
CRYPTO_SAC_PARAMS = {
    'buffer_size': 500000,          # 6 เดือนของข้อมูล daily
    'learning_starts': 2000,        # ~1 เดือนก่อนเรียนรู้
    'train_freq': (8, "step"),      # เทรนทุก 8 steps
    'gradient_steps': 8,            # Multiple gradient steps
    'batch_size': 1024,             # Large batch สำหรับ stability
}
```

---

## 🏢 การปรับปรุง Environment

### 1. Enhanced Action Space

```python
class ImprovedCryptoTradingEnv(gym.Env):
    def __init__(self, df, initial_amount=100000, **kwargs):
        super().__init__()
        
        # Multi-dimensional action space
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]),    # [position_change, confidence]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Enhanced observation space
        self.n_features = len(INDICATORS) + 10  # เพิ่ม portfolio features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_features,), 
            dtype=np.float32
        )
```

### 2. Advanced State Representation

```python
def _get_enhanced_observation(self):
    """Enhanced observation with more portfolio information"""
    current_data = self.df.iloc[self.current_step]
    
    # Technical indicators
    indicators = [current_data[indicator] for indicator in INDICATORS]
    
    # Portfolio features (normalized)
    portfolio_features = [
        self.cash / self.initial_amount,                    # Cash ratio
        self.holdings / self.max_holdings,                  # Holdings ratio
        self.total_value / self.initial_amount,             # Total value ratio
        self._get_portfolio_return(),                       # Portfolio return
        self._get_sharpe_ratio(),                           # Sharpe ratio
        self._get_max_drawdown(),                           # Max drawdown
        self._get_volatility(),                             # Portfolio volatility
        self._get_beta(),                                   # Beta vs market
        self._get_position_duration(),                      # Position holding time
        self._get_trade_frequency()                         # Trading frequency
    ]
    
    # Market context
    market_features = [
        self._get_market_trend(),                           # Market trend
        self._get_volume_profile(),                         # Volume profile
        self._get_volatility_regime()                       # Volatility regime
    ]
    
    obs = np.array(indicators + portfolio_features + market_features, dtype=np.float32)
    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
```

### 3. Dynamic Position Sizing

```python
def step(self, action):
    """Enhanced step function with dynamic position sizing"""
    if self.current_step >= len(self.df) - 1:
        return self._get_observation(), 0, True, True, {}
    
    position_change = action[0]  # -1 to 1
    confidence = action[1]       # 0 to 1
    
    # Dynamic position sizing based on confidence
    max_position_change = confidence * 0.5  # Maximum 50% change
    actual_position_change = position_change * max_position_change
    
    # Risk-adjusted position sizing
    volatility = self._get_current_volatility()
    risk_adjusted_size = actual_position_change / (1 + volatility)
    
    # Execute trade with risk management
    self._execute_trade_with_risk_management(risk_adjusted_size)
    
    # Enhanced reward calculation
    reward = self._calculate_enhanced_reward(action)
    
    return self._get_enhanced_observation(), reward, done, False, info
```

---

## 🎁 การปรับปรุง Reward Function

### 1. Multi-Component Reward

```python
def _calculate_enhanced_reward(self, action):
    """Enhanced reward function with multiple components"""
    
    # 1. Portfolio return (primary)
    portfolio_return = (self.total_value - self.previous_value) / self.previous_value
    
    # 2. Risk-adjusted return (Sharpe-like)
    sharpe_reward = portfolio_return / (self.volatility + 1e-8)
    
    # 3. Benchmark comparison (vs Buy & Hold)
    benchmark_return = self._get_benchmark_return()
    excess_return = portfolio_return - benchmark_return
    
    # 4. Transaction cost penalty
    transaction_penalty = -abs(action[0]) * self.transaction_cost_pct
    
    # 5. Drawdown penalty
    drawdown_penalty = -max(0, self._get_current_drawdown()) * 0.1
    
    # 6. Diversification reward (if multiple assets)
    diversification_reward = self._get_diversification_score() * 0.01
    
    # 7. Risk management reward
    risk_mgmt_reward = self._get_risk_management_score() * 0.05
    
    # Combined reward
    total_reward = (
        portfolio_return * 1.0 +          # Primary objective
        sharpe_reward * 0.3 +             # Risk adjustment
        excess_return * 0.2 +             # Benchmark beating
        transaction_penalty +             # Cost awareness
        drawdown_penalty +                # Risk aversion
        diversification_reward +          # Diversification
        risk_mgmt_reward                  # Risk management
    )
    
    return total_reward
```

### 2. Adaptive Reward Scaling

```python
class AdaptiveRewardScaler:
    """Dynamic reward scaling based on market conditions"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        
    def scale_reward(self, raw_reward, market_volatility):
        """Scale reward based on market conditions"""
        self.reward_history.append(raw_reward)
        
        if len(self.reward_history) < 10:
            return raw_reward
        
        # Adaptive scaling
        reward_std = np.std(self.reward_history)
        volatility_factor = 1.0 / (1.0 + market_volatility)
        
        scaled_reward = raw_reward / (reward_std + 1e-8) * volatility_factor
        return np.clip(scaled_reward, -10, 10)  # Clip extreme values
```

---

## 🔬 การปรับปรุง Feature Engineering

### 1. Advanced Technical Indicators

```python
def add_advanced_technical_indicators(df):
    """Add advanced technical indicators for better signal quality"""
    
    # Existing indicators + advanced ones
    df = add_technical_indicators(df)  # จาก sac.py เดิม
    
    # Advanced momentum indicators
    df['momentum_3d'] = df['close'].pct_change(3)
    df['momentum_7d'] = df['close'].pct_change(7)
    df['momentum_14d'] = df['close'].pct_change(14)
    
    # Advanced volatility indicators
    df['parkinson_volatility'] = calculate_parkinson_volatility(df)
    df['garman_klass_volatility'] = calculate_garman_klass_volatility(df)
    
    # Market microstructure
    df['bid_ask_spread'] = calculate_bid_ask_spread(df)
    df['market_impact'] = calculate_market_impact(df)
    
    # Regime detection
    df['volatility_regime'] = calculate_volatility_regime(df)
    df['trend_regime'] = calculate_trend_regime(df)
    
    # Cross-asset features (if multiple cryptos)
    df['crypto_correlation'] = calculate_crypto_correlation(df)
    df['market_stress'] = calculate_market_stress_indicator(df)
    
    return df

def calculate_parkinson_volatility(df, window=20):
    """Calculate Parkinson volatility estimator"""
    hl_ratio = np.log(df['high'] / df['low'])
    return np.sqrt(hl_ratio.rolling(window).mean() * 252)

def calculate_garman_klass_volatility(df, window=20):
    """Calculate Garman-Klass volatility estimator"""
    log_hl = (np.log(df['high']) - np.log(df['low'])) ** 2
    log_co = (np.log(df['close']) - np.log(df['open'])) ** 2
    
    gk_vol = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk_vol.rolling(window).mean() * 252)
```

### 2. Feature Selection และ Engineering

```python
def optimize_feature_selection(df, target_return):
    """Select optimal features using various methods"""
    
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestRegressor
    
    # ใช้หลายวิธีในการเลือก features
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[features].fillna(0)
    y = target_return
    
    # Method 1: Statistical selection
    selector_stat = SelectKBest(score_func=f_regression, k=15)
    X_stat = selector_stat.fit_transform(X, y)
    selected_features_stat = selector_stat.get_support()
    
    # Method 2: Recursive feature elimination
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    selector_rfe = RFE(estimator=rf, n_features_to_select=15)
    X_rfe = selector_rfe.fit_transform(X, y)
    selected_features_rfe = selector_rfe.get_support()
    
    # Combine results
    final_features = [features[i] for i in range(len(features)) 
                     if selected_features_stat[i] or selected_features_rfe[i]]
    
    return final_features
```

---

## 🚀 Advanced Optimization Techniques

### 1. Curriculum Learning

```python
class CurriculumTrainer:
    """Implement curriculum learning for SAC"""
    
    def __init__(self, easy_env, medium_env, hard_env):
        self.envs = [easy_env, medium_env, hard_env]
        self.current_stage = 0
        self.performance_threshold = [0.05, 0.10, 0.15]  # Return thresholds
        
    def get_current_env(self):
        return self.envs[self.current_stage]
    
    def check_progression(self, performance):
        if (self.current_stage < len(self.envs) - 1 and 
            performance > self.performance_threshold[self.current_stage]):
            self.current_stage += 1
            print(f"🎓 Progressed to stage {self.current_stage + 1}")
            return True
        return False

# Usage in training
curriculum = CurriculumTrainer(easy_env, medium_env, hard_env)
model = SAC(**OPTIMIZED_SAC_PARAMS)

for episode in range(total_episodes):
    current_env = curriculum.get_current_env()
    # Train on current environment
    performance = evaluate_performance(model, current_env)
    curriculum.check_progression(performance)
```

### 2. Population-Based Training

```python
def population_based_training(num_agents=4, generations=10):
    """Implement population-based training for hyperparameter optimization"""
    
    # Initialize population
    population = []
    for i in range(num_agents):
        params = sample_hyperparameters()  # Random sampling
        agent = SAC(env, **params)
        population.append((agent, params))
    
    best_performance = -np.inf
    best_agent = None
    
    for generation in range(generations):
        print(f"🧬 Generation {generation + 1}")
        
        # Evaluate population
        performances = []
        for agent, params in population:
            performance = train_and_evaluate(agent, env, timesteps=10000)
            performances.append(performance)
        
        # Select best performers
        sorted_pop = sorted(zip(population, performances), 
                          key=lambda x: x[1], reverse=True)
        
        # Keep top 50%, mutate others
        elite_size = num_agents // 2
        new_population = []
        
        # Keep elite
        for i in range(elite_size):
            new_population.append(sorted_pop[i][0])
        
        # Mutate and replace bottom 50%
        for i in range(elite_size, num_agents):
            parent = random.choice(sorted_pop[:elite_size])[0]
            mutated_params = mutate_hyperparameters(parent[1])
            new_agent = SAC(env, **mutated_params)
            new_population.append((new_agent, mutated_params))
        
        population = new_population
        
        # Track best
        if max(performances) > best_performance:
            best_performance = max(performances)
            best_agent = sorted_pop[0][0][0]
            print(f"🏆 New best performance: {best_performance:.4f}")
    
    return best_agent, best_performance
```

### 3. Multi-Objective Optimization

```python
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

class SAC_MultiObjective(ElementwiseProblem):
    """Multi-objective optimization for SAC hyperparameters"""
    
    def __init__(self):
        super().__init__(n_var=6,       # 6 hyperparameters to optimize
                         n_obj=3,       # 3 objectives
                         n_constr=0,
                         xl=np.array([1e-5, 1000, 64, 0.001, 0.9, 1]),     # Lower bounds
                         xu=np.array([1e-3, 1000000, 1024, 0.1, 0.999, 16]) # Upper bounds
                        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Extract hyperparameters
        lr, buffer_size, batch_size, tau, gamma, gradient_steps = x
        
        # Create and train SAC agent
        params = {
            'learning_rate': lr,
            'buffer_size': int(buffer_size),
            'batch_size': int(batch_size),
            'tau': tau,
            'gamma': gamma,
            'gradient_steps': int(gradient_steps)
        }
        
        agent = SAC(env, **params)
        
        # Train and evaluate
        metrics = train_and_evaluate_multi_objective(agent)
        
        # Three objectives to optimize:
        out["F"] = [
            -metrics['total_return'],      # Maximize return (minimize negative)
            metrics['max_drawdown'],       # Minimize drawdown
            -metrics['sharpe_ratio']       # Maximize Sharpe (minimize negative)
        ]

# Run multi-objective optimization
problem = SAC_MultiObjective()
algorithm = NSGA2(pop_size=20)
res = minimize(problem, algorithm, ("n_gen", 50), verbose=True)

# Get Pareto optimal solutions
pareto_solutions = res.X
pareto_objectives = res.F
```

---

## 🎯 การปรับปรุง Training Process

### 1. Enhanced Training Loop

```python
def train_sac_with_enhancements(env, total_timesteps=200000):
    """Enhanced SAC training with multiple improvements"""
    
    # 1. Learning rate scheduling
    lr_schedule = linear_schedule(3e-4, 1e-5)
    
    # 2. Create model with optimized parameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        **OPTIMIZED_SAC_PARAMS,
        tensorboard_log="./tensorboard_logs/sac/"
    )
    
    # 3. Callbacks for monitoring and early stopping
    callbacks = [
        EarlyStoppingCallback(patience=10000, min_delta=0.01),
        CheckpointCallback(save_freq=10000, save_path="./checkpoints/"),
        PerformanceMonitorCallback(),
        HyperparameterSchedulerCallback()
    ]
    
    # 4. Progressive training with warm-up
    warm_up_steps = 20000
    main_training_steps = total_timesteps - warm_up_steps
    
    print("🔥 Warm-up training phase...")
    model.learn(total_timesteps=warm_up_steps, callback=callbacks[:2])
    
    print("🚀 Main training phase...")
    model.learn(total_timesteps=main_training_steps, 
               callback=callbacks, reset_num_timesteps=False)
    
    return model

class PerformanceMonitorCallback(BaseCallback):
    """Monitor performance and adjust training dynamically"""
    
    def __init__(self, eval_freq=5000):
        super().__init__()
        self.eval_freq = eval_freq
        self.best_performance = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current performance
            performance = evaluate_model_performance(self.model)
            
            # Log to tensorboard
            self.logger.record("eval/performance", performance)
            
            # Save if best
            if performance > self.best_performance:
                self.best_performance = performance
                self.model.save(f"best_model_step_{self.n_calls}")
                
        return True
```

### 2. Ensemble Training

```python
def train_sac_ensemble(env, n_models=5):
    """Train ensemble of SAC models for robustness"""
    
    models = []
    performances = []
    
    for i in range(n_models):
        print(f"🤖 Training model {i+1}/{n_models}")
        
        # Vary hyperparameters slightly for diversity
        params = OPTIMIZED_SAC_PARAMS.copy()
        params['learning_rate'] *= np.random.uniform(0.5, 1.5)
        params['gamma'] = np.random.uniform(0.99, 0.999)
        params['tau'] = np.random.uniform(0.005, 0.02)
        params['seed'] = 42 + i * 100
        
        # Train model
        model = SAC(env, **params)
        model.learn(total_timesteps=100000)
        
        # Evaluate
        performance = evaluate_model_performance(model)
        
        models.append(model)
        performances.append(performance)
        
        print(f"✅ Model {i+1} performance: {performance:.4f}")
    
    return models, performances

class EnsemblePredictor:
    """Combine predictions from multiple SAC models"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict(self, observation, deterministic=True):
        predictions = []
        
        for model in self.models:
            action, _ = model.predict(observation, deterministic=deterministic)
            predictions.append(action)
        
        # Weighted average
        ensemble_action = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_action, None
```

### 3. Transfer Learning

```python
def transfer_learning_sac(source_model_path, target_env, fine_tune_steps=50000):
    """Apply transfer learning from pre-trained SAC model"""
    
    # Load pre-trained model
    source_model = SAC.load(source_model_path)
    
    # Create new model with same architecture
    target_model = SAC(
        "MlpPolicy",
        target_env,
        **OPTIMIZED_SAC_PARAMS
    )
    
    # Transfer weights (except environment-specific layers)
    transfer_weights(source_model, target_model)
    
    # Fine-tune on target environment with lower learning rate
    target_model.learning_rate = 1e-5  # Lower learning rate for fine-tuning
    
    print("🔄 Fine-tuning pre-trained model...")
    target_model.learn(total_timesteps=fine_tune_steps)
    
    return target_model

def transfer_weights(source_model, target_model):
    """Transfer compatible weights between models"""
    source_params = source_model.get_parameters()
    target_params = target_model.get_parameters()
    
    for key in source_params:
        if key in target_params:
            if source_params[key].shape == target_params[key].shape:
                target_params[key] = source_params[key].copy()
                print(f"✅ Transferred weights for {key}")
            else:
                print(f"⚠️ Shape mismatch for {key}, skipping")
    
    target_model.set_parameters(target_params)
```

---

## 📊 Evaluation และ Benchmarking

### 1. Comprehensive Evaluation Metrics

```python
def comprehensive_sac_evaluation(model, test_env, n_episodes=10):
    """Comprehensive evaluation of SAC model performance"""
    
    results = {
        'returns': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'win_rates': [],
        'profit_factors': [],
        'calmar_ratios': [],
        'sortino_ratios': [],
        'action_distributions': [],
        'trade_statistics': []
    }
    
    for episode in range(n_episodes):
        obs, _ = test_env.reset()
        episode_returns = []
        episode_actions = []
        trades = []
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            
            episode_returns.append(info['total_value'])
            episode_actions.append(action[0])
            
            # Track trades
            if abs(action[0]) > 0.1:  # Significant action
                trades.append({
                    'action': action[0],
                    'price': info['price'],
                    'timestamp': test_env.current_step
                })
        
        # Calculate episode metrics
        total_return = (episode_returns[-1] - episode_returns[0]) / episode_returns[0]
        sharpe = calculate_sharpe_ratio(episode_returns)
        max_dd = calculate_max_drawdown(episode_returns)
        win_rate = calculate_win_rate(trades)
        profit_factor = calculate_profit_factor(trades)
        
        results['returns'].append(total_return)
        results['sharpe_ratios'].append(sharpe)
        results['max_drawdowns'].append(max_dd)
        results['win_rates'].append(win_rate)
        results['profit_factors'].append(profit_factor)
        results['action_distributions'].append(episode_actions)
        results['trade_statistics'].append(trades)
    
    # Aggregate results
    evaluation_summary = {
        'mean_return': np.mean(results['returns']),
        'std_return': np.std(results['returns']),
        'mean_sharpe': np.mean(results['sharpe_ratios']),
        'mean_max_drawdown': np.mean(results['max_drawdowns']),
        'mean_win_rate': np.mean(results['win_rates']),
        'consistency': np.std(results['returns']),  # Lower is better
        'downside_protection': np.mean(results['max_drawdowns'])
    }
    
    return evaluation_summary, results
```

### 2. Benchmark Comparison

```python
def benchmark_sac_against_baselines(sac_model, test_env):
    """Compare SAC against various baseline strategies"""
    
    benchmarks = {
        'SAC': sac_model,
        'Buy_Hold': BuyHoldStrategy(),
        'Random': RandomStrategy(),
        'PPO': load_ppo_baseline(),
        'Mean_Reversion': MeanReversionStrategy(),
        'Momentum': MomentumStrategy()
    }
    
    results = {}
    
    for name, strategy in benchmarks.items():
        print(f"📊 Evaluating {name}...")
        
        obs, _ = test_env.reset()
        account_values = [test_env.initial_amount]
        
        while True:
            if hasattr(strategy, 'predict'):  # RL models
                action, _ = strategy.predict(obs, deterministic=True)
            else:  # Traditional strategies
                action = strategy.get_action(obs, test_env.df.iloc[test_env.current_step])
            
            obs, reward, done, _, info = test_env.step(action)
            account_values.append(info['total_value'])
            
            if done:
                break
        
        # Calculate metrics
        total_return = (account_values[-1] - account_values[0]) / account_values[0]
        sharpe = calculate_sharpe_ratio(account_values)
        max_dd = calculate_max_drawdown(account_values)
        
        results[name] = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_value': account_values[-1]
        }
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('total_return', ascending=False)
    
    print("\n📈 Benchmark Comparison Results:")
    print("=" * 50)
    print(comparison_df.round(4))
    
    return comparison_df
```

---

## 💡 ตัวอย่างการใช้งาน

### 1. การปรับปรุง SAC Model แบบง่าย

```python
# สร้าง improved_sac.py
def create_improved_sac_model():
    """สร้าง SAC model ที่ปรับปรุงแล้ว"""
    
    # โหลดและเตรียมข้อมูล
    df = load_existing_data()
    df = add_advanced_technical_indicators(df)  # ใช้ indicators ที่ปรับปรุง
    
    # สร้าง environment ที่ปรับปรุง
    train_env = ImprovedCryptoTradingEnv(df)
    
    # สร้าง SAC model ด้วย parameters ที่ optimize แล้ว
    model = SAC(
        "MlpPolicy",
        train_env,
        **OPTIMIZED_SAC_PARAMS,
        tensorboard_log="./logs/sac_improved/"
    )
    
    # เทรนด้วย enhanced training loop
    model = train_sac_with_enhancements(train_env, total_timesteps=200000)
    
    # บันทึก model
    model.save("models/sac/improved_sac_agent")
    
    return model

# รันการปรับปรุง
if __name__ == "__main__":
    improved_model = create_improved_sac_model()
    print("✅ Improved SAC model created successfully!")
```

### 2. การใช้งาน Ensemble SAC

```python
# สร้าง ensemble_sac.py
def create_sac_ensemble():
    """สร้าง ensemble ของ SAC models"""
    
    # เตรียมข้อมูล
    df = load_existing_data()
    df = add_advanced_technical_indicators(df)
    
    # แบ่งข้อมูลเป็น train/test
    train_env, test_env, _, _ = create_environment(df)
    
    # สร้าง ensemble
    models, performances = train_sac_ensemble(train_env, n_models=5)
    
    # สร้าง ensemble predictor
    weights = np.array(performances)
    weights = weights / weights.sum()  # Normalize weights
    
    ensemble = EnsemblePredictor(models, weights)
    
    # ทดสอบ ensemble
    evaluation = comprehensive_sac_evaluation(ensemble, test_env)
    
    print(f"🏆 Ensemble Performance: {evaluation['mean_return']:.4f}")
    
    return ensemble

# รันการสร้าง ensemble
if __name__ == "__main__":
    ensemble_model = create_sac_ensemble()
    print("✅ SAC Ensemble created successfully!")
```

### 3. การ Optimize Hyperparameters

```python
# สร้าง optimize_sac.py
import optuna

def objective(trial):
    """Objective function สำหรับ Optuna optimization"""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_int('buffer_size', 50000, 1000000, step=50000)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
    tau = trial.suggest_float('tau', 0.001, 0.1, log=True)
    gamma = trial.suggest_float('gamma', 0.99, 0.999)
    gradient_steps = trial.suggest_int('gradient_steps', 1, 16)
    
    # Create and train model
    params = {
        'learning_rate': lr,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'tau': tau,
        'gamma': gamma,
        'gradient_steps': gradient_steps
    }
    
    model = SAC("MlpPolicy", train_env, **params)
    model.learn(total_timesteps=50000)
    
    # Evaluate performance
    performance = evaluate_model_performance(model, test_env)
    
    return performance

def optimize_sac_hyperparameters():
    """ใช้ Optuna ในการ optimize hyperparameters"""
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("🎯 Best hyperparameters:")
    print(study.best_params)
    print(f"🏆 Best performance: {study.best_value:.4f}")
    
    return study.best_params

# รัน optimization
if __name__ == "__main__":
    best_params = optimize_sac_hyperparameters()
    print("✅ Hyperparameter optimization completed!")
```

---

## 📝 สรุปและข้อแนะนำ

### 🎯 ลำดับความสำคัญในการปรับปรุง

1. **เริ่มจาก Hyperparameters** - ปรับ learning rate, buffer size, batch size
2. **ปรับปรุง Reward Function** - เพิ่ม risk adjustment และ multi-component rewards
3. **เพิ่ม Technical Indicators** - ใช้ advanced indicators สำหรับ signal quality
4. **ปรับปรุง Training Process** - ใช้ learning rate scheduling และ callbacks
5. **ใช้ Ensemble Methods** - รวม multiple models เพื่อ robustness

### 🚀 Quick Wins

- เพิ่ม `buffer_size` จาก 100K เป็น 500K-1M
- ลด `learning_starts` จาก 10K เป็น 5K
- เพิ่ม `gradient_steps` จาก 1 เป็น 4-8
- ใช้ `ent_coef='auto'` สำหรับ automatic entropy tuning
- เพิ่ม training timesteps จาก 50K เป็น 200K+

### ⚠️ ข้อควรระวัง

- อย่าปรับหลาย parameters พร้อมกัน
- ใช้ validation set แยกจาก test set
- Monitor overfitting ด้วย early stopping
- ทดสอบ robustness ในหลาย market conditions
- เก็บ logs และ metrics อย่างละเอียด

---

*คู่มือนี้ครอบคลุมการปรับปรุงประสิทธิภาพ SAC agent แบบครบถ้วน เริ่มจากการปรับปรุงพื้นฐานไปจนถึงเทคนิคขั้นสูง ใช้เป็นแนวทางในการพัฒนา cryptocurrency trading agent ที่มีประสิทธิภาพสูงขึ้น* 