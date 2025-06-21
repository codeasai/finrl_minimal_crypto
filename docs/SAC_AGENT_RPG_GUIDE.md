# SAC Agent RPG Guide: เหมือนตัวละคร Lineage 2! 🎮

> เข้าใจ SAC Agent ผ่านมุมมองเกม RPG - เปรียบเทียบกับอาชีพและระบบ stats ใน Lineage 2

## 🗡️ SAC Agent = Treasure Hunter (อาชีพนักล่าสมบัติ)

**เหตุผลที่เลือก Treasure Hunter:**
- 🎯 **Smart & Adaptive**: ใช้ปัญญาและการวิเคราะห์มากกว่าแรงเข้น
- 💰 **Profit-focused**: เป้าหมายคือหาผลกำไรจากการเทรด (เหมือนหาสมบัติ)
- ⚡ **Quick Decision**: ตัดสินใจเร็วในสถานการณ์เปลี่ยนแปลง
- 📊 **Risk Management**: รู้จักหลีกเลี่ยงความเสี่ยงและจัดการทรัพยากร
- 🔄 **Continuous Learning**: เรียนรู้จากประสบการณ์และปรับปรุงตนเอง

---

## 🎭 SAC Agent Class Variations = Multi-Class Trading Styles

> **SAC Agent สามารถ respec เป็นอาชีพต่างๆ ได้ตาม trading style และ market conditions!**

### 🛡️ **Human Knight (Conservative SAC)**
```python
KNIGHT_CONFIG = {
    'philosophy': 'Protect capital first, profit second',
    'risk_tolerance': 'Very Low (Max 5% drawdown)',
    'trading_style': 'Long-term position holding',
    'preferred_market': 'Stable uptrends',
    'special_skills': [
        'Defense Mastery: Stop-loss ที่เข้มงวด',
        'Shield Wall: Portfolio diversification',
        'Taunt: Attract stable coins/dividends',
        'Guardian: Protect against market crashes'
    ],
    'equipment': {
        'weapon': 'Conservative position sizing',
        'armor': 'Multiple stop-loss layers',
        'shield': 'Hedge positions',
        'accessories': 'Blue-chip focus'
    },
    'weakness': 'Miss opportunity ในช่วง high volatility',
    'strength': 'Survive ทุก market condition'
}
```

**Knight SAC Performance:**
- 📈 **Consistent small gains**: 8-12% annually
- 🛡️ **Ultra-low drawdown**: < 5%
- ⏰ **Long holding periods**: สัปดาห์-เดือน
- 💎 **Diamond hands**: ไม่ panic sell

### 🏹 **Elven Oracle (Technical Analysis SAC)**
```python
ORACLE_CONFIG = {
    'philosophy': 'Predict the future through data divination',
    'risk_tolerance': 'Low-Medium (10-15% drawdown)',
    'trading_style': 'Signal-based systematic trading',
    'preferred_market': 'Trending markets with clear patterns',
    'special_skills': [
        'Prophecy: Advanced forecasting models',
        'Bless: Enhance other agents performance',
        'Resurrect: Recovery from drawdowns',
        'Mass Heal: Portfolio rebalancing'
    ],
    'equipment': {
        'weapon': 'Technical indicators arsenal',
        'armor': 'Pattern recognition',
        'staff': 'Statistical models',
        'accessories': 'Multi-timeframe analysis'
    },
    'weakness': 'Confused by sideways/choppy markets',
    'strength': 'Excellent trend identification'
}
```

**Oracle SAC Performance:**
- 📊 **Data-driven decisions**: 70%+ signal accuracy
- 🔮 **Predictive power**: Catch trends early
- 📈 **Systematic approach**: Eliminate emotion
- 🎯 **Precise entries/exits**: Optimal timing

### 🗡️ **Dark Elf Assassin (Scalping SAC)**
```python
ASSASSIN_CONFIG = {
    'philosophy': 'Strike fast, strike hard, disappear',
    'risk_tolerance': 'High (20-30% drawdown)',
    'trading_style': 'High-frequency micro-profits',
    'preferred_market': 'High volatility, liquid markets',
    'special_skills': [
        'Backstab: Exploit price inefficiencies',
        'Poison: Compound small gains rapidly',
        'Stealth: Avoid detection by algos',
        'Dual Wield: Multiple simultaneous trades'
    ],
    'equipment': {
        'weapon': 'Ultra-low latency execution',
        'armor': 'Tight risk management',
        'boots': 'Speed optimization',
        'accessories': 'Market microstructure data'
    },
    'weakness': 'High transaction costs, burnout',
    'strength': 'Profit in any market condition'
}
```

**Assassin SAC Performance:**
- ⚡ **Lightning speed**: Millisecond execution
- 💰 **High turnover**: 100+ trades/day
- 🎯 **Precision strikes**: Small profit targets
- 🔄 **Market neutral**: Profit from volatility

### 🔧 **Dwarf Artisan (Research SAC)**
```python
ARTISAN_CONFIG = {
    'philosophy': 'Craft the perfect trading system',
    'risk_tolerance': 'Variable (adapts to research)',
    'trading_style': 'Experimental and systematic',
    'preferred_market': 'All markets (for research)',
    'special_skills': [
        'Craft: Build custom indicators',
        'Enchant: Optimize hyperparameters',
        'Repair: Fix broken strategies',
        'Manufacture: Create ensemble systems'
    ],
    'equipment': {
        'weapon': 'Custom algorithms',
        'armor': 'Robust backtesting',
        'tools': 'Statistical software',
        'accessories': 'Research databases'
    },
    'weakness': 'Over-optimization, analysis paralysis',
    'strength': 'Innovation and system improvement'
}
```

**Artisan SAC Performance:**
- 🔬 **R&D focus**: Continuous improvement
- ⚙️ **Custom solutions**: Unique strategies
- 📊 **Systematic testing**: Rigorous validation
- 🏗️ **Long-term building**: Sustainable systems

### 🧙‍♂️ **Human Wizard (Mathematical SAC)**
```python
WIZARD_CONFIG = {
    'philosophy': 'Markets are mathematical puzzles to solve',
    'risk_tolerance': 'Medium (10-20% drawdown)',
    'trading_style': 'Quantitative model-based',
    'preferred_market': 'Complex, multi-asset environments',
    'special_skills': [
        'Fireball: Explosive profit opportunities',
        'Ice Wall: Risk barrier construction',
        'Teleport: Rapid position changes',
        'Meteor: Large position deployment'
    ],
    'equipment': {
        'weapon': 'Advanced mathematics',
        'armor': 'Statistical significance',
        'staff': 'Monte Carlo simulations',
        'accessories': 'Option pricing models'
    },
    'weakness': 'Complex systems can fail spectacularly',
    'strength': 'Theoretical edge over simple strategies'
}
```

**Wizard SAC Performance:**
- 🧠 **Mathematical precision**: Model-driven approach
- 📈 **Complex strategies**: Multi-factor models
- 🎯 **Theoretical edge**: Academic backing
- 🔮 **Predictive models**: Advanced forecasting

### 🏹 **Elven Archer (Swing Trading SAC)**
```python
ARCHER_CONFIG = {
    'philosophy': 'Patient positioning, precise execution',
    'risk_tolerance': 'Medium (15-20% drawdown)',
    'trading_style': 'Medium-term swing trading',
    'preferred_market': 'Trending markets with pullbacks',
    'special_skills': [
        'Eagle Eye: Spot perfect entry points',
        'Power Shot: Large position sizing',
        'Multi-Shot: Diversified positions',
        'Tracking: Follow trends patiently'
    ],
    'equipment': {
        'weapon': 'Swing trading indicators',
        'armor': 'Trend-following discipline',
        'bow': 'Position sizing algorithm',
        'accessories': 'Multi-timeframe charts'
    },
    'weakness': 'Whipsaws in sideways markets',
    'strength': 'Capture major trend movements'
}
```

**Archer SAC Performance:**  
- 🎯 **Precision timing**: Perfect entry/exit points
- 📈 **Trend capture**: Ride major movements
- ⏰ **Medium timeframe**: Days to weeks
- 🏹 **Patient execution**: Wait for setups

### ⚔️ **Human Paladin (Balanced SAC)**
```python
PALADIN_CONFIG = {
    'philosophy': 'Balance between risk and reward',
    'risk_tolerance': 'Medium (12-18% drawdown)',
    'trading_style': 'Balanced multi-strategy approach',
    'preferred_market': 'All market conditions',
    'special_skills': [
        'Divine Protection: Downside protection',
        'Blessing: Performance enhancement',
        'Heal: Portfolio recovery',
        'Sacred Weapon: Ethical investing'
    ],
    'equipment': {
        'weapon': 'Balanced portfolio',
        'armor': 'Risk-reward optimization',
        'shield': 'Correlation management',
        'accessories': 'ESG factors'
    },
    'weakness': 'Master of none syndrome',
    'strength': 'Adaptable to any market'
}
```

**Paladin SAC Performance:**
- ⚖️ **Perfect balance**: Risk-reward harmony
- 🛡️ **Consistent performance**: Steady returns
- 🌟 **Moral compass**: Sustainable strategies
- 🔄 **Adaptability**: Works in all markets

### 💀 **Human Necromancer (Contrarian SAC)**
```python
NECROMANCER_CONFIG = {
    'philosophy': 'Profit from others fear and panic',
    'risk_tolerance': 'High (25-35% drawdown)',
    'trading_style': 'Contrarian/counter-trend',
    'preferred_market': 'Crisis and panic situations',
    'special_skills': [
        'Corpse Explosion: Profit from crashes',
        'Bone Spear: Attack overvalued assets',
        'Summon Skeleton: Create value from decay',
        'Dark Ritual: Sacrifice for greater gains'
    ],
    'equipment': {
        'weapon': 'Contrarian indicators',
        'armor': 'Psychological fortitude',
        'staff': 'Crisis opportunity radar',
        'accessories': 'Fear & greed index'
    },
    'weakness': 'Can be early (market timing)',
    'strength': 'Massive gains during reversals'
}
```

**Necromancer SAC Performance:**
- 💀 **Crisis profiteer**: Buy during panic
- 🔮 **Contrarian signals**: Fade the crowd
- 💰 **Huge opportunities**: Major reversals
- 🧙‍♂️ **Dark magic**: Profit from despair

---

## 🎭 Multi-Class SAC System (Advanced)

### 🔄 **Class Switching Mechanism:**
```python
class MultiClassSAC:
    def __init__(self):
        self.classes = {
            'knight': ConservativeSAC(),
            'oracle': TechnicalSAC(), 
            'assassin': ScalpingSAC(),
            'artisan': ResearchSAC(),
            'wizard': QuantSAC(),
            'archer': SwingSAC(),
            'paladin': BalancedSAC(),
            'necromancer': ContrarianSAC()
        }
        self.current_class = 'paladin'  # Default balanced
        
    def detect_market_regime(self):
        """เปลี่ยน class ตาม market conditions"""
        if self.is_trending_up():
            return 'archer'  # Swing trading
        elif self.is_trending_down():
            return 'necromancer'  # Contrarian
        elif self.is_sideways():
            return 'assassin'  # Scalping
        elif self.is_volatile():
            return 'knight'  # Conservative
        elif self.is_crisis():
            return 'necromancer'  # Crisis trading
        else:
            return 'paladin'  # Balanced default
            
    def switch_class(self, new_class):
        """เปลี่ยนสไตล์การเทรด"""
        self.current_class = new_class
        self.load_class_config(new_class)
        
    def ensemble_vote(self):
        """ให้ทุก class vote หา best action"""
        votes = {}
        for class_name, agent in self.classes.items():
            action = agent.predict_action()
            votes[class_name] = action
        return self.weighted_decision(votes)
```

### 🏰 **Guild Formation (Ensemble Strategy):**
```python
GUILD_COMPOSITION = {
    'Tank': 'Knight SAC (30% allocation)',
    'DPS': 'Assassin SAC (25% allocation)', 
    'Support': 'Oracle SAC (20% allocation)',
    'Healer': 'Paladin SAC (15% allocation)',
    'Specialist': 'Necromancer SAC (10% allocation)'
}

GUILD_BENEFITS = {
    'Risk Distribution': 'ไม่พึ่งพา single strategy',
    'Market Coverage': 'มี specialist สำหรับทุก condition',
    'Performance Boost': 'รวมจุดแข็งของทุก class',
    'Failure Resilience': 'ถ้า 1-2 class fail ยังมีอีก 6-7 class'
}
```

---

## 🎯 Class Selection Guide

### 📊 **ตาม Personality Type:**
```python
PERSONALITY_MAPPING = {
    'Risk Averse': 'Knight → Oracle → Paladin',
    'Balanced': 'Paladin → Archer → Oracle', 
    'Risk Seeking': 'Assassin → Necromancer → Wizard',
    'Analytical': 'Wizard → Artisan → Oracle',
    'Intuitive': 'Archer → Paladin → Assassin',
    'Contrarian': 'Necromancer → Artisan → Knight'
}
```

### 💰 **ตาม Capital Size:**
```python
CAPITAL_REQUIREMENTS = {
    'Knight': '$10K+ (Low frequency, stable)',
    'Oracle': '$25K+ (Medium frequency, signals)',
    'Assassin': '$50K+ (High frequency, scalping)',
    'Artisan': '$100K+ (Research costs, experimentation)',
    'Wizard': '$250K+ (Complex models, data costs)',
    'Archer': '$50K+ (Medium frequency, swings)',
    'Paladin': '$100K+ (Diversification needs)',
    'Necromancer': '$500K+ (Contrarian requires patience)'
}
```

### ⏰ **ตาม Time Commitment:**
```python
TIME_REQUIREMENTS = {
    'Knight': '1-2 hours/week (Set & forget)',
    'Oracle': '1 hour/day (Signal monitoring)',
    'Assassin': '8+ hours/day (Active scalping)',
    'Artisan': '20+ hours/week (Research & dev)',
    'Wizard': '10+ hours/week (Model maintenance)',
    'Archer': '2-3 hours/day (Swing monitoring)',
    'Paladin': '5 hours/week (Balanced approach)',
    'Necromancer': '3-4 hours/week (Crisis watching)'
}
```

### 🎓 **Learning Path Recommendations:**
```
Beginner Path:
Knight → Paladin → Oracle → Archer

Intermediate Path:  
Oracle → Wizard → Artisan → Paladin

Advanced Path:
Paladin → Assassin → Necromancer → Multi-Class

Expert Path:
Multi-Class Ensemble → Guild Master → Strategy Research
```

---

## 📊 SAC Agent Level & Status System

### 🎯 **Level Mapping ตาม Grade:**

```python
LEVEL_SYSTEM = {
    'Grade N (Novice)': {
        'level_range': '1-20',
        'experience_points': '0-50K timesteps',
        'class_unlock': ['Human Fighter'],
        'status_color': '⚪ White (Common)',
        'title': 'Rookie Trader'
    },
    'Grade D (Developing)': {
        'level_range': '21-40', 
        'experience_points': '50K-100K timesteps',
        'class_unlock': ['Human Fighter', 'Human Knight'],
        'status_color': '🔵 Blue (Uncommon)',
        'title': 'Developing Trader'
    },
    'Grade C (Competent)': {
        'level_range': '41-52',
        'experience_points': '100K-200K timesteps', 
        'class_unlock': ['Treasure Hunter', 'Elven Oracle', 'Human Paladin'],
        'status_color': '🟡 Yellow (Rare)',
        'title': 'Competent Trader'
    },
    'Grade B (Proficient)': {
        'level_range': '53-61',
        'experience_points': '200K-500K timesteps',
        'class_unlock': ['Elven Archer', 'Human Wizard', 'Dwarf Artisan'],
        'status_color': '🟣 Purple (Epic)',
        'title': 'Proficient Trader'
    },
    'Grade A (Advanced)': {
        'level_range': '62-76',
        'experience_points': '500K-1M timesteps',
        'class_unlock': ['Dark Elf Assassin', 'Human Necromancer'],
        'status_color': '🔴 Red (Legendary)',
        'title': 'Advanced Trader'
    },
    'Grade S (Supreme)': {
        'level_range': '77-85',
        'experience_points': '1M+ timesteps',
        'class_unlock': ['Multi-Class Master', 'Guild Leader'],
        'status_color': '🟠 Orange (Artifact)',
        'title': 'Supreme Trader'
    }
}
```

### 🎭 **Class-Specific Status Attributes:**

```python
CLASS_STATUS_ATTRIBUTES = {
    'Human Knight (Conservative SAC)': {
        'primary_stats': {
            'CON (Constitution)': 'Max Drawdown Tolerance',
            'DEF (Defense)': 'Risk Management Score', 
            'VIT (Vitality)': 'Portfolio Stability',
            'MEN (Mental)': 'Emotional Discipline'
        },
        'secondary_stats': {
            'ATK (Attack)': 'Profit Generation (Low)',
            'SPD (Speed)': 'Trade Frequency (Low)',
            'LUK (Luck)': 'Market Timing (Medium)'
        },
        'special_abilities': [
            'Shield Wall: -50% drawdown during crashes',
            'Taunt: Attract dividend-paying assets',
            'Guardian: Protect other agents in ensemble'
        ]
    },
    
    'Elven Oracle (Technical SAC)': {
        'primary_stats': {
            'INT (Intelligence)': 'Technical Analysis Mastery',
            'WIS (Wisdom)': 'Pattern Recognition',
            'MEN (Mental)': 'Signal Processing Power',
            'LUK (Luck)': 'Prediction Accuracy'
        },
        'secondary_stats': {
            'ATK (Attack)': 'Profit Generation (Medium)',
            'DEF (Defense)': 'Risk Management (Medium)',
            'SPD (Speed)': 'Signal Response Time'
        },
        'special_abilities': [
            'Prophecy: Predict market direction 70%+ accuracy',
            'Bless: Enhance other agents performance +15%',
            'Resurrect: Recover from drawdowns faster'
        ]
    },
    
    'Dark Elf Assassin (Scalping SAC)': {
        'primary_stats': {
            'DEX (Dexterity)': 'Execution Speed',
            'SPD (Speed)': 'Trade Frequency',
            'ATK (Attack)': 'Profit Per Trade',
            'LUK (Luck)': 'Market Timing Precision'
        },
        'secondary_stats': {
            'DEF (Defense)': 'Risk Management (Low)',
            'VIT (Vitality)': 'Drawdown Recovery',
            'MEN (Mental)': 'Stress Tolerance'
        },
        'special_abilities': [
            'Backstab: 2x profit on perfect entries',
            'Dual Wield: Multiple simultaneous positions',
            'Stealth: Avoid market maker detection'
        ]
    },
    
    'Dwarf Artisan (Research SAC)': {
        'primary_stats': {
            'INT (Intelligence)': 'Algorithm Development',
            'WIS (Wisdom)': 'System Architecture',
            'CON (Constitution)': 'Research Endurance',
            'MEN (Mental)': 'Innovation Capacity'
        },
        'secondary_stats': {
            'ATK (Attack)': 'Profit Generation (Variable)',
            'DEF (Defense)': 'Robust System Design',
            'SPD (Speed)': 'Development Velocity'
        },
        'special_abilities': [
            'Craft: Create custom indicators',
            'Enchant: Optimize hyperparameters +25%',
            'Manufacture: Build ensemble systems'
        ]
    },
    
    'Human Wizard (Mathematical SAC)': {
        'primary_stats': {
            'INT (Intelligence)': 'Mathematical Modeling',
            'WIS (Wisdom)': 'Statistical Understanding',
            'MEN (Mental)': 'Complex System Management',
            'LUK (Luck)': 'Model Generalization'
        },
        'secondary_stats': {
            'ATK (Attack)': 'Theoretical Profit Potential',
            'DEF (Defense)': 'Statistical Significance',
            'VIT (Vitality)': 'Model Robustness'
        },
        'special_abilities': [
            'Fireball: Explosive profit opportunities',
            'Ice Wall: Risk barrier construction',
            'Meteor: Large position deployment'
        ]
    },
    
    'Elven Archer (Swing Trading SAC)': {
        'primary_stats': {
            'DEX (Dexterity)': 'Entry/Exit Precision',
            'WIS (Wisdom)': 'Trend Recognition',
            'LUK (Luck)': 'Market Timing',
            'MEN (Mental)': 'Patience & Discipline'
        },
        'secondary_stats': {
            'ATK (Attack)': 'Trend Capture Ability',
            'DEF (Defense)': 'Stop-Loss Discipline',
            'SPD (Speed)': 'Position Adjustment'
        },
        'special_abilities': [
            'Eagle Eye: Spot perfect entries 90% accuracy',
            'Power Shot: 3x normal position size',
            'Multi-Shot: Diversified trend positions'
        ]
    },
    
    'Human Paladin (Balanced SAC)': {
        'primary_stats': {
            'STR (Strength)': 'Consistent Performance',
            'CON (Constitution)': 'Market Endurance',
            'WIS (Wisdom)': 'Balanced Decision Making',
            'MEN (Mental)': 'Emotional Stability'
        },
        'secondary_stats': {
            'ATK (Attack)': 'Steady Profit Generation',
            'DEF (Defense)': 'Risk-Reward Balance',
            'SPD (Speed)': 'Adaptive Response'
        },
        'special_abilities': [
            'Divine Protection: Reduce max drawdown -30%',
            'Blessing: Enhance all stats +10%',
            'Heal: Portfolio recovery acceleration'
        ]
    },
    
    'Human Necromancer (Contrarian SAC)': {
        'primary_stats': {
            'INT (Intelligence)': 'Contrarian Analysis',
            'WIS (Wisdom)': 'Crisis Opportunity Detection',
            'MEN (Mental)': 'Psychological Fortitude',
            'LUK (Luck)': 'Reversal Timing'
        },
        'secondary_stats': {
            'ATK (Attack)': 'Crisis Profit Potential (Extreme)',
            'DEF (Defense)': 'Drawdown Tolerance (High)',
            'VIT (Vitality)': 'Recovery Speed'
        },
        'special_abilities': [
            'Corpse Explosion: Profit from market crashes',
            'Bone Spear: Short overvalued assets',
            'Dark Ritual: High risk/reward trades'
        ]
    }
}
```

### 📊 **Real-Time Status Display:**

```python
def display_agent_status(agent_name, class_type, level, grade):
    """แสดง Status แบบ RPG Character Sheet"""
    
    status_display = f"""
╔══════════════════════════════════════════════════════════════╗
║                    🎮 SAC AGENT STATUS 🎮                    ║
╠══════════════════════════════════════════════════════════════╣
║ Name: {agent_name:<50} ║
║ Class: {class_type:<49} ║  
║ Level: {level:<3} │ Grade: {grade:<38} ║
║ Title: {LEVEL_SYSTEM[grade]['title']:<49} ║
╠══════════════════════════════════════════════════════════════╣
║                        📊 CORE STATS 📊                      ║
╠══════════════════════════════════════════════════════════════╣
║ 💰 Portfolio Value: ${portfolio_value:>12,.2f}               ║
║ 📈 Total Return: {total_return:>8.2f}% │ 📉 Max Drawdown: {max_drawdown:>6.2f}% ║
║ 🎯 Win Rate: {win_rate:>8.2f}% │ ⚡ Sharpe Ratio: {sharpe_ratio:>8.2f}      ║
║ 🔥 Avg Profit: ${avg_profit:>8.2f} │ 🛡️ Risk Score: {risk_score:>8.2f}        ║
╠══════════════════════════════════════════════════════════════╣
║                    ⚔️ CLASS ATTRIBUTES ⚔️                    ║
╠══════════════════════════════════════════════════════════════╣
"""
    
    # เพิ่ม Class-specific stats
    class_stats = CLASS_STATUS_ATTRIBUTES[class_type]
    
    for stat_name, stat_description in class_stats['primary_stats'].items():
        stat_value = calculate_stat_value(agent_name, stat_name)
        status_display += f"║ {stat_name}: {stat_value:>3}/100 │ {stat_description:<35} ║\n"
    
    status_display += "╠══════════════════════════════════════════════════════════════╣\n"
    status_display += "║                     🌟 SPECIAL ABILITIES 🌟                  ║\n"
    status_display += "╠══════════════════════════════════════════════════════════════╣\n"
    
    for ability in class_stats['special_abilities']:
        status_display += f"║ • {ability:<57} ║\n"
    
    status_display += "╠══════════════════════════════════════════════════════════════╣\n"
    status_display += "║                      🏆 ACHIEVEMENTS 🏆                      ║\n"
    status_display += "╠══════════════════════════════════════════════════════════════╣\n"
    
    achievements = get_agent_achievements(agent_name)
    for achievement in achievements:
        status_display += f"║ 🏅 {achievement:<55} ║\n"
    
    status_display += "╚══════════════════════════════════════════════════════════════╝"
    
    return status_display

# ตัวอย่างการใช้งาน
def show_current_agents():
    """แสดงสถานะ agents ปัจจุบันทั้งหมด"""
    agents = [
        {
            'name': 'sac_agent_20250619_151128_XXBF8G',
            'class': 'Human Paladin (Balanced SAC)',
            'level': 52,
            'grade': 'Grade C (Competent)',
            'portfolio_value': 102456.78,
            'total_return': 2.46,
            'max_drawdown': -22.46,
            'win_rate': 45.2,
            'sharpe_ratio': 0.801,
            'avg_profit': 245.67,
            'risk_score': 73.2
        },
        {
            'name': 'enhanced_sac_grade_A_20250620_012303_XAJI0Y',
            'class': 'Dark Elf Assassin (Scalping SAC)',
            'level': 68,
            'grade': 'Grade A (Advanced)',
            'portfolio_value': 156789.23,
            'total_return': 56.79,
            'max_drawdown': -31.25,
            'win_rate': 62.8,
            'sharpe_ratio': 1.342,
            'avg_profit': 1247.89,
            'risk_score': 45.7
        }
    ]
    
    for agent in agents:
        print(display_agent_status(**agent))
        print("\n" + "="*70 + "\n")
```

### 🎯 **Level Progression System:**

```python
LEVEL_PROGRESSION = {
    'experience_formula': 'timesteps_trained / 1000',
    'level_formula': 'min(85, max(1, int(experience_points ** 0.5)))',
    
    'level_milestones': {
        1: 'First Training Session',
        10: 'Basic Strategy Learned', 
        20: 'Class Selection Available',
        30: 'Advanced Features Unlocked',
        40: 'Multi-Asset Trading',
        50: 'Risk Management Mastery',
        60: 'Ensemble Participation',
        70: 'Research Contribution',
        80: 'Guild Leadership',
        85: 'Legendary Master Status'
    },
    
    'stat_growth_per_level': {
        'Human Knight': {'CON': +2, 'DEF': +2, 'VIT': +1, 'ATK': +0.5},
        'Elven Oracle': {'INT': +2, 'WIS': +2, 'MEN': +1, 'LUK': +1},
        'Dark Elf Assassin': {'DEX': +2, 'SPD': +2, 'ATK': +1.5, 'LUK': +1},
        'Dwarf Artisan': {'INT': +2, 'WIS': +1, 'CON': +1, 'MEN': +1.5},
        'Human Wizard': {'INT': +2.5, 'WIS': +1.5, 'MEN': +1.5, 'LUK': +0.5},
        'Elven Archer': {'DEX': +2, 'WIS': +1.5, 'LUK': +1.5, 'MEN': +1},
        'Human Paladin': {'STR': +1.5, 'CON': +1.5, 'WIS': +1.5, 'MEN': +1.5},
        'Human Necromancer': {'INT': +2, 'WIS': +1.5, 'MEN': +2, 'LUK': +1}
    }
}
```

### 🏆 **Achievement & Title System:**

```python
ACHIEVEMENT_SYSTEM = {
    'trading_achievements': {
        'First Blood': {
            'condition': 'first_profitable_trade',
            'reward': '+5 ATK, Title: "Profit Seeker"',
            'icon': '🩸'
        },
        'Diamond Hands': {
            'condition': 'hold_through_30_percent_drawdown',
            'reward': '+10 VIT, Title: "Diamond Hands"',
            'icon': '💎'
        },
        'Lightning Reflexes': {
            'condition': 'execute_trade_under_10ms',
            'reward': '+15 SPD, Title: "Speed Demon"',
            'icon': '⚡'
        },
        'Market Wizard': {
            'condition': 'beat_benchmark_12_months',
            'reward': '+20 INT, Title: "Market Wizard"',
            'icon': '🧙‍♂️'
        },
        'Risk Master': {
            'condition': 'maintain_under_10_percent_drawdown_1_year',
            'reward': '+25 DEF, Title: "Risk Master"',
            'icon': '🛡️'
        }
    },
    
    'class_specific_achievements': {
        'Human Knight': {
            'Fortress Builder': 'Survive 5 major market crashes',
            'Guardian Angel': 'Protect ensemble from losses 10 times',
            'Immovable Object': 'Never exceed 5% drawdown for 6 months'
        },
        'Dark Elf Assassin': {
            'Speed Demon': 'Execute 1000+ trades in one day',
            'Precision Strike': 'Achieve 95% win rate for one week',
            'Shadow Trader': 'Profit during market close hours'
        },
        'Elven Oracle': {
            'Prophet': 'Predict market direction 20 times correctly',
            'Divine Vision': 'Identify trend reversal 3 days early',
            'Blessing Master': 'Enhance team performance by 50%'
        }
    }
}
```

### 📈 **Stat Calculation Functions:**

```python
def calculate_stat_value(agent_name, stat_name):
    """คำนวณค่า stat ตาม performance metrics"""
    
    stat_mapping = {
        'ATK (Attack)': lambda metrics: min(100, metrics['mean_reward'] / 10),
        'DEF (Defense)': lambda metrics: min(100, 100 - abs(metrics['max_drawdown'])),
        'SPD (Speed)': lambda metrics: min(100, metrics['trade_frequency'] / 10),
        'INT (Intelligence)': lambda metrics: min(100, metrics['sharpe_ratio'] * 50),
        'WIS (Wisdom)': lambda metrics: min(100, metrics['win_rate']),
        'CON (Constitution)': lambda metrics: min(100, metrics['stability_score'] * 100),
        'DEX (Dexterity)': lambda metrics: min(100, metrics['execution_speed'] * 20),
        'VIT (Vitality)': lambda metrics: min(100, metrics['recovery_speed'] * 25),
        'MEN (Mental)': lambda metrics: min(100, metrics['consistency_score'] * 100),
        'LUK (Luck)': lambda metrics: min(100, (metrics['win_rate'] + metrics['profit_factor']) / 2),
        'STR (Strength)': lambda metrics: min(100, metrics['total_return'] * 2)
    }
    
    # โหลด metrics ของ agent
    metrics = load_agent_metrics(agent_name)
    
    if stat_name in stat_mapping:
        return int(stat_mapping[stat_name](metrics))
    else:
        return 50  # Default value
```

---

## 📊 SAC Hyperparameters = Character Stats

### 🧠 **Learning Rate** = **INT (Intelligence)**
- **ต่ำ (0.0001)**: เรียนรู้ช้า แต่มั่นคง (เหมือน Newbie ที่ระวัง)
- **กลาง (0.0003)**: สมดุล เหมาะสำหรับ most players
- **สูง (0.001+)**: เรียนรู้เร็ว แต่อาจ overshoot (เหมือน Pro ที่เสี่ยง)

```python
# เหมือนการ enchant weapon
LEARNING_RATES = {
    'Newbie': 0.0001,     # +0 weapon (ปลอดภัย)
    'Regular': 0.0003,    # +3 weapon (standard)
    'Pro': 0.001,         # +6 weapon (risky but powerful)
    'Whale': 0.01         # +16 weapon (high risk/reward)
}
```

### 💾 **Buffer Size** = **Inventory Slots**
- **เล็ก (100K)**: กระเป๋าเล็ก เก็บของน้อย
- **กลาง (500K)**: กระเป๋าปกติ เพียงพอใช้งาน
- **ใหญ่ (1M+)**: กระเป๋าใหญ่ เก็บประสบการณ์ได้เยอะ

```python
BUFFER_SIZES = {
    'F-Grade': 100_000,    # เหมือน Basic Inventory
    'D-Grade': 250_000,    # เหมือน Expanded Inventory  
    'C-Grade': 500_000,    # เหมือน Premium Inventory
    'B-Grade': 1_000_000,  # เหมือน Warehouse Access
    'A-Grade': 2_000_000,  # เหมือน Guild Warehouse
}
```

### ⚔️ **Batch Size** = **Party Size**
- **เล็ก (32)**: Solo hunting (เรียนรู้ช้า)
- **กลาง (64-128)**: Small party (สมดุล)
- **ใหญ่ (256+)**: Full party (เรียนรู้เร็ว แต่ใช้ทรัพยากรเยอะ)

### 🛡️ **Entropy Coefficient** = **Luck Stat**
- **ต่ำ**: Conservative play (เล่นปลอดภัย)
- **กลาง**: Balanced exploration 
- **สูง**: High risk/reward (เหมือน Critical Rate สูง)

---

## 🎭 Grade System = Class Advancement

### 📈 **Agent Evolution Path:**

```
👶 Grade N (Novice)
├─ เหมือน Human Fighter Level 1-20
├─ 🏋️ Training: 50K timesteps
├─ 💪 Stats: Basic everything
└─ 🎯 Goal: เรียนรู้พื้นฐาน

👨‍🎓 Grade D (Developing) 
├─ เหมือน Human Fighter Level 20-40
├─ 🏋️ Training: 100K timesteps  
├─ 💪 Stats: Improved performance
└─ 🎯 Goal: หา playstyle ที่เหมาะสม

⚔️ Grade C (Competent)
├─ เหมือน Treasure Hunter Level 40-52
├─ 🏋️ Training: 200K timesteps
├─ 💪 Stats: SDE enabled, better risk management
└─ 🎯 Goal: เริ่มทำกำไรได้มั่นคง

🏆 Grade B (Proficient)
├─ เหมือน Treasure Hunter Level 52-61
├─ 🏋️ Training: 500K timesteps
├─ 💪 Stats: High performance, ensemble ready
└─ 🎯 Goal: แข่งขันได้ระดับสูง

👑 Grade A (Advanced)
├─ เหมือน Treasure Hunter Level 61-76
├─ 🏋️ Training: 1M timesteps
├─ 💪 Stats: Advanced features, optimal performance
└─ 🎯 Goal: เป็น top performer

🌟 Grade S (Supreme)
├─ เหมือน Treasure Hunter Level 76+ (3rd Class)
├─ 🏋️ Training: 2M+ timesteps
├─ 💪 Stats: Research-grade, all features unlocked
└─ 🎯 Goal: State-of-the-art performance
```

---

## ⚔️ Performance Metrics = Combat Stats

### 📊 **Agent Status Window:**

```
╔═══════════ SAC Agent Status ═══════════╗
║ Name: sac_agent_20250619_151128_XXBF8G ║
║ Class: Treasure Hunter (Grade C)       ║
║ Level: 52 (200K timesteps)             ║
╠════════════════════════════════════════╣
║ 💰 Wealth (Portfolio): $100,000        ║
║ ⚔️  Attack (Mean Reward): 245.67       ║
║ 🛡️  Defense (Max Drawdown): -22.46%    ║
║ 🎯 Accuracy (Win Rate): 45.2%          ║
║ 🏃 Speed (Sharpe Ratio): 0.801         ║
║ 🍀 Luck (Stability): 0.73              ║
║ 🧠 Intelligence (Learning Rate): 0.0003║
║ 💾 Memory (Buffer Size): 500K          ║
╚════════════════════════════════════════╝

🏆 Achievements:
├─ 🥇 Survived 22% Drawdown (Iron Will)
├─ 💎 Consistent BTC Trading (Diamond Hands)  
├─ 📈 12 Technical Indicators Mastery
└─ ⚡ 568 Training Episodes Completed
```

### 🎖️ **Performance Comparison (PvP Rankings):**

```python
AGENT_RANKINGS = {
    'S-Tier (Hero Grade)': {
        'mean_reward': '> 500',
        'sharpe_ratio': '> 1.5', 
        'max_drawdown': '< 10%',
        'description': 'Legendary players, กำไรสม่ำเสมอ risk ต่ำ'
    },
    'A-Tier (Advanced)': {
        'mean_reward': '300-500',
        'sharpe_ratio': '1.2-1.5',
        'max_drawdown': '10-15%', 
        'description': 'เทพระดับ server, stable profit'
    },
    'B-Tier (Competent)': {
        'mean_reward': '100-300',
        'sharpe_ratio': '0.8-1.2',
        'max_drawdown': '15-20%',
        'description': 'ผู้เล่นดี มีกำไรบ้างขาดทุนบ้าง'
    },
    'C-Tier (Learning)': {
        'mean_reward': '0-100', 
        'sharpe_ratio': '0.5-0.8',
        'max_drawdown': '20-30%',
        'description': 'ยังเรียนรู้อยู่ (เหมือน Agent ปัจจุบัน)'
    },
    'D-Tier (Newbie)': {
        'mean_reward': '< 0',
        'sharpe_ratio': '< 0.5', 
        'max_drawdown': '> 30%',
        'description': 'มือใหม่ ขาดทุนมากกว่ากำไร'
    }
}
```

---

## 🛠️ Equipment & Skills = Model Configuration

### ⚔️ **Primary Weapon: Policy Network**
```python
POLICY_WEAPONS = {
    'Iron Sword (Basic)': 'MlpPolicy with 64 neurons',
    'Mithril Sword (Advanced)': 'MlpPolicy with 256 neurons', 
    'Dragon Sword (S-Grade)': 'Custom Policy with attention',
    'Blessed Weapon (+16)': 'Ensemble of multiple policies'
}
```

### 🛡️ **Armor: Risk Management**
```python
ARMOR_SETS = {
    'Leather Set': 'Basic position sizing (ไม่มี stop-loss)',
    'Chain Set': 'Simple stop-loss (20% drawdown limit)',
    'Plate Set': 'Advanced risk management (multiple stops)',
    'Dragon Set': 'Dynamic position sizing + VaR limits'
}
```

### 💍 **Accessories: Technical Indicators**
```python
INDICATOR_ACCESSORIES = {
    'Ring of SMA': '+5 Trend Following',
    'Ring of RSI': '+10 Mean Reversion', 
    'Necklace of MACD': '+8 Momentum Detection',
    'Earring of Bollinger': '+12 Volatility Sensing',
    'Tattoo of Volume': '+6 Market Strength Reading'
}
```

---

## 🏰 Training Grounds = Market Environments

### 🌍 **Training Locations:**

```python
TRAINING_ENVIRONMENTS = {
    'Talking Island (Newbie)': {
        'market': 'Demo/Paper Trading',
        'difficulty': 'Very Easy',
        'rewards': 'Small but safe',
        'monsters': 'Predictable price movements'
    },
    
    'Elven Forest (Intermediate)': {
        'market': 'BTC Bull Market 2020-2021', 
        'difficulty': 'Easy-Medium',
        'rewards': 'Good profits',
        'monsters': 'Mostly uptrend with small dips'
    },
    
    'Cruma Tower (Advanced)': {
        'market': 'Mixed Market 2022-2023',
        'difficulty': 'Medium-Hard', 
        'rewards': 'Variable profits',
        'monsters': 'Volatile swings, bear/bull mix'
    },
    
    'Dragon Valley (Expert)': {
        'market': 'Crypto Winter 2022',
        'difficulty': 'Very Hard',
        'rewards': 'High risk/reward',
        'monsters': 'Brutal bear market, flash crashes'
    },
    
    'Antharas Lair (Nightmare)': {
        'market': 'Live Trading with Real Money',
        'difficulty': 'Extreme',
        'rewards': 'Real profits/losses',
        'monsters': 'Market makers, whales, black swans'
    }
}
```

---

## 🎯 Quest System = Trading Objectives

### 📜 **Daily Quests (Short-term):**
- 🎯 **"Survive the Day"**: ไม่ขาดทุนเกิน 2%
- 💰 **"Small Gains"**: ทำกำไร 0.5% ในวันนี้  
- 📊 **"Read the Market"**: ทำนายทิศทางถูก 3/5 ครั้ง
- ⚡ **"Quick Reflexes"**: Execute orders ใน < 100ms

### 📋 **Weekly Quests (Medium-term):**
- 🏆 **"Consistent Trader"**: กำไร 5 วันจาก 7 วัน
- 🛡️ **"Risk Manager"**: Max drawdown < 10% ทั้งสัปดาห์
- 📈 **"Trend Follower"**: จับ trend ใหญ่ได้ 1 ครั้ง
- 💎 **"Diamond Hands"**: Hold winning position > 3 วัน

### 🏛️ **Epic Quests (Long-term):**
- 👑 **"Market Master"**: Beat buy-and-hold strategy
- 🌟 **"Sharpe Legend"**: Achieve Sharpe ratio > 1.5
- 🏆 **"Profit King"**: 100% annual return
- 🛡️ **"Fortress Builder"**: Max drawdown < 5% ทั้งปี

---

## 🔄 Skill Tree = Algorithm Improvements

### 🌱 **Passive Skills (Always Active):**
```
Technical Analysis Mastery
├─ Level 1: Basic SMA, EMA
├─ Level 5: RSI, MACD signals  
├─ Level 10: Bollinger Bands strategy
├─ Level 15: Volume analysis
└─ Level 20: Multi-timeframe analysis

Risk Management Expert  
├─ Level 1: Position sizing
├─ Level 5: Stop-loss orders
├─ Level 10: Portfolio diversification
├─ Level 15: VaR calculations
└─ Level 20: Dynamic hedging

Market Psychology Reader
├─ Level 1: Fear & Greed detection
├─ Level 5: Sentiment analysis
├─ Level 10: News impact assessment  
├─ Level 15: Whale movement tracking
└─ Level 20: Market regime identification
```

### ⚡ **Active Skills (Manual Activation):**
```
Profit Strike (Ultimate)
├─ Cooldown: 24 hours
├─ Effect: 2x reward for next 4 hours
├─ Cost: High entropy (more exploration)
└─ Risk: Potential large drawdown

Emergency Exit (Defensive)  
├─ Cooldown: 1 hour
├─ Effect: Close all positions immediately
├─ Cost: Transaction fees
└─ Benefit: Prevent catastrophic loss

Market Scan (Information)
├─ Cooldown: 30 minutes  
├─ Effect: Analyze 100+ assets instantly
├─ Cost: Computational resources
└─ Benefit: Find best trading opportunities
```

---

## 🏆 Boss Battles = Market Challenges

### 🐉 **Dragon Raid: Black Swan Events**
```
Boss: Flash Crash Dragon
├─ HP: Market confidence  
├─ Attack: -50% in 30 minutes
├─ Special: Liquidity drain
├─ Weakness: Stop-loss discipline
├─ Reward: Survival experience +1000
└─ Drop: Risk management wisdom
```

### 👹 **Demon Battle: Bear Market**
```
Boss: The Bear Lord
├─ HP: 18 months duration
├─ Attack: Continuous -80% decline  
├─ Special: Hope crusher (false rallies)
├─ Weakness: Dollar cost averaging
├─ Reward: Diamond hands achievement
└─ Drop: Market cycle knowledge
```

### 🤖 **PvP Arena: Bot Competition**
```
Enemy: High-Frequency Trading Bots
├─ Speed: Microsecond execution
├─ Strategy: Front-running, arbitrage
├─ Advantage: Direct market access
├─ Our edge: Pattern recognition, adaptability
├─ Victory condition: Consistent alpha generation
└─ Reward: Market maker status
```

---

## 📊 Guild System = Ensemble Methods

### 🏰 **Trading Guild Structure:**

```python
ENSEMBLE_GUILD = {
    'Guild Master': 'Meta-learner (เลือก strategy ดีที่สุด)',
    'Officers': [
        'Trend Follower SAC',    # ชำนาญ bull markets
        'Mean Reversion SAC',    # ชำนาญ sideways
        'Momentum SAC',          # ชำนาญ breakouts  
        'Scalping SAC'           # ชำนาญ quick trades
    ],
    'Members': 'Multiple policy variations',
    'Guild Skills': [
        'Collective Intelligence',
        'Risk Diversification', 
        'Strategy Redundancy',
        'Performance Boosting'
    ]
}
```

### 🤝 **Guild Raids (Portfolio Management):**
- **Tank**: Conservative SAC (เป็น anchor, risk management)
- **DPS**: Aggressive SAC (หากำไรหลัก, high risk/reward) 
- **Support**: Balanced SAC (เสริมทั้งคู่, adaptive)
- **Healer**: Hedge SAC (ป้องกันขาดทุน, counter-trend)

---

## 💰 Economy System = Portfolio Management

### 🏦 **Bank Account = Portfolio Value**
```
Starting Capital: $100,000 (เหมือนเงิน spawn ใหม่)
├─ Equipment Budget: 20% (สำหรับ infrastructure)
├─ Training Budget: 30% (สำหรับ backtesting)  
├─ Active Trading: 40% (เงินเทรดจริง)
└─ Emergency Fund: 10% (สำหรับ black swan)
```

### 💎 **Rare Items = Profitable Strategies**
```
Common (White): Basic buy/hold (+5% annual)
Uncommon (Blue): Simple momentum (+8% annual)  
Rare (Yellow): Technical analysis (+12% annual)
Epic (Purple): Advanced SAC (+20% annual)
Legendary (Red): Perfect market timing (+50% annual)
Artifact (Gold): Holy Grail strategy (+100% annual) 
```

---

## 🎮 Character Development Roadmap

### 📈 **Level 1-20: Noob Phase**
```python
NOOB_DEVELOPMENT = {
    'Focus': 'เรียนรู้พื้นฐาน',
    'Training': 'Paper trading เท่านั้น',
    'Goals': [
        'เข้าใจ candlestick charts',
        'รู้จัก basic indicators', 
        'ไม่ blow account ใน demo'
    ],
    'Common Mistakes': [
        'Over-leverage (เหมือนใส่ equipment หนักเกินไป)',
        'FOMO trading (เหมือน rush เข้า dungeon)',
        'No stop-loss (เหมือนไม่ใส่ armor)'
    ]
}
```

### 📈 **Level 20-40: Learning Phase**  
```python
LEARNING_DEVELOPMENT = {
    'Focus': 'หา playstyle',
    'Training': 'Small real money',
    'Goals': [
        'Develop consistent strategy',
        'Learn risk management',
        'Build emotional discipline'
    ],
    'Upgrades': [
        'Better technical indicators',
        'Position sizing rules',
        'Basic portfolio theory'
    ]
}
```

### 📈 **Level 40-60: Competent Phase**
```python
COMPETENT_DEVELOPMENT = {
    'Focus': 'Optimization และ consistency',
    'Training': 'Moderate position sizes', 
    'Goals': [
        'Beat buy-and-hold consistently',
        'Sharpe ratio > 1.0',
        'Max drawdown < 20%'
    ],
    'Advanced Features': [
        'Multi-timeframe analysis',
        'Correlation analysis', 
        'Advanced risk metrics'
    ]
}
```

### 📈 **Level 60+: Master Phase**
```python
MASTER_DEVELOPMENT = {
    'Focus': 'Alpha generation และ innovation',
    'Training': 'Full capital deployment',
    'Goals': [
        'Consistent market outperformance', 
        'Risk-adjusted returns optimization',
        'Strategy research และ development'
    ],
    'Legendary Skills': [
        'Market regime detection',
        'Alternative data integration',
        'Ensemble method mastery'
    ]
}
```

---

## 🏅 Achievement System

### 🥇 **Trading Achievements:**
```
🏆 "First Blood": กำไรครั้งแรก
💎 "Diamond Hands": Hold position ผ่าน 30% drawdown
⚡ "Lightning Reflexes": Execute trade ใน < 10ms  
🎯 "Sniper": 90%+ win rate ใน 1 สัปดาห์
🛡️ "Guardian": ป้องกัน portfolio จาก major crash
🌟 "Market Wizard": Beat benchmark 12 เดือนติด
👑 "Legendary Trader": $1M+ profit ใน 1 ปี
🦄 "Unicorn Hunter": จับ 10x opportunity
```

### 🎖️ **Technical Achievements:**
```
🧠 "Indicator Master": ใช้ 20+ indicators ได้อย่างมีประสิทธิภาพ
⚙️ "Optimizer": Fine-tune hyperparameters เป็น A-grade
🔬 "Researcher": พัฒนา custom features ที่ work
🤖 "AI Trainer": เทรน model ที่มี Sharpe > 1.5
📊 "Data Scientist": สร้าง ensemble ที่ outperform single model
```

---

## 🎯 สรุป: การเล่น SAC Agent เหมือน Lineage 2

### 🎮 **Mindset การเล่น:**
1. **Character Building**: ค่อยๆ พัฒนา hyperparameters เหมือนเพิ่ม stats
2. **Equipment Upgrade**: ปรับปรุง features และ indicators เหมือนเปลี่ยน gear
3. **Skill Training**: เรียนรู้จาก backtest เหมือนฝึก skill ในเกม
4. **Guild Play**: ใช้ ensemble methods เหมือนเล่นเป็นทีม
5. **Boss Fights**: เตรียมพร้อมสำหรับ market crashes และ black swans

### 🏆 **Victory Conditions:**
- **PvE (vs Market)**: Beat buy-and-hold consistently
- **PvP (vs Other Traders)**: Generate alpha และ risk-adjusted returns
- **Guild Wars (vs Institutions)**: Compete กับ hedge funds และ HFT bots
- **Castle Siege (Ultimate)**: สร้าง systematic edge ที่ sustainable

### 💡 **Pro Tips:**
```
🎯 เริ่มจาก Grade N และค่อยๆ level up
⚔️ Focus พัฒนา 1-2 skills ก่อน (ไม่ต้อง master ทุกอย่าง)  
🛡️ Risk management สำคัญกว่า profit maximization
🏰 เมื่อ stable แล้วค่อยคิดถึง ensemble methods
📊 Monitor performance metrics เหมือนดู character stats
🔄 Continuous improvement เหมือนการ farm exp
```

---

*"In the world of crypto trading, every SAC Agent is a hero's journey. Start as a humble newbie, face the dragons of market volatility, learn from defeats, upgrade your strategies, and eventually become a legendary trader who conquers the markets!"* 🗡️📈✨

**Remember**: เหมือนใน Lineage 2, ไม่มี shortcut to greatness - ต้องใช้เวลา, ความอดทน, และการเรียนรู้อย่างต่อเนื่อง! 🎮🚀
</rewritten_file> 