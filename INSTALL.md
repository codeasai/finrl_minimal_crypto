# 🛠️ Installation Guide

## Quick Start (Recommended)

### Option 1: Using pip (Fastest)

```bash
# Clone repository
git clone <repository-url>
cd finrl_minimal_crypto

# Install dependencies
pip install -r requirements.txt

# Run core agent
python main.py
```

### Option 2: Using conda

```bash
# Clone repository
git clone <repository-url>
cd finrl_minimal_crypto

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate finrl_crypto

# Run core agent
python main.py
```

## Detailed Installation

### 1. System Requirements

- **Python:** 3.9.x (tested with 3.9.23)
- **OS:** Windows 10/11, macOS, Linux
- **RAM:** 8GB+ recommended
- **Storage:** 2GB+ free space

### 2. Core Dependencies (Tested Versions)

```
finrl==0.3.7
gymnasium==1.1.1
stable-baselines3==2.6.0
numpy==1.26.4
pandas==2.0.3
yfinance==0.2.61
matplotlib==3.9.4
scikit-learn==1.6.1
```

### 3. Installation Steps

#### Step 1: Clone Repository

```bash
git clone <repository-url>
cd finrl_minimal_crypto
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv finrl_env
source finrl_env/bin/activate  # Linux/macOS
# OR
finrl_env\Scripts\activate     # Windows

# Using conda
conda create -n finrl_crypto python=3.9
conda activate finrl_crypto
```

#### Step 3: Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt

# Development installation (includes Jupyter, TA-Lib, etc.)
pip install -r requirements-dev.txt
```

### 4. Optional: TA-Lib Installation

TA-Lib provides advanced technical indicators. **Recommended method: conda-forge**

#### ✅ Recommended (All Platforms):

```bash
# Using conda-forge (works for Windows, macOS, Linux)
conda install ta-lib -c conda-forge

# Verification
python -c "import talib; print('TA-Lib version:', talib.__version__)"
```

#### Alternative Methods:

**Windows:**

```bash
# Method 1: Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.xx-cp39-cp39-win_amd64.whl

# Method 2: Using conda-forge (recommended)
conda install ta-lib -c conda-forge
```

**macOS:**

```bash
# Method 1: Using conda-forge (recommended)
conda install ta-lib -c conda-forge

# Method 2: Using homebrew + pip
brew install ta-lib
pip install ta-lib
```

**Linux:**

```bash
# Method 1: Using conda-forge (recommended)
conda install ta-lib -c conda-forge

# Method 2: Compile from source
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

#### 🚨 Important for Streamlit Users:

```bash
# Make sure TA-Lib and Streamlit are in the same environment
conda activate your_env
conda install ta-lib streamlit -c conda-forge

# Test both work together
python -c "import streamlit; import talib; print('✅ Ready for advanced indicators!')"
```

### 5. Verification

Test your installation:

```bash
# Test basic imports
python -c "import finrl, gymnasium, stable_baselines3; print('✅ Core packages OK')"

# Test data loading
python -c "import yfinance; print('✅ Data APIs OK')"

# Test visualization
python -c "import matplotlib, plotly; print('✅ Visualization OK')"

# Run verification script
python notebooks/verification_script.py
```

## Environment Setup

### Using conda (Recommended for beginners)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate finrl_crypto

# Update environment (if needed)
conda env update -f environment.yml
```

#### 🪟 Special Notes for Windows Users:

```bash
# If conda activate doesn't work in Git Bash:
# 1. Initialize conda first
conda init bash
source ~/.bash_profile

# 2. Then activate environment
conda activate finrl_crypto

# Alternative: Use PowerShell or Command Prompt
# PowerShell: Works better with conda on Windows
conda activate finrl_crypto

# Command Prompt: Also reliable
conda activate finrl_crypto
```

### Using pip + venv

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Development Setup

For development and advanced features:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Start Jupyter Lab
jupyter lab

# Run Streamlit UI
cd ui
streamlit run app.py
```

## Common Issues & Solutions

### Issue 1: AttributeError with numpy

```bash
# Solution: Use specific numpy version
pip install numpy==1.26.4
```

### Issue 2: TA-Lib installation fails

```bash
# Solution: Skip TA-Lib for now
# The agents work without TA-Lib
# Use simple_advanced_agent.py instead of advanced_crypto_agent.py
```

### Issue 3: CUDA/GPU issues

```bash
# Solution: Install CPU version of PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue 4: Memory issues

```bash
# Solution: Reduce data size in config.py
# Set smaller date ranges or fewer symbols
```

### Issue 5: conda activate ไม่ทำงานบน Windows (Git Bash)

```bash
# Problem: "CondaError: Run 'conda init' before 'conda activate'"

# Solution 1: Initialize conda for bash
conda init bash

# Then reload bash profile
source ~/.bash_profile

# Now activate environment
conda activate your_env_name

# Solution 2: Alternative activation methods
# Method A: Use conda activate directly
C:\ProgramData\miniconda3\Scripts\activate your_env_name

# Method B: Use PowerShell instead of Git Bash
# Open PowerShell and run:
conda activate your_env_name

# Method C: Use conda run (without activation)
conda run -n your_env_name python script.py

# Verification: Check if environment is active
conda info --envs
# You should see * next to your active environment
```

### Issue 6: TA-Lib ไม่ทำงานใน Streamlit

```bash
# Problem: "ModuleNotFoundError: No module named 'talib'" ใน Streamlit
# แม้ว่า TA-Lib จะติดตั้งอยู่ใน environment แล้ว

# Root cause: Streamlit และ TA-Lib อยู่คนละ environment

# Solution 1: ติดตั้ง Streamlit ใน environment เดียวกันกับ TA-Lib
conda activate your_env_name
conda install streamlit -c conda-forge

# Solution 2: ติดตั้ง TA-Lib ใน environment ที่มี Streamlit
conda activate your_streamlit_env
conda install ta-lib -c conda-forge

# Solution 3: ติดตั้งทั้งคู่ใน environment ใหม่
conda create -n finrl_full python=3.9
conda activate finrl_full
conda install ta-lib streamlit -c conda-forge

# Verification: ทดสอบว่าทั้งคู่อยู่ใน environment เดียวกัน
python -c "import streamlit; import talib; print('✅ Both available!')"

# เริ่ม Streamlit
streamlit run ui/app.py
```

## Package Versions Compatibility

### ✅ Tested & Working Combinations:

- **Python 3.9.23** + **numpy 1.26.4** + **pandas 2.0.3**
- **FinRL 0.3.7** + **stable-baselines3 2.6.0** + **gymnasium 1.1.1**

### ⚠️ Known Issues:

- **numpy 2.x** causes AttributeError in FinRL
- **pandas 2.2+** may have compatibility issues
- **Python 3.12+** not fully tested

## Quick Test

After installation, run this quick test:

```bash
# Test conda environment (if using conda)
conda info --envs
# Should show your active environment with *

# Test basic functionality
python -c "
import finrl
import pandas as pd
import numpy as np
print(f'✅ FinRL: {finrl.__version__}')
print(f'✅ Pandas: {pd.__version__}')
print(f'✅ NumPy: {np.__version__}')
print('🎉 Installation successful!')
"
```

## Next Steps

1. **Start with basic agent:** `python main.py`
2. **Try advanced agent:** `python simple_advanced_agent.py`
3. **Explore notebooks:** `jupyter notebook notebooks/`
4. **Use Streamlit UI:** `cd ui && streamlit run app.py`

## Support

If you encounter issues:

1. Check this installation guide
2. Review `install_talib.md` for TA-Lib issues
3. Check GitHub issues
4. Create new issue with error details

---

**🎯 Recommended: Start with `pip install -r requirements.txt` and `python main.py`**
