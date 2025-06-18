# 🎯 คู่มือการประเมิน Agent ใน Streamlit UI

## 🚀 เริ่มต้นใช้งาน

### 1. เปิด Streamlit UI
```bash
cd ui
streamlit run app.py
```
**URL:** http://localhost:8501

### 2. แก้ปัญหา "No trained models found"

ปัญหานี้เกิดจากการแก้ไข evaluate.py แล้ว หากยังพบปัญหา ให้ทำดังนี้:

#### ✅ **วิธีแก้ไขที่ 1: ใช้ Models ที่มีอยู่**
Models ที่พร้อมใช้งาน:
- `minimal_crypto_ppo.zip` - จาก main.py
- `simple_advanced_crypto_ppo.zip` - จาก simple_advanced_agent.py

#### ✅ **วิธีแก้ไขที่ 2: สร้าง Model ใหม่**
```bash
# รัน agent เพื่อสร้าง model
python simple_advanced_agent.py
# หรือ
python main.py
```

## 📊 วิธีการประเมิน Agent

### **Step 1: เลือก Section**
- ไปที่ **Sidebar** → เลือก **"Evaluate"**

### **Step 2: เลือกโมเดล**
Available Models:
```
📋 Models List:
├── minimal_crypto_ppo          # จาก main.py  
└── simple_advanced_crypto_ppo  # จาก simple_advanced_agent.py (แนะนำ)
```

### **Step 3: เลือกข้อมูลประเมิน**
Available Data Files:
```
📁 data/:
├── crypto_data.csv                    # ข้อมูลหลัก (62KB)
├── simple_advanced_crypto_data.csv    # ข้อมูลสำหรับ advanced (522KB)
└── advanced_crypto_data.csv           # ข้อมูลเต็ม (606KB)

📁 simple_data/:
└── simple_crypto_data.csv             # ข้อมูลแบบง่าย (79KB)
```

### **Step 4: กำหนดช่วงเวลา**
- **Start Date:** วันเริ่มต้นประเมิน
- **End Date:** วันสิ้นสุดประเมิน
- **แนะนำ:** 30-90 วันล่าสุด

### **Step 5: ตั้งค่าการลงทุน**
```
🔧 Trading Parameters:
├── Initial Investment: $100,000 (ปรับได้)
├── Transaction Cost: 0.1%
└── Max Holdings: 100 shares
```

### **Step 6: รันการประเมิน**
1. คลิก **"📊 Run Evaluation"**
2. รอผลลัพธ์ (1-3 นาทีขึ้นกับข้อมูล)

## 📈 ตีความผลลัพธ์

### **A. Performance Comparison Graph**
```
📊 กราฟแสดง:
├── เส้น Agent Performance (สีต่างๆ)
├── เส้น Buy & Hold Benchmark (เส้นประ)
└── เส้น Initial Investment (เส้นจุด)
```

### **B. Performance Metrics Table**
| Metric | คำอธิบาย | เป้าหมาย |
|--------|----------|----------|
| **Total Return (%)** | ผลตอบแทนรวม | > 0% |
| **vs. Buy & Hold** | เปรียบเทียบ B&H | > 0% |
| **Sharpe Ratio** | ความคุ้มค่าความเสี่ยง | > 0.5 |
| **Max Drawdown (%)** | การสูญเสียสูงสุด | < 20% |

## 🎯 ตัวอย่างการใช้งาน

### **Scenario 1: ประเมิน Simple Advanced Agent**
```
1. Model: "simple_advanced_crypto_ppo"
2. Data: "simple_crypto_data.csv"
3. Period: 30 วันล่าสุด
4. Investment: $100,000
```

**ผลลัพธ์ที่คาดหวัง:**
- Total Return: ~10.81%
- vs. Buy & Hold: +6.04%
- Sharpe Ratio: ~0.653

### **Scenario 2: ประเมิน Basic Agent**
```
1. Model: "minimal_crypto_ppo"
2. Data: "crypto_data.csv"
3. Period: 60 วันล่าสุด
4. Investment: $50,000
```

### **Scenario 3: เปรียบเทียบข้อมูลแตกต่างกัน**
```
รันการประเมินแยกกัน:
- simple_crypto_data.csv (ข้อมูลง่าย)
- crypto_data.csv (ข้อมูลหลัก)
เพื่อดูว่า agent ทำงานได้ดีกับข้อมูลชุดไหน
```

## ⚠️ แก้ไขปัญหาที่พบบ่อย

### **1. "No trained models found"**
```bash
# ตรวจสอบไฟล์ models
ls models/

# หากไม่มี ให้รัน:
python simple_advanced_agent.py
```

### **2. "ไม่พบไฟล์ข้อมูล"**
```bash
# ตรวจสอบไฟล์ data
ls data/
ls simple_data/

# หากไม่มี ให้รัน agent เพื่อสร้างข้อมูล
python simple_advanced_agent.py
```

### **3. "เกิดข้อผิดพลาดในการโหลดโมเดล"**
- ตรวจสอบว่าไฟล์ .zip ไม่เสียหาย
- ลองรัน agent ใหม่เพื่อสร้าง model ใหม่

### **4. "Memory Error"**
- ลดช่วงเวลาการประเมิน
- ใช้ข้อมูลขนาดเล็กกว่า (crypto_data.csv แทน advanced_crypto_data.csv)

### **5. "Processing ช้า"**
- ลดจำนวนวันที่ประเมิน
- ปิดโปรแกรมอื่นๆ เพื่อเพิ่ม RAM

## 💡 เทคนิคการใช้งาน

### **1. การเปรียบเทียบ Agent**
```
รัน evaluation แยกกัน:
├── simple_advanced_crypto_ppo (ข้อมูลเดียวกัน)
└── minimal_crypto_ppo (ข้อมูลเดียวกัน)

จากนั้นเปรียบเทียบผลลัพธ์
```

### **2. การทดสอบกับช่วงเวลาต่างกัน**
```
ทดสอบ Agent เดียวกันกับ:
├── ช่วง Bull Market
├── ช่วง Bear Market  
└── ช่วง Sideways Market
```

### **3. การปรับแต่ง Initial Investment**
```
ทดสอบกับเงินลงทุนต่างกัน:
├── $10,000 (เงินลงทุนน้อย)
├── $100,000 (มาตรฐาน)
└── $1,000,000 (เงินลงทุนมาก)
```

## 📥 Download และวิเคราะห์เพิ่มเติม

### **1. Download CSV Report**
- คลิก **"📥 Download Comparison Report"**
- ไฟล์ CSV จะมีข้อมูล:
  - วันที่
  - Portfolio Values
  - Benchmark Values

### **2. วิเคราะห์ใน Excel/Python**
```python
import pandas as pd
import matplotlib.pyplot as plt

# โหลดข้อมูล
df = pd.read_csv('version_comparison_report.csv')

# สร้างกราฟเพิ่มเติม
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Portfolio_Value_v1'], label='Agent')
plt.plot(df['Date'], df['Benchmark_Value'], label='Buy & Hold')
plt.legend()
plt.show()
```

## 🎉 สรุป

**Streamlit UI ช่วยให้:**
✅ ประเมิน Agent ได้ง่าย ไม่ต้องเขียนโค้ด  
✅ เปรียบเทียบประสิทธิภาพแบบ Visual  
✅ ปรับพารามิเตอร์ได้ตามต้องการ  
✅ Export ผลลัพธ์เป็น CSV  
✅ ใช้งานแบบ Interactive  

**🚀 เริ่มต้นใช้งาน Streamlit UI เพื่อประเมิน Crypto Trading Agent ของคุณเลย!** 