# 🧠 Crypto RL Agent Dashboard - Streamlit Guide

## การรัน Streamlit Application

ระบบใช้ **Streamlit Multi-Page App** structure ที่ทันสมัยและเป็นมาตรฐาน

### วิธีการรัน

```bash
# เข้าไปยังโฟลเดอร์ ui
cd ui

# รัน Streamlit app
streamlit run app.py
```

Application จะเปิดที่ `http://localhost:8501`

## โครงสร้างแอปพลิเคชัน

### 📁 โครงสร้างไฟล์

```
ui/
├── app.py                     # หน้าแรก (Main Dashboard)
├── pages/                     # หน้าต่างๆ ของแอปพลิเคชัน
│   ├── 1_Data_Loader.py      # โหลดข้อมูล
│   ├── 2_Data_Prepare.py     # เตรียมข้อมูล
│   ├── 3_Train_Agent.py      # เทรน Agent
│   ├── 4_Test_Agent.py       # ทดสอบ Agent
│   ├── 5_Evaluate_Performance.py  # ประเมินผล
│   └── 6_Manage_Agents.py    # จัดการ Agents
└── pipeline/                  # โมดูล backend (ยังคงใช้งานอยู่)
    ├── data_loader.py
    ├── data_prepare.py
    ├── data_prepare_ui.py
    ├── train.py
    ├── test.py
    ├── evaluate.py
    └── agent_manager.py
```

### 🗺️ Navigation

Streamlit จะสร้าง navigation menu อัตโนมัติจากไฟล์ในโฟลเดอร์ `pages/`:

- **🏠 Crypto RL Agent Dashboard** (หน้าแรก)
- **Data Loader** - โหลดข้อมูล cryptocurrency
- **Data Prepare** - เตรียมข้อมูลและเพิ่ม technical indicators
- **Train Agent** - เทรน RL agent ด้วย PPO algorithm
- **Test Agent** - ทดสอบประสิทธิภาพของ agent
- **Evaluate Performance** - วิเคราะห์และเปรียบเทียบผลการเทรด
- **Manage Agents** - จัดการ agents ที่มีอยู่

## คำแนะนำการใช้งาน

### 🚀 ขั้นตอนการเริ่มต้น

1. **หน้าแรก (Dashboard)**
   - แสดงสถานะระบบโดยรวม
   - ตรวจสอบจำนวนไฟล์ข้อมูลและโมเดล
   - ตรวจสอบสถานะ GPU

2. **📊 Data Loader**
   - โหลดข้อมูลจาก exchanges (Binance, Bybit, OKX)
   - กำหนดช่วงเวลาและคู่เทรด
   - บันทึกข้อมูลในรูปแบบ CSV

3. **🔧 Data Prepare**
   - เลือกไฟล์ข้อมูลที่จะประมวลผล
   - เพิ่ม Technical Indicators (SMA, EMA, RSI, MACD, BB)
   - Normalize ข้อมูลและตรวจสอบคุณภาพ
   - บันทึกข้อมูลที่ประมวลผลแล้ว

4. **🎯 Train Agent**
   - เลือก Grade การเทรน (N, D, C, B, A, S)
   - ตั้งค่า parameters หรือใช้ค่า default
   - เทรนโมเดลใหม่หรือเทรนต่อจากโมเดลเดิม
   - ติดตามความคืบหน้าการเทรน

5. **🧪 Test Agent**
   - ทดสอบ agent ที่เทรนแล้ว
   - ดูผลการเทรดแบบ real-time simulation

6. **📈 Evaluate Performance**
   - วิเคราะห์ผลการเทรดอย่างละเอียด
   - เปรียบเทียบกับ Buy & Hold strategy
   - แสดงกราฟและ metrics ต่างๆ

7. **⚙️ Manage Agents**
   - ดูรายการ agents ทั้งหมด
   - ลบ agents ที่ไม่ต้องการ
   - จัดการไฟล์โมเดล

### 📝 Best Practices

#### การเตรียมข้อมูล
- ใช้ข้อมูลคุณภาพสูงที่มีความต่อเนื่อง
- ตรวจสอบข้อมูลที่ missing หรือผิดปกติ
- ใช้ technical indicators ที่เหมาะสมกับ strategy

#### การเทรน Agent
- เริ่มต้นด้วย Grade N หรือ D สำหรับการทดสอบ
- ใช้ Grade B, A, S สำหรับการเทรนจริง
- ตรวจสอบ GPU memory ก่อนเทรน Grade สูง
- บันทึก checkpoint เป็นระยะ

#### การประเมินผล
- เปรียบเทียบกับ benchmark เสมอ
- ดู metrics หลายๆ ตัว (Sharpe Ratio, Max Drawdown, etc.)
- ทดสอบกับข้อมูล out-of-sample

### ⚠️ ข้อควรระวัง

#### Performance
- การเทรน Grade S (100,000 steps) ใช้เวลานาน
- แนะนำให้ใช้ GPU สำหรับการเทรน
- CPU อาจใช้เวลาหลายชั่วโมง

#### Memory Management
- ตรวจสอบ RAM และ GPU memory
- ปิดแอปพลิเคชันอื่นๆ ระหว่างเทรน
- ใช้ batch size ที่เหมาะสม

#### Data Quality
- ข้อมูลที่มี gaps หรือ outliers อาจส่งผลต่อการเทรน
- ตรวจสอบ data normalization
- ระวัง data leakage

### 🔧 Troubleshooting

#### ปัญหาที่พบบ่อย

1. **Import Error**
   ```
   ModuleNotFoundError: No module named 'finrl'
   ```
   **แก้ไข:** ติดตั้ง dependencies ตาม requirements.txt

2. **GPU Not Available**
   ```
   CUDA out of memory
   ```
   **แก้ไข:** 
   - ลด batch_size
   - ปิดแอปพลิเคชันอื่นๆ
   - ใช้ CPU แทน

3. **Data Loading Error**
   ```
   File not found error
   ```
   **แก้ไข:** ตรวจสอบ path และ permissions

4. **Training Stuck**
   **แก้ไข:**
   - ตรวจสอบ data quality
   - ลด learning rate
   - ปรับ parameters

### 📊 System Requirements

#### Minimum Requirements
- RAM: 8GB
- CPU: 4 cores
- Storage: 5GB free space
- Python: 3.8+

#### Recommended Requirements
- RAM: 16GB+
- GPU: 6GB+ VRAM (RTX 3060 หรือเทียบเท่า)
- CPU: 8+ cores
- Storage: 20GB+ free space
- Python: 3.9+

### 🔄 Updates และ Maintenance

#### การอัพเดทระบบ
```bash
# อัพเดท dependencies
pip install -r requirements.txt --upgrade

# ตรวจสอบเวอร์ชั่น
pip list | grep -E "(streamlit|finrl|stable-baselines3)"
```

#### การสำรองข้อมูล
- สำรอง models/ directory
- สำรอง data/ directory  
- สำรอง configuration files

### 📞 การติดต่อและความช่วยเหลือ

หากพบปัญหาในการใช้งาน:
1. ตรวจสอบ logs ใน terminal
2. ดู error messages ใน Streamlit interface
3. ตรวจสอบ system requirements
4. อ่าน documentation ใน notebooks/

---

**🧠 Crypto RL Agent Dashboard** - Powered by FinRL & Streamlit 