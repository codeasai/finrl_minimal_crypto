# Changelog

## [v1.1.0] — 2026-06-03

### Overview
Release นี้ทำให้ Streamlit UI ครบ full pipeline จริงครั้งแรก — ตั้งแต่โหลดข้อมูลจาก exchange
ไปจนถึงเทรน backtest และวิเคราะห์ผล โดยไม่ต้องเขียน code

### ✨ New Features

#### Train Agent (หน้า 3) — เทรนจริงแล้ว
- ปุ่ม **Create Agent** และ **Continue Training** เทรน PPO ได้จริง (เดิมเป็น stub)
- รองรับ Training Grade N–S (5,000–100,000 steps)
- รองรับข้อมูล intraday (1h, 15m ฯลฯ) — resample เป็น daily อัตโนมัติ
- ส่ง hyperparameters จาก UI เข้า model จริง (learning_rate, batch_size ฯลฯ)
- Detect symbols จากไฟล์ข้อมูลอัตโนมัติ ไม่ต้องแก้ `config.py`
- บันทึก JSON sidecar (`models/minimal_crypto_<name>.json`) พร้อม exchange, grade, symbols, steps

#### Test Agent (หน้า 4) — Backtest จริงแล้ว
- `st.status` แสดง 3 ขั้นตอน: load model spec → สร้าง env → รัน `DRL_prediction`
- **Auto-detect model spec**: อ่าน `n_symbols` และ `obs_dim` จาก model `.zip` โดยตรง
  แล้วกรอง/ปรับ data ให้ตรงกับ model อัตโนมัติ — แก้ปัญหา obs shape mismatch
- Resample intraday → daily พร้อม day-based index ที่ FinRL ต้องการ
- ไฟล์ข้อมูล: จัดลำดับ `data_prepare/` ก่อน raw files, warning เมื่อเลือก raw file
- บันทึกผลเป็น CSV ใน `models/backtest_results/`
- ส่งผลต่อ session_state ให้หน้า Evaluate โดยอัตโนมัติ

#### Evaluate Performance (หน้า 5) — วิเคราะห์ผลจริงแล้ว
- **8 performance metrics**: Total Return, Final Value, Sharpe Ratio, Max Drawdown,
  Sortino Ratio, Calmar Ratio, Win Rate, Volatility (annualised)
- กราฟ **Portfolio vs Buy & Hold** (equal-weight) สำหรับทุก symbol
- กราฟ **Drawdown** area chart
- ตาราง **B&H Return per Symbol** พร้อม Alpha
- โหลดจาก session (ทันที) หรือจาก CSV ที่บันทึกไว้ (reload-safe)
- Export account value และ actions เป็น CSV
- Summary verdict พร้อม Sharpe interpretation

#### Manage Agents (หน้า 6) — แสดง metadata จาก sidecar
- อ่าน `.json` sidecar แสดง Exchange, Grade, Steps, Symbols ใน expander

#### Help Page (หน้า 0) — หน้าใหม่
- คู่มือการใช้งานครบถ้วน: pipeline overview, คำแนะนำทีละขั้นตอน, troubleshooting table
- Training Grades table พร้อมเวลาโดยประมาณ
- Performance metrics อธิบายพร้อม target
- FAQ 6 ข้อ
- System Status แบบ live (นับ data files, models, backtest results)

#### Inline Help ทุกหน้า
- เพิ่ม `st.expander("💡 วิธีใช้งาน")` ที่ด้านบนของหน้า 1–5

### 🐛 Bug Fixes

| Bug | สาเหตุ | แก้ไข |
|-----|--------|--------|
| `numpy.float64 has no attribute 'values'` ใน FinRL env | ข้อมูล intraday (1h) มีหลาย rows ต่อวัน ทำให้ `df.loc[day]` คืน Series แทน DataFrame | Resample → daily + day-based index |
| `Unexpected observation shape (1,29) for Box (15,)` | Model เทรนด้วย 1 symbol แต่ data file มี 2 symbols | `detect_model_spec()` อ่าน n_symbols จาก model แล้วกรอง data ให้ตรง |
| Backtest เลือก raw data file เป็น default | `csv_candidates` sort ตาม alphabet ทำให้ raw file ขึ้นก่อน | จัดลำดับ `data_prepare/` ก่อน raw files เสมอ |
| Train Agent ไม่รู้จัก `tic` column | Data Prepare ผลิต `symbol` แต่ FinRL ต้องการ `tic` | `prepare_df_for_training()` rename อัตโนมัติ |
| `bb_std` หายจากไฟล์ที่ prepare | Data Prepare ไม่ได้ export `bb_std` เป็น column แยก | Derive จาก `(bb_upper - bb_middle) / 2` อัตโนมัติ |

### 📝 Documentation
- **README.md** — อัพเดทให้ Streamlit UI เป็นวิธีหลัก พร้อมตาราง pipeline และ Training Grades
- **CHANGELOG.md** — ไฟล์นี้

### 🔧 Technical Notes
- FinRL `StockTradingEnv` ต้องการ integer index แบบ day-based (แต่ละ day มี n_symbols rows)
  — ต้องตั้งค่านี้ทุกครั้งที่สร้าง environment จากข้อมูล intraday
- Models เก่า (ก่อน v1.1.0) ไม่มี `.json` sidecar — Manage Agents ยังแสดงได้แต่ไม่มี metadata

---

## [v1.0.0] — 2025-06-18

- Initial release: Data Loader (CCXT), Data Prepare (indicators), Train Agent (stub),
  Evaluate (stub), Manage Agents
- CLI scripts: `main.py`, `simple_advanced_agent.py`, `advanced_crypto_agent.py`
