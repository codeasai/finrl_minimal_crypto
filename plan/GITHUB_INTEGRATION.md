# 🐙 GitHub Integration Guide

*คู่มือการใช้งาน GitHub Issues แบบมาตรฐานสำหรับ Feature Requests และ Bug Reports*

---

## 📋 GitHub Issues Overview

### 🎯 วัตถุประสงค์
ใช้ **GitHub Issues แบบมาตรฐาน** เป็นเครื่องมือหลักในการ:
1. **Feature Requests** - ขอความสามารถใหม่
2. **Bug Reports** - รายงานปัญหาและข้อผิดพลาด
3. **Project Tracking** - ติดตามความคืบหน้า
4. **Community Collaboration** - ให้ชุมชนมีส่วนร่วม

---

## 🚀 Feature Request Process

### 📍 วิธีการสร้าง Feature Request

1. **ไปที่ GitHub Repository**
   ```
   https://github.com/codeasai/finrl_minimal_crypto
   ```

2. **คลิกที่ Issues Tab**
   - ไปที่ **Issues** > **New Issue**

3. **สร้าง Issue ใหม่**
   - คลิก **"New Issue"**
   - ใช้ title format: **[FEATURE] Request ID: FR-2024-XXX - Title**

### 📝 Format สำหรับ Feature Request

```markdown
## 🚀 Feature Request: FR-2024-XXX

### 📋 Request Details
- **Request ID**: FR-2024-XXX
- **Priority**: 🔥 High / 🟡 Medium / 🟢 Low
- **Category**: Performance Enhancement / New Feature / Technical Improvement
- **Timeline**: X weeks

### 🎯 Objective
[อธิบายเป้าหมายหลักของ feature]

### 📊 Current Situation
[อธิบายสถานะปัจจุบันและข้อจำกัด]

### 💡 Proposed Solution
[อธิบายวิธีแก้ไขที่เสนอ]

### 📈 Success Metrics
- Metric 1: Target value
- Metric 2: Target value
- Metric 3: Target value

### 🛠️ Implementation Plan (Optional)
Phase 1: Description
Phase 2: Description

### 🔗 Dependencies
[รายการ dependencies หรือ requirements]

### 🚨 Potential Risks
[ความเสี่ยงที่อาจเกิดขึ้น]
```

---

## 🐛 Bug Report Process

### 📍 วิธีการรายงาน Bug

1. **ไปที่ GitHub Repository**
2. **คลิกที่ Issues Tab**
3. **สร้าง Issue ใหม่**
   - ใช้ title format: **[BUG] Bug ID: BUG-2024-XXX - Title**

### 📝 Format สำหรับ Bug Report

```markdown
## 🐛 Bug Report: BUG-2024-XXX

### 📋 Bug Details
- **Bug ID**: BUG-2024-XXX
- **Severity**: 🔴 Critical / 🟠 High / 🟡 Medium / 🟢 Low

### 📝 Description
[คำอธิบายปัญหาอย่างชัดเจน]

### ✅ Expected Behavior
[พฤติกรรมที่คาดหวัง]

### ❌ Actual Behavior
[พฤติกรรมที่เกิดขึ้นจริง]

### 🔄 Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Run command '...'
4. See error

### 🖥️ Environment
- OS: [e.g., Windows 10, macOS 12.0, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Conda environment: [e.g., tfyf]
- GPU: [e.g., NVIDIA RTX 3080, None]
- RAM: [e.g., 16GB, 32GB]

### 📋 Error Logs
```
[Paste error logs here]
```

### 📝 Additional Context
[ข้อมูลเพิ่มเติมที่เกี่ยวข้อง]
```

---

## 🏷️ Labels และ Manual Tagging

### 📌 การติด Labels ด้วยตนเอง
เนื่องจากใช้ GitHub Issues แบบมาตรฐาน ให้ติด labels ด้วยตนเอง:

#### Feature Request Labels:
- `enhancement`
- `feature request`
- `priority: high` / `priority: medium` / `priority: low`
- `performance` / `data` / `agent` / `ui/ux`

#### Bug Report Labels:
- `bug`
- `critical` / `high` / `medium` / `low`
- `help wanted`
- `good first issue`

---

## 🔄 Issue Lifecycle

### 📋 สถานะของ Issue

1. **Open** - Issue ใหม่ที่ยังไม่ได้ดำเนินการ
2. **In Progress** - กำลังดำเนินการ (comment update)
3. **Testing** - กำลังทดสอบ (comment update)
4. **Completed** - เสร็จสิ้นแล้ว
5. **Closed** - ปิดโดยไม่ดำเนินการ

### 🔄 การอัพเดทสถานะ
ใช้ **Comments** เพื่ออัพเดทสถานะ:
```markdown
## 🔄 Status Update
**Status**: In Progress
**Progress**: 30%
**ETA**: 2024-12-25
**Notes**: Working on data collection phase
```

---

## 🎯 Integration กับ Plan Directory

### 🔗 การเชื่อมโยง

1. **GitHub Issue** → **Plan Document**
   - สร้าง GitHub Issue ก่อน
   - สร้าง plan document และ reference GitHub Issue number

2. **Development Branch** → **GitHub Issue**
   - สร้าง branch จาก issue number
   - Commit message reference issue number

### 📝 ตัวอย่าง Workflow

```bash
# 1. สร้าง GitHub Issue #42 (FR-2024-002)
# 2. สร้าง plan document
echo "# Feature Request FR-2024-002 (GitHub Issue #42)" > plan/FR-2024-002.md

# 3. สร้าง development branch
git checkout -b feature/FR-2024-002

# 4. Development work
# ...

# 5. Commit with issue reference
git commit -m "feat: implement feature FR-2024-002

Closes #42"

# 6. Push และ create PR
git push origin feature/FR-2024-002
```

---

## 📊 Example: FR-2024-001

### 🎯 การสร้าง GitHub Issue สำหรับ FR-2024-001

1. **ไปที่**: https://github.com/codeasai/finrl_minimal_crypto/issues
2. **คลิก**: New Issue
3. **Title**: `[FEATURE] Request ID: FR-2024-001 - Data Improvement Strategy`
4. **Body**: ใช้ format ข้างต้น พร้อมข้อมูลจาก `plan/DATA_IMPROVEMENT_STRATEGY.md`
5. **Labels**: `enhancement`, `feature request`, `priority: high`, `performance`, `data`

### 📋 Issue Content ตัวอย่าง:
```markdown
## 🚀 Feature Request: FR-2024-001

### 📋 Request Details
- **Request ID**: FR-2024-001
- **Priority**: 🔥 High
- **Category**: Performance Enhancement
- **Timeline**: 8 weeks

### 🎯 Objective
ปรับปรุงประสิทธิภาพของ cryptocurrency trading agents ด้วย advanced data และ features

### 📈 Success Metrics
- Sharpe Ratio: จาก negative → > 1.0
- Maximum Drawdown: ลดลงเป็น < 15%
- Win Rate: เพิ่มเป็น > 55%
- Trading Frequency: ลดลงเป็น < 500 trades per period

### 🔗 Related Documents
- Plan Document: `plan/DATA_IMPROVEMENT_STRATEGY.md`
- Implementation Phases: 4 phases over 8 weeks
```

---

## 📈 Benefits ของ Standard GitHub Issues

### ✅ ข้อดี:
- **ไม่ซับซ้อน** - ใช้ระบบที่มีอยู่แล้ว
- **ยืดหยุ่น** - สามารถปรับแต่งได้ตามต้องการ
- **เข้าใจง่าย** - ทุกคนคุ้นเคยกับ GitHub Issues
- **ไม่ต้องบำรุงรักษา** - ไม่มี custom templates ที่ต้องดูแล

### 🎯 การใช้งาน:
- **Manual Formatting** - ใช้ markdown templates
- **Manual Labeling** - ติด labels ด้วยตนเอง
- **Comment Updates** - อัพเดทสถานะผ่าน comments
- **Standard Workflow** - ใช้ GitHub workflow มาตรฐาน

---

## 📋 Best Practices

### ✅ สำหรับ Feature Requests
- **ใช้ title format ที่ชัดเจน**: `[FEATURE] Request ID: FR-2024-XXX - Title`
- **ใช้ markdown template** ที่กำหนดไว้
- **ติด labels ที่เหมาะสม** ด้วยตนเอง
- **Reference plan documents** ถ้ามี
- **อัพเดทสถานะ** ผ่าน comments

### ✅ สำหรับ Bug Reports
- **ใช้ title format**: `[BUG] Bug ID: BUG-2024-XXX - Title`
- **ให้ steps to reproduce ชัดเจน**
- **ระบุ environment ครบถ้วน**
- **แนบ error logs** ถ้ามี
- **ระบุ severity** ที่เหมาะสม

### ✅ สำหรับ Issue Management
- **ใช้ labels อย่างสม่ำเสมอ**
- **อัพเดทสถานะ** เป็นประจำ
- **ตอบกลับ comments** อย่างสร้างสรรค์
- **ปิด issues** เมื่อเสร็จสิ้น
- **Link related issues** เมื่อเกี่ยวข้อง

---

## 🔧 Simplified Workflow

### 📁 ไม่มี Custom Templates
- ไม่มี `.github/ISSUE_TEMPLATE/`
- ใช้ GitHub Issues แบบมาตรฐาน
- Manual formatting ด้วย markdown templates
- Manual labeling และ assignment

### 🎯 Focus บน Content
- **Quality Content** มากกว่า fancy forms
- **Clear Communication** ผ่าน markdown
- **Consistent Formatting** ด้วย templates
- **Manual but Flexible** approach

---

*คู่มือนี้อธิบายการใช้งาน GitHub Issues แบบมาตรฐานเป็นเครื่องมือหลักในการจัดการ Feature Requests และ Bug Reports ของโปรเจค finrl_minimal_crypto โดยไม่ต้องใช้ custom templates* 