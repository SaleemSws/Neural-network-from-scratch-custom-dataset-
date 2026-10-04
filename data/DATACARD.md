# 🗃️ Dataset Card: PSU Reservoir Road Lane & Track Dataset

## 1. Dataset Summary
- **Dataset Name:** PSU Reservoir Road Lane & Track Dataset
- **Domain:** Autonomous Driving / Intelligent Transportation Systems (ITS)
- **Primary Tasks:**
  1. **Lane & Track Instance Segmentation (polygon)** (`dataset_seg`)
  2. **Lane & Road Markings Object Detection  (Bbox)** (`dataset_bb`)
  3. **Lane Centerline Estimation / Trajectory Planning (Polyline)** (`dataset_polyline`)
- **Location:** ถนนรอบอ่างเก็บน้ำ มหาวิทยาลัยสงขลานครินทร์ วิทยาเขตหาดใหญ่ (Prince of Songkla University, Hat Yai Campus)
- **Total Selected Images:** 1,000 frames (จากวิดีโอต้นฉบับ 1,300 เฟรม)
- **Data Splits:** Train 800 frames (80%), Validation 200 frames (20%)
- **Source Video:** `psu-reservoir-2026Aug06_121427.mp4`

---

## 2. Dataset Structure & Formats

```text
lane_project/
│
├── dataset_seg/                     <-- (Task 1: polygon)
│   ├── data_seg.yaml                <-- Ultralytics YOLO Segmentation config
│   ├── train.txt                    <-- รายชื่อรูป 800 รูป
│   ├── val.txt                      <-- รายชื่อรูป 200 รูป
│   ├── images/all_images/           <-- รูปต้นฉบับ 1,000 รูป (.jpg)
│   └── labels/all_images/           <-- ไฟล์ .txt รูปแบบ Polygon
│
├── dataset_bb/                      <-- (Task 2: Object Detection)
│   ├── data_bb.yaml                 <-- Ultralytics YOLO Detection config
│   ├── train.txt                    <-- รายชื่อรูป 800 รูป
│   ├── val.txt                      <-- รายชื่อรูป 200 รูป
│   ├── images/all_images/           <-- รูปต้นฉบับ 1,000 รูป (.jpg)
│   └── labels/all_images/           <-- ไฟล์ .txt รูปแบบ Bounding Box (cx, cy, w, h)
│
├── dataset_polyline/                <-- (Task 3: Centerline Polyline)
│   ├── data_polyline.yaml           <-- Ultralytics Polyline config
│   ├── train.txt                    <-- รายชื่อรูป 800 รูป
│   ├── val.txt                      <-- รายชื่อรูป 200 รูป
│   ├── images/all_images/           <-- รูปต้นฉบับ 1,000 รูป (.jpg)
│   └── labels/all_images/           <-- ไฟล์ .txt รูปแบบ Polyline จุดกึ่งกลางเลน
│
└── visualizations/                  <-- โฟลเดอร์รูปภาพสำหรับตรวจสอบผลลัพธ์ (ครบ 1,000 รูปทุกโฟลเดอร์)
    ├── sample_seg/                  <-- ภาพวาดหน้ากากโปร่งแสง (polygon) ทับเลน (1,000 รูป)
    ├── sample_bb/                   <-- ภาพวาดกรอบสี่เหลี่ยม Bounding Box พร้อมชื่อคลาส (1,000 รูป)
    └── sample_polyline/             <-- ภาพวาดเส้นกึ่งกลาง Polyline ความหนา 1px ไม่มีจุดปลาย (1,000 รูป)
```

---

## 3. Class Ontology & Specifications

### 3.1 ตารางนิยาม Class ในแต่ละ Task

| Class ID | Class Name | คำอธิบายวัตถุ | มีใน `dataset_seg` | มีใน `dataset_bb` | มีใน `dataset_polyline` | รหัสสี (BGR) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **0** | `line_left` | เส้นแบ่งขอบถนนฝั่งซ้าย | ✅ | ✅ | ✅ | `(0, 0, 255)` แดง |
| **1** | `line_center`| เส้นแบ่งกึ่งกลางถนน (เส้นประ/ทึบ) | ✅ | ✅ | ✅ | `(0, 255, 0)` เขียว |
| **2** | `line_right` | เส้นแบ่งขอบถนนฝั่งขวา | ✅ | ✅ | ✅ | `(255, 0, 0)` น้ำเงิน |
| **3** | `lane` | พื้นผิวผิวจราจรที่รถวิ่งได้ | ✅ | ✅ | ❌ | `(0, 255, 255)` เหลือง |
| **4** | `non-track area` | ขอบทาง ฟุตบาท ต้นไม้ เกาะกลาง | ✅ | ❌ | ❌ | `(180, 105, 255)` ชมพู |

> [!NOTE]
> - ใน `dataset_bb` ตัดคลาส `4: non-track area` ออก เพื่อโฟกัสที่เส้นและช่องทางจราจร
> - ใน `dataset_polyline` คัดเฉพาะคลาส `0, 1, 2` ซึ่งเป็นโครงสร้างเส้นนำร่อง (Line dividing markers)

---

## 4. Annotation Methodology & Provenance
1. **AI-Assisted Video Tracking:** ใช้ **SAM2 (Segment Anything Model 2 - `facebook/sam2.1-hiera-tiny`)** ติดตั้งผ่าน `cvat-cli` เชื่อมต่อระบบ Agent กับ CVAT Online (`app.cvat.ai`)
2. **Track-based Propagation:** วาด Polygon บนเฟรมหลัก แปลงเป็น Shape Track แล้วรัน AI Tracker ติดตามวัตถุตลอดลำดับภาพ
3. **Automated Polygon to BB Conversion:** คำนวณขอบเขต $\min(x), \max(x), \min(y), \max(y)$ และหาค่ากึ่งกลางความกว้าง/ความสูง Normalized YOLO Bbox
4. **Automated Polygon to Polyline Conversion:** ใช้อัลกอริทึมค้นหาจุดปลายหัว-ท้าย แบ่งขอบ 2 ฝั่ง และหาจุดกึ่งกลาง (Midline) ผ่าน Arc-Length Resampling Parameterization

---

## 5. PSU Cloud Storage Download & Access Instructions

ชุดข้อมูลทั้งหมดถูกบีบอัดและอัปโหลดขึ้นสู่ระบบคลาวด์ของมหาวิทยาลัยสงขลานครินทร์ (PSU Storage / OneDrive PSU)

- 📦 **Download Link:** [PSU Storage - lane_project_assignment_dataset.zip](https://storage.psu.ac.th/drive/d/f/19qLvAPqMpMLcH0BTbKiQhLBwZajpY91)
- **ขนาดไฟล์บีบอัด:** ประมาณ 280 MB

