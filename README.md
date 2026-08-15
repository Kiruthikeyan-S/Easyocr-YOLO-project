# 🖥️Easyocr-YOLO-project (LCD & Document OCR)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

An **AI & Data Science pipeline** for Optical Character Recognition (OCR) fine-tuned to detect and reconstruct digital text from **LCD monitor screens, digital meters**, as well as **printed ID cards and documents**.
<h1>Demo & Test link=https://easyocr-yolo-azm8w2htwpwgpo5x3tchyq.streamlit.app/<h1>
---

## 📌 Project Overview

Traditional OCR engines (such as Tesseract) often fail on **LCD displays and 7-segment digital screens** due to segment gaps, reflection glare, and unique font structures. 

This project solves this challenge by leveraging a fine-tuned **YOLO11x Object Detection Neural Network** paired with a **4-step Data Science spatial reconstruction algorithm** to convert raw 2D bounding boxes into clean, human-readable text strings.

---

## 🚀 Key Features

- **🎯 Fine-Tuned YOLO11x Model**: Trained on 43 distinct digital character classes (`0-9`, `A-Z`, `%`, `*`, `+`, `-`, `.`, `/`, `=`).
- **🤖 Universal Dual-Engine Architecture**:
  - **`🖥️ Digital LCD Monitor Engine (YOLO11)`**: For digital meters, calculators, 7-segment LED numbers, and LCD screens.
  - **`📄 ID Card & Document Engine (EasyOCR)`**: For printed student ID cards, badges, certificates, and paper text.
- **📊 4-Step Data Science Post-Processing**:
  1. 2D Bounding box coordinate extraction $[x_{min}, y_{min}, x_{max}, y_{max}]$.
  2. Vertical line grouping ($y_{threshold} = 15\text{px}$).
  3. Left-to-right horizontal sorting ($x_{min}$).
  4. String assembly and visual annotation.
- **🌐 Interactive Web Application (`app.py`)**: Built with Streamlit, supporting multi-image upload, live camera snapshots, detection parameter tuning, and 1-click **`.txt`** / **`.json`** data exports.
- **🎥 Real-Time Desktop Feed (`live_cam_ocr.py`)**: Live video stream OCR with real-time on-screen banner text overlay.

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (UI / UX)                              │
│                      Built with Streamlit (app.py)                         │
│                                                                           │
│   • Input Modes: File/Folder Upload & Live Camera Snapshot                │
│   • Sidebar Controls: OCR Engine Switcher & Sensitivity Sliders           │
│   • Display: Original Image vs. AI Annotated Image Side-by-Side           │
│   • Output & Exports: Formatted Text Box + Download Buttons (.txt/.json) │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
                             (Data Flow & Image Array)
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           BACKEND (AI & Data Science)                     │
│                        Python + PyTorch + OpenCV                          │
│                                                                           │
│  ┌─────────────────────────────────┐   ┌───────────────────────────────┐  │
│  │ ENGINE 1: YOLO11x (LCD Screens) │   │ ENGINE 2: EasyOCR (ID Cards)  │  │
│  └─────────────────────────────────┘   └───────────────────────────────┘  │
│                                    │                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Data Science Post-Processing Pipeline:                              │  │
│  │ 1. Extract 2D Bounding Box Coordinates (x_min, y_min, x_max, y_max) │  │
│  │ 2. Vertical Line Grouping Algorithm (y_threshold = 15px)            │  │
│  │ 3. Horizontal Left-to-Right Sorting (x_min)                         │  │
│  │ 4. Reconstruct Text String & Render Green Bounding Boxes            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🔢 Supported Character Classes (`nc: 43`)

The model detects 43 individual character classes:
- **Digits (10)**: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
- **Letters (26)**: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`
- **Symbols (7)**: `%`, `*`, `+`, `-`, `.`, `/`, `=`

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vlarjun20/-Digital-Character-Recognition-.git
cd -Digital-Character-Recognition-
```

### 2. Install Required Packages
```bash
pip install ultralytics opencv-python streamlit easyocr pillow numpy
```

---

## 💻 How to Run

### 🌐 Option 1: Launch Interactive Web Application
```bash
python -m streamlit run app.py
```
*Navigating to `http://localhost:8501` will open the full web interface.*

### 📄 Option 2: Run Python Script Output on Image (`chumma.jpeg`)
```bash
python test1.py
```
*Outputs recognized text string in console and saves `output_test1.jpeg`.*

### 🎥 Option 3: Run Real-Time Camera Reader Stream
```bash
python live_cam_ocr.py
```
*Press `'s'` to save recognized live text to `live_scan_output.txt`, or `'q'` to quit.*

### 🏋️‍♂️ Option 4: Train / Fine-Tune YOLO Model
```bash
python model.py
```

---

## 📁 Repository Directory Structure

```
.
├── dataset.yaml            # YOLO dataset configuration & 43 class names
├── model.py                # YOLO11x model training script
├── test1.py                # Standalone 5-step OCR pipeline script
├── app.py                  # Streamlit Web Application (Dual Engine)
├── live_cam_ocr.py         # Real-time desktop camera feed OCR script
├── YOLO OCR.pt             # Fine-tuned PyTorch model weights (~114.5 MB)
├── chumma.jpeg             # Sample test image of a blue LCD monitor
├── README.md               # Project documentation
├── train/                  # Training images & YOLO annotation label files (.txt)
└── val/                    # Validation images & YOLO label files (.txt)
```

---

## ⚡ Performance Metrics

- **Inference Speed**: ~0.3s per image on CPU | ~15-30ms on GPU (60 FPS).
- **RAM Usage**: ~350 MB – 650 MB.
- **Model Disk Footprint**: ~114.5 MB.

---

## 📦 Large File Note (Git LFS)

Because `YOLO OCR.pt` (114.5 MB) exceeds GitHub's 100 MB single file limit, use **Git LFS** when pushing:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add .
git commit -m "Add fine-tuned YOLO model weights"
git push origin main
```
