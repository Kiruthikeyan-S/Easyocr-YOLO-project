import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
import json

# Set page configuration
st.set_page_config(
    page_title="Universal OCR Suite (LCD & ID Card Reader)",
    page_icon="🔍",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stAppHeader { background-color: transparent; }
    .result-card {
        background: linear-gradient(135deg, #1e2638 0%, #0d131f 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        margin-top: 15px;
    }
    .ocr-text-box {
        font-family: 'Courier New', monospace;
        background-color: #030712;
        color: #00ffcc;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        letter-spacing: 2px;
        border: 1px solid #00f2fe;
    }
    .engine-badge {
        background: #1e3a8a;
        color: #93c5fd;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Engines
# ---------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("YOLO OCR.pt")

@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['en'], gpu=False)

try:
    yolo_model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

try:
    easyocr_reader = load_easyocr_reader()
except Exception as e:
    st.error(f"Error loading EasyOCR engine: {e}")

# ---------------------------------------------------------
# Engine 1: YOLO OCR (For Digital LCD Monitors & Meters)
# ---------------------------------------------------------
def process_yolo_ocr(image_np, conf_threshold=0.25, y_threshold=15):
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = yolo_model.predict(source=img_bgr, conf=conf_threshold, save=False, show=False, verbose=False)
    
    img_draw = img_bgr.copy()
    detections = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()
        labels = result.names
        
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box[:4]
            label = labels[int(cls_ids[i])]
            detections.append((x_min, y_min, x_max, y_max, label))
            
            cv2.rectangle(img_draw, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(img_draw, label, (int(x_min), max(15, int(y_min) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    img_draw_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    if not detections:
        return img_draw_rgb, "", 0

    detections = sorted(detections, key=lambda x: (x[1], x[0]))
    lines = []
    current_line = [detections[0]]
    prev_y_min = detections[0][1]

    for det in detections[1:]:
        x_min, y_min, x_max, y_max, label = det
        if abs(y_min - prev_y_min) > y_threshold:
            lines.append(current_line)
            current_line = [det]
        else:
            current_line.append(det)
        prev_y_min = y_min

    if current_line:
        lines.append(current_line)

    final_text_lines = []
    for line in lines:
        line.sort(key=lambda x: x[0])
        line_str = "".join([char[4] for char in line])
        final_text_lines.append(line_str)

    full_reconstructed_text = "\n".join(final_text_lines)
    return img_draw_rgb, full_reconstructed_text, len(detections)

# ---------------------------------------------------------
# Engine 2: EasyOCR (For ID Cards, Paper & Printed Text)
# ---------------------------------------------------------
def process_easy_ocr(image_np, conf_threshold=0.20):
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = easyocr_reader.readtext(image_np)
    
    img_draw = img_bgr.copy()
    extracted_lines = []
    
    for (bbox, text, prob) in results:
        if prob >= conf_threshold:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(img_draw, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            x_min = int(min([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            cv2.putText(img_draw, f"{text}", (x_min, max(15, y_min - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            
            extracted_lines.append(text)

    img_draw_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    full_text = "\n".join(extracted_lines)
    return img_draw_rgb, full_text, len(extracted_lines)


# --- SIDEBAR NAVIGATION & MODES ---
st.sidebar.title("🔍 Navigation & Engine Choice")

ocr_engine = st.sidebar.selectbox(
    "🤖 Choose OCR Model Engine:",
    [
        "🖥️ Digital LCD Screen Mode (YOLO)",
        "📄 ID Card & Document Mode (EasyOCR)"
    ]
)

app_mode = st.sidebar.radio(
    "Select Input Mode:",
    [
        "📁 Upload Image / Folder",
        "📸 Camera Snapshot (Take Photo)",
        "🎥 Continuous Live Video Stream"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Tuning Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.80, 0.20, 0.05)
y_thresh = st.sidebar.slider("Line Vertical Threshold (y_threshold)", 5, 50, 15, 1)

# Header Section
st.title("🔍 Universal OCR System")
st.markdown(f"Selected Model Engine: <span class='engine-badge'>{ocr_engine}</span>", unsafe_allow_html=True)

def execute_ocr(image_np):
    if "LCD Screen" in ocr_engine:
        return process_yolo_ocr(image_np, conf_threshold=conf_thresh, y_threshold=y_thresh)
    else:
        return process_easy_ocr(image_np, conf_threshold=conf_thresh)

# Session State for Live Stream Control
if "stream_active" not in st.session_state:
    st.session_state.stream_active = False

# =========================================================
# MODE 1: UPLOAD IMAGE / FOLDER
# =========================================================
if app_mode == "📁 Upload Image / Folder":
    st.subheader("📁 Upload Images (LCD Screens, ID Cards, Documents)")
    uploaded_files = st.file_uploader("Choose image file(s)...", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
    
    if uploaded_files:
        for idx, file in enumerate(uploaded_files):
            st.markdown(f"### 📄 Image {idx+1}: `{file.name}`")
            image = Image.open(file).convert("RGB")
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            image_np = np.array(image, dtype=np.uint8)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Input Image", width=600)
                
            with st.spinner(f"Processing with {ocr_engine}..."):
                processed_img, recognized_text, num_dets = execute_ocr(image_np)
                
            with col2:
                st.image(processed_img, caption="AI Detections", width=600)

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("#### 🔤 Recognized Text Output:")
            if recognized_text:
                st.markdown(f"<div class='ocr-text-box'>{recognized_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                st.caption(f"Total Text Detections: {num_dets}")
                
                st.download_button(
                    label="📥 Download Recognized String (.txt)",
                    data=recognized_text,
                    file_name=f"{file.name}_recognized.txt",
                    mime="text/plain",
                    key=f"txt_{idx}"
                )
            else:
                st.warning("No text detected. Try adjusting the Confidence Threshold slider in the sidebar.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")

# =========================================================
# MODE 2: CAMERA SNAPSHOT (TAKE PHOTO)
# =========================================================
elif app_mode == "📸 Camera Snapshot (Take Photo)":
    st.subheader("📸 Camera Snapshot Reader")
    camera_image = st.camera_input("Take a photo")

    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        image_np = np.array(image, dtype=np.uint8)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Captured Snapshot", width=600)
            
        with st.spinner(f"Processing with {ocr_engine}..."):
            processed_img, recognized_text, num_dets = execute_ocr(image_np)
            
        with col2:
            st.image(processed_img, caption="AI Detections", width=600)
            
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("#### 🔤 Snapshot Recognized Output:")
        if recognized_text:
            st.markdown(f"<div class='ocr-text-box'>{recognized_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Save Recognized Text to File",
                data=recognized_text,
                file_name="snapshot_ocr.txt",
                mime="text/plain"
            )
        else:
            st.warning("No text detected in snapshot.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# MODE 3: CONTINUOUS LIVE VIDEO STREAM (START / STOP BUTTONS)
# =========================================================
elif app_mode == "🎥 Continuous Live Video Stream":
    st.subheader("🎥 Real-Time Continuous Live Camera Stream")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🟢 START Live Camera (ON)", type="primary", use_container_width=True):
            st.session_state.stream_active = True
    with col_btn2:
        if st.button("🔴 STOP Live Camera (OFF)", type="secondary", use_container_width=True):
            st.session_state.stream_active = False

    if st.session_state.stream_active:
        st.success("🟢 Live Camera Stream is ON and running continuously. Click '🔴 STOP Live Camera' above to turn OFF.")
        FRAME_WINDOW = st.image([])
        text_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while st.session_state.stream_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access webcam feed.")
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_img, recognized_text, num_dets = execute_ocr(frame_rgb)
            
            FRAME_WINDOW.image(processed_img, width=600)
            
            if recognized_text:
                text_placeholder.markdown(f"<div class='result-card'><h4>🔴 Live Text Output:</h4><div class='ocr-text-box'>{recognized_text.replace(chr(10), '<br>')}</div></div>", unsafe_allow_html=True)
            else:
                text_placeholder.markdown("<div class='result-card'><h4>🔴 Live Text Output:</h4><i>Scanning live camera feed...</i></div>", unsafe_allow_html=True)

        cap.release()
    else:
        st.info("💡 Click **🟢 START Live Camera (ON)** to start continuous live video processing.")
