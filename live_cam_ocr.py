import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Load both models
print("Loading YOLO LCD Model...")
yolo_model = YOLO("YOLO OCR.pt")

print("Loading EasyOCR General Text Engine...")
reader = easyocr.Reader(['en'], gpu=False)

# Open Live Camera Feed (0 = default webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mode = "EASYOCR"  # Modes: 'EASYOCR' or 'YOLO'

def reconstruct_yolo_text(detections, y_threshold=15, space_threshold=25):
    if not detections:
        return ""
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
        
    full_text = []
    for line in lines:
        line.sort(key=lambda x: x[0])
        line_str = ""
        for i, char in enumerate(line):
            if i > 0 and (char[0] - line[i-1][2]) > space_threshold:
                line_str += " "
            line_str += char[4]
        full_text.append(line_str)
        
    return " | ".join(full_text)

print("\n" + "="*50)
print("🎥 LIVE CAMERA OCR STARTED")
print("• Press 'm' to TOGGLE ENGINE (YOLO vs EasyOCR)")
print("• Press 's' to SAVE CURRENT TEXT to file")
print("• Press 'q' to QUIT")
print("="*50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab camera frame.")
        break
        
    recognized_string = ""
    
    if mode == "EASYOCR":
        # Run EasyOCR on current live frame
        results = reader.readtext(frame)
        extracted = []
        for (bbox, text, prob) in results:
            if prob > 0.25:
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                x_min = int(min([p[0] for p in bbox]))
                y_min = int(min([p[1] for p in bbox]))
                cv2.putText(frame, text, (x_min, max(15, y_min - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                extracted.append(text)
        recognized_string = " | ".join(extracted)
        
    else:  # YOLO Mode
        results = yolo_model.predict(source=frame, conf=0.25, save=False, show=False, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.names
            cls_ids = result.boxes.cls.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = box[:4]
                label = labels[int(cls_ids[i])]
                detections.append((x_min, y_min, x_max, y_max, label))
                
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x_min), max(15, int(y_min) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        recognized_string = reconstruct_yolo_text(detections)

    # Top Banner
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 0), -1)
    banner_text = f"[{mode}] RECOGNIZED: {recognized_string}"
    cv2.putText(frame, banner_text[:110], (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.imshow("Live Camera OCR Stream (Press 'm' to switch mode)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mode = "YOLO" if mode == "EASYOCR" else "EASYOCR"
        print(f"Switched engine mode to: {mode}")
    elif key == ord('s'):
        with open("live_scan_output.txt", "a") as f:
            f.write(f"[{mode}] {recognized_string}\n")
        print(f"Saved to file: {recognized_string}")

cap.release()
cv2.destroyAllWindows()
