import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load fine-tuned YOLO model
model = YOLO("YOLO OCR.pt")

# 2. Open Live Camera Feed (0 = default laptop webcam / USB camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def reconstruct_text(detections, y_threshold=15, space_threshold=25):
    if not detections:
        return ""
    
    # Sort vertically first
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
        
    # Sort horizontally and build final string
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

print("Starting Live Camera OCR... Press 'q' to quit, 's' to save recognized text.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab video frame.")
        break
        
    # Run YOLO prediction
    results = model.predict(source=frame, conf=0.35, save=False, show=False, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.names
        cls_ids = result.boxes.cls.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box[:4]
            label = labels[int(cls_ids[i])]
            detections.append((x_min, y_min, x_max, y_max, label))
            
            # Draw green bounding box & label
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    # Reconstruct text string
    recognized_string = reconstruct_text(detections)
    
    # Overlay black top banner with yellow recognized text string
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(frame, f"RECOGNIZED: {recognized_string}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    
    # Show live stream window
    cv2.imshow("Digital Character Recognition - Live Feed", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        with open("live_scan_output.txt", "a") as f:
            f.write(recognized_string + "\n")
        print(f"Saved to file: {recognized_string}")

cap.release()
cv2.destroyAllWindows()
