from ultralytics import YOLO
import cv2
import numpy as np
import sys

# Load the custom YOLO model
model = YOLO("YOLO OCR.pt")

# Open webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Function to group characters into rows based on their y_min position
def group_by_lines(detections, y_threshold=20): # slightly increased threshold for live camera jitter
    if not detections:
        return []
    lines = []
    current_line = []
    prev_y_min = detections[0][1]
    
    for det in detections:
        x_min, y_min, x_max, y_max, label = det
        if abs(y_min - prev_y_min) > y_threshold:
            # New line starts
            lines.append(current_line)
            current_line = [det]
        else:
            # Same line
            current_line.append(det)
        
        prev_y_min = y_min
        
    if current_line:
        lines.append(current_line)
    return lines

print("Starting live camera feed. Press 'q' in the window to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Perform prediction (verbose=False to avoid flooding the console)
    results = model.predict(source=frame, save=False, show=False, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.names
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box[:4]
            label = labels[int(result.boxes.cls[i].item())]
            detections.append((x_min, y_min, x_max, y_max, label))
            
    # Sort detections by y_min first, then x_min
    if detections:
        detections = sorted(detections, key=lambda x: (x[1], x[0]))
        lines = group_by_lines(detections, y_threshold=20)
        
        # Sort each line horizontally and reconstruct text
        final_text = []
        for line in lines:
            line.sort(key=lambda x: x[0])
            line_text = ''.join([char[4] for char in line])
            final_text.append(line_text)
            
        # Draw bounding boxes and text
        for det in detections:
            x_min, y_min, x_max, y_max, label = det
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # Optional: Print text to console (with a separator for readability)
        sys.stdout.write("\r" + " | ".join(final_text).ljust(80))
        sys.stdout.flush()
    else:
        sys.stdout.write("\r" + "No characters detected...".ljust(80))
        sys.stdout.flush()
        
    # Display the image in a window
    cv2.imshow('Live YOLO OCR', frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
print("\nClosing camera...")
cap.release()
cv2.destroyAllWindows()
