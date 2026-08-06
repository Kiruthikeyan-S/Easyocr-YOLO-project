from ultralytics import YOLO
import cv2
import numpy as np

# Load the custom YOLO model
model = YOLO("YOLO OCR.pt")

# Perform the prediction
results = model.predict(source="chumma.jpeg", save=False, show=False)

# Load the image to display results
img = cv2.imread("chumma.jpeg")

image_height, image_width = img.shape[:2]

# Initialize an empty list to store detected characters and their bounding boxes
detections = []

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    labels = result.names  # Class labels (characters)
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box[:4]  # Get bounding box coordinates
        label = labels[int(result.boxes.cls[i].item())]  # Get the class (character) label
        detections.append((x_min, y_min, x_max, y_max, label))  # Add to detections list

# Sort detections by y_min (vertical position) first, and then by x_min (horizontal position)
detections = sorted(detections, key=lambda x: (x[1], x[0]))

# Function to group characters into rows based on their y_min position
def group_by_lines(detections, y_threshold=15):
    lines = []
    current_line = []
    prev_y_min = detections[0][1]  # Set the first y_min as reference
    
    for det in detections:
        x_min, y_min, x_max, y_max, label = det
        if abs(y_min - prev_y_min) > y_threshold:
            # New line starts
            lines.append(current_line)
            current_line = [det]  # Start a new line
        else:
            # Same line
            current_line.append(det)
        
        prev_y_min = y_min
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    return lines

# Group the detections into lines
lines = group_by_lines(detections, y_threshold=15)

# Sort each line by x_min (horizontal position)
for line in lines:
    line.sort(key=lambda x: x[0])  # Sort within each line by x_min

# Reconstruct the text line by line
final_text = []
for line in lines:
    line_text = ''.join([char[4] for char in line])  # Concatenate the characters in the line
    final_text.append(line_text)

# Print the final text in the correct order
print("Recognized Text from Image:")
for text_line in final_text:
    print(text_line)

# Display the image with bounding boxes for visualization
def draw_bounding_boxes(detections, img):
    for det in detections:
        x_min, y_min, x_max, y_max, label = det
        # Draw the bounding box
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # Put the label (character) above the bounding box
        cv2.putText(img, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Draw bounding boxes on the image
draw_bounding_boxes(detections, img)

# Save output image to file
cv2.imwrite("output_test1.jpeg", img)
print("Saved annotated image to 'output_test1.jpeg'")

# Display the image in a window
try:
    cv2.imshow('Recognized Characters', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    pass