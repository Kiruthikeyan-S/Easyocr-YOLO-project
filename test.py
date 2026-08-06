from ultralytics import YOLO

model = YOLO("YOLO OCR.pt")

model.predict(source="chumma.jpeg", show=True)
