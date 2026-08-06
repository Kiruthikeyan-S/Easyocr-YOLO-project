from ultralytics import YOLO

model=YOLO("yolo11x.pt")

model.train(data="dataset.yaml", imgsz = 640, batch = 8, epochs=100, workers = 0, device='cpu')