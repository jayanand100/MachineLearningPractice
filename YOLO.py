from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
model.predict(source='IMG_20211001_162923.jpg')