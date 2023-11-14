from ultralytics import YOLO
import os

# Load the model.
model = YOLO('yolov8m.pt')

save_dir = '/home/fadhel/Workspaces/test/halotec/logs-2'
os.makedirs(save_dir, exist_ok=True)
 
# Training.
results = model.train(
   data='/home/fadhel/Workspaces/test/halotec/uhfhlsw.v1i.yolov8/data.yaml',
   imgsz=640,
   epochs=50,
   batch=8,
   patience=25,
   verbose=True,
   name='receipt_extractor',
   project=save_dir)