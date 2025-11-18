import os

from ultralytics import YOLO
from src.utils.config import Config

def train_model():

    model = YOLO("yolov8s.pt")
    data_yaml = os.path.join(Config.DATA_DIR, "dataset_parking","dataset_car_truck","parking.yaml")
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=416,
        batch=8,
        lr0=0.001,
        patience=15,
        device="cpu",
        workers=0,
        augment=True,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        shear=1.0,
        perspective=0.0002,
        flipud=0.2,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        optimizer='Adam',
        weight_decay=0.0001,
        warmup_epochs=3.0,
        warmup_bias_lr=0.05,
        warmup_momentum=0.8,
        close_mosaic=10,
        name='car_truck_detection_v8'
    )


train_model()