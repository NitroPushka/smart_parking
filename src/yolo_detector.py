from mpmath import degrees
from ultralytics import YOLO

def train_car_truck_detection():
    model = YOLO("yolov8n.pt")

    model.train(
        data = "C:/Users/ilyap/PycharmProjects/smart_parking/data/dataset_parking/dataset_car_truck/parking.yaml",
        epochs = 50,
        imgsz = 640,
        batch = 8,
        lr0 = 0.01,
        patience = 10,
        device = "cpu",
        workers = 0,
        augment = True,
        degrees = 5.0,
        fliplr = 0.5,
        name = 'car_truck_detection_v1'
    )
train_car_truck_detection()