import os


class Config:
    # Пути к данным
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # Пути к моделям
    CLASSIFIER_MODEL = os.path.join(MODELS_DIR, "parking_classifier.pth")
    YOLO_MODEL = os.path.join(MODELS_DIR, "car_truck_detection_v7", "weights", "best.pt")

    # Пути к данным
    POLYGON_JSON = os.path.join(DATA_DIR, "dataset_parking", "dataset_car_truck", "parking_zone.json")
    DATASET_PATH = os.path.join(DATA_DIR, "dataset_parking", "parking_dataset")

    # Настройки видео
    VIDEO_URL = "http://94.72.19.56/mjpg/video.mjpg"

    # Настройки детекции
    DETECTION_CONF = 0.02
    DETECTION_IOU = 0.4

    # Настройки классификации
    CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.65