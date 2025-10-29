import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.classification.parking_classifier import ParkingClassifier
from src.detection.vehicle_detector import VehicleDetector
from src.geometry.polygon_manager import PolygonManager
from src.utils.config import Config


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    model = ParkingClassifier(num_classes=2)
    model.load_state_dict(torch.load(Config.CLASSIFIER_MODEL, map_location=device))
    model.to(device)
    model.eval()

    yolo_model = VehicleDetector(Config.YOLO_MODEL)
    polygon_manager = PolygonManager(Config.POLYGON_JSON)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    cap = cv2.VideoCapture(Config.VIDEO_URL)

    if not cap.isOpened():
        print("Ошибка подключения к камере")
        return

    print("Запуск классификации === 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            break

        display_frame = frame.copy()
        free_by_classifier = 0
        occupied_by_classifier = 0

        if 1 in polygon_manager.parking_polygons:
            total_count = len(polygon_manager.parking_polygons[1])

            for i, polygon in enumerate(polygon_manager.parking_polygons[1]):
                try:
                    spot_image = polygon_manager.extract_parking_spot(frame, polygon)

                    # Классификация
                    image_rgb = cv2.cvtColor(spot_image, cv2.COLOR_BGR2RGB)
                    transformed = transform(image=image_rgb)
                    input_tensor = transformed["image"].unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(probabilities, 1).item()
                        confidence = probabilities[0][predicted_class].item()

                    if predicted_class == 1 and confidence > Config.CLASSIFICATION_CONFIDENCE_THRESHOLD:
                        status = "OCCUPIED"
                        color = (0, 0, 255)
                        occupied_by_classifier += 1
                    else:
                        status = "FREE"
                        color = (0, 255, 0)
                        free_by_classifier += 1

                    pts = np.array(polygon, dtype=np.int32)
                    cv2.polylines(display_frame, [pts], True, color, 1)

                    centroid = np.mean(pts, axis=0)
                    label = f"{i + 1}: {status} ({confidence:.2f})"
                    cv2.putText(display_frame, label,
                                (int(centroid[0]), int(centroid[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    print(label)

                except Exception as e:
                    print(f"Ошибка обработки парковочного места {i + 1}: {e}")
                    continue
            print("=" * 50)

        # Детекция транспорта
        car_boxes, vehicle_in_roi = yolo_model.detect_vehicles(frame)

        # статистика
        car_count = sum(1 for box in car_boxes)  # Упрощенный подсчет
        stats = [
            f"Classified - Free: {free_by_classifier}",
            f"Classified - Occupied: {occupied_by_classifier}",
            f"YOLO - Cars: {car_count}",
            f"YOLO - Trucks: {vehicle_in_roi - car_count}",
            f"YOLO - Total: {vehicle_in_roi}",
            f"Discrepancy: {occupied_by_classifier - vehicle_in_roi}"
        ]

        for i, text in enumerate(stats):
            cv2.putText(display_frame, text, (20, 40 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Parking Classification", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Классификация завершена")


if __name__ == "__main__":
    main()