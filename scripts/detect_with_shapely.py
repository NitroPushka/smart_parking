import cv2
import numpy as np
import json
import sys
import os

from shapely.geometry import Polygon, box
from shapely.affinity import scale

from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config


def main():
    # Используем Config для всех параметров
    VIDEO_URL = Config.VIDEO_URL
    POLYGON_JSON_PATH = Config.POLYGON_JSON
    MODEL_PATH = Config.YOLO_MODEL
    DETECTION_CONF = Config.DETECTION_CONF
    DETECTION_IOU = Config.DETECTION_IOU

    with open(POLYGON_JSON_PATH) as json_file:
        data = json.load(json_file)

    model = YOLO(MODEL_PATH)

    # ключ = id картинки, значение = список полигонов \\ у меня две картинки = два ключа
    parking_polygon = {}
    # print(data["annotations"])

    # список аннотаций (annotations),
    for ann in data["annotations"]:
        if ann["category_id"] == 1:
            img_id = ann["image_id"]
            coords = ann["segmentation"][0]  # segmentation хранит координаты вершин.
            points = np.array(coords).reshape(-1,
                                              2)  # преобразуем матрицу -1 автоматически определяет кол-во строк и 2 столбца

            # мы группируем для каждого изображения - кол-во полигонов
            if img_id not in parking_polygon:
                parking_polygon[img_id] = []
            parking_polygon[img_id].append(points)
            # Добавляет текущий полигон (points) в список полигонов для этого изображения

    camera = cv2.VideoCapture(VIDEO_URL)

    if not camera.isOpened():
        print("Ошибка! Не удалось открыть видеопоток")
        exit()

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            print("Ошибка чтения видеопотока")
            break

        high, width = frame.shape[:2]

        roi_width = width // 2
        roi_height = high // 4

        roi_x = width - roi_width
        roi_y = high - roi_height

        cv2.rectangle(frame, (roi_x, roi_y), (width, high), (0, 255, 255), 2)
        cv2.putText(frame, "Range Of Interest", (roi_x + 10, roi_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                    2)

        # Используем параметры из Config
        results = model(frame, conf=DETECTION_CONF,
                        iou=DETECTION_IOU,
                        imgsz=640,
                        augment=False,
                        verbose=False)[0]
        #   print(results.boxes)

        car_boxes = []
        vehicle_in_roi = 0

        free_spots = 0
        total_spots = 0

        if results.boxes is not None and len(results.boxes) > 0:
            # zip() объединяет соответствующие элементы из трех списков:
            for xyxy, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                xyxy = xyxy.cpu().numpy()
                cls = int(cls.cpu().numpy())
                conf = float(conf.cpu().numpy())

                # получаем название класса
                label = model.names[cls]
                if label in ["car", "truck"]:
                    x1, y1, x2, y2 = xyxy
                    # создаем Shapely box из координат [x1,y1, x2, y2]
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2

                    in_roi = (roi_x <= x_center <= width) and (roi_y <= y_center <= high)
                    if in_roi:
                        car_bbox = box(x1, y1, x2, y2)
                        car_boxes.append(car_bbox)
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1

                        bbox_area = bbox_width * bbox_height
                        if conf < 0.1 and label == "car" and bbox_area > 4000:
                            label = "truck"
                            color = (0, 255, 255)
                        elif label == "truck":
                            color = (0, 255, 255)
                        elif label == "car":
                            color = (0, 255, 0)
                        vehicle_in_roi += 1

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                        # text = f"{label} {conf:.2f}"
                        # cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            print("Объекты не обнаружены!")

        if 1 in parking_polygon:
            free_spots = 0
            total_spots = len(parking_polygon[1])

            # Перебирает все парковочные места для камеры 1
            for i, pts in enumerate(parking_polygon[1]):
                parking_poly = Polygon(pts)  # создает объект полигона из библиотеки Shapely
                occupied = False

                for car_box in car_boxes:
                    intersection_area = parking_poly.intersection(car_box).area

                    if intersection_area > min(parking_poly.area, car_box.area) * 0.65:
                        occupied = True
                        break
                pts_array = np.array(pts, dtype=np.int32)

                if occupied:
                    color = (0, 0, 255)
                    status_text = f"Zone {i + 1}: OCCUPIED"

                else:
                    color = (0, 255, 0)
                    status_text = f"Zone {i + 1}: FREE"
                    free_spots += 1

                # Рисуем контур полигона
                cv2.polylines(frame, [pts_array], True, color, 1)

                # cv2.putText(frame, status_text,
                #            (pts_array[0][0], pts_array[0][1] - 5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            stats_text = f"Free: {free_spots}/{total_spots}"
            cv2.putText(frame, stats_text, (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        roi_info = f"Vehicle: {vehicle_in_roi}"
        roi_occup = f"OCCUPIED ZONE: {total_spots - free_spots}"
        roi_zone = f"TOTAL ZONE: {total_spots}"
        errors = f"Errors: {int((total_spots - free_spots) - vehicle_in_roi)}"

        cv2.putText(frame, roi_info, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        print("=" * 50)
        print(roi_info)
        print(roi_zone)
        print(roi_occup)
        print(errors)
        print("=" * 50)

        cv2.imshow("Smart Parking System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
