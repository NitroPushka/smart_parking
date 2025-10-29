import cv2
import numpy as np
import json
import sys
import os
from shapely.geometry import Polygon

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import Config
from src.utils.video_process import VideoProcessor
from src.detection.vehicle_detector import VehicleDetector


class ShapelyDetectionProcessor(VideoProcessor):
    def __init__(self):
        super().__init__(Config.VIDEO_URL)

        # Используем Config для всех параметров
        self.detector = VehicleDetector(Config.YOLO_MODEL)

        with open(Config.POLYGON_JSON) as json_file:
            data = json.load(json_file)

        # ключ = id картинки, значение = список полигонов \\ у меня две картинки = два ключа
        self.parking_polygons = {}
        # print(data["annotations"])

        # список аннотаций (annotations),
        for ann in data["annotations"]:
            if ann["category_id"] == 1:
                img_id = ann["image_id"]
                coords = ann["segmentation"][
                    0
                ]  # segmentation хранит координаты вершин.
                points = np.array(coords).reshape(
                    -1, 2
                )  # преобразуем матрицу -1 автоматически определяет кол-во строк и 2 столбца

                # мы группируем для каждого изображения - кол-во полигонов
                if img_id not in self.parking_polygons:
                    self.parking_polygons[img_id] = []
                self.parking_polygons[img_id].append(points)
                # Добавляет текущий полигон (points) в список полигонов для этого изображения

    def process_frame(self, frame):
        display_frame = frame.copy()
        high, width = frame.shape[:2]

        # Настройка и отрисовка ROI (Region of Interest)
        roi_width = width // 2
        roi_height = high // 4
        roi_x = width - roi_width
        roi_y = high - roi_height

        cv2.rectangle(
            display_frame, (roi_x, roi_y), (width, high), (0, 255, 255), 2
        )
        cv2.putText(
            display_frame,
            "Range Of Interest",
            (roi_x + 10, roi_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        car_boxes, vehicle_in_roi = self.detector.detect_vehicles(frame)
        #   print(results.boxes)

        free_spots = 0
        total_spots = 0

        if 1 in self.parking_polygons:
            free_spots = 0
            total_spots = len(self.parking_polygons[1])

            # Перебирает все парковочные места для камеры 1
            for i, pts in enumerate(self.parking_polygons[1]):
                parking_poly = Polygon(
                    pts
                )  # создает объект полигона из библиотеки Shapely
                occupied = False

                for car_box in car_boxes:
                    intersection_area = parking_poly.intersection(car_box).area

                    if (
                        intersection_area
                        > min(parking_poly.area, car_box.area)
                        * Config.CLASSIFICATION_CONFIDENCE_THRESHOLD
                    ):
                        occupied = True
                        break
                pts_array = np.array(pts, dtype=np.int32)

                if occupied:
                    color = (0, 0, 255)
                    # status_text = f"Zone {i + 1}: OCCUPIED"

                else:
                    color = (0, 255, 0)
                    # status_text = f"Zone {i + 1}: FREE"
                    free_spots += 1

                # Рисуем контур полигона
                cv2.polylines(display_frame, [pts_array], True, color, 1)

                # cv2.putText(display_frame, status_text,
                #            (pts_array[0][0], pts_array[0][1] - 5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            stats_text = f"Free: {free_spots}/{total_spots}"
            cv2.putText(
                display_frame,
                stats_text,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

        roi_info = f"Vehicle: {vehicle_in_roi}"
        roi_occup = f"OCCUPIED ZONE: {total_spots - free_spots}"
        roi_zone = f"TOTAL ZONE: {total_spots}"
        errors = f"Errors: {int((total_spots - free_spots) - vehicle_in_roi)}"

        cv2.putText(
            display_frame,
            roi_info,
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        print("=" * 50)
        print(roi_info)
        print(roi_zone)
        print(roi_occup)
        print(errors)
        print("=" * 50)

        return display_frame


def main():
    """Основная функция запуска детекции"""
    try:
        processor = ShapelyDetectionProcessor()
        processor.run()
    except Exception as e:
        print(f"Ошибка запуска детектора: {e}")


if __name__ == "__main__":
    main()