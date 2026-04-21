from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon

from app import crud
from parking_engine.src.detection.vehicle_detector import VehicleDetector
from parking_engine.src.utils.config import Config

_detector: VehicleDetector | None = None


def get_vehicle_detector() -> VehicleDetector:
    global _detector
    if _detector is None:
        _detector = VehicleDetector(Config.YOLO_MODEL)
    return _detector


def analyze_parking_image(db, lot_id: int, image_path: str) -> list[dict[str, str | int]]:
    spots = crud.get_parking_spots_by_lot(db, lot_id, skip=0, limit=10_000)
    if not spots:
        raise ValueError("Parking spots for this lot were not found")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Image could not be read")

    detector = get_vehicle_detector()
    car_boxes, _, _ = detector.detect_vehicles(
        frame,
        conf_threshold=Config.DETECTION_CONF,
        iou_threshold=Config.DETECTION_IOU,
    )

    results: list[dict[str, str | int]] = []

    for spot in spots:
        parking_polygon = Polygon(np.array(spot.polygon, dtype=float))
        occupied = False

        for car_box in car_boxes:
            intersection_area = parking_polygon.intersection(car_box).area
            threshold = (
                min(parking_polygon.area, car_box.area)
                * Config.CLASSIFICATION_CONFIDENCE_THRESHOLD
            )
            if intersection_area > threshold:
                occupied = True
                break

        results.append(
            {
                "spot_id": spot.id,
                "spot_number": spot.spot_number,
                "status": "occupied" if occupied else "free",
            }
        )

    return results
