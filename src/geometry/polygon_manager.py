import cv2
import numpy as np
import json
import os
from shapely.geometry import Polygon, box


class PolygonManager:
    def __init__(self, polygon_json_path):
        self.parking_polygons = self._load_parking_polygons(polygon_json_path)

    def _load_parking_polygons(self, json_path):
        with open(json_path) as json_file:
            data = json.load(json_file)

        parking_polygon = {}

        for ann in data["annotations"]:
            if ann["category_id"] == 1:
                img_id = ann["image_id"]
                coords = ann["segmentation"][0]
                points = np.array(coords).reshape(-1, 2)

                if img_id not in parking_polygon:
                    parking_polygon[img_id] = []
                parking_polygon[img_id].append(points)

        return parking_polygon

    def extract_parking_spot(self, image, polygon_points):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        x, y, w, h = cv2.boundingRect(pts)
        cropped = image[y:y + h, x:x + w]
        mask_cropped = mask[y:y + h, x:x + w]

        result = cv2.bitwise_and(cropped, cropped, mask=mask_cropped)
        white_bg = np.ones_like(cropped) * 255
        result_with_bg = cv2.bitwise_or(white_bg, white_bg, mask=cv2.bitwise_not(mask_cropped))
        result_with_bg = cv2.bitwise_or(result, result_with_bg)

        return result_with_bg
