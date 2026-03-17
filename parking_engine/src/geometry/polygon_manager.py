import numpy as np
import json


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
