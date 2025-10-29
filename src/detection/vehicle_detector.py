from shapely.geometry import box
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def _calculate_roi(self, frame_shape):
        """Вычисляет ROI координаты """
        height, width = frame_shape[:2]
        roi_width = width // 2
        roi_height = height // 4
        roi_x = width - roi_width
        roi_y = height - roi_height
        return roi_x, roi_y, width, height

    def detect_vehicles(self, frame, conf_threshold=0.02, iou_threshold=0.6):
        results = self.model(frame,
                             conf=conf_threshold,
                             iou=iou_threshold,
                             imgsz=640,
                             augment=False,
                             verbose=False)[0]

        car_boxes = []
        vehicle_in_roi = 0
        detections_info = []  # Новая переменная для хранения информации о детекциях

        if results.boxes is not None and len(results.boxes) > 0:
            roi_x, roi_y, roi_x2, roi_y2 = self._calculate_roi(frame.shape)

            for xyxy, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                xyxy = xyxy.cpu().numpy()
                cls = int(cls.cpu().numpy())
                conf = float(conf.cpu().numpy())

                label = self.model.names[cls]
                if label in ["car", "truck"]:
                    x1, y1, x2, y2 = xyxy

                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2

                    in_roi = (roi_x <= x_center <= roi_x2) and (roi_y <= y_center <= roi_y2)

                    if in_roi:
                        car_bbox = box(x1, y1, x2, y2)
                        car_boxes.append(car_bbox)

                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        bbox_area = bbox_width * bbox_height

                        if conf < 0.1 and label == "car" and bbox_area > 4000:
                            label = "truck"

                        vehicle_in_roi += 1

                    # Сохраняем информацию о всех детекциях для отрисовки
                    detections_info.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_name': label,
                        'in_roi': in_roi
                    })

        return car_boxes, vehicle_in_roi, detections_info