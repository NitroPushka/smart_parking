import json
import numpy as np
import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from parking_engine.scripts.detect_with_shapely import ShapelyDetectionProcessor


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_shapely_detection_processor_stub(tmp_path, dummy_frame):
    test_polygons = {
        "annotations": [
            {
                "category_id": 1,
                "image_id": 1,
                "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]]
            }
        ]
    }
    json_file = tmp_path / "test_polygons.json"
    json_file.write_text(json.dumps(test_polygons))

    with patch('parking_engine.scripts.detect_with_shapely.Config') as mock_config:
        mock_config.VIDEO_URL = "mock_video"
        mock_config.YOLO_MODEL = "mock_model.pt"
        mock_config.POLYGON_JSON = str(json_file)
        mock_config.CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5

        with patch('parking_engine.scripts.detect_with_shapely.VehicleDetector') as mock_vehicle_class:
            mock_detector = Mock()
            mock_vehicle_class.return_value = mock_detector

            mock_detector.detect_vehicles.return_value = ([], 0, [])  # car_boxes, vehicle_in_roi, detections_info

            with patch('parking_engine.scripts.detect_with_shapely.VideoProcessor.__init__') as mock_video_init:
                mock_video_init.return_value = None

                processor = ShapelyDetectionProcessor()

                assert processor.detector is mock_detector

                assert 1 in processor.parking_polygons
                assert len(processor.parking_polygons[1]) == 1
                assert isinstance(processor.parking_polygons[1][0], np.ndarray)

                result = processor.process_frame(dummy_frame)

                assert isinstance(result, np.ndarray)
                assert result.shape == dummy_frame.shape