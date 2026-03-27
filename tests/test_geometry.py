import json
import numpy as np
from parking_engine.src.geometry.polygon_manager import PolygonManager

def test_load_polygons_stub(tmp_path):
    test_data = {
        "annotations": [
            {
                "category_id": 1,
                "image_id": 1,
                "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]]
            },
            {
                "category_id": 1,
                "image_id": 2,
                "segmentation": [[30, 30, 40, 30, 40, 40, 30, 40]]
            }
        ]
    }
    json_file = tmp_path / "test_polygons.json"
    json_file.write_text(json.dumps(test_data))

    manager = PolygonManager(str(json_file))
    polygons = manager.parking_polygons  # Вот здесь берём словарь

    assert isinstance(polygons, dict)
    assert 1 in polygons
    assert 2 in polygons
    assert len(polygons[1]) == 1
    assert isinstance(polygons[1][0], np.ndarray)
    assert polygons[1][0].shape == (4, 2)