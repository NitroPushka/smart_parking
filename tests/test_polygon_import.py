from app.services import parse_parking_spots_from_coco


def test_parse_parking_spots_from_coco_single_image():
    payload = b"""
    {
      "images": [{"id": 1, "file_name": "night_parking_900.jpg"}],
      "annotations": [
        {
          "id": 1,
          "image_id": 1,
          "category_id": 1,
          "segmentation": [[445.16, 470.69, 479.17, 472.41, 481.5, 483.9, 446.7, 482.6]]
        },
        {
          "id": 2,
          "image_id": 1,
          "category_id": 1,
          "segmentation": [[447.9, 484.5, 483.4, 486.0, 485.6, 495.8, 449.1, 494.9]]
        }
      ]
    }
    """

    image_id, spots = parse_parking_spots_from_coco(payload, image_name="night_parking_900.jpg")

    assert image_id == 1
    assert len(spots) == 2
    assert spots[0]["spot_number"] == "P1"
    assert spots[1]["spot_number"] == "P2"
    assert spots[0]["polygon"] == [
        [445.16, 470.69],
        [479.17, 472.41],
        [481.5, 483.9],
        [446.7, 482.6],
    ]
