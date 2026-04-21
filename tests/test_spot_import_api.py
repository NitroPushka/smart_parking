from app import crud


def test_import_spots_from_polygon_json(client, db_session, test_lot):
    polygon_json = """
    {
      "images": [{"id": 7, "file_name": "night_parking_900.jpg"}],
      "annotations": [
        {
          "id": 1,
          "image_id": 7,
          "category_id": 1,
          "segmentation": [[10, 20, 30, 20, 30, 40, 10, 40]]
        },
        {
          "id": 2,
          "image_id": 7,
          "category_id": 1,
          "segmentation": [[50, 60, 70, 60, 70, 80, 50, 80]]
        }
      ]
    }
    """

    response = client.post(
        f"/parking-lots/{test_lot}/spots/import",
        data={"replace_existing": "true"},
        files={
            "polygons_file": (
                "parking_zone.json",
                polygon_json.encode("utf-8"),
                "application/json",
            )
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data == {"lot_id": test_lot, "imported_spots": 2, "image_id": 7}

    spots = crud.get_parking_spots_by_lot(db_session, test_lot)
    assert len(spots) == 2
    assert [spot.spot_number for spot in spots] == ["P1", "P2"]
