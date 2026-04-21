from app import crud
from app import models


def test_create_analysis_task(client, test_lot, monkeypatch, tmp_path):
    import app.main as app_main

    calls = []

    def fake_send_task(name, args=None, task_id=None, **kwargs):
        calls.append({"name": name, "args": args, "task_id": task_id})

    monkeypatch.setattr(app_main, "upload_dir", tmp_path)
    monkeypatch.setattr(app_main.celery_app, "send_task", fake_send_task)

    spot_response = client.post(
        "/spots",
        json={
            "lot_id": test_lot,
            "spot_number": "A1",
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "status": "free",
        },
    )
    assert spot_response.status_code == 201

    response = client.post(
        "/analyses",
        data={"lot_id": str(test_lot)},
        files={"image": ("test.jpg", b"fake-image-content", "image/jpeg")},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert data["task_id"]
    assert len(calls) == 1
    assert calls[0]["name"] == "app.tasks.process_analysis"
    assert calls[0]["args"] == [data["task_id"]]
    assert calls[0]["task_id"] == data["task_id"]

    saved_file = tmp_path / f'{data["task_id"]}.jpg'
    assert saved_file.exists()


def test_create_analysis_task_with_polygon_json(client, db_session, test_lot, monkeypatch, tmp_path):
    import app.main as app_main

    calls = []
    polygon_json = """
    {
      "images": [{"id": 1, "file_name": "parking.jpg"}],
      "annotations": [
        {
          "id": 1,
          "image_id": 1,
          "category_id": 1,
          "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]]
        }
      ]
    }
    """

    def fake_send_task(name, args=None, task_id=None, **kwargs):
        calls.append({"name": name, "args": args, "task_id": task_id})

    monkeypatch.setattr(app_main, "upload_dir", tmp_path)
    monkeypatch.setattr(app_main.celery_app, "send_task", fake_send_task)

    response = client.post(
        "/analyses",
        data={"lot_id": str(test_lot)},
        files={
            "image": ("parking.jpg", b"fake-image-content", "image/jpeg"),
            "polygons_file": ("parking_zone.json", polygon_json.encode("utf-8"), "application/json"),
        },
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert len(calls) == 1

    saved_spots = crud.get_parking_spots_by_lot(db_session, test_lot)
    assert len(saved_spots) == 1
    assert saved_spots[0].spot_number == "P1"
    assert saved_spots[0].polygon == [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]


def test_get_analysis_task_result(client, db_session, test_lot, monkeypatch, tmp_path):
    import app.main as app_main

    monkeypatch.setattr(app_main, "upload_dir", tmp_path)
    monkeypatch.setattr(app_main.celery_app, "send_task", lambda *args, **kwargs: None)

    spot_response = client.post(
        "/spots",
        json={
            "lot_id": test_lot,
            "spot_number": "B1",
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "status": "free",
        },
    )
    assert spot_response.status_code == 201
    spot_id = spot_response.json()["id"]

    create_response = client.post(
        "/analyses",
        data={"lot_id": str(test_lot)},
        files={"image": ("test.jpg", b"fake-image-content", "image/jpeg")},
    )
    task_id = create_response.json()["task_id"]

    crud.update_analysis_task_status(
        db_session,
        task_id=task_id,
        status="completed",
        result=[{"spot_id": spot_id, "spot_number": "B1", "status": "free"}],
        error_message=None,
    )

    response = client.get(f"/analyses/{task_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == "completed"
    assert data["result"] == [{"spot_id": spot_id, "spot_number": "B1", "status": "free"}]

    saved_results = (
        db_session.query(models.AnalysisResult)
        .filter(models.AnalysisResult.task_id == task_id)
        .all()
    )
    assert len(saved_results) == 1
    assert saved_results[0].spot_id == spot_id
    assert saved_results[0].spot_number == "B1"
    assert saved_results[0].status == "free"
