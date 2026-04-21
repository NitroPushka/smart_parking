from app import crud, schemas
from app.tasks import process_analysis


def test_process_analysis_saves_results(db_session, monkeypatch):
    lot = crud.create_parking_lot(
        db_session,
        lot=schemas.ParkingLotCreate(
            name="Worker Lot",
            coordinates=[[0, 0], [1, 0], [1, 1], [0, 1]],
        ),
    )

    spot = crud.create_parking_spot(
        db_session,
        spot=schemas.ParkingSpotCreate(
            lot_id=lot.id,
            spot_number="W1",
            polygon=[[0, 0], [2, 0], [2, 2], [0, 2]],
            status="free",
        ),
    )
    spot_id = spot.id

    task = crud.create_analysis_task(
        db_session,
        task_id="task-worker-1",
        lot_id=lot.id,
        image_path="uploads/task-worker-1.jpg",
    )

    monkeypatch.setattr(
        "app.tasks.analyze_parking_image",
        lambda db, lot_id, image_path: [
            {"spot_id": spot.id, "spot_number": "W1", "status": "occupied"}
        ],
    )
    monkeypatch.setattr("app.tasks.Session", lambda: db_session)

    result = process_analysis(task.task_id)
    db_session.expire_all()

    db_task = crud.get_analysis_task(db_session, task.task_id)
    saved_results = crud.get_analysis_results(db_session, task.task_id)

    assert result["status"] == "completed"
    assert db_task.status == "completed"
    assert len(saved_results) == 1
    assert saved_results[0].spot_id == spot_id
    assert saved_results[0].spot_number == "W1"
    assert saved_results[0].status == "occupied"
