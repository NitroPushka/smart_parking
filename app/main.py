from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app import crud, schemas
from app.config import settings
from app.celery_app import celery_app
from app.database import get_db

app = FastAPI(title="SmartParkingAPI", version="1.0.0")
upload_dir = Path(settings.UPLOAD_DIR)
upload_dir.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/parking-lots", response_model=schemas.ParkingLot, status_code=status.HTTP_201_CREATED)
def create_parking_lot(lot: schemas.ParkingLotCreate, db: Session = Depends(get_db)):
    return crud.create_parking_lot(db, lot)


@app.post("/spots", response_model=schemas.ParkingSpot, status_code=status.HTTP_201_CREATED)
def create_parking_spot(spot: schemas.ParkingSpotCreate, db: Session = Depends(get_db)):
    lot = crud.get_parking_lot(db, spot.lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Parking lot not found")
    return crud.create_parking_spot(db, spot)


@app.get("/parking-lots/{lot_id}", response_model=schemas.ParkingLot)
def read_parking_lot(lot_id: int, db: Session = Depends(get_db)):
    lot = crud.get_parking_lot(db, lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Parking lot not found")
    return lot


@app.get("/parking-lots/{lot_id}/spots", response_model=List[schemas.ParkingSpot])
def read_parking_spots(
    lot_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    lot = crud.get_parking_lot(db, lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Parking lot not found")
    return crud.get_parking_spots_by_lot(db, lot_id, skip=skip, limit=limit)


@app.delete("/spots/{spot_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_parking_spot(spot_id: int, db: Session = Depends(get_db)):
    success = crud.delete_parking_spot(db, spot_id)
    if not success:
        raise HTTPException(status_code=404, detail="Parking spot not found")
    return None


@app.delete("/parking-lots/{lot_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_parking_lot(lot_id: int, db: Session = Depends(get_db)):
    success = crud.delete_parking_lot(db, lot_id)
    if not success:
        raise HTTPException(status_code=404, detail="Parking lot not found")
    return None


@app.patch("/spots/{spot_id}", response_model=schemas.ParkingSpot)
def update_parking_spot(
    spot_id: int,
    spot_update: schemas.ParkingSpotUpdate,
    db: Session = Depends(get_db),
):
    updated_spot = crud.update_parking_spot(db, spot_id, spot_update)
    if not updated_spot:
        raise HTTPException(status_code=404, detail="Parking spot not found")
    return updated_spot


@app.post(
    "/analyses",
    response_model=schemas.AnalysisTaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_analysis_task(
    lot_id: int = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    lot = crud.get_parking_lot(db, lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Parking lot not found")

    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename is empty")

    task_id = str(uuid4())
    image_suffix = Path(image.filename).suffix or ".jpg"
    image_path = upload_dir / f"{task_id}{image_suffix}"
    image_bytes = await image.read()
    image_path.write_bytes(image_bytes)

    crud.create_analysis_task(db, task_id=task_id, lot_id=lot_id, image_path=str(image_path))
    celery_app.send_task("app.tasks.process_analysis", args=[task_id], task_id=task_id)

    return schemas.AnalysisTaskCreateResponse(task_id=task_id, status="queued")


@app.get("/analyses/{task_id}", response_model=schemas.AnalysisTaskStatusResponse)
def get_analysis_task(task_id: str, db: Session = Depends(get_db)):
    db_task = crud.get_analysis_task(db, task_id)
    if not db_task:
        raise HTTPException(status_code=404, detail="Analysis task not found")

    result = [
        schemas.SpotAnalysisResult(
            spot_id=item.spot_id,
            spot_number=item.spot_number,
            status=item.status,
        )
        for item in crud.get_analysis_results(db, task_id)
    ]

    return schemas.AnalysisTaskStatusResponse(
        task_id=db_task.task_id,
        status=db_task.status,
        lot_id=db_task.lot_id,
        result=result or None,
        error_message=db_task.error_message,
        created_at=db_task.created_at,
        updated_at=db_task.updated_at,
    )
