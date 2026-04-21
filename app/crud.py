from sqlalchemy.orm import Session
from app import models, schemas

# вывод соответстующей парковки
def get_parking_lot(db: Session, lot_id: int):
    return db.query(models.ParkingLot).filter(models.ParkingLot.id == lot_id).first()

def create_parking_lot(db: Session, lot: schemas.ParkingLotCreate):
    db_lot = models.ParkingLot(**lot.model_dump()) # преобразует Pydantic-схему в словарь ->  распаковывает словарь в именованные аргументы
    db.add(db_lot) # добавляет объект в сессию
    db.commit()
    db.refresh(db_lot)
    return db_lot

def create_parking_spot(db: Session, spot: schemas.ParkingSpotCreate):
    db_spot = models.ParkingSpot(**spot.model_dump())
    db.add(db_spot)
    db.commit()
    db.refresh(db_spot)
    return db_spot


def create_parking_spots(db: Session, spots: list[schemas.ParkingSpotCreate]):
    db_spots = [models.ParkingSpot(**spot.model_dump()) for spot in spots]
    db.add_all(db_spots)
    db.commit()
    for db_spot in db_spots:
        db.refresh(db_spot)
    return db_spots

def get_parking_spots_by_lot(db: Session, lot_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.ParkingSpot).filter(models.ParkingSpot.lot_id == lot_id).offset(skip).limit(limit).all()

def get_parking_spot(db: Session, spot_id: int):
    return db.query(models.ParkingSpot).filter(models.ParkingSpot.id == spot_id).first()

def update_parking_spot(db: Session, spot_id: int, spot_update: schemas.ParkingSpotUpdate):
    db_spot = get_parking_spot(db, spot_id)
    if not db_spot:
        return None

    update_data = spot_update.dict(exclude_unset=True)
    # преобразует Pydantic-схему в словарь, исключая поля, которые не были переданы

    # проходим по всем переданным полям
    for field, value in update_data.items():
        setattr(db_spot, field, value) # устанавливаем новое значение атрибута SQLAlchemy-объекта
    db.commit()
    db.refresh(db_spot)
    return db_spot

def delete_parking_spot(db: Session, spot_id: int) -> bool:
    db_spot = get_parking_spot(db, spot_id)
    if db_spot:
        db.delete(db_spot)
        db.commit()
        return True
    return False

def delete_parking_lot(db: Session, lot_id: int) -> bool:
    db_lot = get_parking_lot(db, lot_id)
    if db_lot:
        db.delete(db_lot)
        db.commit()
        return True
    return False


def delete_parking_spots_by_lot(db: Session, lot_id: int) -> int:
    deleted = (
        db.query(models.ParkingSpot)
        .filter(models.ParkingSpot.lot_id == lot_id)
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted


def create_analysis_task(db: Session, task_id: str, lot_id: int, image_path: str):
    db_task = models.AnalysisTask(
        task_id=task_id,
        lot_id=lot_id,
        image_path=image_path,
        status="queued",
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


def get_analysis_task(db: Session, task_id: str):
    return db.query(models.AnalysisTask).filter(models.AnalysisTask.task_id == task_id).first()


def get_analysis_results(db: Session, task_id: str):
    return (
        db.query(models.AnalysisResult)
        .filter(models.AnalysisResult.task_id == task_id)
        .order_by(models.AnalysisResult.id.asc())
        .all()
    )


def update_analysis_task_status(
    db: Session,
    task_id: str,
    status: str,
    result=None,
    error_message: str | None = None,
):
    db_task = get_analysis_task(db, task_id)
    if not db_task:
        return None

    db_task.status = status
    db_task.result = result
    db_task.error_message = error_message

    db.query(models.AnalysisResult).filter(
        models.AnalysisResult.task_id == task_id
    ).delete(synchronize_session=False)

    if result:
        for item in result:
            db.add(
                models.AnalysisResult(
                    task_id=task_id,
                    spot_id=item["spot_id"],
                    spot_number=item["spot_number"],
                    status=item["status"],
                )
            )

    db.commit()
    db.refresh(db_task)
    return db_task
