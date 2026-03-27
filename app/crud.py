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