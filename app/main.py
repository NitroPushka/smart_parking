from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
# Depends — используется для внедрения зависимостей -- сессии
from app import crud, models, schemas
from app.database import engine, get_db

app = FastAPI(title="SmartParkingAPI", version="1.0.0")

@app.post("/parking-lots", response_model=schemas.ParkingLot, status_code=status.HTTP_201_CREATED)
def create_parking_lot(lot: schemas.ParkingLotCreate, db: Session = Depends(get_db)):
    return crud.create_parking_lot(db, lot)

@app.post("/spots", response_model=schemas.ParkingSpot, status_code=status.HTTP_201_CREATED)
def create_parking_spot(spot: schemas.ParkingSpotCreate, db: Session = Depends(get_db)):
    # Проверяем, существует ли парковка
    lot = crud.get_parking_lot(db, spot.lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Parking lot not found")
    return crud.create_parking_spot(db, spot)

# response_model=schemas.ParkingLot - ответ должен соответствовать схеме ParkingLot
@app.get("/parking-lots/{lot_id}", response_model=schemas.ParkingLot)
def read_parking_lot(lot_id: int, db: Session = Depends(get_db)):
    lot = crud.get_parking_lot(db, lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Parking lot not found")
    return lot

@app.get("/parking-lots/{lot_id}/spots", response_model=List[schemas.ParkingSpot])
def read_parking_spots(lot_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    # Можно сначала проверить существование парковки
    lot = crud.get_parking_lot(db, lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Parking lot not found")
    spots = crud.get_parking_spots_by_lot(db, lot_id, skip=skip, limit=limit)
    return spots


@app.delete("/spots/{spot_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_parking_spot(spot_id: int, db: Session = Depends(get_db)):
    success = crud.delete_parking_spot(db, spot_id)
    if not success:
        raise HTTPException(status_code=404, detail="Parking spot not found")
    return None

@app.delete("/parking-lots/{lot_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_parking_spot(spot_id: int, db: Session = Depends(get_db)):
    success = crud.delete_parking_lot(db, spot_id)
    if not success:
        raise HTTPException(status_code=404, detail="Parking spot not found")
    return None

# Частичное обновление полигонов парковочного места
@app.patch("/spots/{spot_id}", response_model=schemas.ParkingSpot)
def update_parking_spot(spot_id: int, spot_update: schemas.ParkingSpotUpdate, db: Session = Depends(get_db)):
    updated_spot = crud.update_parking_spot(db, spot_id, spot_update)
    if not updated_spot:
        raise HTTPException(status_code=404, detail="Parking spot not found")
    return updated_spot