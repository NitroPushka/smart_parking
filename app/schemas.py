from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class ParkingLotBase(BaseModel):
    name: str
    coordinates: List[List[float]] # наш JSON файл с геометрией

class ParkingLot(ParkingLotBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None # либо datetime либо None

    class Config:
        orm_mode = True

class ParkingLotCreate(ParkingLotBase):
    pass


class ParkingSpotBase(BaseModel):
    spot_number: str
    polygon: List[List[float]]
    status: str = "free"

class ParkingSpotCreate(ParkingSpotBase):
    lot_id: int

class ParkingSpotUpdate(ParkingSpotBase):
    polygon: Optional[List[List[float]]] = None
    status: Optional[str] = None

# для ответа .. какие поля получит клиент
class ParkingSpot(ParkingSpotBase):
    id: int
    lot_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    class Config:
        orm_mode = True
