from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class ParkingLotBase(BaseModel):
    name: str
    coordinates: List[List[float]]


class ParkingLot(ParkingLotBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ParkingLotCreate(ParkingLotBase):
    pass


class ParkingSpotBase(BaseModel):
    spot_number: str
    polygon: List[List[float]]
    status: str = "free"


class ParkingSpotCreate(ParkingSpotBase):
    lot_id: int


class ParkingSpotUpdate(ParkingSpotBase):
    spot_number: Optional[str] = None
    polygon: Optional[List[List[float]]] = None
    status: Optional[str] = None


class ParkingSpot(BaseModel):
    id: int
    lot_id: int
    spot_number: str
    polygon: List[List[float]]
    status: str = "free"
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AnalysisTaskCreateResponse(BaseModel):
    task_id: str
    status: str


class SpotAnalysisResult(BaseModel):
    spot_id: int
    spot_number: str
    status: str


class AnalysisTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    lot_id: int
    result: Optional[List[SpotAnalysisResult]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
