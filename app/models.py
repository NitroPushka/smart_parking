from sqlalchemy import Column, Integer, String, JSON, DateTime, func, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class ParkingLot(Base):
    __tablename__ = "parking_lots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    coordinates = Column(JSON, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Отношение "один ко многим" к парковочным местам
    spots = relationship("ParkingSpot", back_populates="lot", cascade="all, delete-orphan")

class ParkingSpot(Base):
    __tablename__ = "parking_spots"

    id = Column(Integer, primary_key=True, index=True)
    lot_id = Column(Integer, ForeignKey("parking_lots.id"))
    spot_number = Column(String)
    polygon = Column(JSON)
    status = Column(String, default="free")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Обратная связь к парковке
    lot = relationship("ParkingLot", back_populates="spots")