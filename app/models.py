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


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    task_id = Column(String, primary_key=True, index=True)
    lot_id = Column(Integer, ForeignKey("parking_lots.id"), nullable=False)
    image_path = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    result = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    lot = relationship("ParkingLot")
    analysis_results = relationship(
        "AnalysisResult",
        back_populates="task",
        cascade="all, delete-orphan",
    )


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, ForeignKey("analysis_tasks.task_id"), nullable=False, index=True)
    spot_id = Column(Integer, ForeignKey("parking_spots.id"), nullable=False)
    spot_number = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    task = relationship("AnalysisTask", back_populates="analysis_results")
    spot = relationship("ParkingSpot")
