from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    echo=True, # логируем SQL-запросы!
) # создали пул соединении

Session = sessionmaker(autocommit=False, autoflush=False, bind=engine) # рабочий стол для БД. пишем SQL-запросы и сохраняем...

Base = declarative_base() # Базовый класс для моделей - от него наследуются все таблицы
# используется Alembic для генерации миграции

# предоставление сессия БД в эндпоинтах
def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()