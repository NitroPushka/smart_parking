import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, get_db

TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(TEST_DATABASE_URL)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def db_engine():
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestingSession(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
def client(db_session):
    def mock_get_db():
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = mock_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def test_lot(client):
    response = client.post(
        "/parking-lots",
        json={
            "name": "Test Parking",
            "coordinates": [
                [55.75, 37.61],
                [55.76, 37.61],
                [55.76, 37.62],
                [55.75, 37.62]
            ]
        }
    )
    assert response.status_code == 201
    return response.json()["id"]