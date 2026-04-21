from fastapi import status

def test_create_parking_lot(client):
    response = client.post(
        "/parking-lots",
        json={
            "name": "Test",
            "coordinates": [
                [55.75, 37.61],
                [55.76, 37.61],
                [55.76, 37.62],
                [55.75, 37.62]
            ]
        }
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["name"] == "Test"
    assert "id" in data

def test_read_parking_lot(client, test_lot):
    response = client.get(f"/parking-lots/{test_lot}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["name"] == "Test Parking"

def test_read_parking_lot_not_found(client):
    response = client.get("/parking-lots/999")
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_create_parking_spot(client, test_lot):
    spot_data = {
        "lot_id": test_lot,
        "spot_number": "A1",
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "status": "free"
    }
    response = client.post("/spots", json=spot_data)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["spot_number"] == "A1"
    assert data["lot_id"] == test_lot
