import json
import time
from pathlib import Path

import requests


BASE_URL = "http://localhost:8000"
IMAGE_PATH = Path(r"C:\full\path\to\parking.jpg")
POLYGONS_PATH = Path(r"C:\full\path\to\parking_zone.json")


def create_parking_lot():
    payload = {
        "name": "Script parking",
        "coordinates": [
            [55.75, 37.61],
            [55.76, 37.61],
            [55.76, 37.62],
            [55.75, 37.62],
        ],
    }
    response = requests.post(f"{BASE_URL}/parking-lots", json=payload)
    response.raise_for_status()
    return response.json()["id"]


def start_analysis(lot_id: int):
    with IMAGE_PATH.open("rb") as image_file, POLYGONS_PATH.open("rb") as polygons_file:
        response = requests.post(
            f"{BASE_URL}/analyses",
            data={
                "lot_id": str(lot_id),
                "replace_existing_spots": "true",
            },
            files={
                "image": (IMAGE_PATH.name, image_file, "image/jpeg"),
                "polygons_file": (POLYGONS_PATH.name, polygons_file, "application/json"),
            },
        )
    response.raise_for_status()
    return response.json()


def wait_result(task_id: str, timeout_sec: int = 120):
    started = time.time()
    while time.time() - started < timeout_sec:
        response = requests.get(f"{BASE_URL}/analyses/{task_id}")
        response.raise_for_status()
        data = response.json()

        print("task status:", data["status"])

        if data["status"] in {"completed", "failed"}:
            return data

        time.sleep(2)

    raise TimeoutError("Task did not finish in time")


def main():
    health = requests.get(f"{BASE_URL}/health")
    health.raise_for_status()
    print("health:", health.json())

    lot_id = create_parking_lot()
    print("lot_id:", lot_id)

    task = start_analysis(lot_id)
    print("task:", json.dumps(task, ensure_ascii=False, indent=2))

    result = wait_result(task["task_id"])
    print("final result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
