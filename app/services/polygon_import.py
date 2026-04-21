from __future__ import annotations

import json
from pathlib import Path


def parse_parking_spots_from_coco(
    json_bytes: bytes,
    *,
    image_name: str | None = None,
    image_id: int | None = None,
) -> tuple[int, list[dict[str, object]]]:
    data = json.loads(json_bytes)
    annotations = data.get("annotations", [])
    images = data.get("images", [])

    if not annotations:
        raise ValueError("Polygon JSON does not contain annotations")

    target_image_id = image_id
    if target_image_id is None:
        image_ids = {ann["image_id"] for ann in annotations if ann.get("category_id") == 1}
        if not image_ids:
            raise ValueError("Polygon JSON does not contain parking polygons")

        if len(image_ids) == 1:
            target_image_id = next(iter(image_ids))
        elif image_name:
            normalized_name = Path(image_name).name
            matched_image = next(
                (item for item in images if Path(item.get("file_name", "")).name == normalized_name),
                None,
            )
            if not matched_image:
                raise ValueError(
                    "Polygon JSON contains multiple images and none matches the uploaded image name"
                )
            target_image_id = matched_image["id"]
        else:
            raise ValueError(
                "Polygon JSON contains multiple images; specify polygon_image_id or upload the matching image"
            )

    spots: list[dict[str, object]] = []
    spot_index = 1

    for ann in annotations:
        if ann.get("category_id") != 1 or ann.get("image_id") != target_image_id:
            continue

        segmentation = ann.get("segmentation") or []
        if not segmentation or not segmentation[0]:
            continue

        coords = segmentation[0]
        if len(coords) < 6 or len(coords) % 2 != 0:
            raise ValueError("Polygon JSON contains an invalid segmentation")

        polygon = []
        for idx in range(0, len(coords), 2):
            polygon.append([float(coords[idx]), float(coords[idx + 1])])

        spots.append(
            {
                "spot_number": f"P{spot_index}",
                "polygon": polygon,
                "status": "free",
            }
        )
        spot_index += 1

    if not spots:
        raise ValueError("No parking polygons found for the selected image")

    return target_image_id, spots
