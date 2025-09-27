import cv2
import json

frame = cv2.imread("C:/Users/ilyap/PycharmProjects/smart_parking/data/night_parking.jpg")

rois = cv2.selectROIs("Select Parking Slots", frame, showCrosshair=True, fromCenter=False)
rois = rois.tolist()

with open("C:/Users/ilyap/PycharmProjects/smart_parking/data/parking_slots.json", "w") as f:
    json.dump(rois, f)

cv2.destroyAllWindows()