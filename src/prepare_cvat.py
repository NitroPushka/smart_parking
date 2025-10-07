import cv2
import os

camera = cv2.VideoCapture("http://94.72.19.56/mjpg/video.mjpg")
output_dir = "C:/Users/ilyap/PycharmProjects/smart_parking/data/cvat_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count_pictures = 20
saved_intervals = 10

fps = camera.get(cv2.CAP_PROP_FPS)
saved_count = 0
frame_count = 0

while camera.isOpened() and saved_count < count_pictures:
    ret, frame = camera.read()
    if not ret:
        print("Error reading frame")
        break
    # если каждый кадр делится на 60й кадр без остатка
    if frame_count % int(fps * saved_intervals) == 0:
        filename = f"evening_parking_{frame_count}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        print(f"Saved {filename}")

        saved_count += 1

    frame_count += 1
camera.release()
print(f"Done saving {saved_count} images")


