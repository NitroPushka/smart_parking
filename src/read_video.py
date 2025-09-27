import cv2
import time

camera = cv2.VideoCapture("http://94.72.19.56/mjpg/video.mjpg")
time.sleep(2)
if not camera.isOpened():
    print("Could not open camera")
    exit()


while True:
    ret, frame = camera.read()
    if not ret:
        print("Не удались получить кадр")
        break

    frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("Parking Live Stream", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()