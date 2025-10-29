import cv2


class VideoProcessor:
    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot open video source: {video_source}")

    def process_frame(self, frame):
        """Обработка кадра"""
        return frame

    def run(self):
        """Основной цикл обработки видео"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow(self.__class__.__name__, processed_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()