import cv2


class VideoProcessor:
    """
    Базовый класс для обработки видео потоков
    """

    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Невозможно открыть видеопоток: {video_source}")

    def process_frame(self, frame):
        """
        Обработка кадра
        :param frame: входной кадр
        :return: обработанный кадр
        """
        return frame

    def run(self):
        """
        Основной цикл обработки видео
        """
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Ошибка чтения видеопотока")
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow(self.__class__.__name__, processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()