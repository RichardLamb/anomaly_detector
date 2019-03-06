import cv2


class VideoHandler:
    def __init__(self, data_source, *, filepath=None, output_path=None, end_time_seconds=None):
        self.data_source = data_source
        self.filepath = filepath
        self.output_path = output_path

    def data_generator(self):
        if self.data_source == 'video_stream':
            self.cap = cv2.VideoCapture(0)
        elif self.data_source == 'video_clip':
            self.cap = cv2.VideoCapture(self.filepath)
        return self._create_image_generator()

    def _create_image_generator(self):
        while True:
            ret, frame = self.cap.read()

            if ret == False:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            yield frame

        self.kill_stream()

    def kill_stream(self):
        self.cap.release()
