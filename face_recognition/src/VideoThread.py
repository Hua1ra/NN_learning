import cv2
import torch
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    error_signal = pyqtSignal(str)
    def __init__(self,
                 detector,
                 normalizer,
                 extractor,
                 dbadmin,
                 device):
        super().__init__()
        self.is_running = True
        self.detector = detector
        self.normalizer = normalizer
        self.extractor = extractor
        self.dbadmin = dbadmin
        self.device = device

    def run(self):
        print('Video thread is running...')
        # Get the flow from the camera
        cap = cv2.VideoCapture(0)
        while self.is_running:
            # Get the image
            ret, frame = cap.read()
            if ret:
                # Convert the image to a proper format
                rgb_image = cv2.cvtColor(frame,
                                         cv2.COLOR_BGR2RGB)
                # Check if the models loaded correctly
                if self.detector is None:
                    print('No detector loaded.')
                    # noinspection PyUnresolvedReferences
                    self.error_signal.emit('No detector loaded.')
                    cap.release()
                    print('Video thread is closed.')
                    return
                if self.extractor is None:
                    print('No extractor loaded.')
                    # noinspection PyUnresolvedReferences
                    self.error_signal.emit('No extractor loaded.')
                    cap.release()
                    print('Video thread is closed.')
                    return
                if self.normalizer is None:
                    print('No normalizer loaded.')
                    # noinspection PyUnresolvedReferences
                    self.error_signal.emit('No normalizer loaded.')
                    cap.release()
                    print('Video thread is closed.')
                    return
                if self.dbadmin is None:
                    print('No database loaded.')
                    # noinspection PyUnresolvedReferences
                    self.error_signal.emit('No database loaded.')
                    cap.release()
                    print('Video thread is closed.')
                    return
                full_name = 'Undefined'
                with torch.no_grad():
                    detector_image = Image.fromarray(rgb_image)
                    cropped_image, left_eye, right_eye, rectangle = self.detector(detector_image)
                    # TODO: normalize_transform?
                    # Get the closest vector if at least one face was found
                    if cropped_image is not None:
                        cropped_image = self.normalizer.transform(cropped_image)
                        cropped_image = cropped_image.to(self.device)
                        embedding = self.extractor(cropped_image)[0]
                        # TODO: reaccess bd every n frames
                        closest = self.dbadmin.get_closest(embedding.tolist(),
                                                           threshold=0.6)
                        # Get the name
                        if closest:
                            closest = closest[0]
                            person_id = closest[1]
                            full_name = ' '.join(self.dbadmin.get_person(person_id)[1:3])
                        # Print results
                        if cropped_image is not None:
                            cv2.putText(rgb_image,
                                        full_name,
                                        (rectangle[0], rectangle[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 255, 0),
                                        2)
                            cv2.rectangle(rgb_image,
                                          (rectangle[0], rectangle[1]),
                                          (rectangle[2], rectangle[3]),
                                          (0, 255, 0),
                                          2)
                # Create QImage
                h, w, ch = rgb_image.shape
                bytes_per_line = rgb_image.strides[0]
                qt_image = QImage(rgb_image.data,
                                  w,
                                  h,
                                  bytes_per_line,
                                  QImage.Format.Format_RGB888).copy()
                # Return it to the main window
                # noinspection PyUnresolvedReferences
                self.change_pixmap_signal.emit(qt_image)
        cap.release()
        print('Video thread is closed.')

    def stop(self):
        self.is_running = False