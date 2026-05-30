import dotenv
import os
import sys
import torch
import torch_directml
from PIL import Image
from src.DBAdmin import DBAdmin
from src.Detector import Detector
from src.Extractor import Extractor
from src.Normalizer import Normalizer
from src.AddFaceDialog import AddFaceDialog
from src.VideoThread import VideoThread
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtWidgets import QFileDialog, QLabel, QMessageBox, QPushButton

class FaceApp(QMainWindow):
    def __init__(self,
                 detector_path='./models/yolov8n-face.pt',
                 extractor_path='./models/my_extractor4.pth'):
        super().__init__()
        self.extractor_path = extractor_path
        self.detector_path = detector_path
        dotenv.load_dotenv()
        self.db_config = {
            'DB_HOST': os.getenv('DB_HOST'),
            'DB_NAME': os.getenv('DB_NAME'),
            'DB_USER': os.getenv('DB_USER'),
            'DB_PASSWORD': os.getenv('DB_PASSWORD'),
            'DB_PORT': os.getenv('DB_PORT'),
            'DB_ADMIN_PASSWORD': os.getenv('DB_ADMIN_PASSWORD')
        }
        self.device = torch_directml.device(0)
        # Models
        self.dbadmin = None
        self.detector = None
        self.extractor = None
        self.normalizer = None
        # Video
        self.VideoThread = None
        # Application
        self.setWindowTitle('Face Recognition System')
        self.setGeometry(100, 100, 400, 300)
        # UI
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        # Display area
        self.display_label = QLabel('No models where loaded.')
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setStyleSheet('background-color: #2e2e2e; color: white; font-size: 16px;')
        layout.addWidget(self.display_label)
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton('Reload models')
        self.btn_start = QPushButton('Start')
        self.btn_stop = QPushButton('Stop')
        self.btn_add = QPushButton('Add face')
        for btn in [self.btn_load, self.btn_start, self.btn_stop, self.btn_add]:
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)
        # Connections
        # noinspection PyUnresolvedReferences
        self.btn_load.clicked.connect(lambda: self.load_model(self.detector_path,
                                                              self.extractor_path))
        # noinspection PyUnresolvedReferences
        self.btn_start.clicked.connect(self.start_processing)
        # noinspection PyUnresolvedReferences
        self.btn_stop.clicked.connect(self.stop_processing)
        # noinspection PyUnresolvedReferences
        self.btn_add.clicked.connect(self.open_add_face_dialog)
        # Load models
        self.load_model(self.detector_path,
                        self.extractor_path)

    def load_model(self, detector_path, extractor_path):
        # Try to load every file.
        if self.VideoThread is not None and self.VideoThread.is_running:
            print('Cannot reload models while running video.')
            return
        try:
            print('Loading Database...')
            self.dbadmin = DBAdmin(db_config=self.db_config)
            print('Database loaded.')
            print('Loading Detector...')
            self.detector = Detector(model_path=detector_path)
            print('Detector loaded.')
            print('Loading Extractor...')
            self.extractor = Extractor(model_path='./models/feature_extractor.pth')
            checkpoint = torch.load(extractor_path,
                                    weights_only=True)
            self.extractor.load_state_dict(checkpoint)
            self.extractor.eval()
            self.extractor = self.extractor.to(self.device)
            print('Extractor loaded.')
            print('Loading Normalizer...')
            self.normalizer = Normalizer()
            print('Normalizer loaded.')
        except Exception as e:
            print('Failed to load models.')
            print(e)
            self.print_info('Failed to load models.')
        finally:
            print('Ready to work.')
            self.print_info('Ready to work.')

    def start_processing(self):
        # Start the thread
        self.VideoThread = VideoThread(self.detector,
                                       self.normalizer,
                                       self.extractor,
                                       self.dbadmin,
                                       self.device)
        self.VideoThread.change_pixmap_signal.connect(self.update_image)
        self.VideoThread.error_signal.connect(self.print_info)
        self.VideoThread.start()
        print('Starting camera...')
        self.print_info('Starting camera...')

    def stop_processing(self):
        if self.VideoThread and self.VideoThread.is_running:
            self.VideoThread.stop()
            self.VideoThread.change_pixmap_signal.disconnect()
            self.display_label.clear()
            print('Terminated.')
            self.print_info('Terminated.')
            self.VideoThread = None

    def open_add_face_dialog(self):
        # Check errors
        if self.detector is None:
            print('No detector loaded.')
            self.print_info('No detector loaded.')
            return
        if self.extractor is None:
            print('No extractor loaded.')
            self.print_info('No extractor loaded.')
            return
        if self.normalizer is None:
            print('No normalizer loaded.')
            self.print_info('No normalizer loaded.')
            return
        if self.dbadmin is None:
            print('No database loaded.')
            self.print_info('No database loaded.')
            return
        # TODO: Check if person exists
        print('Adding faces...')
        self.print_info('Adding faces...')
        files, _ = QFileDialog.getOpenFileNames(self,
                                                'Choose 5 - 15 photos',
                                                '',
                                                'Images (*.jpg *.png)')
        # Check files
        if 1 <= len(files) <= 15:
            dialog = AddFaceDialog(self)
            if dialog.exec():
                # Get the information
                data = dialog.get_data()
                # Push a person into the database
                index = self.dbadmin.add_person(data['first_name'],
                                                data['last_name'],
                                                data['surname'],
                                                data['birthdate'])
                with torch.no_grad():
                    # Try to add all records
                    error_text = ''
                    for i, file in enumerate(files):
                        img = Image.open(file).convert('RGB')
                        # Try to detect faces
                        cropped_image, left_eye, right_eye, rectangle = self.detector(img)
                        if cropped_image is not None:
                            # TODO: normalize_transform?
                            cropped_image = self.normalizer.transform(cropped_image)
                            cropped_image = cropped_image.to(self.device)
                            embedding = self.extractor(cropped_image)[0]
                            self.dbadmin.add_record(index, embedding.cpu().tolist())
                        else:
                            print(f'No face detected on the image indexed: ({i})')
                            error_text = error_text + f'No face detected on the image indexed: ({i})\n'
                    if error_text:
                            self.print_info(error_text)
                    else:
                        print(f'Added {len(files)} photos successfully.')
                        self.print_info(f'Added {len(files)} photos successfully.')
        else:
            print('Error. Not enough photos.')
            QMessageBox.warning(self,
                                'Error',
                                'Choose 1 - 15 photos.')

    def update_image(self, qt_img):
        # Update QLabel
        self.display_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))

    def print_info(self, error_message):
        self.display_label.setText(error_message)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceApp()
    window.show()
    sys.exit(app.exec())