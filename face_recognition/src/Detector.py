import torch
from ultralytics import YOLO

class Detector(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.detector = YOLO(model_path)

    def forward(self, img):
        results = self.detector(img, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None, None, None, None
        x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
        img = img.crop((x1 - 10, y1 - 10, x2 + 10, y2 + 10))
        left_eye = list(map(int, results[0].keypoints.xy[0][0].tolist()))
        right_eye = list(map(int, results[0].keypoints.xy[0][1].tolist()))
        left_eye[0] -= x1
        left_eye[1] -= y1
        right_eye[0] -= x1
        right_eye[1] -= y1
        return img, tuple(left_eye), tuple(right_eye), tuple([x1, y1, x2, y2])