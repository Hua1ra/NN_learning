import math
import torchvision

class Normalizer:
    def __init__(self, mode='transform', transformer=None):
        self.mode = mode
        self.transformer = transformer
        if self.transformer is None:
            self.transformer = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])
    def normalize(self, image, left_eye, right_eye):
        angle = -math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        image = image.rotate(-math.degrees(angle), center=center)
        new_left_eye = [0, 0]
        new_right_eye = [0, 0]
        new_left_eye[0] = center[0] + (left_eye[0] - center[0]) * math.cos(angle) - (left_eye[1] - center[1]) * math.sin(angle)
        new_left_eye[1] = center[1] + (left_eye[0] - center[0]) * math.sin(angle) + (left_eye[1] - center[1]) * math.cos(angle)
        new_right_eye[0] = center[0] + (right_eye[0] - center[0]) * math.cos(angle) - (right_eye[1] - center[1]) * math.sin(angle)
        new_right_eye[1] = center[1] + (right_eye[0] - center[0]) * math.sin(angle) + (right_eye[1] - center[1]) * math.cos(angle)
        if self.mode == 'transform':
            image = self.transformer(image)
        return image, tuple(new_left_eye), tuple(new_right_eye)

    def transform(self):
        self.mode = 'transform'

    def rotate(self):
        self.mode = 'rotate'