import json
import os
import torch
import tqdm
import random
from collections import defaultdict
from PIL import Image

# Create Data Processor Class
class DataProcessor:
    def __init__(self, cnn_output=20,
                 base_path='./resources/tt100k_2021/other/',
                 annotations_path='./resources/tt100k_2021/annotations_all.json',
                 save_path='./resources/tt100k_2021/train/',
                 labels_path='./resources/tt100k_2021/labels_all.json',):
        self.CNN_output = cnn_output # We have a 20 * 20 grid after CNN layers
        self.nums_per_pixel = 5 # For each pixel in the grid we have (probability, x, y, width, height)
        self.base_path = base_path # Base path for images_old
        self.annotations_path = annotations_path # Path for annotations
        self.save_path = save_path # Path to save processed data
        self.labels_path = labels_path
    def create_labels(self, image, objects):
        # Base tensor, which would be transformed into a label tensor
        labels = torch.zeros([self.CNN_output, self.CNN_output, self.nums_per_pixel], dtype=torch.float32)
        size = image.size
        for bbox in objects: # For each sign on the picture
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
            x_center = int(xmin + xmax) // 2 # Sign center coordinate x
            y_center = int(ymin + ymax) // 2 # Sign center coordinate x
            block_x = int(self.CNN_output / size[0] * x_center) # Block id (x)
            block_y = int(self.CNN_output / size[1] * y_center) # Block id (y)
            # Clamp values (0, CNN_output)
            block_x = max(0, block_x)
            block_y = max(0, block_y)
            block_x = min(self.CNN_output - 1, block_x)
            block_y = min(self.CNN_output - 1, block_y)
            # Relative coords
            x_center = (x_center % (size[0] / self.CNN_output)) / (size[0] / self.CNN_output)
            y_center = (y_center % (size[1] / self.CNN_output)) / (size[1] / self.CNN_output)
            width = (xmax - xmin) / size[0]
            height = (ymax - ymin) / size[1]
            labels[block_y, block_x] = torch.tensor([1, x_center, y_center, width, height])
        return labels.view(-1)
    def get_tiles(self, image, objects):
        # Split image into a 2 * 2 image grid
        tiles = [[1 for _ in range(2)] for _ in range(2)]
        signs = [[list() for _ in range(2)] for _ in range(2)]
        labels = [[1 for _ in range(2)] for _ in range(2)]
        size = image.size
        half_w = size[0] // 2
        half_h = size[1] // 2
        # Get subimages
        tiles[0][0] = (image.crop((0, 0, half_w, half_h)))
        tiles[0][1] = (image.crop((half_w, 0, size[0], half_h)))
        tiles[1][0] = (image.crop((0, half_h, half_w, size[1])))
        tiles[1][1] = (image.crop((half_w, half_h, size[0], size[1])))
        for bbox in objects: # For each sign on the whole image
            xmin = float(bbox['bbox']['xmin'])
            ymin = float(bbox['bbox']['ymin'])
            xmax = float(bbox['bbox']['xmax'])
            ymax = float(bbox['bbox']['ymax'])
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            tx = 1 if x_center >= half_w else 0
            ty = 1 if y_center >= half_h else 0
            offset_x = tx * half_w
            offset_y = ty * half_h
            # Transform coordinates (depends on the tile id)
            new_xmin = max(0, min(half_w, xmin - offset_x))
            new_xmax = max(0, min(half_w, xmax - offset_x))
            new_ymin = max(0, min(half_h, ymin - offset_y))
            new_ymax = max(0, min(half_h, ymax - offset_y))
            signs[ty][tx].append({'xmin': new_xmin,
                                  'ymin': new_ymin,
                                  'xmax': new_xmax,
                                  'ymax': new_ymax})
        # Fill labels
        for i in range(2):
            for j in range(2):
                labels[i][j] = self.create_labels(tiles[i][j], signs[i][j])
        return tiles, labels, signs
    def process_all(self):
        save_index = 0
        labels_json = {} # New labels
        with open(self.annotations_path, 'r') as j:
            annotations = json.load(j) # Old labels
        for obj in tqdm(os.listdir(self.base_path)):
            id = obj.split('.')[0] # Img id
            img = Image.open(self.base_path + obj).convert('RGB')
            # Get tiles and new labels
            if not annotations['imgs'].get(id):
                continue
            tiles, labels, signs = self.get_tiles(img, annotations['imgs'][id]['objects'])
            # Flatten tiles and labels lists and save
            tiles = [item for sublist in tiles for item in sublist]
            labels = [item for sublist in labels for item in sublist]
            signs = [item for sublist in signs for item in sublist]
            for j in range(4):
                if len(signs[j]) == 0 and random.random() > 0.3:
                    continue
                tiles[j].save(f'{self.save_path}{save_index}.jpg')
                labels_json[save_index] = labels[j].tolist()
                save_index += 1
        with open(self.labels_path, 'w') as j:
            json.dump(labels_json, j)

    # We need to convert data from TXT to JSON
    def process_labels(self):
        with open('./resources/TestIJCNN2013Download/labels.txt', 'r') as f:
            data = f.readlines()
        with open('./resources/TestIJCNN2013Download/labels.json', 'w') as j:
            new_j = defaultdict(list)
            for line in data:
                objects = line.split(';')
                id = int(objects[0][:-4].strip())
                xmin = int(objects[1].strip())
                ymin = int(objects[2].strip())
                xmax = int(objects[3].strip())
                ymax = int(objects[4].strip())
                new_j[id].append({'xmin': xmin,
                                  'ymin': ymin,
                                  'xmax': xmax,
                                  'ymax': ymax})
            json.dump(new_j, j)