import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from src.Processor import DataProcessor

# Process all images_old
processor = DataProcessor()
processor.process_labels()
processor.process_all()

# Let's check if we have correct labels
# Load nesseccary data
with open('./resources/tt100k_2021/labels_all.json', 'r') as j:
    labels_json_vis = json.load(j)
# Visualize the data
id_to_check = 13
img_vis = Image.open(f'./resources/tt100k_2021/train/{id_to_check}.jpg') # Image id=13
labels_vis = labels_json_vis[str(id_to_check)] # Labels id=13
labels_vis = torch.tensor(labels_vis).view(20, 20, 5)
# Image
plt.axis('off')
plt.imshow(img_vis)
plt.show()
# Confidence
sns.heatmap(labels_vis[:, :, 0])
plt.show()
# Borders
draw = ImageDraw.Draw(img_vis)
for i in range(20):
    for j in range(20):
        if labels_vis[i][j][0] == 0:
            continue
        else:
            size_vis = img_vis.size
            x_center_vis = (j + labels_vis[i][j][1]) * (size_vis[0] // 20)
            y_center_vis = (i + labels_vis[i][j][2]) * (size_vis[1] // 20)
            x_min_vis = x_center_vis - labels_vis[i][j][3] * size_vis[0] // 2
            x_max_vis = x_center_vis + labels_vis[i][j][3] * size_vis[0] // 2
            y_min_vis = y_center_vis - labels_vis[i][j][4] * size_vis[1] // 2
            y_max_vis = y_center_vis + labels_vis[i][j][4] * size_vis[1] // 2
            draw.line((x_min_vis, y_min_vis, x_min_vis, y_max_vis), 'red', width=2)
            draw.line((x_min_vis, y_min_vis, x_max_vis, y_min_vis), 'red', width=2)
            draw.line((x_max_vis, y_min_vis, x_max_vis, y_max_vis), 'red', width=2)
            draw.line((x_min_vis, y_max_vis, x_max_vis, y_max_vis), 'red', width=2)
plt.axis('off')
plt.imshow(img_vis)
plt.show()