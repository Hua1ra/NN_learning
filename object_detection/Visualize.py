import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch_directml
from src.SignDetectionModel import SignDetectionModel
from src.TrainValidateFn import *
from src.CCTSDB import CCTSDB # type: ignore
from src.TT100K import TT100KDataset # type: ignore

# Device declaration
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('cuda')
elif torch_directml.is_available():
    device = torch_directml.device(0)
    print(torch_directml.device_name(0))
else:
    device = torch.device('cpu')
    print('cpu')

last_checkpoint = 'model62' # Need to be a path:str
dataset = TT100KDataset()
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=10,
                                          pin_memory=True)
model = SignDetectionModel().to(device)
# Load the existing model
if last_checkpoint is not None:
    checkpoint = torch.load(f'./models/{last_checkpoint}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

# Some of the examples
dataset = TT100KDataset()
with open('./resources/tt100k_2021/labels_all.json', 'r') as j:
    labels_json = json.load(j)
for img_id in range(13, len(dataset), 1000):
    # Open the image
    if isinstance(dataset, TT100KDataset):
        img_path = f'./resources/tt100k_2021/train/{img_id}.jpg'
    else:
        img_path = f'./resources/CCTSDB/test/{str(img_id)}.png'
    img = Image.open(img_path)
    # Get the detected image with confidence heatmap
    original_detected = get_original_detected(img_path, labels_json)
    img_detected, predictions_confidence = get_detected(model, device, img_path, threshold=0.15, iou_threshold=0.1)
    # Visualize the result
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(original_detected)
    axs[0].axis('off')

    sns.heatmap(np.round(predictions_confidence, 1), vmin=0, vmax=1, annot=True, ax=axs[1])
    axs[1].axis('on')

    axs[2].imshow(img_detected)
    axs[2].axis('off')

    plt.show()