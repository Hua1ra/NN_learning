import torch_directml
from src.SignDetectionModel import SignDetectionModel
from src.TrainValidateFn import *
from src.TT100K import TT100KDataset

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

# Object declaration for validation
last_checkpoint = 'model62' # Need to be a path:str
dataset = TT100KDataset()
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=10,
                                          pin_memory=True)
model = SignDetectionModel().to(device)
# Load the existing model
if last_checkpoint is not None:
    checkpoint = torch.load(f'./resources/models/{last_checkpoint}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

# Final validation
results = final_validation(model, device, data_loader, min_confidence=0.15)

# Get the metrics for the test dataset (metrics can be worse due to format diferences)
precision = results['tp'] / (results['tp'] + results['fp'])
recall = results['tp'] / (results['tp'] + results['fn'])
print(f'TP: {results["tp"]}\tFP: {results["fp"]}\tFN: {results["fn"]}\tTN: {results["tn"]}')
print(f'Precision: {precision * 100}')
print(f'Recall: {recall * 100}')
print(f'Average IoU: {sum(results["ious"]) / len(results["ious"]) * 100}')

