import torch_directml
from src.LossFn import LossFN
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

# Create objects of necessary classes
last_checkpoint = 'model62' # Need to be a path:str if we want to load the existing model
last_epoch = 0
num_epochs = 100
dataset = TT100KDataset()
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          pin_memory=True)
model = SignDetectionModel().to(device)
criterion = LossFN(device)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=1e-4,
                            momentum=0.9,
                            weight_decay=0.01)
scheduler_1 = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                start_factor=0.1,
                                                end_factor=1.0,
                                                total_iters=5)
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=10,
                                                                 T_mult=2,
                                                                 eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                  schedulers=[scheduler_1, scheduler_2],
                                                  milestones=[5])
# Load the existing model if needed
if last_checkpoint is not None:
    checkpoint = torch.load(f'./resources/models/{last_checkpoint}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    last_epoch = checkpoint['epoch']
    num_epochs = checkpoint['num_epochs']

# Get the training done
train(data_loader, model, optimizer, criterion, scheduler, num_epochs, last_epoch)