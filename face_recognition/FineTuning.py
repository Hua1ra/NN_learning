import torch
import torch_directml
from src.ArcFace import ArcFace
from src.CasiaWebface import CasiaWebface
from src.Extractor import Extractor
from src.Functions import fine_tune, validate
from src.ValDataset import ValDataset

# Device
device = ''
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda')
elif torch_directml.is_available():
    device = torch_directml.device(0)
    print(torch_directml.device_name(0))
else:
    device = torch.device('cpu')
    print('cpu')

# Data
dataset = CasiaWebface(rec_path='./data/casia-webface/train.rec',
                       idx_path='./data/casia-webface/train.idx',
                       transformer=None,
                       mode='transformer')
dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=25,
                                         shuffle=True,
                                         pin_memory=True)
val_dataset = ValDataset(root='./data/eval/lfw.bin')
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=25,
                                             shuffle=False,
                                             pin_memory=True)
# Model
extractor = Extractor(model_path='./models/feature_extractor.pth',
                      embed_size=512)
extractor.eval()
extractor = extractor.to(device)
for param in extractor.extractor.parameters():
    param.requires_grad = False
for param in extractor.extractor[6].parameters():
    param.requires_grad = True
for param in extractor.extractor[7].parameters():
    param.requires_grad = True
for param in extractor.fc.parameters():
    param.requires_grad = True
for param in extractor.batch_norm.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(extractor.parameters(),
                            lr=1e-3,
                            momentum=0.9,
                            weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=50,
                                                       eta_min=1e-5)
criterion = ArcFace(in_features=512,
                    num_classes=10571,
                    m=0.5,
                    s=64.0)
criterion = criterion.to(device)

# Other
start_epoch = 0
last_epoch = 50
epoch_loss_dynamic = []

# Checkpoint
chechpoint_path: str | None = 'checkpoint2'
if chechpoint_path is not None:
    checkpoint = torch.load('./checkpoints/' + chechpoint_path + '.pth')
    extractor.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion.load_state_dict(checkpoint['criterion'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch_loss_dynamic = checkpoint['epoch_loss_dynamic']

# Training
# fine_tune(model=extractor,
#           dataloader=dataloader,
#           optimizer=optimizer,
#           criterion=criterion,
#           lr_scheduler=scheduler,
#           device=device,
#           start_epoch=start_epoch,
#           last_epoch=last_epoch,
#           val_dataloader=val_dataloader,
#           epoch_loss_dynamic=epoch_loss_dynamic)

accuracy = validate(model=extractor,
         val_dataloader=val_dataloader,
         device=device,
         threshold=0.87)
print(f'Accuracy: {accuracy}%')