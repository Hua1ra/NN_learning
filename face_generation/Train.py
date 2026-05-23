import torch
import torch_directml
from src.CelebA import CelebA
from src.Critic import Critic
from src.Functions import train
from src.Generator import Generator
from src.Loss import Loss
# Get the device
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda')
elif torch_directml.is_available():
    device = torch_directml.device(0)
    print(torch_directml.device_name(0))
else:
    print('cpu')
# Create objects
dataset = CelebA()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
generator = Generator().to(device)
generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
generator_lr_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=5, gamma=0.5)
critic = Critic().to(device)
critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=5e-5)
critic_lr_scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=5, gamma=0.5)
criterion = Loss()
start_epoch = 1
epoch_num = 101
n_critic = 5
generator_losses = []
critic_losses = []
# Load checkpoint
checkpoint_name: str | None = None
if checkpoint_name is not None:
    checkpoint = torch.load('./models/' + checkpoint_name + '.pth')
    generator.load_state_dict(checkpoint['generator'])
    critic.load_state_dict(checkpoint['critic'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    generator_lr_scheduler.load_state_dict(checkpoint['generator_lr_scheduler'])
    critic_lr_scheduler.load_state_dict(checkpoint['critic_lr_scheduler'])
    start_epoch = checkpoint['start_epoch']
    epoch_num = checkpoint['epoch_num']
    n_critic  = checkpoint['n_critic']
    generator_losses = checkpoint['generator_losses']
    critic_losses = checkpoint['critic_losses']
# Train
train(generator=generator,
      critic=critic,
      device=device,
      data_loader=data_loader,
      generator_optimizer=generator_optimizer,
      critic_optimizer=critic_optimizer,
      generator_lr_scheduler=generator_lr_scheduler,
      critic_lr_scheduler=critic_lr_scheduler,
      criterion=criterion,
      start_epoch=start_epoch,
      epoch_num=epoch_num,
      n_critic=n_critic,
      generator_losses=generator_losses,
      critic_losses=critic_losses)