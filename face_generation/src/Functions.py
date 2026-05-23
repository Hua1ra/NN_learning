import matplotlib.pyplot as plt
import torch
import torchvision
from typing import Any
from tqdm import tqdm

# Plot losses history
def plot_losses(generator_losses, critic_losses, save_path='./generated_images/loss_history.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(generator_losses, alpha=0.8, color='red', label='Generator Loss')
    plt.plot(critic_losses, alpha=0.8, color='blue', label='Critic Loss')
    plt.xlabel('Iteration (*500)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig(save_path)
    plt.close()

def train(generator, critic, device, data_loader,
          generator_optimizer, critic_optimizer, generator_lr_scheduler, critic_lr_scheduler,
          criterion, start_epoch, epoch_num, n_critic, generator_losses, critic_losses):
    generator.train()
    critic.train()
    generator_10k_loss = 0
    critic_10k_loss = 0
    critic_iteration_count = 0
    generator_iteration_count = 0
    for epoch in range(start_epoch, epoch_num):
        running_critic_loss = 0
        running_generator_loss = 0
        tqdm_loader: Any = tqdm(data_loader)
        num_generator_updates = 0
        for i, (real_images, flags) in enumerate(tqdm_loader):
            critic_iteration_count += 1
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            flags = flags.to(device)
            # Get critic score on real images
            critic_optimizer.zero_grad()
            real_score = critic(real_images, flags)
            # Get generated images
            z = torch.randn(batch_size, 100, device=device)
            fake_images = generator(z, flags)
            # Get critic score on generated images
            fake_score = critic(fake_images.detach(), flags)
            # Get critic loss
            critic_loss_base = criterion.critic_loss(real_score, fake_score)
            critic_grad_penalty = criterion.gradient_penalty(critic, real_images, fake_images.detach(), flags)
            critic_loss = critic_loss_base + critic_grad_penalty
            running_critic_loss += critic_loss.item()
            critic_10k_loss += critic_loss.item()

            critic_loss.backward()
            critic_optimizer.step()

            if i % n_critic == 0:
                num_generator_updates += 1
                generator_iteration_count += 1
                generator_optimizer.zero_grad()
                # Get generator loss
                fake_score_generator = critic(fake_images, flags)
                generator_loss = criterion.generator_loss(fake_score_generator)
                running_generator_loss += generator_loss.item()
                generator_10k_loss += generator_loss.item()
                generator_loss.backward()
                generator_optimizer.step()
                if i % (20 * n_critic) == 0:
                    tqdm_loader.set_postfix(critic_loss=critic_loss.item(), generator_loss=generator_loss.item())

            if critic_iteration_count % 500 == 0 and critic_iteration_count != 0:
                generator_losses.append(generator_10k_loss / generator_iteration_count)
                critic_losses.append(critic_10k_loss / critic_iteration_count)
                plot_losses(generator_losses, critic_losses)
                generator_10k_loss = 0
                critic_10k_loss = 0
                generator_iteration_count = 0
                critic_iteration_count = 0
        generator_lr_scheduler.step()
        critic_lr_scheduler.step()
        # Get info on every batch
        print(f'Epoch: {epoch}')
        print(f'Running Critic Loss: {running_critic_loss / len(data_loader)}')
        print(f'Running Generator Loss: {running_generator_loss / num_generator_updates}')
        print()
        # Save model on every batch
        chechpoint = {
            'generator': generator.state_dict(),
            'critic': critic.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'critic_optimizer': critic_optimizer.state_dict(),
            'generator_lr_scheduler': generator_lr_scheduler.state_dict(),
            'critic_lr_scheduler': critic_lr_scheduler.state_dict(),
            'start_epoch': epoch + 1,
            'epoch_num': epoch_num,
            'n_critic': n_critic,
            'generator_losses': generator_losses,
            'critic_losses': critic_losses
        }
        torch.save(chechpoint, f'./models/model{epoch}.pth')
        # Save generated images on every epoch
        generator.eval()
        with torch.no_grad():
            z = torch.randn(16, 100, device=device)
            flags = torch.randint(0, 2, (16, 7), device=device).float()
            fake_images = generator(z, flags)
            torchvision.utils.save_image(fake_images.cpu(),
                                         f'./generated_images/epoch{epoch}.png',
                                         nrow=4,
                                         normalize=True)
        generator.train()