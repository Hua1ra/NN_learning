import torch

class Loss(torch.nn.Module):
    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp=lambda_gp

    @staticmethod
    def critic_loss(real_score, fake_score):
        # Maximize E(real_score) - E(fake_score)
        return torch.mean(fake_score) - torch.mean(real_score)

    @staticmethod
    def generator_loss(fake_score):
        # Maximize E(fake_score)
        return -torch.mean(fake_score)

    def gradient_penalty(self, critic, real_images, fake_images, flags):
        batch_size = real_images.size(0)
        device = real_images.device
        alpha = torch.rand((batch_size, 1, 1, 1), device=device)
        # Interpolate real and fake images
        interpolated_images = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
        # Get the scores
        mixed_scores = critic(interpolated_images, flags)
        # Calculate grad d(mixed_scores)/d(interpolated_images) with optimizztions
        gradients = torch.autograd.grad(outputs=mixed_scores,
                            inputs=interpolated_images,
                            grad_outputs=torch.ones(mixed_scores.size(), device=device),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0].view(batch_size, -1)
        # Get L2 Norm
        gradients = torch.sqrt(torch.sum(torch.pow(gradients, 2), dim=-1) + 1e-12)
        # Get MSE
        gradients = torch.mean(torch.pow(gradients - 1, 2))
        return gradients * self.lambda_gp