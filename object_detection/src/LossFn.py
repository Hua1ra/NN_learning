import torch

# Custom Loss Class
class LossFN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    def forward(self, predictions, labels):
        # Reshape predictions and labels to ensure
        predictions = predictions.view(-1, 20 * 20, 5)
        labels = labels.view(-1, 20 * 20, 5)
        # Get a mask to calculate loss only for p != 0
        labels_coords_mask = (labels[:, :, 0] != 0).bool().unsqueeze(-1).expand_as(predictions[:, :, 1:])
        # BCE loss with logits for p
        probs = torch.sigmoid(predictions[:, :, 0])
        p_loss = -(labels[:, :, 0] * torch.log(probs + 1e-7) + (1 - labels[:, :, 0]) * torch.log((1 - probs + 1e-7))).mean()
        # Smooth L1 loss for (x, y, w, h)
        if labels_coords_mask.sum() == 0:
            xywh_loss = torch.tensor(0.0).to(self.device)
        else:
            xywh_loss = torch.nn.functional.smooth_l1_loss(predictions[:, :, 1:][labels_coords_mask],
                                                           labels[:, :, 1:][labels_coords_mask])
        # Weighted sum of losses
        return p_loss + 5 * xywh_loss