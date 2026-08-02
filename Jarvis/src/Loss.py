import dotenv
import os
from pathlib import Path
import torch



class Loss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load .env.client parameters
        dotenv.load_dotenv(Path(__file__).resolve().parent.parent / '.env.client')
        # Weights for different tokens
        self.token_weights = torch.tensor(list(map(float, os.getenv('TOKEN_WEIGHTS').split(','))), dtype=torch.float).to(device)
        self.loss_ratio = float(os.getenv('LOSS_RATIO'))

    # Calculate loss (softmax included)
    def forward(self, predicted_intent, predicted_labels, true_intent, true_labels):
        intent_loss = torch.nn.functional.cross_entropy(predicted_intent,
                                                        true_intent)
        token_loss = torch.nn.functional.cross_entropy(predicted_labels.view(-1, int(os.getenv('TOKENS_NUM'))),
                                                       true_labels.view(-1), ignore_index=int(os.getenv('IGNORE_INDEX')), weight=self.token_weights)
        return self.loss_ratio * intent_loss + token_loss