import dotenv
import os
from pathlib import Path
import sys
import torch
from transformers import AutoModel



class IntentTokenClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load .env.client parameters
        self.base_path = self.get_base_path()
        dotenv.load_dotenv(self.base_path / '.env.client')
        # Base BERT model
        self.bert = AutoModel.from_pretrained(str((self.base_path / os.getenv('BERT_MODEL')).resolve()))
        for param in self.bert.parameters():
            param.requires_grad = True
        self.hidden_size = self.bert.config.hidden_size
        # 2 layer intent classifier
        self.intent_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(float(os.getenv('DROPOUT'))),
            torch.nn.Linear(self.hidden_size, int(os.getenv('INTENTS_NUM')))
        )
        # 2 layer token classifier
        self.token_extractor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(float(os.getenv('DROPOUT'))),
            torch.nn.Linear(self.hidden_size, int(os.getenv('TOKENS_NUM')))
        )

    # Get the base path for data loading
    @staticmethod
    def get_base_path():
        if getattr(sys, 'frozen', False):
            return Path(getattr(sys, '_MEIPASS', ''))
        else:
            return Path(__file__).resolve().parent.parent

    # Forward function for training
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        intent_logits = self.intent_classifier(outputs.last_hidden_state[:, 0, :])
        token_logits = self.token_extractor(outputs.last_hidden_state)
        return intent_logits, token_logits

    # Predict function for inference (includes softmax)
    def predict(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        intent_logits = self.intent_classifier(outputs.last_hidden_state[:, 0, :])
        token_logits = self.token_extractor(outputs.last_hidden_state)
        return torch.softmax(intent_logits, dim=-1), torch.softmax(token_logits, dim=-1)