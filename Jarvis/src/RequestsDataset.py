import dotenv
import json
import os
from pathlib import Path
import random
import torch
from transformers import AutoTokenizer

class RequestsDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        dotenv.load_dotenv(Path(__file__).resolve().parent.parent / '.env.client')
        self.is_train = is_train
        self.tokenizer = AutoTokenizer.from_pretrained(os.getenv('BERT_MODEL'))
        with open(os.getenv('INTENTS_PATH'), 'r') as j:
            self.intents = json.load(j)
        with open(os.getenv('TOKENS_PATH'), 'r') as j:
            self.tokens = json.load(j)
        with open(os.getenv('REQUESTS_TRAIN_PATH' if is_train else 'REQUESTS_TEST_PATH'), 'r') as j:
            self.requests = json.load(j)

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        item = self.requests[idx]
        if self.is_train and random.random() < float(os.getenv('AUG_RATE')):
            words = [w.lower() for w in item['words']]
        else:
            words = item['words']
        embedding = self.tokenizer(text=words,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=int(os.getenv('EMBEDDING_LENGTH')),
                                   return_tensors=None,
                                   is_split_into_words=True)
        word_ids = embedding.word_ids()
        labels_ids = []
        previous_word_id = None
        raw_slots = [self.tokens[t] for t in item['slots']]
        intent = self.intents[item['intent']]
        for word_id in word_ids:
            if word_id is None:
                labels_ids.append(int(os.getenv('IGNORE_INDEX')))
            elif word_id != previous_word_id:
                labels_ids.append(raw_slots[word_id])
                previous_word_id = word_id
            else:
                labels_ids.append(int(os.getenv('IGNORE_INDEX')))
        return (
            torch.tensor(embedding['input_ids'], dtype=torch.long),
            torch.tensor(embedding['attention_mask'], dtype=torch.long),
            (
                torch.tensor(intent, dtype=torch.long),
                torch.tensor(labels_ids, dtype=torch.long)
            )
        )