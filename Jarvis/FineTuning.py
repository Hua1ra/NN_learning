import dotenv
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import torch
import torch_directml
import tqdm
import transformers
from sklearn.metrics import precision_score, recall_score
from Jarvis.src.BERT import IntentTokenClassifier
from Jarvis.src.Loss import Loss
from Jarvis.src.RequestsDataset import RequestsDataset

def train_epoch(model,
                dataloader,
                criterion,
                optimizer,
                device,
                loss_dynamic):
    model.train()
    total_epoch_loss = 0
    for i, (input_ids, attention_mask, (intent, labels)) in tqdm.tqdm(enumerate(dataloader),
                                                                                      postfix='Training'):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        intent = intent.to(device)
        labels = labels.to(device)

        predicted_intent, predicted_labels = model(input_ids, attention_mask)
        loss = criterion(predicted_intent, predicted_labels, intent, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.item()
    loss_dynamic.append(total_epoch_loss / len(dataloader))

def validate_epoch(model,
                   dataloader,
                   device,
                   loss_dynamic):
    true_intent = []
    true_predicted_intent = []
    true_labels = []
    true_predicted_labels = []

    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_mask, (intent, labels)) in tqdm.tqdm(enumerate(dataloader),
                                                                                          postfix='Validation'):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            intent = intent.to(device)
            labels = labels.to(device)
            predicted_intent, predicted_labels = model(input_ids, attention_mask)
            predicted_intent = torch.argmax(predicted_intent, dim=-1).cpu().flatten().tolist()
            intent = intent.cpu().flatten().tolist()
            predicted_labels = torch.argmax(predicted_labels, dim=-1).cpu().flatten().tolist()
            labels = labels.cpu().flatten().tolist()

            for j in range(len(intent)):
                true_intent.append(intent[j])
                true_predicted_intent.append(predicted_intent[j])
            for j in range(len(labels)):
                if labels[j] != int(os.getenv('IGNORE_INDEX')):
                    true_labels.append(labels[j])
                    true_predicted_labels.append(predicted_labels[j])
    intent_precision = precision_score(true_intent,
                                       true_predicted_intent,
                                       average=os.getenv('AVERAGING'),
                                       zero_division=0)
    intent_recall = recall_score(true_intent,
                                 true_predicted_intent,
                                 average=os.getenv('AVERAGING'),
                                 zero_division=0)
    labels_precision = precision_score(true_labels,
                                       true_predicted_labels,
                                       average=os.getenv('AVERAGING'),
                                       zero_division=0,
                                       labels=list(map(int, os.getenv('PR_LABELS').split(','))))
    labels_recall = recall_score(true_labels,
                                 true_predicted_labels,
                                 average=os.getenv('AVERAGING'),
                                 zero_division=0,
                                 labels=list(map(int, os.getenv('PR_LABELS').split(','))))
    loss_dynamic[0].append((intent_precision, intent_recall))
    loss_dynamic[1].append((labels_precision, labels_recall))

def plot(train_loss_dynamic,
         test_loss_dynamic):
    x_list = [i + 1 for i in range(len(train_loss_dynamic))]
    test_loss_dynamic_np = np.array(test_loss_dynamic)

    plt.figure(figsize=(12, 6), dpi=200)
    sns.lineplot(x=x_list, y=train_loss_dynamic, color='red')
    plt.title('Train Loss Dynamic')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.savefig(os.getenv('TRAIN_LOSS'))
    plt.close()

    plt.figure(figsize=(12, 6), dpi=200)
    sns.lineplot(x=x_list, y=test_loss_dynamic_np[0, :, 0], color='red', label='Intent')
    sns.lineplot(x=x_list, y=test_loss_dynamic_np[1, :, 0], color='blue', label='Tokens')
    plt.title('Test Precision Dynamic')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.savefig(os.getenv('TEST_P'))
    plt.close()

    plt.figure(figsize=(12, 6), dpi=200)
    sns.lineplot(x=x_list, y=test_loss_dynamic_np[0, :, 1], color='red', label='Intent')
    sns.lineplot(x=x_list, y=test_loss_dynamic_np[1, :, 1], color='blue', label='Tokens')
    plt.title('Test Recall Dynamic')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.savefig(os.getenv('TEST_R'))
    plt.close()

def save_model(model,
               optimizer,
               device,
               train_loss_dynamic,
               test_loss_dynamic,
               last_epoch):
    checkpoint = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'device' : device,
        'train_loss_dynamic' : train_loss_dynamic,
        'test_loss_dynamic' : test_loss_dynamic,
        'last_epoch' : last_epoch
    }
    torch.save(checkpoint, f'./models/model{last_epoch + 1}.pth')



def main(checkpoint_path=None):
    dotenv.load_dotenv(Path(__file__).resolve().parent / '.env.client')
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch_directml.is_available():
        device = torch_directml.device(0)
    epochs = int(os.getenv('EPOCHS'))
    model = IntentTokenClassifier()
    model = model.to(device)
    train_dataset = RequestsDataset(is_train=True)
    test_dataset = RequestsDataset(is_train=False)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=int(os.getenv('TRAIN_BATCH_SIZE')),
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=int(os.getenv('TEST_BATCH_SIZE')))
    criterion = Loss(device)
    optimizer = torch.optim.AdamW([
        { 'params' : model.bert.parameters(), 'lr' : float(os.getenv('BERT_LR')) },
        { 'params' : model.intent_classifier.parameters(), 'lr' : float(os.getenv('HEAD_LR')) } ,
        { 'params' : model.token_extractor.parameters(), 'lr' : float(os.getenv('HEAD_LR')) }
    ], weight_decay=float(os.getenv('WEIGHT_DECAY')))
    train_loss_dynamic = []
    test_loss_dynamic = [[], []]
    start_epoch = 0

    if checkpoint_path is not None:
        checkpoint = torch.load('./models/' + checkpoint_path + '.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        device = checkpoint['device']
        train_loss_dynamic = checkpoint['train_loss_dynamic']
        test_loss_dynamic = checkpoint['test_loss_dynamic']
        start_epoch = checkpoint['last_epoch'] + 1
        del checkpoint

    for epoch in range(start_epoch, epochs):
        train_epoch(model,
                    train_dataloader,
                    criterion,
                    optimizer,
                    device,
                    train_loss_dynamic)
        print(f'Epoch: {epoch + 1}/{epochs}')
        print(f'Loss: {train_loss_dynamic[-1]}')
        validate_epoch(model,
                       test_dataloader,
                       device,
                       test_loss_dynamic)
        print(f'Intent precision: {test_loss_dynamic[0][-1][0]}')
        print(f'Intent recall: {test_loss_dynamic[0][-1][1]}')
        print(f'Token precision: {test_loss_dynamic[1][-1][0]}')
        print(f'Token recall: {test_loss_dynamic[1][-1][1]}')
        print()
        plot(train_loss_dynamic,
             test_loss_dynamic)
        if epoch % int(os.getenv('SAVE_EACH')) == (int(os.getenv('SAVE_EACH')) - 1):
            save_model(model,
                       optimizer,
                       device,
                       train_loss_dynamic,
                       test_loss_dynamic,
                       epoch)



if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    main(checkpoint_path=None)