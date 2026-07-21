import dotenv
import os
import torch
import torch_directml # type: ignore
from Jarvis.src.BERT import IntentTokenClassifier

def get_model_from_checkpoint(start_path, end_path):
    checkpoint = torch.load(start_path)
    classifier_model = IntentTokenClassifier()
    classifier_model.load_state_dict(checkpoint['model'])
    classifier_model = classifier_model.to('cpu')
    classifier_model.eval()
    torch.save(classifier_model.state_dict(), end_path)



def main():
    dotenv.load_dotenv()
    start_path = os.getenv('CHECKPOINT')
    end_path = os.getenv('MODEL')
    get_model_from_checkpoint(start_path, end_path)



if __name__ == '__main__':
    main()