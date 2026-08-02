import dotenv
import os
from pathlib import Path
import sys
import torch
import torch_directml # type: ignore
from Jarvis.src.BERT import IntentTokenClassifier



# Get the base path
def get_base_path():
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', ''))
    else:
        return Path(__file__).resolve().parent.parent

# Get the model from checkpoint
def get_model_from_checkpoint(start_path, end_path):
    checkpoint = torch.load(start_path)
    classifier_model = IntentTokenClassifier()
    classifier_model.load_state_dict(checkpoint['model'])
    classifier_model = classifier_model.to('cpu')
    classifier_model.eval()
    torch.save(classifier_model.state_dict(), end_path)



# Save only the model
def main():
    dotenv.load_dotenv(Path(__file__).resolve().parent / '.env.client')
    basic_path = get_base_path()
    start_path = (basic_path / os.getenv('CHECKPOINT')).resolve()
    end_path = (basic_path / os.getenv('MODEL')).resolve()
    get_model_from_checkpoint(start_path, end_path)



if __name__ == '__main__':
    main()