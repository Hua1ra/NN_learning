import dotenv
import io
import json
import logging
import numpy as np
import os
from pathlib import Path
import re
import sys
import torch
import transformers
import wave
from openwakeword import Model
from faster_whisper import WhisperModel
from src.BERT import IntentTokenClassifier #type: ignore



class MicController:
    def __init__(self):
        try:
            transformers.logging.set_verbosity_error()
            # Load .env.client parameters
            self.basic_path = self.get_base_path()
            dotenv.load_dotenv(self.basic_path / '.env.client')
            # Load intent-to-id and id-to-intent, token-to-id and id-to-token
            with open((self.basic_path / os.getenv('INTENTS_PATH')).resolve(), 'r') as j:
                self.intent_to_id = json.load(j)
            with open((self.basic_path / os.getenv('TOKENS_PATH')).resolve(), 'r') as j:
                self.token_to_id = json.load(j)
            self.id_to_intent = {v: k for k, v in self.intent_to_id.items()}
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            # Load models and tokenizer
            self.pr_labels = list(map(int, os.getenv('PR_LABELS').split(',')))
            self.oww_model = Model(wakeword_models=[os.getenv('OWW_MODEL')], inference_framework=os.getenv('OWW_FRAMEWORK'))
            self.rec_model = WhisperModel(str((self.basic_path / os.getenv('WHISPER_MODEL')).resolve()), device='cpu', compute_type='int8')
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(str((self.basic_path / os.getenv('BERT_MODEL')).resolve()))
            self.classifier_model = IntentTokenClassifier()
            self.classifier_model.load_state_dict(torch.load((self.basic_path / os.getenv('MODEL')).resolve(), weights_only=True))
            # Service variables
            self.oww_recognized = False
            self.command_chunks = []
            self.silent_chunks = 0
            self.command_chunks_number = 0
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    # Get the base path
    @staticmethod
    def get_base_path():
        if getattr(sys, 'frozen', False):
            return Path(getattr(sys, '_MEIPASS', ''))
        else:
            return Path(__file__).resolve().parent.parent

    # Speech to text for a chunk
    def stt(self, audio_data):
        try:
            # Write audio chunk to a Bytes object
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as b:
                b.setnchannels(int(os.getenv('CHANNELS')))
                b.setsampwidth(int(os.getenv('SAMP_WIDTH')))
                b.setframerate(int(os.getenv('RATE')))
                b.writeframes(audio_data.tobytes())
            buffer.seek(0)
            # Get the text command
            segments, _ = self.rec_model.transcribe(buffer, beam_size=int(os.getenv('BEAM_SIZE')))
            command = ''
            for seg in segments:
                command = command + seg.text
            # Leave only letters and numbers
            return re.sub(r'[^a-zA-Zа-яА-Яё0-9\s]', ' ', command)
        except Exception as e:
            logging.error(e)

    # Process audio chunk
    def listen(self, audio_chunk):
        try:
            # Has the dim=2 (channels, rate)
            audio_chunk = audio_chunk.flatten()
            # If wake word has not been recognied
            if not self.oww_recognized:
                # Try to recognize
                oww_prediction = self.oww_model.predict(audio_chunk)
                if oww_prediction[os.getenv('OWW_MODEL')] >= float(os.getenv('OWW_CONFIDENCE')):
                    logging.info('Jarvis recognized')
                    self.oww_recognized = True
                    return True
            # If wake word has been recognized
            if self.oww_recognized:
                # Collect comman chunks
                self.command_chunks_number += 1
                self.command_chunks.append(audio_chunk)
                rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
                # Check if it is a silence chunk and count them
                if rms < int(os.getenv('SILENCE_THRESHOLD')):
                    self.silent_chunks += 1
                else:
                    self.silent_chunks = 0
                # If the command is complete
                if self.silent_chunks > int(os.getenv('SILENCE_DURATION_CHUNK')) or self.command_chunks_number > int(os.getenv('MAX_CHUNKS')):
                    # Get the complete command
                    command = np.concat(self.command_chunks)
                    command = self.stt(command)
                    logging.info(f'Command: {command}')
                    # Process the command
                    with torch.no_grad():
                        embedding = self.tokenizer(text=command,
                                                   padding='max_length',
                                                   truncation=True,
                                                   max_length=int(os.getenv('EMBEDDING_LENGTH')),
                                                   is_split_into_words=False)
                        output = self.classifier_model.predict(torch.tensor(embedding.input_ids).unsqueeze(0),
                                                               torch.tensor(embedding.attention_mask).unsqueeze(0))
                        # Get the index of the potentional command
                        index = torch.argmax(output[0], dim=-1).item()
                        tokens_list = []
                        # If the model is sure
                        if output[0][0][index] > float(os.getenv('CONFIDENCE_THRESHOLD')):
                            # Get the intent
                            intent = self.id_to_intent[index]
                            # Get the tokens
                            command = command.split()
                            prev_id = -1
                            # For each predicted label for each word token
                            for i, arg in enumerate(torch.argmax(output[1], dim=-1).flatten()):
                                # If the label is a valid token and not a part of a previous word,
                                # get the whole word into a token list
                                if arg in self.pr_labels and embedding.word_ids()[i] is not None and \
                                        embedding.word_ids()[i] != prev_id:
                                    tokens_list.append(command[embedding.word_ids()[i]])
                                prev_id = embedding.word_ids()[i]
                        else:
                            # If the model is not sure
                            intent = 'other'
                        # Reset all the service variables
                        self.command_chunks = []
                        self.oww_model.reset()
                        self.silent_chunks = 0
                        self.command_chunks_number = 0
                        self.oww_recognized = False
                        return intent, tokens_list
            return False
        except Exception as e:
            logging.error(e)