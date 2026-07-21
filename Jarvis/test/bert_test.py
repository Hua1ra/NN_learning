import io
import json
import os
import dotenv
import keyboard
import re
import torch
import torch_directml
import transformers
import vlc
import wave
import yandex_music
import numpy as np
import sounddevice as sd
from openwakeword import Model
from faster_whisper import WhisperModel
from Jarvis.src.BERT import IntentTokenClassifier



print('Loading models...')
transformers.logging.set_verbosity_error()
dotenv.load_dotenv()

with open('../data/intents.json', 'r') as j:
    intents_copy = json.load(j)
with open('../data/tokens.json', 'r') as j:
    tokens_copy = json.load(j)

intents = {v : k for k, v in intents_copy.items()}
tokens = {v : k for k, v in tokens_copy.items()}

device = torch_directml.device(0)

tokenizer = transformers.AutoTokenizer.from_pretrained(os.getenv('BERT_MODEL'))
classifier = IntentTokenClassifier()
classifier.load_state_dict(torch.load(os.getenv('MODEL'), weights_only=True))
oww_model = Model(wakeword_models=[os.getenv('OWW_MODEL')], inference_framework=os.getenv('OWW_FRAMEWORK'))
rec_model = WhisperModel(os.getenv('WHISPER_MODEL'), device='cpu', compute_type='int8')

client = yandex_music.Client(os.getenv('BASE_TOKEN')).init()

instance = vlc.Instance()
player = instance.media_player_new()
print('Loaded')

def stt(audio_data):
    print('Recognizing...')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as b:
        b.setnchannels(int(os.getenv('CHANNELS')))
        b.setsampwidth(int(os.getenv('SAMP_WIDTH')))
        b.setframerate(int(os.getenv('RATE')))
        b.writeframes(audio_data.tobytes())
    buffer.seek(0)
    segments, _ = rec_model.transcribe(buffer, beam_size=int(os.getenv('BEAM_SIZE')))
    command = ''
    print('Recognized')
    for seg in segments:
        command = command + seg.text
    return re.sub(r'[^\w\s]', '', command).strip().lower()

def play_track(song_tokens):
    command = ' '.join(song_tokens)
    print(f'Playing the song "{command}"')
    results = client.search(text=command)
    if results.tracks is not None:
        track = results.tracks.results[0]
        direct_link = client.tracks_download_info(track_id=track.id, get_direct_links=True)[0]['direct_link']
        media = instance.media_new(direct_link)
        player.set_media(media)
        player.play()
    else:
        print('No results')

def play_artist(song_tokens):
    command = ' '.join(song_tokens)
    print(f'Playing the song "{command}"')
    results = client.search(text=command)
    if results.tracks is not None:
        track = results.tracks.results[0]
        direct_link = client.tracks_download_info(track_id=track.id, get_direct_links=True)[0]['direct_link']
        media = instance.media_new(direct_link)
        player.set_media(media)
        player.play()
    else:
        print('No results')



def main():
    print('Ready')
    while True:
        if keyboard.is_pressed(' '):
            break
    print('\nListening...')

    stream = sd.InputStream(samplerate=int(os.getenv('RATE')),
                            channels=int(os.getenv('CHANNELS')),
                            dtype='int16',
                            blocksize=int(os.getenv('CHUNK')))
    with stream:
        while True:
            audio_chunk, _ = stream.read(int(os.getenv('CHUNK')))
            audio_chunk = audio_chunk.flatten()
            prediction = oww_model.predict(audio_chunk)
            if prediction[os.getenv('OWW_MODEL')] > float(os.getenv('OWW_CONFIDENCE')):
                print('Hey Jarvis detected')
                if player.is_playing():
                    print('Stopping...')
                    player.stop()
                print('Recording...')
                command_chunks = []
                silent_chunks_count = 0
                while True:
                    command_chunk, _ = stream.read(int(os.getenv('CHUNK')))
                    command_chunks.append(command_chunk)
                    rms = np.sqrt(np.mean(command_chunk.astype(np.float32) ** 2))
                    if rms < int(os.getenv('SILENCE_THRESHOLD')):
                        silent_chunks_count += 1
                    else:
                        silent_chunks_count = 0
                    if silent_chunks_count > int(os.getenv('SILENCE_DURATION_CHUNK')):
                        print('Recorded')
                        break
                command = np.concat(command_chunks, axis=0)
                command = stt(command)
                oww_model.reset()
                # Test part
                with torch.no_grad():
                    embedding = tokenizer(text=command,
                                          padding='max_length',
                                          truncation=True,
                                          max_length=int(os.getenv('EMBEDDING_LENGTH')),
                                          is_split_into_words=False)
                    output = classifier(torch.tensor(embedding.input_ids).unsqueeze(0),
                                        torch.tensor(embedding.attention_mask).unsqueeze(0))
                    intent = intents[torch.argmax(output[0], dim=-1).item()]
                    tokens_list = [command.split()[embedding.word_ids()[i]] for i, arg in enumerate(torch.argmax(output[1], dim=-1).flatten()) if arg in list(map(int, os.getenv('PR_LABELS').split(','))) and embedding.word_ids()[i] is not None]
                    print(intent)
                    print(tokens_list)

                # Yandex music
                if intent == 'play_track':
                    play_track(tokens_list)

                print('Listening...')


if __name__ == '__main__':
    main()