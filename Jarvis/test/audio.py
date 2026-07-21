import dotenv
import os
import keyboard
import io
import vlc
import re
import wave
import yandex_music
import numpy as np
import sounddevice as sd
from openwakeword import Model
from faster_whisper import WhisperModel

dotenv.load_dotenv()
# Yandex
TOKEN = os.getenv('BASE_TOKEN')
# Models
OWW_MODEL = os.getenv('OWW_MODEL')
WHISPER_MODEL = os.getenv('WHISPER_MODEL')
# Sound
RATE = int(os.getenv('RATE'))
CHUNK = int(os.getenv('CHUNK'))
CHANNELS = int(os.getenv('CHANNELS'))
# Silence
SILENCE_THRESHOLD = int(os.getenv('SILENCE_THRESHOLD'))
SILENCE_DURATION_SEC = int(os.getenv('SILENCE_DURATION_SEC'))
SILENCE_DURATION_CHUNK = int(os.getenv('SILENCE_DURATION_CHUNK'))

print('Loading models...')
oww_model = Model(wakeword_models=[OWW_MODEL], inference_framework=os.getenv('INFERENCE_FRAMEWORK'))
rec_model = WhisperModel(WHISPER_MODEL, device='cpu', compute_type='int8')
client = yandex_music.Client(TOKEN).init()
instance = vlc.Instance()
player = instance.media_player_new()
print('Loaded')

def stt(audio_data):
    print('Recognizing...')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as b:
        b.setnchannels(CHANNELS)
        b.setsampwidth(int(os.getenv('SAMP_WIDTH')))
        b.setframerate(RATE)
        b.writeframes(audio_data.tobytes())
    buffer.seek(0)
    segments, _ = rec_model.transcribe(buffer, beam_size=int(os.getenv('BEAM_SIZE')))
    command = ''
    print('Recognized')
    for seg in segments:
        command = command + seg.text
    return re.sub(r'[^\w\s]', '', command)



def main():
    print('Ready')
    while True:
        if keyboard.is_pressed(' '):
            break
    print('\nListening...')

    stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16', blocksize=CHUNK)
    with stream:
        while True:
            audio_chunk, _ = stream.read(CHUNK)
            audio_chunk = audio_chunk.flatten()
            prediction = oww_model.predict(audio_chunk)
            if prediction[OWW_MODEL] > float(os.getenv('OWW_CONFIDENCE')):
                print('Hey Jarvis detected')
                if player.is_playing():
                    print('Stopping...')
                    player.stop()
                print('Recording...')
                command_chunks = []
                silent_chunks_count = 0
                while True:
                    command_chunk, _ = stream.read(CHUNK)
                    command_chunks.append(command_chunk)
                    rms = np.sqrt(np.mean(command_chunk.astype(np.float32) ** 2))
                    if rms < SILENCE_THRESHOLD:
                        silent_chunks_count += 1
                    else:
                        silent_chunks_count = 0
                    if silent_chunks_count > SILENCE_DURATION_CHUNK:
                        print('Recorded')
                        break
                command = np.concat(command_chunks, axis=0)
                command = stt(command)
                oww_model.reset()
                # Test part

                # Exit
                if 'exit' in command.lower() or 'выход' in command.lower():
                    print('Exitting...')
                    break

                if 'stop' in command.lower() or 'стоп' in command.lower():
                    print('Stopping...')
                    player.stop()
                    print('Listening...')
                    continue

                if 'continue' in command.lower() or 'продолжи' in command.lower():
                    print('Continuing...')
                    player.play()
                    print('Listening...')
                    continue

                # Yandex music
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

                print('Listening...')


if __name__ == '__main__':
    main()