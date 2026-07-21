import dotenv
import os
import sounddevice as sd
from Jarvis.src.Brain import Brain

dotenv.load_dotenv()

brain = Brain(token=os.getenv('YM_TOKEN'), device=os.getenv('YM_DEVICE'))
# stream = sd.InputStream(samplerate=int(os.getenv('RATE')),
#                                 channels=int(os.getenv('CHANNELS')),
#                                 dtype='int16',
#                                 blocksize=int(os.getenv('chunk')))
# print('Listening')
# with stream:
#     while True:
#         audio_chunk, _ = stream.read(int(os.getenv('CHUNK')))
#         result = brain.listen(audio_chunk)
brain.player()