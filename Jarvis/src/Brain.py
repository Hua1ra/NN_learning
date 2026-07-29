import json
import dotenv
import keyboard
import logging
import os
from pathlib import Path
import sys
import random
import time
import yandex_music
import vlc
from src.AudioController import AudioController #type: ignore
from src.MicController import MicController #type: ignore
from src.YMController import YMController #type: ignore



class Brain:
    def __init__(self, token, device):
        dotenv.load_dotenv(Path(__file__).resolve().parent.parent / '.env.client')
        try:
            self.token = token
            self.device = device
            self.client = yandex_music.Client(token)
            self.ymcontroller = YMController(self.client, self.device)
            self.device = device

            self.audiocontroller = AudioController()
            self.event_manager = self.audiocontroller.player.event_manager()

            self.miccontroller = MicController()

            self.basic_path = self.get_base_path()
            with open((self.basic_path / os.getenv('RADIO_PATH')).resolve(), 'r') as j:
                self.radio = json.load(j)
            self.is_track_ended = False
            self.was_playing = False
            self.event_manager.event_attach(getattr(vlc.EventType, 'MediaPlayerEndReached'), self.track_ended)
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    @staticmethod
    def get_base_path():
        if getattr(sys, 'frozen', False):
            return Path(getattr(sys, '_MEIPASS', ''))
        else:
            return Path(__file__).resolve().parent.parent

    def processor(self, intent, tokens=None):
        has_params = (intent in ('play_track', 'play_artist', 'play_album', 'play_playlist', 'set_volume'))
        try:
            method = getattr(self, intent)
            if has_params:
                method(tokens)
            else:
                method()
        except Exception as e:
            logging.error(e)

    def play_track(self, tokens):
        query = ' '.join(tokens)
        logging.info(f'Searching track {query}')
        queue_copy = self.ymcontroller.current_queue.copy()
        self.ymcontroller.play_search_track(query)
        if self.ymcontroller.current_queue != queue_copy:
            self.track_next()

    def play_artist(self, tokens):
        query = ' '.join(tokens)
        logging.info(f'Searching artist {query}')
        queue_copy = self.ymcontroller.current_queue.copy()
        self.ymcontroller.play_search_artist(query)
        if self.ymcontroller.current_queue != queue_copy:
            self.track_next()

    def play_album(self, tokens):
        query = ' '.join(tokens)
        logging.info(f'Searching album {query}')
        queue_copy = self.ymcontroller.current_queue.copy()
        self.ymcontroller.play_search_album(query)
        if self.ymcontroller.current_queue != queue_copy:
            self.track_next()

    def play_playlist(self, tokens):
        query = ' '.join(tokens)
        logging.info(f'Searching playlist {query}')
        queue_copy = self.ymcontroller.current_queue.copy()
        self.ymcontroller.play_search_playlist(query)
        if self.ymcontroller.current_queue != queue_copy:
            self.track_next()

    def play_wave(self):
        logging.info('Playing wave')
        self.ymcontroller.play_wave()
        self.track_next()

    def play_favourite(self):
        logging.info('Playing favourite')
        self.ymcontroller.play_favorite_batch()
        if len(self.ymcontroller.current_queue) > 0:
            self.track_next()

    def play_random_radio(self):
        radio = self.radio[str(random.randint(0, int(os.getenv('RADIO_NUMBER')) - 1))]
        logging.info(f'Playing radio {radio}')
        self.ymcontroller.set_station(radio)
        self.track_next()

    def track_next(self):
        self.is_track_ended = False
        track = self.ymcontroller.play_next()
        logging.info(f'Playing track: {track.title}')
        track = self.ymcontroller.get_link(track.id)
        self.audiocontroller.play(track)

        logging.info(f'Queue: {[item.title for item in self.ymcontroller.current_queue]}')
        logging.info(f'Length: {len(self.ymcontroller.current_queue)}')
        logging.info(f'Index: {self.ymcontroller.batch_index}')

    def track_prev(self):
        self.is_track_ended = False
        track = self.ymcontroller.play_prev()
        logging.info(f'Playing track: {track.title}')
        track = self.ymcontroller.get_link(track.id)
        self.audiocontroller.play(track)

        logging.info(f'Queue: {[item.title for item in self.ymcontroller.current_queue]}')
        logging.info(f'Length: {len(self.ymcontroller.current_queue)}')
        logging.info(f'Index: {self.ymcontroller.batch_index}')

    def track_pause(self):
        if self.audiocontroller.player.is_playing():
            logging.info('Pause')
            self.audiocontroller.pause()

    def track_resume(self):
        if not self.audiocontroller.player.is_playing():
            logging.info('Resume')
            self.audiocontroller.pause()
            self.was_playing = False

    def volume_up(self):
        self.audiocontroller.volume_up()

    def volume_down(self):
        self.audiocontroller.volume_down()

    def set_volume(self, tokens):
        self.audiocontroller.set_volume(int(' '.join(tokens)))

    def like(self):
        logging.info('Like')
        self.ymcontroller.like(self.ymcontroller.current_track.id)

    def dislike(self):
        logging.info('Dislike')
        self.ymcontroller.dislike(self.ymcontroller.current_track.id)

    def ask_current_track(self):
        logging.info('Asking current track')
        if self.ymcontroller.current_track is not None and self.audiocontroller.player.is_playing():
            return {
                'id' : self.ymcontroller.current_track.id,
                'title' : self.ymcontroller.current_track.title,
                'artists' : ', '.join([artist.name for artist in self.ymcontroller.current_track.artists]),
                'albums' : ', '.join([album.title for album in self.ymcontroller.current_track.albums])
            }
        else:
            return None

    @staticmethod
    def other():
        logging.info('Undefined command')
        return 'undefined command'

    def exit(self):
        self.audiocontroller.exit()

    def track_ended(self, _):
        logging.info('Track ended')
        self.is_track_ended = True

    def listen(self, audio_chunk):
        # Process audio chunk
        try:
            result = self.miccontroller.listen(audio_chunk)
            if result is True:
                if self.audiocontroller.player.is_playing():
                    self.track_pause()
                    self.was_playing = True
                return 'Recognized', None
            elif result is False:
                return 'None', self.is_track_ended
            else:
                intent, tokens = result
                logging.info(intent + ' ' + ' '.join(tokens))
                if intent == 'exit':
                    return intent, 'exit'
                elif intent == 'ask_current_track':
                    if self.was_playing:
                        self.track_resume()
                        self.was_playing = False
                    return intent, self.ask_current_track()
                elif intent == 'other':
                    if self.was_playing:
                        self.track_resume()
                        self.was_playing = False
                    return intent, self.other()
                else:
                    self.processor(intent, tokens)
                    if intent != 'track_pause' and self.was_playing:
                        self.track_resume()
                        self.was_playing = False
                    return intent, tokens
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    def player(self):
        # Function for demonstration YM Controller abilities
        try:
            print('+---------------------------------+')
            print('| Controls                        |')
            print('+---------------------------------+')
            print('| w   - play wave                 |')
            print('| n   - next trac                 |')
            print('| p   - previous track            |')
            print('| " " - pause                     |')
            print('| s   - search track              |')
            print('| f   - favourite                 |')
            print('| kUp - volume up                 |')
            print('| kDn - volume down               |')
            print('| q   - exit                      |')
            print('+---------------------------------+')
            print()
            print('Ready')
            print()
            while True:
                if keyboard.is_pressed('w'):
                    self.processor('play_wave')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                if keyboard.is_pressed('n'):
                    self.processor('track_next')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                if keyboard.is_pressed('p'):
                    self.processor('track_prev')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                if keyboard.is_pressed(' '):
                    self.processor('track_pause')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                if keyboard.is_pressed('s'):
                    q = 'break it off rihanna'
                    self.processor('play_track', q.split())
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                if keyboard.is_pressed('f'):
                    self.processor('play_favourite')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                if keyboard.is_pressed('r'):
                    self.processor('play_radio')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                if keyboard.is_pressed('up'):
                    self.processor('volume_up')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                if keyboard.is_pressed('down'):
                    self.processor('volume_down')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                if keyboard.is_pressed('q'):
                    self.processor('exit')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                    exit(0)
                if self.is_track_ended:
                    self.processor('track_next')
                    time.sleep(float(os.getenv('SMALL_DELAY')))
                time.sleep(float(os.getenv('TINY_DELAY')))
        except Exception as e:
            logging.error(e)
            sys.exit(1)