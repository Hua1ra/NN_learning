import dotenv
import logging
import os
from pathlib import Path
import sys
import vlc



class AudioController:
    def __init__(self):
        dotenv.load_dotenv(Path(__file__).resolve().parent.parent / '.env.client')
        try:
            self.instance = vlc.Instance()
            self.player = self.instance.media_player_new()
            self.volume = self.player.audio_get_volume()
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    def play(self, direct_link):
        try:
            media = self.instance.media_new(direct_link)
            self.player.set_media(media)
            self.player.play()
        except Exception as e:
            logging.error(e)

    def pause(self):
        try:
            self.player.pause()
        except Exception as e:
            logging.error(e)

    def volume_up(self):
        try:
            self.volume = min(self.player.audio_get_volume() + int(os.getenv('VOLUME_STEP')), int(os.getenv('VOLUME_MAX')))
            self.player.audio_set_volume(self.volume)
            logging.info(f'Volume up to {self.volume}')
        except Exception as e:
            logging.error(e)

    def volume_down(self):
        try:
            self.volume = max(self.player.audio_get_volume() - int(os.getenv('VOLUME_STEP')), int(os.getenv('VOLUME_MIN')))
            self.player.audio_set_volume(self.volume)
            logging.info(f'Volume down to {self.volume}')
        except Exception as e:
            logging.error(e)

    def set_volume(self, tokens):
        try:
            self.volume = min(max(int(tokens), int(os.getenv('VOLUME_MIN'))), int(os.getenv('VOLUME_MAX')))
            self.player.audio_set_volume(self.volume)
            logging.info(f'Volume set to {self.volume}')
        except Exception as e:
            logging.error(e)

    def exit(self):
        try:
            logging.info('Exit')
            if self.player.is_playing():
                self.player.stop()
            if self.player:
                self.player.release()
            if self.instance:
                self.instance.release()
        except Exception as e:
            logging.error(e)