import dotenv
import logging
import os
from pathlib import Path
import sys
import random



class YMController:
    def __init__(self, client, device):
        dotenv.load_dotenv(Path(__file__).resolve().parent.parent / '.env.client')
        self.client = client # Client for yandex_music requests
        self.device = device # Current user device

        self.current_track = None # Current track playing (class Track)
        self.current_queue = [] # Current batch of tracks
        self.batch_index = 0 # Current track index in the batch
        self.current_station_id = 'user:onion' # Station to connect

    def play_next(self):
        # Play the next track in the current batch.
        # Get the new batch if needed.
        try:
            if self.current_track is not None:
                self.__send_play_end_track(self.current_track, self.play_id)
            if self.batch_index + 1 >= len(self.current_queue):
                self.current_queue = self.__get_radio_batch(self.current_track.id if self.current_track is not None else None)
            self.batch_index += 1
            self.current_track = self.__update_current_track()
            return self.current_track
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    def play_prev(self):
        # Play the previous track in the current batch (if possible).
        try:
            if self.current_track is not None:
                self.__send_play_end_track(self.current_track, self.play_id)
            self.batch_index -= 1
            if self.batch_index < 0:
                logging.warning('Cannot play prev track')
                self.batch_index = 0
            self.current_track = self.__update_current_track()
            return self.current_track
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    def play_search_track(self, query):
        self.current_queue = self.__get_search_batch(query, 'track')

    def play_search_artist(self, query):
        self.current_queue = self.__get_search_batch(query, 'artist')

    def play_search_album(self, query):
        self.current_queue = self.__get_search_batch(query, 'album')

    def play_search_playlist(self, query):
        self.current_queue = self.__get_search_batch(query, 'playlist')

    def play_favorite_batch(self):
        self.current_queue = self.__get_favorite_batch()

    def play_wave(self):
        self.set_station('user:onion')

    def set_station(self, current_station_id):
        # Set different station to connect.
        self.current_station_id = current_station_id
        self.current_queue = self.__get_radio_batch()

    def get_link(self, track_id):
        try:
            return self.client.tracks_download_info(track_id=track_id, get_direct_links=True)[0]['direct_link']
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    def like(self, track_id):
        try:
            self.client.users_likes_tracks_add(track_id)
        except Exception as e:
            logging.error(e)

    def dislike(self, track_id):
        try:
            self.client.users_likes_tracks_remove(track_id)
        except Exception as e:
            logging.error(e)

    def __get_radio_batch(self, queue=None):
        # Get the new batch from the current radio.
        try:
            if self.current_queue is None or self.current_queue == []:
                old_tracks = []
            else:
                start = max(self.batch_index + 1 - int(os.getenv('LAST_TRACKS_COUNT')), 0)
                old_tracks = self.current_queue[start : self.batch_index + 1]
            self.batch_index = len(old_tracks) - 1
            new_tracks = [item.track for item in self.client.rotor_station_tracks(self.current_station_id, queue=queue).sequence]
        except Exception as e:
            logging.error(e)
            sys.exit(1)
        return old_tracks + new_tracks

    def __get_favorite_batch(self):
        # Get the new batch from the favourite playlist.
        try:
            if self.current_queue is None or self.current_queue == []:
                old_tracks = []
            else:
                start = max(self.batch_index + 1 - int(os.getenv('LAST_TRACKS_COUNT')), 0)
                old_tracks = self.current_queue[start: self.batch_index + 1]
            self.batch_index = len(old_tracks) - 1
            new_tracks = [item.id for item in self.client.users_likes_tracks()]
            new_tracks = self.client.tracks(new_tracks)
            random.shuffle(new_tracks)
        except Exception as e:
            logging.error(e)
            return self.current_queue
        return old_tracks + new_tracks

    def __get_search_batch(self, query, target):
        try:
            if self.current_queue is None or self.current_queue == []:
                old_tracks = []
            else:
                start = max(self.batch_index + 1 - int(os.getenv('LAST_TRACKS_COUNT')), 0)
                old_tracks = self.current_queue[start : self.batch_index + 1]
            self.batch_index = len(old_tracks) - 1
            if target == 'track':
                new_tracks = self.client.search(query).tracks.results
                new_tracks = new_tracks[:min(int(os.getenv('SEARCH_TRACKS_COUNT')), len(new_tracks))]
            elif target == 'artist':
                artist = self.client.search(query).artists.results[0]
                new_tracks = artist.get_tracks(page_size=int(os.getenv('SEARCH_ARTIST_COUNT'))).tracks
            elif target == 'album':
                album = self.client.search(query).albums.results[0]
                new_tracks = [item for row in self.client.albums_with_tracks(album.id).volumes for item in row]
            elif target == 'playlist':
                playlist = self.client.search(query).playlists.results[0]
                new_tracks = [item.id for item in self.client.users_playlists(playlist.kind, playlist.uid).tracks]
                new_tracks = self.client.tracks(new_tracks)
            else:
                new_tracks = []
        except Exception as e:
            logging.error(e)
            return self.current_queue
        return old_tracks + new_tracks

    def __update_current_track(self):
        # Get the next track.
        self.play_id = self.__generate_play_id()
        self.__send_play_start_track(self.current_queue[self.batch_index], self.play_id)
        return self.current_queue[self.batch_index]

    def __send_play_start_track(self, track, play_id):
        # Send start reply.
        try:
            total_seconds = track.duration_ms / 1000
            album_id = track.albums[0].id if track.albums else 0
            self.client.play_audio(
                from_=self.device,
                track_id=str(track.id),
                album_id=str(album_id),
                play_id=play_id,
                track_length_seconds=0,
                total_played_seconds=0,
                end_position_seconds=total_seconds
            )
        except Exception as e:
            logging.error(e)

    def __send_play_end_track(self, track, play_id):
        # Send end reply.
        try:
            total_seconds = track.duration_ms / 1000
            played_seconds = total_seconds
            album_id = track.albums[0].id if track.albums else 0
            self.client.play_audio(
                from_=self.device,
                track_id=str(track.id),
                album_id=str(album_id),
                play_id=play_id,
                track_length_seconds=int(total_seconds),
                total_played_seconds=int(played_seconds),
                end_position_seconds=int(total_seconds),
            )
        except Exception as e:
            logging.error(e)

    @staticmethod
    def __generate_play_id():
        # Get random play id.
        return '%s-%s-%s' % (int(random.random() * 100000), int(random.random() * 100000), int(random.random() * 100000))