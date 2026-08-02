import asyncio
from datetime import datetime
import dotenv
import flet as ft
import logging
import os
from pathlib import Path
import requests
import sounddevice as sd
import sys
import threading
import hashlib
import yandex_music
from src.Brain import Brain



# Get the path for data
def get_base_path():
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', ''))
    else:
        return Path(__file__).resolve().parent
# Get the path for logs
def get_writable_path():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent / 'logs'
    return Path(__file__).resolve().parent / 'logs'

# Load .env.client parameters
dotenv.load_dotenv(get_base_path() / '.env.client')
# Set logs
log_path = get_writable_path()
log_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=log_path / f'log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding='utf-8'
)



# Application
class JarvisApp:
    def __init__(self):
        try:
            # Main page
            self.page: ft.Page | None = None
            # Components
            self.brain: Brain | None = None
            self.loop: None | asyncio.AbstractEventLoop = None
            self.db_url = os.getenv('DB_URL')
            self.is_playing = False
            self.stop_worker = threading.Event()
            self.ym_token = None
            self.client = yandex_music.Client()
            # Visual components
            self.width = int(os.getenv('WIDTH', 300))
            self.height = int(os.getenv('HEIGHT', 500))
            self.status_led = ft.Icon(ft.Icons.CIRCLE,
                                      color=ft.Colors.RED_ACCENT,
                                      size=10)
            self.status_text = ft.Text('Loading',
                                       size=12,
                                       color='#808080',
                                       weight=ft.FontWeight.W_400)
            self.last_action_label = ft.Text('Waiting...',
                                             size=12,
                                             color='#606060',
                                             italic=True,
                                             text_align=ft.TextAlign.CENTER)
            self.play_pause_icon = ft.Icon(ft.Icons.PLAY_ARROW,
                                           color='#101012',
                                           size=40)
            self.red_glow = ft.BoxShadow(spread_radius=1,
                                         blur_radius=20,
                                         color=ft.Colors.with_opacity(0.3, ft.Colors.RED_ACCENT),
                                         offset=ft.Offset(0, 0))
            self.green_glow = ft.BoxShadow(spread_radius=1,
                                           blur_radius=20,
                                           color=ft.Colors.with_opacity(0.4, ft.Colors.GREEN_ACCENT),
                                           offset=ft.Offset(0, 0))
            self.central_button = ft.Container(
                content=self.play_pause_icon,
                width=100,
                height=100,
                border_radius=50,
                bgcolor=ft.Colors.WHITE,
                alignment=ft.Alignment.CENTER,
                shadow=self.red_glow,
                animate_scale=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
                on_click=self.play_pause_clicked,
                disabled=True,
            )
            self.reload_button = ft.Container(
                content=ft.Text('Reload',
                                color=ft.Colors.BLACK,
                                weight=ft.FontWeight.NORMAL),
                width=100,
                height=20,
                border_radius=5,
                bgcolor=ft.Colors.WHITE,
                alignment=ft.Alignment.CENTER,
                on_click=self.reload_models,
                disabled=True,
            )
            self.auth_login = ft.TextField(label='Login',
                                           width=200,
                                           text_size=14)
            self.auth_password = ft.TextField(label='Password',
                                              password=True,
                                              can_reveal_password=True,
                                              width=200,
                                              text_size=14)
            self.auth_error = ft.Text(value='',
                                      width=200,
                                      size=7,
                                      text_align=ft.TextAlign.CENTER)
            self.auth_btn = ft.Button('Enter',
                                      on_click=self.handle_login,
                                      width=200)
            self.auth_error_msg = ft.Text(value='',
                                          color=ft.Colors.RED_ACCENT,
                                          size=12,
                                          text_align=ft.TextAlign.CENTER)
            self.ym_link_btn = ft.TextButton(
                content='',
                url='',
                style=ft.ButtonStyle(color=ft.Colors.BLUE_400),
            )
            self.ym_code_text = ft.Text(selectable=True,
                                        weight=ft.FontWeight.BOLD,
                                        size=16,
                                        text_align=ft.TextAlign.CENTER)
            self.auth_dialog = ft.AlertDialog(
                modal=True,
                title=ft.Text('Enter code via link',
                              size=18,
                              weight=ft.FontWeight.BOLD),
                content=ft.Column(
                    [
                        ft.Container(height=10),
                        self.ym_link_btn,
                        self.ym_code_text,
                        ft.Text('Waiting for user action', size=10, color=ft.Colors.GREY_500, text_align=ft.TextAlign.CENTER)
                    ],
                    tight=True,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                actions=[
                    ft.TextButton(
                        'Close',
                        on_click=self.close_yandex_auth_dialog
                    )
                ]
            )
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    # Build the start screen
    def build_ui(self, page: ft.Page):
        try:
            self.page = page

            self.page.window.width = self.width
            self.page.window.height = self.height
            self.page.window.min_width = self.width
            self.page.window.max_width = self.width
            self.page.window.min_height = self.height
            self.page.window.max_height = self.height
            self.page.window.resizable = False

            self.page.title = 'Jarvis UI'
            self.page.theme_mode = ft.ThemeMode.DARK
            self.page.bgcolor = '#101012'
            self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
            self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
            self.page.padding = 20

            self.page.window.prevent_close = True
            self.page.window.on_event = self.window_event_handler

            self.page.overlay.append(self.auth_dialog)
            # Start authentification
            self.show_auth_screen()
        except Exception as e:
            logging.error(e)
            self.page.window.destroy()

    # Show the authentification screen
    def show_auth_screen(self):
        try:
            self.page.controls.clear()
            auth_layout = ft.Column(
                [
                    ft.Text('Authorization', size=20, weight=ft.FontWeight.BOLD),
                    ft.Container(height=10),
                    self.auth_login,
                    self.auth_password,
                    ft.Container(height=10),
                    self.auth_btn,
                    self.auth_error
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
            self.page.add(auth_layout)
            self.page.update()
        except Exception as e:
            logging.error(e)
            self.page.window.destroy()

    # Handle login button pressed
    async def handle_login(self, _):
        try:
            # Get the event loop
            self.loop = asyncio.get_event_loop()
            self.page.update()
            # Get the data
            username = self.auth_login.value
            password = self.auth_password.value

            is_authenticated = await asyncio.to_thread(self.check_user_in_db, username, password)
            # If user is valid
            if is_authenticated:
                self.auth_btn.disabled = True
                self.ym_token = await asyncio.to_thread(self.get_token_from_db, username, password)
                self.show_main_screen()
            # if user is not valid
            else:
                # Check existing
                if await asyncio.to_thread(self.check_login_in_db, username):
                    self.auth_error.value = 'Wrong username or password'
                    self.page.update()
                    return
                # Check password
                if len(password) < int(os.getenv('PASS_MIN_LEN')):
                    self.auth_error.value = 'Password too short'
                    self.page.update()
                    return
                # check username
                if username == '':
                    self.auth_error.value = 'Empty username'
                    self.page.update()
                    return
                # Yandex authentification
                self.auth_error.value = ''
                self.auth_btn.disabled = True
                self.ym_link_btn.content = 'Loading...'
                self.ym_link_btn.url = ''
                self.ym_code_text.value = ''

                self.auth_dialog.open = True
                self.auth_btn.disabled = False
                self.page.update()
                self.page.run_task(self.process_yandex_auth, username, password)
        except Exception as e:
            logging.error(e)
            await self.page.window.destroy()
            raise Exception('Login error')

    # Yandex authentification window
    async def process_yandex_auth(self, username, password):
        try:
            # Getting yandex token
            if not self.auth_dialog.open:
                return
            await asyncio.to_thread(self.client.device_auth, on_code=self.on_code)
            self.ym_token = self.client.token
            await asyncio.to_thread(self.save_user_to_db, username, password)
            self.close_yandex_auth_dialog(None)
            self.show_main_screen()
        except Exception as e:
            if not self.auth_dialog.open:
                return
            logging.error(f'Yandex auth error: {e}')
            self.ym_link_btn.content = 'Authorization error'
            self.ym_link_btn.disabled = True
            self.ym_code_text.value = 'Unknow error'
            self.auth_dialog.open = True
            self.ym_token = 'Error'
            self.page.update()
            raise Exception('Authorization error')

    # Set the yandex authentification window
    def on_code(self, code):
        try:
            self.ym_link_btn.content = code.verification_url
            self.ym_link_btn.url = code.verification_url
            self.ym_code_text.value = f'Code: {code.user_code}'
            self.loop.call_soon_threadsafe(self.page.update) # type: ignore
        except Exception as e:
            logging.error(e)
            raise Exception('Authorization error (Yandex Music API)')

    # Close authentification window
    def close_yandex_auth_dialog(self, _):
        self.auth_dialog.open = False
        self.page.update()

    # Show the main screen
    def show_main_screen(self):
        try:
            if self.page is None:
                return
            self.page.controls.clear()
            self.status_led.color = ft.Colors.RED_ACCENT
            self.status_text.value = 'Loading'
            ready_indicator = ft.Row([self.status_text, self.status_led],
                                     alignment=ft.MainAxisAlignment.CENTER,
                                     spacing=6)
            layout = ft.Column(
                [
                    ready_indicator,
                    ft.Container(height=30),
                    self.central_button,
                    ft.Container(height=30),
                    self.last_action_label,
                    ft.Container(height=20),
                    self.reload_button,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
            self.page.add(layout)
            self.page.update()
            # Load models
            self.page.run_task(self.load_brain)
        except Exception as e:
            logging.error(e)
            self.page.window.destroy()
            raise Exception('Loading main screen error')

    # Exit instructions
    async def window_event_handler(self, e):
        if e.type == ft.WindowEventType.CLOSE:
            logging.info('Closing app')
            if self.brain is not None:
                self.stop_worker.set()
                self.brain.abort()
            await self.page.window.destroy()
            os._exit(0)

    # Start / stop button
    def play_pause_clicked(self, _):
        try:
            self.is_playing = not self.is_playing
            if self.is_playing:
                self.play_pause_icon.name = ft.Icons.PAUSE
                self.central_button.shadow = self.green_glow
                self.stop_worker.clear()
                self.page.run_task(self.listen)
            else:
                self.play_pause_icon.name = ft.Icons.PLAY_ARROW
                self.central_button.shadow = self.red_glow
                self.stop_worker.set()
            self.play_pause_icon.update()
            self.central_button.update()
        except Exception as e:
            logging.error(e)
            self.page.window.destroy()
            raise Exception('Button play/pause error')

    # Reload all the models
    def reload_models(self):
        try:
            logging.info('Reloading models')
            self.central_button.disabled = True
            self.is_playing = False
            self.play_pause_icon.name = ft.Icons.PLAY_ARROW
            self.central_button.shadow = self.red_glow
            self.last_action_label.value = 'exit'
            self.stop_worker.set()
            self.reload_button.disabled = True
            self.show_main_screen()
            logging.info('Reloading finished')
        except Exception as e:
            logging.error(e)
            self.page.window.destroy()
            raise Exception('Button reload error')

    # Load models function
    async def load_brain(self):
        try:
            token = self.ym_token
            device = 'windows_desktop'
            self.brain = await asyncio.to_thread(Brain, token=token, device=device)
            self.status_led.color = ft.Colors.GREEN_ACCENT
            self.status_text.value = 'Ready'
            self.central_button.disabled = False
            self.reload_button.disabled = False
            self.page.update()
        except Exception as e:
            logging.error(e)
            self.status_led.color = ft.Colors.RED_ACCENT
            self.status_text.value = 'Error'
            self.central_button.disabled = True
            self.reload_button.disabled = False
            self.last_action_label.value = 'Try to reload'
            self.page.update()
            raise Exception('Load brain error')

    # Start listening
    async def listen(self):
        try:
            # Listening stream
            stream = sd.InputStream(
                samplerate=int(os.getenv('RATE')),
                channels=int(os.getenv('CHANNELS')),
                dtype='int16',
                blocksize=int(os.getenv('CHUNK'))
            )
            with stream:
                # While is on
                while not self.stop_worker.is_set():
                    # Get the chunk
                    audio_chunk, _ = await asyncio.to_thread(stream.read, int(os.getenv('CHUNK', 1024)))
                    # Get the result
                    result = await asyncio.to_thread(self.brain.listen, audio_chunk)
                    # If the intent == 'exit'
                    if result[0] == 'exit':
                        self.play_pause_icon.name = ft.Icons.PLAY_ARROW
                        self.central_button.shadow = self.red_glow
                        self.last_action_label.value = 'exit'
                        self.stop_worker.set()
                        self.page.update()
                    # If the intent == 'ask_current_track'
                    elif result[0] == 'ask_current_track':
                        if result[1] is None:
                            self.last_action_label.value = f'Nothing is playing right now'
                        else:
                            self.last_action_label.value = f'{result[1]["title"]}, {result[1]["artists"]}'
                        self.page.update()
                    # If the intent == 'other'
                    elif result[0] == 'other':
                        self.last_action_label.value = result[1]
                        self.page.update()
                    # If the wake word was recognized
                    elif result[0] == 'Recognized':
                        self.last_action_label.value = 'Listening'
                        self.page.update()
                    # If the track is ended
                    elif result[0] == 'None':
                        if result[1]:
                            self.brain.track_next()
                        else:
                            continue
                    # Else print intent with tokens
                    else:
                        intent = result[0].replace('_', ' ')
                        tokens = ' '.join(result[1])
                        self.last_action_label.value = f'{intent}{": " if tokens != "" else " "}{tokens}'
                        self.page.update()
                    await asyncio.sleep(float(os.getenv('TINY_DELAY', 0.1)))
                self.brain.track_pause()
        except Exception as e:
            logging.error(e)
            raise Exception('Listening error')

    # Hash the password
    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    # Check if the user in db (using API)
    def check_login_in_db(self, username):
        try:
            response = requests.post(
                f'{self.db_url}/auth/login',
                json={'username': username}
            )
            if response.status_code == 200:
                return response.json().get('exists', False)
            return False
        except Exception as e:
            logging.error(e)
            raise Exception('Check login error')

    # Check if the user authorized in db (using API)
    def check_user_in_db(self, username, password):
        try:
            password = self.hash_password(password)
            response = requests.post(
                f'{self.db_url}/auth/check',
                json={'username': username, 'password': password}
            )
            if response.status_code == 200:
                return response.json().get('exists', False)
            return False
        except Exception as e:
            logging.error(e)
            raise Exception('Check user error')

    # Get the token for the user from db (using API)
    def get_token_from_db(self, username, password):
        password = self.hash_password(password)
        try:
            response = requests.post(
                f'{self.db_url}/auth/token',
                json={'username': username, 'password': password}
            )
            if response.status_code == 200:
                return response.json().get('token')
            else:
                logging.error('Token error')
                sys.exit(1)
        except Exception as e:
            logging.error(e)
            raise Exception('Get token error')

    # Save the user to db (using API)
    def save_user_to_db(self, username, password):
        try:
            password = self.hash_password(password)
            requests.post(
                f'{self.db_url}/auth/save',
                json={'username': username, 'password': password, 'token': self.ym_token}
            )
        except Exception as e:
            logging.error(e)
            raise Exception('Save user error')



if __name__ == '__main__':
    app = JarvisApp()
    ft.run(main=app.build_ui)