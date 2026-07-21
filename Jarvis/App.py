import asyncio
import dotenv
import flet as ft
import logging
import os
import sounddevice as sd
import sys
import threading
import hashlib
import yandex_music
from Jarvis.src.Brain import Brain
from Jarvis.src.Database import Database



dotenv.load_dotenv()
logging.basicConfig(
    filename=os.getenv('LOGGING'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding='utf-8'
)



class JarvisApp:
    def __init__(self):
        self.page: ft.Page | None = None

        self.brain: Brain | None = None
        self.db = Database()
        self.is_playing = False
        self.stop_worker = threading.Event()
        self.ym_token = None
        self.client = yandex_music.Client()

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
            on_click=self.play_pause_clicked, disabled=True,
        )

        self.auth_login = ft.TextField(label='Login',
                                       width=200,
                                       text_size=14)
        self.auth_password = ft.TextField(label='Password',
                                          password=True,
                                          can_reveal_password=True,
                                          width=200,
                                          text_size=14)
        self.auth_btn = ft.ElevatedButton('Enter',
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
            title=ft.Text('Enter code via link', size=18, weight=ft.FontWeight.BOLD),
            content=ft.Column(
                [
                    ft.Container(height=10),
                    self.ym_link_btn,
                    self.ym_code_text,
                    ft.Text('Waiting for user action', size=10, color=ft.Colors.GREY_500, text_align=ft.TextAlign.CENTER)
                ],
                tight=True,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER
            )
        )

    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def check_user_in_db(self, username, password):
        password = self.hash_password(password)
        return self.db.check_user_in_db(username, password)

    def save_user_to_db(self, username, password):
        password = self.hash_password(password)
        self.db.save_user_to_db(username, password, self.ym_token)

    def get_token_from_db(self, username, password):
        password = self.hash_password(password)
        try:
            return self.db.get_token(username, password)
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    def on_code(self, code):
        self.ym_link_btn.content = code.verification_url
        self.ym_link_btn.url = code.verification_url
        self.ym_code_text.value = f'Code: {code.user_code}'
        if self.page:
            self.page.update()

    async def process_yandex_auth(self, username, password):
        try:
            await asyncio.to_thread(self.client.device_auth, on_code=self.on_code)
            self.ym_token = self.client.token
            await asyncio.to_thread(self.save_user_to_db, username, password)
            self.auth_dialog.open = False
            self.show_main_screen()
        except Exception as e:
            logging.error(f'Yandex auth error: {e}')
            self.ym_link_btn.content = 'Authorization error'
            self.ym_link_btn.url = ''
            self.ym_code_text.value = ''
            if self.page:
                self.page.update()

    def build_ui(self, page: ft.Page):
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

        self.show_auth_screen()

    def show_auth_screen(self):
        self.page.controls.clear()
        auth_layout = ft.Column(
            [
                ft.Text('Authorization', size=20, weight=ft.FontWeight.BOLD),
                ft.Container(height=10),
                self.auth_login,
                self.auth_password,
                ft.Container(height=10),
                self.auth_btn,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.page.add(auth_layout)
        self.page.update()

    async def handle_login(self, _):
        self.auth_btn.disabled = True
        self.page.update()

        username = self.auth_login.value
        password = self.auth_password.value

        is_authenticated = await asyncio.to_thread(self.check_user_in_db, username, password)
        if is_authenticated:
            self.ym_token = await asyncio.to_thread(self.get_token_from_db, username, password)
            self.show_main_screen()
        else:
            self.ym_link_btn.content = 'Loading...'
            self.ym_link_btn.url = ''
            self.ym_code_text.value = ''

            self.page.overlay.append(self.auth_dialog)
            self.auth_dialog.open = True
            self.auth_btn.disabled = False
            self.page.update()
            self.page.run_task(self.process_yandex_auth, username, password)

    def show_main_screen(self):
        if self.page is None:
            return
        self.page.controls.clear()
        ready_indicator = ft.Row([self.status_text, self.status_led],
                                 alignment=ft.MainAxisAlignment.CENTER,
                                 spacing=6)
        layout = ft.Column(
            [
                ready_indicator,
                ft.Container(height=35),
                self.central_button,
                ft.Container(height=35),
                self.last_action_label,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.page.add(layout)
        self.page.update()
        self.page.run_task(self.load_brain)

    async def window_event_handler(self, e):
        if e.type == ft.WindowEventType.CLOSE:
            logging.info('Closing app')
            self.db.close()
            if self.brain is not None:
                self.stop_worker.set()
                self.brain.exit()
            await self.page.window.destroy()
            os._exit(0)

    async def load_brain(self):
        try:
            token = self.ym_token
            device = 'windows_desktop'
            self.brain = await asyncio.to_thread(Brain, token=token, device=device)

            self.status_led.color = ft.Colors.GREEN_ACCENT
            self.status_text.value = 'Ready'
            self.central_button.disabled = False
            self.page.update()
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    async def listen(self):
        try:
            stream = sd.InputStream(
                samplerate=int(os.getenv('RATE')),
                channels=int(os.getenv('CHANNELS')),
                dtype='int16',
                blocksize=int(os.getenv('CHUNK'))
            )
            with stream:
                while not self.stop_worker.is_set():
                    audio_chunk, _ = await asyncio.to_thread(stream.read, int(os.getenv('CHUNK', 1024)))
                    result = await asyncio.to_thread(self.brain.listen, audio_chunk)
                    if result[0] == 'exit':
                        self.play_pause_icon.name = ft.Icons.PLAY_ARROW
                        self.central_button.shadow = self.red_glow
                        self.stop_worker.set()
                        self.page.update()
                    elif result[0] == 'ask_current_track':
                        if result[1] is None:
                            self.last_action_label.value = f'Nothing is playing right now'
                        else:
                            self.last_action_label.value = f'{result[1]["title"]}, {result[1]["artists"]}'
                        self.page.update()
                    elif result[0] == 'other':
                        self.last_action_label.value = result[1]
                        self.page.update()
                    elif result[0] == 'Recognized':
                        self.last_action_label.value = 'Listening'
                        self.page.update()
                    elif result[0] == 'None':
                        if result[1]:
                            self.brain.track_next()
                        else:
                            continue
                    else:
                        intent = result[0].replace('_', ' ')
                        tokens = ' '.join(result[1])
                        self.last_action_label.value = f'{intent}{": " if tokens != "" else " "}{tokens}'
                        self.page.update()
                    await asyncio.sleep(float(os.getenv('TINY_DELAY', 0.1)))
                self.brain.track_pause()
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    def play_pause_clicked(self, _):
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



if __name__ == '__main__':
    app = JarvisApp()
    ft.run(main=app.build_ui)