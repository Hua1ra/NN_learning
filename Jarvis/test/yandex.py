import dotenv
import os # type: ignore
import yandex_music
import vlc
import keyboard

def on_code(code):
    print(f"Open {code.verification_url} and enter: {code.user_code}")

dotenv.load_dotenv()
client = yandex_music.Client()
token = client.device_auth(on_code=on_code)
print(f"Token: {token.access_token}")

# token = os.getenv('YM_TOKEN')
token = token.access_token

client = yandex_music.Client(token).init()
query = str(input())

search_result = client.search(text=query)
track = search_result.tracks.results[0]
download_info = client.tracks_download_info(track_id=track.id, get_direct_links=True)
direct_link = download_info[0]['direct_link']
if direct_link:
    print("Link available")
    instance = vlc.Instance()
    player = instance.media_player_new()
    media = instance.media_new(direct_link)
    player.set_media(media)
    player.play()
    while True:
        if keyboard.is_pressed('q'):
            player.stop()
            break
else:
    print("Link not available")