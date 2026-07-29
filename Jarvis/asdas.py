from faster_whisper import download_model
path = download_model('medium', output_dir='./models/whisper-medium')
print(path)