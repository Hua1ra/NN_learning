@echo off
flet pack App.py ^
  -D ^
  --name Jarvis ^
  --add-data "src:src" ^
  --add-data ".env.client:." ^
  --add-data "data:data" ^
  --add-data "models\extractor.pth:models" ^
  --hidden-import keyboard ^
  --pyinstaller-build-args="--collect-all=torch" ^
  --pyinstaller-build-args="--collect-all=transformers" ^
  --pyinstaller-build-args="--collect-all=faster_whisper" ^
  --pyinstaller-build-args="--collect-all=openwakeword" ^
  --pyinstaller-build-args="--collect-all=sounddevice" ^
  --pyinstaller-build-args="--collect-all=yandex_music"