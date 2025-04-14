#/bin/bash
. ~/tts/bin/activate
python wsl_piper_aplay_server.py --model ~/workspace/piper-voices/en_US-amy-medium.onnx --port 5002 --cuda --length-scale 0.75
