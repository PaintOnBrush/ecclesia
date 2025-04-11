make llamacpp with cuda support in c:\users\<users>\workspace\llamacpp

windows python 3.12
needs to pip install somethings. 

run both servers bat files

run third python script. latest version is 3.3 (ai33.py)



```# Basic usage (uses defaults, no file saving)
python conversation_simulator.py

# Custom prompt, 3 turns, enable human roleplay, default file saving
python conversation_simulator.py -p "Let's discuss quantum physics." -t 3 --human --file_location conversation_output.md

# Specify all parameters, including different delimiter and file path
python conversation_simulator.py \
    --prompt "Create a short story outline together." \
    --turns 4 \
    --timeout 60 \
    --max_tokens 1000 \
    --temperature 0.8 \
    --file_location ./story_blocks.txt \
    --file_delimiter COLLABORATE_WRITE

# Just enable default file saving with default delimiter
python conversation_simulator.py --file_location conversation_output.md

# Get help
python conversation_simulator.py --help
```
