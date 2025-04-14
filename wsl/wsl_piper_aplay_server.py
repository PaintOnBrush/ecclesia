#!/usr/bin/env python3
import subprocess
import sys
import argparse
import traceback
import tempfile
import os
import uuid # For unique filenames
import re   # For audio_id validation
from flask import Flask, request, jsonify, abort, Response
try:
    from waitress import serve
    HAVE_WAITRESS = True
except ImportError:
    HAVE_WAITRESS = False
    print("WARNING: Waitress package not found. Falling back to Flask development server.")
    print("         'pip install waitress' is recommended for better performance.")

app = Flask(__name__)

# --- Configuration ---
PIPER_MODEL_PATH = None # Will be set from args
APLAY_CMD = "aplay"     # Will be set from args
PIPER_CMD = "piper"     # Will be set from args
USE_CUDA = False        # Will be set from args
LENGTH_SCALE = 1.0      # Will be set from args

# Directory for temporary audio files
AUDIO_TEMP_DIR = "/tmp/piper_audio" # Using /tmp which is usually available in Linux/WSL
# --- End Configuration ---

def initialize_server():
    """Performs one-time setup tasks when the server starts."""
    print("Performing server initialization...")
    # Ensure the temporary directory exists at startup
    try:
        os.makedirs(AUDIO_TEMP_DIR, exist_ok=True) # Use exist_ok=True
        print(f"Ensured temporary audio directory exists: {AUDIO_TEMP_DIR}")
        # Basic permission check (optional but helpful)
        if not os.access(AUDIO_TEMP_DIR, os.W_OK | os.R_OK | os.X_OK):
             print(f"WARNING: Possible permission issue with {AUDIO_TEMP_DIR}. Ensure write/read/execute access.")
    except OSError as e:
        print(f"FATAL ERROR: Could not create/access temporary audio directory {AUDIO_TEMP_DIR}: {e}")
        # Exit if we can't create the essential directory
        sys.exit(f"Failed to initialize directory {AUDIO_TEMP_DIR}")
    except Exception as e:
        print(f"FATAL ERROR: Unexpected error during directory initialization: {e}")
        sys.exit("Failed to initialize server")

    # Validate essential commands (optional but good practice)
    # These rely on global variables being set *before* this function is called
    if not PIPER_MODEL_PATH or not os.path.isfile(PIPER_MODEL_PATH):
         print(f"FATAL ERROR: Piper model file not found or not specified: {PIPER_MODEL_PATH}")
         sys.exit(f"Model file missing or invalid path.")
    else:
         print(f"Checked: Piper model exists at: {PIPER_MODEL_PATH}")

    try:
        subprocess.run([APLAY_CMD, '--version'], check=True, capture_output=True, text=True, timeout=5)
        print(f"Checked: '{APLAY_CMD}' seems available.")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"FATAL ERROR: '{APLAY_CMD}' command not found, failed check, or timed out: {e}")
        sys.exit(f"Playback command '{APLAY_CMD}' unavailable or timed out.")
    try:
        # Piper might not have a --version flag that exits cleanly, check presence differently
        # Attempting to run it without input might hang or error, use check=False
        # We mainly want to know if the command itself exists.
        subprocess.run([PIPER_CMD, '--help'], check=True, capture_output=True, text=True, timeout=5)
        print(f"Checked: '{PIPER_CMD}' command seems available (via --help).")
    except (FileNotFoundError) as e:
        print(f"FATAL ERROR: '{PIPER_CMD}' command not found: {e}")
        sys.exit(f"Generation command '{PIPER_CMD}' unavailable.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
         # If --help fails or times out, it might still exist but is behaving unexpectedly
         print(f"WARNING: '{PIPER_CMD} --help' failed or timed out, but command might still exist. Proceeding cautiously. Error: {e}")


    print("Server Initialization Checks Complete.")


# === Endpoint for Generation ===
@app.route('/generate', methods=['POST'])
def handle_generate_request():
    if not request.data:
        print("ERROR: Received /generate request with empty body.")
        abort(400, description="Request body (text) cannot be empty.")

    text_to_speak = request.data.decode('utf-8')
    if not text_to_speak.strip(): # Check if empty after stripping whitespace
        print("ERROR: Received /generate request with blank text.")
        abort(400, description="Text to speak cannot be effectively empty.")

    print(f"Received request to GENERATE: '{text_to_speak[:100]}...'")

    # <<< --- FILTERING ADDED HERE --- >>>
    text_for_piper = text_to_speak.replace('*', '')
    if text_for_piper != text_to_speak:
        print(f"Filtered asterisks. Text for Piper: '{text_for_piper[:100]}...'")
    # <<< --- END FILTERING --- >>>


    # Generate a unique filename
    audio_id = str(uuid.uuid4()) + ".wav"
    temp_wav_path = os.path.join(AUDIO_TEMP_DIR, audio_id)
    print(f"Target temporary WAV path: {temp_wav_path}")

    try:
        # Use list form for subprocess to avoid shell injection risks and handle quotes better
        piper_gen_command_list = [PIPER_CMD, '--model', PIPER_MODEL_PATH, '--output_file', temp_wav_path]
        if USE_CUDA:
            piper_gen_command_list.append("--cuda")
        if LENGTH_SCALE is not None and LENGTH_SCALE != 1.0: # Only add if not default
             piper_gen_command_list.extend(["--length-scale", str(LENGTH_SCALE)])

        print(f"Executing Piper generation command list: {piper_gen_command_list}")

        # Pipe the CLEANED text to piper's stdin
        gen_process = subprocess.run(
            piper_gen_command_list,
            input=text_for_piper, # <<< USE CLEANED TEXT HERE >>>
            check=True,          # Raise exception on non-zero exit code
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE, # Capture stdout too just in case
            text=True,           # Work with text streams
            encoding='utf-8',
            timeout=60           # Add a timeout for generation
        )
        print("Piper generation command finished.")
        if gen_process.stderr: # Log any stderr output even on success
             print(f"Piper Stderr: {gen_process.stderr.strip()}")
        if gen_process.stdout: # Log any stdout output even on success
             print(f"Piper Stdout: {gen_process.stdout.strip()}")


        # Verify file existence and size *after* process completes
        if not os.path.exists(temp_wav_path):
             stderr_info = f" Stderr: {gen_process.stderr.strip()}" if gen_process.stderr else ""
             raise FileNotFoundError(f"Piper command succeeded but WAV file missing: {temp_wav_path}.{stderr_info}")
        if os.path.getsize(temp_wav_path) == 0:
             stderr_info = f" Stderr: {gen_process.stderr.strip()}" if gen_process.stderr else ""
             # Attempt to remove the empty file
             try: os.remove(temp_wav_path)
             except Exception: pass
             raise ValueError(f"Piper command succeeded but WAV file is empty: {temp_wav_path}.{stderr_info}")

        # Return the ID (filename) for later playback
        print(f"Successfully generated audio file: {audio_id}")
        return jsonify({"status": "generated", "audio_id": audio_id})

    except subprocess.TimeoutExpired:
         error_message = f"Piper generation command timed out after 60s."
         print(f"ERROR: {error_message}")
         # Clean up potentially incomplete file
         if os.path.exists(temp_wav_path):
             try: os.remove(temp_wav_path)
             except Exception as e_del: print(f"Warning: Failed to clean up timed-out file {temp_wav_path}: {e_del}")
         return jsonify({"status": "error", "message": error_message}), 500 # Internal Server Error for timeout

    except subprocess.CalledProcessError as e:
        error_message = f"Piper generation command failed (Code {e.returncode})"
        stderr_output = e.stderr.strip() if e.stderr else "No stderr captured."
        stdout_output = e.stdout.strip() if e.stdout else "No stdout captured."
        # Try to create a command string representation for logging
        try: cmd_str = ' '.join(e.cmd)
        except TypeError: cmd_str = str(e.cmd) # Fallback if list contains non-strings
        print(f"ERROR: {error_message} - Command: {cmd_str}")
        print(f"Stderr: {stderr_output}")
        print(f"Stdout: {stdout_output}")
        if os.path.exists(temp_wav_path): # Cleanup failed generation attempt
             try: os.remove(temp_wav_path)
             except Exception as e_del: print(f"Warning: Failed to clean up failed generation file {temp_wav_path}: {e_del}")
        return jsonify({"status": "error", "message": error_message, "stderr": stderr_output, "stdout": stdout_output}), 500 # Internal Server Error

    except Exception as e:
        error_message = f"Unexpected error during generation: {type(e).__name__}: {e}"
        print(f"ERROR: {error_message}")
        traceback.print_exc()
        if os.path.exists(temp_wav_path): # Cleanup if file exists after unexpected error
             try: os.remove(temp_wav_path)
             except Exception as e_del: print(f"Warning: Failed to clean up file {temp_wav_path} after unexpected error: {e_del}")
        # Return a generic 500 error for truly unexpected issues
        return jsonify({"status": "error", "message": "An unexpected internal error occurred during audio generation."}), 500

# === Endpoint for Playback ===
@app.route('/play', methods=['POST'])
def handle_play_request():
    try: # Wrap the whole thing to catch JSON errors early
        req_data = request.get_json()
        if not req_data or 'audio_id' not in req_data:
            print("ERROR: Received /play request with invalid/missing JSON or missing 'audio_id'.")
            abort(400, description="Request body must be JSON with 'audio_id' key.")
    except Exception as json_e: # Catch potential JSON decoding errors
         print(f"ERROR: Failed to decode JSON in /play request: {json_e}")
         abort(400, description=f"Invalid JSON in request body: {json_e}")

    audio_id = req_data['audio_id']

    # Basic validation to prevent path traversal and ensure expected format
    # Allows UUIDs with hyphens, standard alphanumeric, and ends in .wav
    if not re.match(r'^[a-zA-Z0-9-]+\.wav$', audio_id) or '..' in audio_id or '/' in audio_id or '\\' in audio_id:
         print(f"ERROR: Received /play request with invalid audio_id format: {audio_id}")
         abort(400, description="Invalid audio_id format.")

    temp_wav_path = os.path.join(AUDIO_TEMP_DIR, audio_id)
    print(f"Received request to PLAY audio_id: {audio_id}")
    print(f"Expecting audio file at: {temp_wav_path}")

    if not os.path.exists(temp_wav_path):
        print(f"ERROR: Audio file not found for playback: {temp_wav_path}")
        # Return 404 Not Found if the specific audio ID doesn't exist
        abort(404, description=f"Audio file not found: {audio_id}")
    if not os.path.isfile(temp_wav_path):
         print(f"ERROR: Path exists but is not a file: {temp_wav_path}")
         abort(500, description=f"Server error: Expected audio path is not a file.") # Should not happen

    try:
        # Use list form for subprocess, safer than shell=True
        aplay_command_list = [APLAY_CMD, temp_wav_path]
        print(f"Executing Aplay playback command list: {aplay_command_list}")

        play_process = subprocess.run(
            aplay_command_list,
            check=True,          # Raise exception on non-zero exit code
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE, # Capture stdout too
            text=True,           # Work with text streams
            encoding='utf-8',
            timeout=300          # Generous timeout for playback (5 minutes)
        )
        print(f"Aplay playback command finished successfully for {audio_id}.")
        if play_process.stderr: # Log any stderr output even on success
             print(f"Aplay Stderr: {play_process.stderr.strip()}")
        if play_process.stdout: # Log any stdout output even on success
             print(f"Aplay Stdout: {play_process.stdout.strip()}")

        # *** SUCCESS RESPONSE ***
        return jsonify({"status": "played", "message": "Audio playback completed successfully."})

    except subprocess.TimeoutExpired:
         error_message = f"Aplay playback command timed out after 300s for {audio_id}."
         print(f"ERROR: {error_message}")
         # Don't delete file on timeout, might be useful for debugging *why* it timed out
         return jsonify({"status": "error", "message": error_message}), 500 # Internal Server Error

    except subprocess.CalledProcessError as e:
        error_message = f"Aplay playback command failed (Code {e.returncode})"
        stderr_output = e.stderr.strip() if e.stderr else "No stderr captured."
        stdout_output = e.stdout.strip() if e.stdout else "No stdout captured."
        # Try to create a command string representation for logging
        try: cmd_str = ' '.join(e.cmd)
        except TypeError: cmd_str = str(e.cmd) # Fallback if list contains non-strings
        print(f"ERROR: {error_message} - Command: {cmd_str}")
        print(f"Stderr: {stderr_output}")
        print(f"Stdout: {stdout_output}")
        # Don't delete the file on error, keep it for inspection
        return jsonify({"status": "error", "message": error_message, "stderr": stderr_output, "stdout": stdout_output}), 500 # Internal Server Error

    except Exception as e:
        # Catch any other unexpected errors during playback attempt
        error_message = f"Unexpected error during playback of {audio_id}: {type(e).__name__}: {e}"
        print(f"ERROR: {error_message}")
        traceback.print_exc()
        # Don't delete the file
        return jsonify({"status": "error", "message": "An unexpected internal error occurred during audio playback."}), 500

    # Files will now persist in AUDIO_TEMP_DIR until manually cleaned or system tmp cleanup.

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Piper TTS server generating temp files and playing via aplay.")
    parser.add_argument("-m", "--model", required=True, help="Path to the Piper ONNX model file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind to (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on (e.g., 5001, 5002, 5003).")
    parser.add_argument("--aplay-cmd", default="aplay", help="Audio playback command (default: aplay).")
    parser.add_argument("--piper-cmd", default="piper", help="Piper executable command (default: piper).")
    parser.add_argument("--cuda", action='store_true', help="Use --cuda flag for piper.")
    parser.add_argument("--length-scale", type=float, default=1.0, help="Phoneme length scale.")

    args = parser.parse_args()

    # Assign parsed arguments to global variables FIRST
    PIPER_MODEL_PATH = args.model
    APLAY_CMD = args.aplay_cmd
    PIPER_CMD = args.piper_cmd
    USE_CUDA = args.cuda
    LENGTH_SCALE = args.length_scale

    # Call initialization function HERE
    initialize_server() # Run setup checks *after* args are parsed and assigned

    print("--- WSL Piper Aplay Server (Generate/Play Mode) ---")
    # Print config details *after* initialization checks passed
    print(f"Model Path:       {PIPER_MODEL_PATH}")
    print(f"Piper Command:    {PIPER_CMD}")
    print(f"Aplay Command:    {APLAY_CMD}")
    print(f"Use CUDA:         {USE_CUDA}")
    print(f"Length Scale:     {LENGTH_SCALE}")
    print(f"Audio Temp Dir:   {AUDIO_TEMP_DIR}")
    print(f"Listening on:     http://{args.host}:{args.port}")
    print(f"Generate endpoint: /generate (POST text/plain)")
    print(f"Playback endpoint: /play     (POST application/json {'{'} \"audio_id\": \"<uuid>.wav\" {'}'})")
    print("--------------------------------------------------")

    # Use waitress if available, otherwise fallback to Flask dev server
    if HAVE_WAITRESS:
        print(f"Starting server with Waitress on {args.host}:{args.port}...")
        serve(app, host=args.host, port=args.port, threads=4) # Adjust threads as needed
    else:
        print(f"Starting server with Flask development server on {args.host}:{args.port} (Waitress not found)...")
        app.run(host=args.host, port=args.port, debug=False) # DO NOT use debug=True in anything resembling production
