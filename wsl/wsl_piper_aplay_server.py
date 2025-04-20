#!/usr/bin/env python3
import subprocess
import sys
import argparse
import traceback
import tempfile
import os
import uuid # For unique filenames
import re   # For audio_id validation
import json
from typing import Optional # ** ADDED IMPORT **
from flask import Flask, request, jsonify, abort, Response, make_response

# Attempt to import waitress
try:
    from waitress import serve
    HAVE_WAITRESS = True
except ImportError:
    HAVE_WAITRESS = False
    print("WARNING: Waitress package not found. Falling back to Flask development server.", file=sys.stderr)
    print("         'pip install waitress' is recommended for production.", file=sys.stderr)

app = Flask(__name__)

# --- Configuration (Set via argparse) ---
ALLOWED_VOICE_DIR : Optional[str] = None # Type hint requires Optional
APLAY_CMD : str = "aplay"
PIPER_CMD : str = "piper"
USE_CUDA : bool = False
LENGTH_SCALE : float = 1.0
AUDIO_TEMP_DIR : str = "/tmp/piper_audio_server" # Slightly different default
# --- End Configuration ---

# --- Utility ---
def log_info(message: str):
    """Standard info logging."""
    print(f"INFO: {message}", flush=True)

def log_error(message: str, include_traceback: bool = False):
    """Standard error logging."""
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    if include_traceback:
        traceback.print_exc(file=sys.stderr)

def log_warning(message: str):
    """Standard warning logging."""
    print(f"WARNING: {message}", file=sys.stderr, flush=True)

def make_error_response(message: str, status_code: int):
    """Creates a JSON error response."""
    log_error(f"Responding with {status_code}: {message}")
    response = jsonify({"status": "error", "message": message})
    response.status_code = status_code
    return response

# --- Initialization ---
def initialize_server():
    """Performs one-time setup tasks when the server starts."""
    log_info("Performing server initialization...")
    global ALLOWED_VOICE_DIR, AUDIO_TEMP_DIR # Allow modification

    # 1. Validate and Create Temporary Audio Directory
    try:
        AUDIO_TEMP_DIR = os.path.abspath(AUDIO_TEMP_DIR)
        os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)
        log_info(f"Ensured temporary audio directory exists: {AUDIO_TEMP_DIR}")
        # Basic permission check
        if not os.access(AUDIO_TEMP_DIR, os.W_OK | os.R_OK | os.X_OK):
             log_warning(f"Potential permission issue with temporary directory {AUDIO_TEMP_DIR}. Check ownership/permissions.")
    except OSError as e:
        log_error(f"Could not create/access temporary audio directory {AUDIO_TEMP_DIR}: {e}", include_traceback=True)
        sys.exit(f"FATAL: Failed to initialize directory {AUDIO_TEMP_DIR}")
    except Exception as e:
        log_error(f"Unexpected error during temporary directory initialization: {e}", include_traceback=True)
        sys.exit("FATAL: Failed to initialize server (directory setup)")

    # 2. Validate Allowed Voices Directory
    if not ALLOWED_VOICE_DIR or not os.path.isdir(ALLOWED_VOICE_DIR):
        log_error(f"Allowed voices directory not specified, not found, or not a directory: '{ALLOWED_VOICE_DIR}'")
        sys.exit("FATAL: --voices-dir is required and must point to a valid directory.")
    else:
        ALLOWED_VOICE_DIR = os.path.abspath(ALLOWED_VOICE_DIR)
        log_info(f"Validated: Allowed voices directory set to: {ALLOWED_VOICE_DIR}")

    # 3. Check Playback Command (aplay) Availability
    try:
        log_info(f"Checking playback command: '{APLAY_CMD}'...")
        result = subprocess.run([APLAY_CMD, '--version'], check=True, capture_output=True, text=True, timeout=5)
        log_info(f"Playback command '{APLAY_CMD}' check successful (version: {result.stdout.splitlines()[0]}).")
    except FileNotFoundError:
         log_error(f"Playback command '{APLAY_CMD}' not found in PATH.")
         sys.exit(f"FATAL: Playback command '{APLAY_CMD}' unavailable.")
    except subprocess.TimeoutExpired:
         log_error(f"Checking playback command '{APLAY_CMD}' timed out after 5s.")
         sys.exit(f"FATAL: Playback command '{APLAY_CMD}' check timed out.")
    except subprocess.CalledProcessError as e:
         log_error(f"Playback command '{APLAY_CMD} --version' failed (Code {e.returncode}):\nStderr: {e.stderr}\nStdout: {e.stdout}")
         sys.exit(f"FATAL: Playback command '{APLAY_CMD}' check failed.")
    except Exception as e:
       log_error(f"Unexpected error checking playback command '{APLAY_CMD}': {e}", include_traceback=True)
       sys.exit(f"FATAL: Failed to check playback command '{APLAY_CMD}'.")

    # 4. Check TTS Command (piper) Availability (Basic Check)
    try:
        log_info(f"Checking TTS command: '{PIPER_CMD}' (using --help)...")
        result = subprocess.run([PIPER_CMD, '--help'], check=True, capture_output=True, text=True, timeout=5)
        log_info(f"TTS command '{PIPER_CMD}' seems available (via --help check).")
    except FileNotFoundError:
         log_error(f"TTS command '{PIPER_CMD}' not found in PATH.")
         sys.exit(f"FATAL: TTS command '{PIPER_CMD}' unavailable.")
    except subprocess.TimeoutExpired:
         log_warning(f"Checking TTS command '{PIPER_CMD} --help' timed out after 5s. Command might still work.")
    except subprocess.CalledProcessError as e:
         log_warning(f"TTS command '{PIPER_CMD} --help' failed (Code {e.returncode}). Command might still work.\nStderr: {e.stderr}")
    except Exception as e:
         log_warning(f"Unexpected error checking TTS command '{PIPER_CMD} --help': {e}. Command might still work.")

    log_info("Server Initialization Checks Complete.")


# === Endpoint for Generation ===
@app.route('/generate', methods=['POST', 'OPTIONS'])
def handle_generate_request():
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    # Actual POST request handling
    if not request.is_json:
        return make_error_response("Request body must be JSON.", 415) # Unsupported Media Type

    try:
        req_data = request.get_json()
        text_to_speak = req_data.get('text')
        model_path_relative = req_data.get('model_path') # Expect relative path from voices-dir

        if not text_to_speak or not isinstance(text_to_speak, str) or not text_to_speak.strip():
            return make_error_response("JSON must include a non-empty 'text' field (string).", 400)
        if not model_path_relative or not isinstance(model_path_relative, str) or not model_path_relative.endswith(".onnx"):
             return make_error_response("JSON must include a 'model_path' field (string ending in .onnx).", 400)

    except Exception as json_e:
        return make_error_response(f"Invalid JSON in request body: {json_e}", 400)

    log_info(f"Received GENERATE request: Model='{model_path_relative}', Text='{text_to_speak[:60]}...'")
    text_for_piper = text_to_speak.replace('*', '') # Basic sanitization for '*' if problematic

    # --- Validate Model Path ---
    try:
        # Prevent path traversal by joining and checking prefix
        requested_model_path = os.path.abspath(os.path.join(ALLOWED_VOICE_DIR, model_path_relative))
        if not requested_model_path.startswith(ALLOWED_VOICE_DIR + os.sep):
             log_error(f"SECURITY VIOLATION: Path traversal attempt detected. Relative path '{model_path_relative}' resolved to '{requested_model_path}' which is outside allowed directory '{ALLOWED_VOICE_DIR}'")
             return make_error_response("Invalid model_path specified (Path traversal denied).", 400)

        # Check if model file exists
        if not os.path.isfile(requested_model_path):
            log_error(f"Model file not found at resolved path: {requested_model_path}")
            return make_error_response(f"Specified model_path '{model_path_relative}' not found.", 404) # Not Found

        # Check if corresponding config file exists
        config_path = requested_model_path + ".json"
        if not os.path.isfile(config_path):
            log_error(f"Model config file (.json) not found for model: {requested_model_path}")
            return make_error_response(f"Model config file (.json) not found for '{model_path_relative}'.", 400)

        piper_model_to_use = requested_model_path # Use the validated absolute path
        log_info(f"Validated model path: {piper_model_to_use}")

    except Exception as path_e:
        log_error(f"Internal error during model path validation: {path_e}", include_traceback=True)
        return make_error_response("Internal server error validating model path.", 500)

    # --- Generate Audio ---
    audio_id = str(uuid.uuid4()) + ".wav" # Unique ID for the audio file
    temp_wav_path = os.path.join(AUDIO_TEMP_DIR, audio_id)
    log_info(f"Target temporary WAV path: {temp_wav_path}")

    try:
        # Build Piper command list
        piper_gen_command_list = [PIPER_CMD, '--model', piper_model_to_use, '--output_file', temp_wav_path]
        if USE_CUDA:
            piper_gen_command_list.append("--cuda")
        if LENGTH_SCALE is not None and LENGTH_SCALE != 1.0:
            piper_gen_command_list.extend(["--length-scale", str(LENGTH_SCALE)])

        log_info(f"Executing Piper command: {' '.join(piper_gen_command_list)}")
        # Use text=True for automatic encoding/decoding
        gen_process = subprocess.run(
            piper_gen_command_list,
            input=text_for_piper,
            check=True, # Raise CalledProcessError on non-zero exit code
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8', # Be explicit
            timeout=60 # Timeout for generation
        )
        log_info("Piper generation command finished successfully.")
        if gen_process.stderr: log_info(f"Piper Stderr: {gen_process.stderr.strip()}")
        # Piper might output useful info to stdout sometimes
        if gen_process.stdout: log_info(f"Piper Stdout: {gen_process.stdout.strip()}")

        # --- Validate Output File ---
        if not os.path.exists(temp_wav_path):
            # This shouldn't happen if check=True worked, but safety check
            log_error(f"Piper command succeeded but output WAV file is missing: {temp_wav_path}")
            raise FileNotFoundError(f"Piper command succeeded but WAV file missing: {temp_wav_path}.")

        if os.path.getsize(temp_wav_path) == 0:
             log_error(f"Piper command succeeded but generated WAV file is empty: {temp_wav_path}. Removing file.")
             # ** Corrected indentation for cleanup **
             try:
                 os.remove(temp_wav_path)
             except Exception as e_del:
                 log_warning(f"Failed to remove empty WAV file {temp_wav_path}: {e_del}")
             raise ValueError(f"Piper generated an empty WAV file: {temp_wav_path}.")

        log_info(f"Successfully generated audio file: {audio_id} ({os.path.getsize(temp_wav_path)} bytes)")
        # Add CORS headers to the success response
        response = jsonify({"status": "generated", "audio_id": audio_id})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except subprocess.TimeoutExpired:
         error_message = f"Piper generation command timed out after 60s."
         log_error(error_message)
         # ** Corrected indentation for cleanup **
         if os.path.exists(temp_wav_path):
              try:
                  os.remove(temp_wav_path)
                  log_info(f"Cleaned up timed-out file: {temp_wav_path}")
              except Exception as e_del:
                   log_warning(f"Failed to clean up timed-out file {temp_wav_path}: {e_del}")
         return make_error_response(error_message, 500)

    except subprocess.CalledProcessError as e:
        error_message = f"Piper generation command failed (Exit Code {e.returncode})"
        stderr_output = e.stderr.strip() if e.stderr else "No stderr captured."
        stdout_output = e.stdout.strip() if e.stdout else "No stdout captured."
        # ** Corrected indentation for command string **
        cmd_str = "N/A"
        try:
            cmd_str = ' '.join(e.cmd)
        except TypeError: # If cmd is not iterable list/tuple
            cmd_str = str(e.cmd)
        log_error(f"{error_message} - Command: {cmd_str}\nStderr: {stderr_output}\nStdout: {stdout_output}")
        # ** Corrected indentation for cleanup **
        if os.path.exists(temp_wav_path):
             try:
                 os.remove(temp_wav_path)
                 log_info(f"Cleaned up file after failed generation: {temp_wav_path}")
             except Exception as e_del:
                 log_warning(f"Failed to clean up file {temp_wav_path} after failed generation: {e_del}")
        # Include stderr in the response if available
        response_data = {"status": "error", "message": error_message}
        if stderr_output: response_data["stderr"] = stderr_output
        if stdout_output: response_data["stdout"] = stdout_output # Sometimes useful
        response = jsonify(response_data)
        response.status_code = 500
        return response

    except FileNotFoundError as e: # Catch specific error from manual raise
        log_error(f"File not found error during generation: {e}")
        return make_error_response(f"Generation succeeded but output file handling failed: {e}", 500)
    except ValueError as e: # Catch specific error from manual raise
        log_error(f"Value error during generation (e.g., empty file): {e}")
        return make_error_response(f"Generation failed: {e}", 500)
    except Exception as e:
        error_message = f"Unexpected error during generation: {type(e).__name__}: {e}"
        log_error(error_message, include_traceback=True)
        # ** Corrected indentation for cleanup **
        if os.path.exists(temp_wav_path):
             try:
                 os.remove(temp_wav_path)
                 log_info(f"Cleaned up file after unexpected error: {temp_wav_path}")
             except Exception as e_del:
                 log_warning(f"Failed to clean up file {temp_wav_path} after unexpected error: {e_del}")
        return make_error_response("An unexpected internal error occurred during audio generation.", 500)

# === Endpoint for Playback ===
@app.route('/play', methods=['POST', 'OPTIONS'])
def handle_play_request():
     # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    # Actual POST request handling
    if not request.is_json:
        return make_error_response("Request body must be JSON.", 415)

    try:
        req_data = request.get_json()
        if not req_data or 'audio_id' not in req_data:
            return make_error_response("Request body must be JSON with an 'audio_id' key.", 400)
        audio_id = req_data['audio_id']

        # Validate audio_id format to prevent path traversal / invalid chars
        # Allows UUIDs, letters, numbers, hyphen, MUST end in .wav
        if not isinstance(audio_id, str) or not re.match(r'^[a-zA-Z0-9-]+\.wav$', audio_id):
             return make_error_response("Invalid audio_id format. Must be alphanumeric/hyphen and end in .wav.", 400)

    except Exception as json_e:
        return make_error_response(f"Invalid JSON in request body: {json_e}", 400)

    # Construct full path and check existence
    temp_wav_path = os.path.abspath(os.path.join(AUDIO_TEMP_DIR, audio_id))

    # Security check: ensure resolved path is still within the temp dir
    if not temp_wav_path.startswith(AUDIO_TEMP_DIR + os.sep):
         log_error(f"SECURITY VIOLATION: Playback path traversal attempt detected. audio_id '{audio_id}' resolved to '{temp_wav_path}' which is outside allowed directory '{AUDIO_TEMP_DIR}'")
         return make_error_response("Invalid audio_id (Path traversal denied).", 400)

    log_info(f"Received PLAY request for audio_id: {audio_id} -> {temp_wav_path}")

    if not os.path.exists(temp_wav_path):
        log_error(f"Audio file not found for playback: {temp_wav_path} (audio_id: {audio_id})")
        return make_error_response(f"Audio file not found: {audio_id}", 404) # Not Found

    if not os.path.isfile(temp_wav_path):
         # Should not happen if generation worked, but indicates server state issue
         log_error(f"Server state error: Expected audio path is not a file: {temp_wav_path}")
         return make_error_response(f"Server error: Audio path is not a file for {audio_id}.", 500)

    # --- Play Audio ---
    try:
        aplay_command_list = [APLAY_CMD, temp_wav_path]
        log_info(f"Executing Playback command: {' '.join(aplay_command_list)}")
        play_process = subprocess.run(
            aplay_command_list,
            check=True, # Raise error on failure
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            timeout=300 # Timeout for playback (adjust if needed)
        )
        log_info(f"Playback command finished successfully for {audio_id}.")
        if play_process.stderr: log_info(f"Aplay Stderr: {play_process.stderr.strip()}")
        if play_process.stdout: log_info(f"Aplay Stdout: {play_process.stdout.strip()}")

        # Add CORS headers to the success response
        response = jsonify({"status": "played", "message": "Audio playback completed successfully."})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except subprocess.TimeoutExpired:
         error_message = f"Playback command timed out after 300s for {audio_id}."
         log_error(error_message)
         # Don't remove the file on playback timeout, it might still be valid
         return make_error_response(error_message, 500)

    except subprocess.CalledProcessError as e:
        error_message = f"Playback command failed (Exit Code {e.returncode})"
        stderr_output = e.stderr.strip() if e.stderr else "No stderr captured."
        stdout_output = e.stdout.strip() if e.stdout else "No stdout captured."
        # ** Corrected indentation for command string **
        cmd_str = "N/A"
        try:
            cmd_str = ' '.join(e.cmd)
        except TypeError:
            cmd_str = str(e.cmd)
        log_error(f"{error_message} - Command: {cmd_str}\nStderr: {stderr_output}\nStdout: {stdout_output}")
        # Include stderr in the response if available
        response_data = {"status": "error", "message": error_message}
        if stderr_output: response_data["stderr"] = stderr_output
        if stdout_output: response_data["stdout"] = stdout_output
        response = jsonify(response_data)
        response.status_code = 500
        return response

    except Exception as e:
        error_message = f"Unexpected error during playback of {audio_id}: {type(e).__name__}: {e}"
        log_error(error_message, include_traceback=True)
        return make_error_response("An unexpected internal error occurred during audio playback.", 500)

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="WSL Piper TTS Server: Generates audio using Piper and plays using aplay via HTTP requests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--voices-dir", required=True, help="Path to the base directory containing allowed Piper voice (.onnx / .onnx.json) files. This server restricts generation to files within this directory.")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address to bind the server to.")
    parser.add_argument("--port", type=int, required=True, help="Port number for the server to listen on (e.g., 5001).")
    parser.add_argument("--aplay-cmd", default="aplay", help="Path or command name for the audio playback utility (e.g., aplay).")
    parser.add_argument("--piper-cmd", default="piper", help="Path or command name for the Piper TTS executable.")
    parser.add_argument("--temp-dir", default=AUDIO_TEMP_DIR, help="Directory to store temporary generated WAV files.")
    parser.add_argument("--cuda", action='store_true', help="Pass the --cuda flag to the piper command for GPU acceleration.")
    parser.add_argument("--length-scale", type=float, default=1.0, help="Phoneme length scale factor passed to piper (e.g., <1 for faster speech, >1 for slower).")
    parser.add_argument("--waitress-threads", type=int, default=8, help="Number of worker threads for Waitress server (if installed).")

    args = parser.parse_args()

    # Apply arguments to global config variables
    ALLOWED_VOICE_DIR = args.voices_dir
    APLAY_CMD = args.aplay_cmd
    PIPER_CMD = args.piper_cmd
    USE_CUDA = args.cuda
    LENGTH_SCALE = args.length_scale
    AUDIO_TEMP_DIR = args.temp_dir # Update temp dir from args

    # Perform checks and setup
    initialize_server()

    # Print server configuration summary
    print("\n--- WSL Piper Aplay Server Configuration ---")
    print(f" Allowed Voices Dir: {ALLOWED_VOICE_DIR}")
    print(f" Piper Command:      {PIPER_CMD}")
    print(f" Aplay Command:      {APLAY_CMD}")
    print(f" Use CUDA:           {USE_CUDA}")
    print(f" Length Scale:       {LENGTH_SCALE}")
    print(f" Audio Temp Dir:     {AUDIO_TEMP_DIR}")
    print(f" Listening on:       http://{args.host}:{args.port}")
    print(f" Generate endpoint:  /generate (POST application/json {'{'} \"text\":\"...\", \"model_path\":\"relative/path/to/voice.onnx\" {'}'})")
    print(f" Playback endpoint:  /play     (POST application/json {'{'} \"audio_id\": \"<uuid>.wav\" {'}'})")
    print("-------------------------------------------\n", flush=True)

    # Start the server
    try:
        if HAVE_WAITRESS:
            log_info(f"Starting server with Waitress (Threads: {args.waitress_threads}) on {args.host}:{args.port}...")
            serve(app, host=args.host, port=args.port, threads=args.waitress_threads)
        else:
            log_info(f"Starting server with Flask development server on {args.host}:{args.port}...")
            # Note: Flask dev server is not recommended for production
            # Use debug=False for production fallback
            app.run(host=args.host, port=args.port, debug=False)
    except OSError as e:
        if "address already in use" in str(e).lower():
            log_error(f"Port {args.port} is already in use. Please choose a different port or stop the existing process.")
            sys.exit(f"FATAL: Address already in use: {args.host}:{args.port}")
        else:
            log_error(f"Failed to start server due to OS error: {e}", include_traceback=True)
            sys.exit("FATAL: Failed to start server.")
    except Exception as e:
        log_error(f"An unexpected error occurred while starting the server: {e}", include_traceback=True)
        sys.exit("FATAL: Unexpected error starting server.")
