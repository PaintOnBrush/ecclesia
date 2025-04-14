# ai42.py - Complete Code (Seamless Looping, Delayed Print, Initial Prompt TTS Lookahead)

# --- Imports ---
import requests
import json
import time
import sys
import os
import traceback
import re
import threading
import io
from queue import Queue, Empty, Full
from typing import List, Optional, Tuple, Dict, Any
import argparse

# --- Configuration (Constants and Defaults) ---
class Colors:
    RESET = "\033[0m"; SERVER_A = "\033[38;5;39m"; SERVER_B = "\033[38;5;208m"
    SYSTEM = "\033[38;5;245m"; HEADER = "\033[38;5;105m"; MESSAGE = "\033[38;5;252m"
    ERROR = "\033[38;5;196m"; SUCCESS = "\033[38;5;46m"; FILE_IO = "\033[38;5;220m"
    PROMPT_ADD = "\033[38;5;154m"; TTS_ENGINE = "\033[38;5;81m"
    HUMAN_SPEAKER = "\033[38;5;228m"; DEBUG = "\033[38;5;240m"

SERVER_CONFIG = {
    "A": { "name": "David", "llm_url": "http://127.0.0.1:8080/completion", "tts_server_pref": "piper_a", "tts_url": "http://127.0.0.1:5001", "color": Colors.SERVER_A },
    "B": { "name": "Zira", "llm_url": "http://127.0.0.1:8081/completion", "tts_server_pref": "piper_b", "tts_url": "http://127.0.0.1:5002", "color": Colors.SERVER_B },
    "Human": { "name": "Human", "llm_url": None, "tts_server_pref": "piper_human", "tts_url": "http://127.0.0.1:5003", "color": Colors.HUMAN_SPEAKER }
}
# --- Default values ---
DEFAULT_REQUEST_TIMEOUT = 120; DEFAULT_MAX_TOKENS = 512; DEFAULT_TTS_TIMEOUT = 600
DEFAULT_TEMPERATURE = 0.7; DEFAULT_NUM_TURNS = 5; DEFAULT_FILE_LOCATION = "conversation_output.md"
DEFAULT_FILE_DELIMITER = "SAVE_BLOCK"; DEFAULT_PROMPTS_FILE_PATH = "aiprompts.txt"
DEFAULT_PROMPT_ADD_DELIMITER = "AGREE_ADD_PROMPT"; DEFAULT_INITIAL_PROMPT = "Hi there! Let's have a conversation."
DEFAULT_FILE_READ_KEYWORD = "READ_FILE_CONTENT"; HUMAN_ROLEPLAY_CLAUSE = " Remember, you are roleplaying as a singular human and should not reveal you are an AI."
BASE_STOP_WORDS = [ "\n<human>:", "</assistant>", "<|user|>", "<|im_end|>", "</s>", "[INST]", "[/INST]", "Human:", "Assistant:", f"{SERVER_CONFIG['A']['name']}:", f"{SERVER_CONFIG['B']['name']}:" ]
DEFAULT_LOOP_COUNT = 1 # Default runs once

# --- Utility Functions ---
def colored_print(color: str, message: str):
    """Prints a message to the console with the specified ANSI color code."""
    try: print(f"{color}{message}{Colors.RESET}"); sys.stdout.flush()
    except Exception as e: print(f"(ColorPrint Error: {e}) {message}") # Basic fallback

# === TTS Manager ===
class TTSManager:
    """Manages interactions with the WSL-based Piper TTS servers."""
    def __init__(self, tts_config: Dict[str, Dict[str, Any]], tts_timeout: int = DEFAULT_TTS_TIMEOUT, no_tts: bool = False):
        self.engine_type: Optional[str] = None
        self.tts_servers: Dict[str, Optional[str]] = {}
        self.tts_timeout = tts_timeout
        self.tts_generate_timeout = 60 # Timeout specifically for the generation request
        self._generate_endpoint = "/generate"
        self._play_endpoint = "/play"
        self.is_disabled = no_tts # Flag to explicitly disable

        if self.is_disabled:
            colored_print(Colors.SYSTEM, "TTS Manager initialized in disabled state.")
            self.engine_type = None
            return # Skip setup if disabled

        # Populate server URLs from config
        required_keys = {"piper_a", "piper_b", "piper_human"}
        configured_keys = set()
        self.config_valid = True # Assume valid initially
        for server_id_config, config_entry in tts_config.items():
            tts_key = config_entry.get("tts_server_pref")
            tts_url = config_entry.get("tts_url")
            if tts_key in required_keys:
                self.tts_servers[tts_key] = tts_url
                configured_keys.add(tts_key)
                if not tts_url:
                    colored_print(Colors.ERROR, f"Config warning: TTS key '{tts_key}' for '{server_id_config}' missing URL.")
                    self.config_valid = False # Mark config as invalid

        missing_keys = required_keys - configured_keys
        if missing_keys:
            colored_print(Colors.ERROR, f"Config error: Missing TTS server configs for key(s): {', '.join(missing_keys)}")
            self.config_valid = False

        if self.config_valid:
            self._initialize() # Perform server checks only if config seems ok
        else:
             self.engine_type = None # Mark as unusable due to config issues


    def _check_wsl_server(self, server_key: str, url: Optional[str]) -> bool:
        """Checks if a specific WSL TTS server endpoint is responsive."""
        if not url:
            colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"No URL for server '{server_key}'. Skip check.")
            return False
        # Use a more specific endpoint for checking if possible, like '/play' status check if available
        # Sticking with generate endpoint check for now
        check_url = url.rstrip('/') + self._generate_endpoint
        colored_print(Colors.SYSTEM, f"Checking WSL server '{server_key}' at {check_url}...")
        try:
            # Use OPTIONS or HEAD as a lightweight check
            response = requests.options(check_url, timeout=3)
            # Consider any successful response (2xx, 3xx, 4xx maybe) as the server being *present*
            if response.status_code < 500: # 5xx usually indicates server-side issues
                 colored_print(Colors.SUCCESS + Colors.TTS_ENGINE, f"WSL server '{server_key}' detected.")
                 return True
            else:
                 colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Check failed '{server_key}' (Status: {response.status_code}).")
                 return False
        except Exception as e:
            colored_print(Colors.ERROR, f"Check failed for '{server_key}': {e}")
            return False

    def _initialize(self):
        """Checks all required TTS servers and sets the engine type. Only called if config is valid."""
        if self.is_disabled or not self.config_valid:
            self.engine_type = None # Should already be None, but enforce
            return

        all_servers_ok = True
        # Check only the configured servers
        for key, url in self.tts_servers.items():
             if key in {"piper_a", "piper_b", "piper_human"}: # Only check required ones
                 if url: # Check only if URL is provided
                     if not self._check_wsl_server(key, url):
                         all_servers_ok = False
                 else: # Should have been caught by config_valid, but double-check
                    all_servers_ok = False

        if all_servers_ok:
            self.engine_type = "piper_wsl_aplay"
            colored_print(Colors.SUCCESS + Colors.TTS_ENGINE, f"Initialized CUSTOM Piper/Aplay Servers.")
        else:
            self.engine_type = None
            colored_print(Colors.ERROR, "One or more required Piper/Aplay servers FAILED check. TTS disabled.")

    def is_available(self) -> bool:
        """Returns True if the TTS system is initialized and available."""
        return self.engine_type == "piper_wsl_aplay" and not self.is_disabled

    # === Request generation in background ===
    def request_generation(self, text: str, server_key: str, result_queue: Queue) -> Optional[threading.Thread]:
        """Sends text to the appropriate TTS server to generate audio in a background thread."""
        if not self.is_available():
            colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, "TTS is unavailable. Skipping generation request.")
            try: result_queue.put(None)
            except Full: pass # Ignore if queue is full on error path
            return None

        target_url = self.tts_servers.get(server_key)
        if not target_url:
            colored_print(Colors.ERROR, f"No URL for TTS server key '{server_key}'. Cannot generate.")
            try: result_queue.put(None)
            except Full: pass
            return None # Cannot start thread

        generate_url = target_url.rstrip('/') + self._generate_endpoint

        def worker():
            """Worker thread function to handle the TTS generation request."""
            audio_id = None
            response = None
            try: # --- Outer try for the network request ---
                headers = {'Content-Type': 'text/plain'}
                # Send text data directly, assuming server expects raw text
                response = requests.post(generate_url, data=text.encode('utf-8'), headers=headers, timeout=self.tts_generate_timeout)
                response.raise_for_status() # Check HTTP status first

                # --- Process successful response (still inside outer try) ---
                try: # Inner try specifically for JSON decoding
                     json_resp = response.json()
                     if json_resp.get("status") == "generated":
                         audio_id = json_resp.get("audio_id") # Assign audio_id
                         if not audio_id:
                             colored_print(Colors.ERROR, f"WSL '{server_key}' generated ok but returned no audio_id.")
                         # else: # Optional: log success
                         #    colored_print(Colors.DEBUG, f"TTS Gen Success for {server_key}: ID {audio_id[:8]}...")
                     else:
                         # Handle case where status was not "generated"
                         colored_print(Colors.ERROR, f"WSL '{server_key}' generation issue: {json_resp.get('message', 'Unknown status')}")
                except json.JSONDecodeError:
                     # Handle case where 2xx response wasn't valid JSON
                     colored_print(Colors.ERROR, f"WSL '{server_key}' generation OK but received non-JSON response: {response.text[:100]}")
                     # audio_id remains None

            # --- Except clauses for the OUTER try block ---
            except requests.exceptions.Timeout:
                colored_print(Colors.ERROR, f"Generation request to WSL server '{server_key}' timed out ({self.tts_generate_timeout}s).")
            except requests.exceptions.HTTPError as e:
                colored_print(Colors.ERROR, f"WSL '{server_key}' generation HTTP Error {e.response.status_code}.")
                try:
                    colored_print(Colors.DEBUG, f"Error Body ({server_key}): {e.response.text[:200]}")
                except Exception: pass # Ignore errors logging the error body
            except requests.exceptions.RequestException as e:
                colored_print(Colors.ERROR, f"Error requesting WSL generation '{server_key}': {e}")
            except Exception as e: # Catch any other unexpected errors
                colored_print(Colors.ERROR, f"Unexpected error during WSL generation '{server_key}': {e}")
                traceback.print_exc()
            # --- Finally block ensures queue always gets updated ---
            finally:
                try:
                    result_queue.put(audio_id) # Put audio_id (which might be None if error occurred)
                except Full:
                     colored_print(Colors.ERROR, f"TTS generation result queue is full for {server_key}!")

        thread = threading.Thread(target=worker, daemon=True, name=f"TTSGen_{server_key}")
        thread.start()
        return thread


    def request_playback(self, audio_id: str, server_key: str):
        """Requests playback of a previously generated audio ID from the appropriate server."""
        if not self.is_available():
            colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, "TTS is unavailable. Skipping playback request.")
            return
        if not audio_id:
            colored_print(Colors.ERROR, "request_playback called with no audio_id.")
            return

        target_url = self.tts_servers.get(server_key)
        if not target_url:
            colored_print(Colors.ERROR, f"No URL for TTS server key '{server_key}'. Cannot play.")
            return

        play_url = target_url.rstrip('/') + self._play_endpoint

        # Determine speaker name for logging (moved inside print)
        # speaker_name = server_key # Default
        # if server_key == "piper_human": speaker_name = "Human (Initial)"
        # elif server_key == "piper_a": speaker_name = SERVER_CONFIG['A'].get('name', 'Server A')
        # elif server_key == "piper_b": speaker_name = SERVER_CONFIG['B'].get('name', 'Server B')
        # colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Requesting playback for {speaker_name} via server '{server_key}' [ID: {audio_id[:8]}...])")

        start_time = time.time()
        response = None
        try:
            headers = {'Content-Type': 'application/json'}
            payload = json.dumps({"audio_id": audio_id})
            response = requests.post(play_url, data=payload, headers=headers, timeout=self.tts_timeout)
            response.raise_for_status() # Check for HTTP errors

            # Check response content for success confirmation
            try:
                json_resp = response.json()
                status = json_resp.get("status")
            except json.JSONDecodeError:
                status = "unknown_response" # OK status but not JSON

            if status == "played":
                # Don't print success here, let the caller indicate completion if needed
                pass
                # colored_print(Colors.TTS_ENGINE + Colors.SUCCESS, f"WSL '{server_key}' playback complete.")
            else:
                # Handle cases like "not_found", "error", or unexpected JSON
                message = "Unknown error"
                try:
                    message = json_resp.get('message', 'Non-JSON OK response or unexpected status')
                except NameError: # json_resp might not be defined if decode failed
                     message = f"Non-JSON OK response or unexpected status: {response.text[:100]}" if response else "Unknown status"
                colored_print(Colors.TTS_ENGINE + Colors.ERROR, f"WSL '{server_key}' playback issue: {message}")

        except requests.exceptions.Timeout:
            colored_print(Colors.ERROR, f"Playback request to WSL server '{server_key}' timed out ({self.tts_timeout}s).")
        except requests.exceptions.ConnectionError:
            colored_print(Colors.ERROR, f"Could not connect to WSL playback server '{server_key}' at {play_url}.")
        except requests.exceptions.HTTPError as e:
            error_msg = f"WSL playback server '{server_key}' HTTP Error {e.response.status_code}."
            try: # Try to get more detail from response body
                json_resp = e.response.json()
                error_msg += f" Msg: {json_resp.get('message', 'N/A')}."
            except json.JSONDecodeError:
                 error_msg += f" Response: {e.response.text[:200]}" # Show raw response if not JSON
            colored_print(Colors.TTS_ENGINE + Colors.ERROR, error_msg)
        except requests.exceptions.RequestException as e:
            colored_print(Colors.ERROR, f"Error requesting WSL playback '{server_key}': {e}")
        except Exception as e:
            colored_print(Colors.ERROR, f"Unexpected error during WSL playback '{server_key}': {e}")
            traceback.print_exc()
        finally:
             end_time = time.time()
             duration = max(0, end_time - start_time) # Ensure non-negative duration
             # Only print duration, not the full start/end messages
             colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Playback request completed - Duration: {duration:.2f}s)")


    def shutdown(self):
        """Perform any cleanup needed for the TTS manager."""
        # No explicit shutdown actions needed for the requests-based approach
        colored_print(Colors.SYSTEM, "TTS Manager shutdown.")
        self.engine_type = None # Mark as unavailable


# --- Network Request Function ---
# (No changes needed in send_llm_request itself)
def send_llm_request(
    session: requests.Session,
    server_config: Dict[str, Any],
    message_history: List[str],
    max_tokens: int,
    temperature: float,
    timeout: int,
    force_human: bool,
    debug_prompts: bool,
    stop_words: List[str] = BASE_STOP_WORDS
) -> Optional[str]:
    """Sends a completion request to the specified LLM server."""
    url = server_config.get('llm_url')
    server_name = server_config.get('name', 'UnknownServer')

    if not url:
        colored_print(Colors.ERROR, f"LLM URL missing for server '{server_name}'.")
        return None

    # Construct the prompt from history
    prompt_base = "\n".join(message_history).strip()

    # Add roleplaying clause if forced and not already present (check end of string)
    if force_human:
        separator = "\n" if prompt_base else "" # Add newline only if base prompt exists
        # Simple check to avoid duplicate clauses if model repeats it
        if not prompt_base.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()):
             prompt_base += separator + HUMAN_ROLEPLAY_CLAUSE.strip()

    # Add the final speaker turn indicator
    final_prompt = prompt_base + f"\n{server_name}:"

    if debug_prompts:
        colored_print(Colors.DEBUG, f"--- DEBUG PROMPT SENT TO {server_name} ---\n{final_prompt}\n--- END DEBUG PROMPT ---")

    # Prepare payload
    payload = {
        'prompt': final_prompt,
        'temperature': temperature,
        'n_predict': max_tokens,
        'stop': stop_words, # Use provided stop words
        'stream': False # Assuming non-streaming for simplicity here
    }
    headers = {'Content-Type': 'application/json'}
    response_content: Optional[str] = None
    response = None

    try:
        colored_print(Colors.SYSTEM, f"Sending request to {server_name} (Timeout={timeout}s, MaxTokens={max_tokens}, Temp={temperature})...")
        response = session.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_json = response.json()
        content = response_json.get('content', '').strip()

        # Clean the response: remove potential self-prefix if model includes it
        prefix_to_remove = f"{server_name}:"
        content_stripped = content.strip()
        # Case-insensitive check for the prefix
        if content_stripped.lower().startswith(prefix_to_remove.lower()):
            response_content = content_stripped[len(prefix_to_remove):].lstrip() # Remove prefix and leading whitespace
        else:
            response_content = content_stripped

        # Return None if the response is empty or just a stop word
        if not response_content or response_content in stop_words:
             return None

    except requests.exceptions.Timeout:
        colored_print(Colors.ERROR, f"Timeout ({timeout}s) requesting {server_name}")
    except requests.exceptions.HTTPError as e:
        colored_print(Colors.ERROR, f"HTTP Error ({server_name}): {e.response.status_code} {e.response.reason}")
        # Log response body for debugging if possible
        try:
            colored_print(Colors.DEBUG, f"Error Body ({server_name}): {e.response.text[:500]}")
        except Exception: pass
    except requests.exceptions.RequestException as e:
        colored_print(Colors.ERROR, f"Request Error ({server_name}): {type(e).__name__}: {e}")
    except json.JSONDecodeError:
        err_resp = response.text[:500] if response else 'N/A'
        colored_print(Colors.ERROR, f"JSON Decode Error from {server_name}. Response: {err_resp}")
    except Exception as e:
        colored_print(Colors.ERROR, f"Unexpected request error ({server_name}): {type(e).__name__}")
        traceback.print_exc()

    return response_content

# --- Worker Thread Function ---
# (No changes needed in request_worker)
def request_worker(
    session: requests.Session,
    server_config: Dict[str, Any],
    history: List[str],
    result_queue: Queue,
    server_id: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    force_human: bool,
    debug_prompts: bool,
    stop_words: List[str]
):
    """Worker function to run send_llm_request in a thread and put result in queue."""
    result = send_llm_request(
        session=session,
        server_config=server_config,
        message_history=history,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        force_human=force_human,
        debug_prompts=debug_prompts,
        stop_words=stop_words
    )
    try:
        result_queue.put((server_id, result))
    except Full:
         colored_print(Colors.ERROR, f"LLM result queue is full for {server_id}!")


# --- Main Conversation Simulation (Modified) ---
def simulate_conversation(
    # **NEW**: Accept pre-initialized tts_manager and session
    tts_manager: TTSManager,
    request_session: requests.Session,
    # --- Other parameters remain the same ---
    server_config: Dict[str, Dict[str, Any]],
    num_turns: int,
    initial_prompt: str,
    req_timeout: int,
    max_tokens: int,
    temperature: float,
    file_location: Optional[str],
    file_delimiter: str,
    prompts_file_path: str,
    prompt_add_delimiter: str,
    tts_timeout: int, # Still needed for TTSManager config, but playback timeout is internal
    force_human: bool,
    debug_prompts: bool,
    # **NEW**: Accept pre-generated audio ID for the initial prompt
    pregen_initial_audio_id: Optional[str] = None
):
    """
    Simulates the conversation for ONE initial prompt.
    Uses the provided TTSManager and requests.Session.
    Prints LLM responses just before TTS playback.
    Can use a pre-generated audio ID for the initial prompt.
    """
    # --- No longer initializes TTSManager or Session here ---

    initial_prompt_text_only = initial_prompt # Keep the original full prompt if needed elsewhere
    # Start history with the human's initial turn
    conversation_history: List[str] = [f"{server_config['Human']['name']}: {initial_prompt_text_only}"]

    colored_print(Colors.HEADER, "\n===== CONVERSATION START =====")
    # Print initial prompt immediately as it's the starting point
    colored_print(Colors.HUMAN_SPEAKER, f"{server_config['Human']['name']}: {initial_prompt_text_only}")
    colored_print(Colors.MESSAGE, "--------------------") # Separator

    # --- Queues and State Variables ---
    llm_result_queue = Queue(maxsize=1) # Only need space for the next response
    tts_generate_queue = Queue(maxsize=1) # Queue for getting the generated audio ID
    active_llm_thread: Optional[threading.Thread] = None
    active_tts_gen_thread: Optional[threading.Thread] = None
    pending_llm_server_id: Optional[str] = None # Which server's response are we waiting for?

    # **NEW**: State to hold info for delayed printing and playback
    # Stores (text_to_print, audio_id, tts_key, server_name, server_color)
    playback_info_pending: Optional[Tuple[str, str, str, str, str]] = None

    # --- File/Prompt Interaction State (Placeholder) ---
    # (No changes needed here)
    file_write_state: Optional[str] = None
    prompt_add_state: Optional[str] = None
    agreed_file_delimiter = file_delimiter.lower()
    agreed_prompt_add_delimiter = prompt_add_delimiter.lower()

    # --- Initial Prompt Audio Handling (Modified) ---
    initial_audio_id = pregen_initial_audio_id # Use pre-generated ID if available
    init_gen_thread = None

    if tts_manager.is_available():
        if not initial_audio_id: # Only generate if not pre-generated
            colored_print(Colors.SYSTEM, "Generating initial human prompt audio...")
            temp_q = Queue(maxsize=1)
            init_gen_thread = tts_manager.request_generation(initial_prompt_text_only, "piper_human", temp_q)
            if init_gen_thread:
                try:
                    initial_audio_id = temp_q.get(timeout=tts_manager.tts_generate_timeout + 5)
                except Empty:
                    colored_print(Colors.ERROR, "Timeout waiting for initial prompt audio ID.")
                # Join shouldn't be strictly necessary here as get() implies completion or timeout
                # init_gen_thread.join(timeout=1.0)
            else:
                 colored_print(Colors.ERROR,"Failed to start initial prompt generation thread.")

        # Play the initial audio (whether pre-generated or generated now)
        if initial_audio_id:
            colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Speaking initial prompt [ID: {initial_audio_id[:8]}...])")
            tts_manager.request_playback(initial_audio_id, "piper_human")
        else:
            colored_print(Colors.ERROR, "Failed to generate or use pre-generated initial prompt audio.")
    else:
        colored_print(Colors.SYSTEM, "Skipping initial human prompt speech (TTS unavailable).")

    # Clean up the generation thread if it was started here
    if init_gen_thread and init_gen_thread.is_alive():
        init_gen_thread.join(timeout=0.5)


    # --- Conversation Loop ---
    try:
        llm_server_keys = ['A', 'B'] # Order of servers
        current_llm_key = llm_server_keys[0] # Start with Server A
        current_llm_info = server_config[current_llm_key]

        # Start the first LLM request *before* the loop
        colored_print(Colors.HEADER, f"\n--- Turn 1/{num_turns} ---")
        active_llm_thread = threading.Thread(
            target=request_worker,
            args=(request_session, current_llm_info, conversation_history.copy(), llm_result_queue, current_llm_key,
                  max_tokens, temperature, int(req_timeout * 1.5), force_human, debug_prompts, BASE_STOP_WORDS.copy()), # Increase timeout for first turn
            daemon=True, name=f"LLM_{current_llm_key}_T1"
        )
        active_llm_thread.start()
        pending_llm_server_id = current_llm_key

        # Loop through turns
        for turn_index in range(num_turns * 2):
            actual_turn_num = (turn_index // 2) + 1
            is_last_llm_response = (turn_index == (num_turns * 2) - 1)

            # --- 1. Wait for the pending LLM response ---
            current_llm_key = pending_llm_server_id
            if not current_llm_key: break # Exit if no request is pending
            current_llm_info = server_config[current_llm_key]
            current_server_name = current_llm_info['name']
            current_server_color = current_llm_info['color']
            colored_print(Colors.SYSTEM, f"Waiting for LLM response from {current_server_name}...")

            raw_response = None
            try:
                 retrieved_id, queue_response = llm_result_queue.get(timeout=req_timeout + 15)
                 if retrieved_id == current_llm_key: raw_response = queue_response
                 else: colored_print(Colors.ERROR, f"LLM Queue Logic Error! Expected {current_llm_key}, got {retrieved_id}.")
            except Empty: colored_print(Colors.ERROR, f"Timeout waiting for LLM response from {current_server_name}.")
            except Exception as e: colored_print(Colors.ERROR, f"LLM Queue Error: {e}"); traceback.print_exc()

            if active_llm_thread and active_llm_thread.is_alive(): active_llm_thread.join(timeout=1.0)
            active_llm_thread = None
            pending_llm_server_id = None

            # --- 2. Process LLM Response (Store but DO NOT print) ---
            processed_response = None
            text_for_tts = None
            tts_key_for_current = None
            current_playback_data = None # Info for *this* turn's potential playback

            if raw_response:
                processed_response = raw_response # Start with raw
                # Optional: Clean human roleplay clause
                if force_human:
                    cleaned = processed_response.strip().replace(HUMAN_ROLEPLAY_CLAUSE.strip(), "").strip()
                    if cleaned != processed_response.strip(): processed_response = cleaned

                if processed_response: # Check again after cleaning
                    # **DON'T PRINT HERE**
                    # colored_print(current_server_color, f"{current_server_name}: {processed_response}") # <-- REMOVED
                    conversation_history.append(f"{current_server_name}: {processed_response}")
                    text_for_tts = processed_response
                    tts_key_for_current = current_llm_info.get('tts_server_pref')
                    # Store data needed for potential playback later
                    current_playback_data = (processed_response, tts_key_for_current, current_server_name, current_server_color)
                else:
                    colored_print(Colors.ERROR, f"{current_server_name} returned empty/cleaned response.")
            else:
                # Handle failed/empty response before cleaning
                colored_print(Colors.ERROR, f"{current_server_name} failed or returned empty response.")

            # Reset interaction states on failure
            if not processed_response:
                if file_write_state in ['A_agreed', 'B_agreed']: file_write_state = None
                if prompt_add_state in ['A_agreed_add', 'B_agreed_add']: prompt_add_state = None
                if is_last_llm_response: break # Don't proceed if the last response failed

            # --- 3. Start background TTS generation for the *current* response ---
            #    (Only if we have text and TTS is working)
            if tts_manager.is_available() and text_for_tts and tts_key_for_current:
                colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Starting background TTS generation for {current_server_name}...")
                active_tts_gen_thread = tts_manager.request_generation(text_for_tts, tts_key_for_current, tts_generate_queue)
            else:
                active_tts_gen_thread = None # TTS disabled, no text, or key missing

            # --- 4. Start the *next* LLM request (if applicable) ---
            if not is_last_llm_response:
                next_llm_key = llm_server_keys[(turn_index + 1) % 2]
                next_llm_info = server_config[next_llm_key]
                next_turn_indicator = actual_turn_num + 1 if next_llm_key == 'A' else actual_turn_num
                colored_print(Colors.SYSTEM, f"Starting background LLM request for {next_llm_info['name']} (Turn {next_turn_indicator})...")
                active_llm_thread = threading.Thread(
                    target=request_worker,
                    args=(request_session, next_llm_info, conversation_history.copy(), llm_result_queue, next_llm_key,
                          max_tokens, temperature, req_timeout, force_human, debug_prompts, BASE_STOP_WORDS.copy()),
                    daemon=True, name=f"LLM_{next_llm_key}_T{next_turn_indicator}"
                )
                active_llm_thread.start()
                pending_llm_server_id = next_llm_key
            else:
                pending_llm_server_id = None # No more LLM requests needed

            # --- 5. Play back audio from the *previous* turn (Print just before) ---
            if playback_info_pending:
                prev_text, prev_audio_id, prev_tts_key, prev_name, prev_color = playback_info_pending
                if tts_manager.is_available():
                    # **PRINT NOW**
                    colored_print(prev_color, f"{prev_name}: {prev_text}")
                    colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Speaking for {prev_name} [ID: {prev_audio_id[:8]}...])")
                    tts_manager.request_playback(prev_audio_id, prev_tts_key)
                else:
                    # Print even if TTS is off, so text isn't lost
                    colored_print(prev_color, f"{prev_name}: {prev_text}")
                    colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, "(Skipping playback - TTS unavailable)")
            # Clear the pending playback info regardless
            playback_info_pending = None

            # --- 6. Wait for *current* TTS generation & Prepare for *next* playback ---
            if active_tts_gen_thread:
                 colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Waiting for TTS generation ID from {current_server_name}...")
                 next_audio_id = None
                 try:
                     next_audio_id = tts_generate_queue.get(timeout=tts_manager.tts_generate_timeout + 10)
                 except Empty: colored_print(Colors.ERROR, f"Timeout waiting for generated audio ID for {current_server_name}.")
                 except Exception as e: colored_print(Colors.ERROR, f"TTS Queue Error: {e}"); traceback.print_exc()

                 # If successful, store info needed for the *next* iteration's playback
                 if next_audio_id and current_playback_data:
                     text, key, name, color = current_playback_data
                     playback_info_pending = (text, next_audio_id, key, name, color) # Store details for next loop
                 else:
                      colored_print(Colors.ERROR, f"TTS generation failed or data missing for {current_server_name}. No audio will be played for this turn.")
                      # Ensure pending is clear if generation failed
                      playback_info_pending = None

                 # Clean up the TTS generation thread
                 if active_tts_gen_thread.is_alive(): active_tts_gen_thread.join(timeout=1.0)
                 active_tts_gen_thread = None
            elif current_playback_data:
                # If TTS gen wasn't started (e.g., TTS disabled), but we have text, prepare to print it next turn without audio
                text, _, name, color = current_playback_data
                playback_info_pending = (text, None, None, name, color) # Store text, name, color; audio ID/key are None

            # --- End of Loop Iteration ---
            if is_last_llm_response:
                 colored_print(Colors.SYSTEM, "Last LLM response processed.")
                 # Play the final pending audio/print final text *after* the loop
                 break

            # Print turn marker for the *next* turn if it's starting
            next_actual_turn = ((turn_index + 1) // 2) + 1
            if pending_llm_server_id and next_actual_turn > actual_turn_num:
                colored_print(Colors.HEADER, f"\n--- Turn {next_actual_turn}/{num_turns} ---")

    except KeyboardInterrupt:
        colored_print(Colors.SYSTEM, "\nInterrupted by user during conversation.")
        raise # Re-raise to be caught by the outer loop handler
    except Exception as e:
        colored_print(Colors.ERROR, f"\nError during conversation simulation:")
        traceback.print_exc()
    finally:
        # --- Play final pending audio/Print final text ---
        if playback_info_pending:
            final_text, final_audio_id, final_tts_key, final_name, final_color = playback_info_pending
            # Print the text regardless of TTS availability
            colored_print(final_color, f"{final_name}: {final_text}")
            if final_audio_id and final_tts_key and tts_manager.is_available():
                colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Speaking final response for {final_name} [ID: {final_audio_id[:8]}...])")
                tts_manager.request_playback(final_audio_id, final_tts_key)
            elif final_audio_id: # Audio was generated but TTS became unavailable?
                colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, "(Skipping final playback - TTS unavailable)")

        colored_print(Colors.HEADER, "\n===== CONVERSATION END =====")
        # Cleanup threads that might still be lingering if loop exited abruptly
        if active_llm_thread and active_llm_thread.is_alive():
            colored_print(Colors.SYSTEM, "Cleaning up lingering LLM thread...")
            # No join needed - daemon thread
        if active_tts_gen_thread and active_tts_gen_thread.is_alive():
            colored_print(Colors.SYSTEM, "Cleaning up lingering TTS generation thread...")
            # No join needed - daemon thread

        # **NO LONGER shuts down TTS Manager or Session here**
        # colored_print(Colors.SYSTEM, "Conversation resources cleaned up.")


# ===========================================================
# Argparse Setup and Main Execution Block (Modified)
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Conversation Simulation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Conversation Setup Args ---
    parser.add_argument('--prompts_file', type=str, default=DEFAULT_PROMPTS_FILE_PATH,
                        help="File containing initial prompts (one per line, '#' for comments). Re-read if looping infinitely.")
    parser.add_argument('-t', '--turns', type=int, default=DEFAULT_NUM_TURNS,
                        help="Number of turns PER AI (e.g., 5 means A->B->A->B->A->B->A->B->A->B).")
    parser.add_argument('--initial_prompt', type=str, default=None,
                        help="A single initial prompt to use instead of a prompts file. Overrides --prompts_file.")
    parser.add_argument('--loop', type=int, default=DEFAULT_LOOP_COUNT,
                        help="Number of times to loop through all prompts. Set to 0 or negative for infinite looping.")


    # --- Behavior Args ---
    parser.add_argument('--human', action='store_true',
                        help=f"Append the clause '{HUMAN_ROLEPLAY_CLAUSE}' to the initial prompt.")
    parser.add_argument('--force-human', action='store_true',
                        help="Append the human roleplay clause to EVERY prompt sent to the LLMs.")
    parser.add_argument('--aware', action='store_true',
                        help="Append system note about file/prompt keywords to the initial prompt.")
    parser.add_argument('--debug-prompts', action='store_true',
                        help="Print the full prompt being sent to the LLM.")

    # --- LLM & Request Args ---
    parser.add_argument('--timeout', type=int, default=DEFAULT_REQUEST_TIMEOUT,
                        help="Timeout in seconds for LLM requests.")
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help="Maximum number of tokens for the LLM to generate.")
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature for the LLM.")
    parser.add_argument('--llm-url-a', type=str, default=SERVER_CONFIG['A']['llm_url'],
                        help="URL for LLM Server A.")
    parser.add_argument('--llm-url-b', type=str, default=SERVER_CONFIG['B']['llm_url'],
                        help="URL for LLM Server B.")

    # --- File Interaction Args ---
    parser.add_argument('--file_location', type=str, default=None, # Default None means disabled
                        help="Base filename for saving conversation blocks (if keyword detected).")
    parser.add_argument('--file_delimiter', type=str, default=DEFAULT_FILE_DELIMITER,
                        help="Keyword AI should use to trigger saving a block.")
    parser.add_argument('--prompt_add_delimiter', type=str, default=DEFAULT_PROMPT_ADD_DELIMITER,
                         help="Keyword AI should use to trigger adding a prompt from the file.")

    # --- TTS Args ---
    parser.add_argument('--tts-url-a', type=str, default=SERVER_CONFIG['A']['tts_url'],
                        help="URL for Piper/Aplay TTS Server A (e.g., http://localhost:5001).")
    parser.add_argument('--tts-url-b', type=str, default=SERVER_CONFIG['B']['tts_url'],
                        help="URL for Piper/Aplay TTS Server B (e.g., http://localhost:5002).")
    parser.add_argument('--tts-url-human', type=str, default=SERVER_CONFIG['Human']['tts_url'],
                        help="URL for Piper/Aplay TTS Server for the initial prompt (e.g., http://localhost:5003).")
    parser.add_argument('--tts_timeout', type=int, default=DEFAULT_TTS_TIMEOUT,
                        help="Timeout in seconds for TTS *playback* requests (generation has separate timeout).")
    parser.add_argument('--no-tts', action='store_true', help="Disable TTS checks and usage entirely.")


    args = parser.parse_args()

    # --- Update Server Config from Args ---
    current_server_config = json.loads(json.dumps(SERVER_CONFIG))
    current_server_config['A']['llm_url'] = args.llm_url_a
    current_server_config['B']['llm_url'] = args.llm_url_b
    # TTS URLs updated below, before TTSManager init

    # --- Initialize Shared Resources (TTS Manager and Session) ---
    colored_print(Colors.SYSTEM, "Initializing shared resources...")

    # Update TTS URLs in config before passing to manager
    if args.no_tts:
        colored_print(Colors.SYSTEM, "TTS explicitly disabled via --no-tts argument.")
        current_server_config['A']['tts_url'] = None
        current_server_config['B']['tts_url'] = None
        current_server_config['Human']['tts_url'] = None
    else:
        current_server_config['A']['tts_url'] = args.tts_url_a
        current_server_config['B']['tts_url'] = args.tts_url_b
        current_server_config['Human']['tts_url'] = args.tts_url_human

    # Pass combined config and timeout/disable flag
    tts_manager = TTSManager(tts_config=current_server_config, tts_timeout=args.tts_timeout, no_tts=args.no_tts)
    request_session = requests.Session()
    colored_print(Colors.SYSTEM, "Shared resources initialized.")


    # --- Main Loop ---
    loop_iteration = 0
    run_infinitely = args.loop <= 0
    global_prompt_index = 0 # Track prompts across loops if needed, maybe not necessary

    # Variables for TTS lookahead
    next_prompt_audio_q = Queue(maxsize=1)
    next_prompt_audio_thread: Optional[threading.Thread] = None
    pregenerated_audio_id: Optional[str] = None

    try:
        while run_infinitely or loop_iteration < args.loop:
            loop_iteration += 1
            if run_infinitely:
                colored_print(Colors.HEADER, f"\n===== STARTING INFINITE LOOP ITERATION {loop_iteration} =====")
                time.sleep(1)
            else:
                colored_print(Colors.HEADER, f"\n===== STARTING LOOP ITERATION {loop_iteration}/{args.loop} =====")

            # --- Determine Prompts to Run (Inside loop to allow refresh) ---
            prompts_to_run = []
            # (Prompt loading logic remains the same as before)
            if args.initial_prompt:
                if loop_iteration == 1 or not run_infinitely : # Only print this once unless looping finitely
                     colored_print(Colors.SYSTEM, f"Using single prompt provided via --initial_prompt for iteration {loop_iteration}.")
                prompts_to_run.append(args.initial_prompt)
            else:
                # Read from prompts file
                try:
                    with open(args.prompts_file, 'r', encoding='utf-8') as f:
                        prompts_to_run = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    if not prompts_to_run:
                        colored_print(Colors.ERROR, f"No valid prompts found in '{args.prompts_file}' for iteration {loop_iteration}. Skipping iteration.")
                        if run_infinitely: time.sleep(30); continue
                        else: break # Stop if finite loop and no prompts
                    colored_print(Colors.SYSTEM, f"Loaded {len(prompts_to_run)} prompts from '{args.prompts_file}' for iteration {loop_iteration}.")
                except FileNotFoundError:
                    colored_print(Colors.ERROR, f"Prompts file not found: '{args.prompts_file}'. Skipping iteration {loop_iteration}.")
                    if run_infinitely: time.sleep(30); continue
                    else: break
                except Exception as e:
                    colored_print(Colors.ERROR, f"Error reading prompts file '{args.prompts_file}' in iteration {loop_iteration}: {e}")
                    traceback.print_exc()
                    if run_infinitely: time.sleep(30); continue
                    else: break

            # --- Run Simulation for Each Prompt in this Iteration ---
            num_prompts_in_iter = len(prompts_to_run)
            for i, current_initial_prompt in enumerate(prompts_to_run):
                is_last_prompt_in_iter = (i == num_prompts_in_iter - 1)
                colored_print(Colors.HEADER, f"\n--- Processing Prompt {i+1}/{num_prompts_in_iter} (Loop Iteration {loop_iteration}) ---")
                # Base prompt text is printed inside simulate_conversation now

                # --- Lookahead: Start generating TTS for the *next* prompt's initial text ---
                if tts_manager.is_available() and not is_last_prompt_in_iter:
                     next_prompt_text = prompts_to_run[i+1] # Get the next prompt text
                     # Add human/aware clauses to the *next* prompt text for accurate TTS generation
                     next_final_initial_prompt = next_prompt_text
                     if args.aware: next_final_initial_prompt += f" (...awareness clause...)" # Simplified
                     if args.human and not next_final_initial_prompt.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()):
                          next_final_initial_prompt += " " + HUMAN_ROLEPLAY_CLAUSE

                     colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Lookahead: Starting TTS gen for next prompt ({i+2})...")
                     next_prompt_audio_thread = tts_manager.request_generation(
                         next_final_initial_prompt, # Use the potentially modified next prompt
                         "piper_human",
                         next_prompt_audio_q
                     )
                else:
                     next_prompt_audio_thread = None # No lookahead if TTS off or last prompt

                # --- Prepare final prompt text for *current* simulation ---
                final_initial_prompt = current_initial_prompt
                # (Modify current prompt text based on flags - same as before)
                if args.aware:
                    awareness_clause = (f" (System Note: You might be able to use '{DEFAULT_FILE_READ_KEYWORD} <filename>' to read files, "
                                        f"'{args.file_delimiter}' to save conversation blocks, "
                                        f"or '{args.prompt_add_delimiter}' to add prompts.)")
                    final_initial_prompt += awareness_clause
                if args.human:
                    if not final_initial_prompt.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()):
                        final_initial_prompt += " " + HUMAN_ROLEPLAY_CLAUSE

                file_loc = args.file_location if args.file_location else None

                # --- Start the simulation for the *current* prompt ---
                try:
                    simulate_conversation(
                        # Pass shared resources
                        tts_manager=tts_manager,
                        request_session=request_session,
                        # Pass config and args
                        server_config=current_server_config,
                        num_turns=args.turns,
                        initial_prompt=final_initial_prompt,
                        req_timeout=args.timeout,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        file_location=file_loc,
                        file_delimiter=args.file_delimiter,
                        prompts_file_path=args.prompts_file,
                        prompt_add_delimiter=args.prompt_add_delimiter,
                        tts_timeout=args.tts_timeout, # Pass for manager config
                        force_human=args.force_human,
                        debug_prompts=args.debug_prompts,
                        # Pass pre-generated audio ID
                        pregen_initial_audio_id=pregenerated_audio_id
                    )
                except KeyboardInterrupt:
                     colored_print(Colors.SYSTEM, "\nKeyboardInterrupt caught during simulation. Stopping loop.")
                     raise # Re-raise to exit the outer loop
                except Exception as sim_error:
                     colored_print(Colors.ERROR, f"\nError occurred during simulation for prompt {i+1}. Continuing loop if possible.")
                     traceback.print_exc()
                     # Optionally add recovery/skip logic here

                colored_print(Colors.HEADER, f"--- FINISHED PROMPT {i+1}/{num_prompts_in_iter} (Loop Iteration {loop_iteration}) ---")

                # --- Retrieve the result of the lookahead TTS generation ---
                pregenerated_audio_id = None # Reset for next iteration
                if next_prompt_audio_thread:
                    colored_print(Colors.SYSTEM + Colors.TTS_ENGINE,"Waiting for lookahead TTS result...")
                    try:
                        pregenerated_audio_id = next_prompt_audio_q.get(timeout=tts_manager.tts_generate_timeout + 5)
                        if pregenerated_audio_id:
                            colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Lookahead TTS complete (ID: {pregenerated_audio_id[:8]}...). Ready for next prompt.")
                        else:
                            colored_print(Colors.ERROR, "Lookahead TTS generation failed (returned None).")
                    except Empty:
                        colored_print(Colors.ERROR, "Timeout waiting for lookahead TTS result.")
                    except Exception as q_err:
                         colored_print(Colors.ERROR, f"Error getting lookahead TTS result: {q_err}")
                    # Ensure thread is joined if it's still alive
                    if next_prompt_audio_thread.is_alive():
                        next_prompt_audio_thread.join(timeout=0.5)
                next_prompt_audio_thread = None # Clear thread variable

                # Add a small delay between prompts unless it's the very last one
                if not is_last_prompt_in_iter:
                    time.sleep(1) # Reduced delay

            # Add a longer delay between full loop iterations if looping
            if run_infinitely or loop_iteration < args.loop:
                 colored_print(Colors.SYSTEM, f"Finished loop iteration {loop_iteration}. Pausing before next...")
                 time.sleep(3) # Reduced delay

    except KeyboardInterrupt:
        colored_print(Colors.SYSTEM, "\nLoop interrupted by user. Exiting.")
    except Exception as main_loop_error:
         colored_print(Colors.ERROR, f"\nUnhandled error in main execution loop:")
         traceback.print_exc()
    finally:
        # --- Cleanup Shared Resources ---
        colored_print(Colors.SYSTEM, "Cleaning up shared resources...")
        if tts_manager:
            tts_manager.shutdown()
        if request_session:
            request_session.close()
        # Wait for any final lookahead thread if interrupted mid-wait
        if next_prompt_audio_thread and next_prompt_audio_thread.is_alive():
             colored_print(Colors.SYSTEM, "Waiting for final lookahead TTS thread...")
             # No join needed - daemon
        colored_print(Colors.SUCCESS, "\nScript execution finished.")
