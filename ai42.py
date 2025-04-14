# ai42.py - Complete Code (Corrected request_generation worker JSON handling FINAL FINAL FINAL + 8 + Indentation Fix)

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
DEFAULT_FILE_READ_KEYWORD = "READ_FILE_CONTENT"; HUMAN_ROLEPLAY_CLAUSE = " Remember, you are roleplaying as a human and should not reveal you are an AI."
BASE_STOP_WORDS = [ "\n<human>:", "</assistant>", "<|user|>", "<|im_end|>", "</s>", "[INST]", "[/INST]", "Human:", "Assistant:", f"{SERVER_CONFIG['A']['name']}:", f"{SERVER_CONFIG['B']['name']}:" ]

# --- Utility Functions ---
def colored_print(color: str, message: str):
    """Prints a message to the console with the specified ANSI color code."""
    try: print(f"{color}{message}{Colors.RESET}"); sys.stdout.flush()
    except Exception as e: print(f"(ColorPrint Error: {e}) {message}") # Basic fallback

# === TTS Manager ===
class TTSManager:
    """Manages interactions with the WSL-based Piper TTS servers."""
    def __init__(self, tts_config: Dict[str, Dict[str, Any]], tts_timeout: int = DEFAULT_TTS_TIMEOUT):
        self.engine_type: Optional[str] = None
        self.tts_servers: Dict[str, Optional[str]] = {}
        self.tts_timeout = tts_timeout
        self.tts_generate_timeout = 60 # Timeout specifically for the generation request
        self._generate_endpoint = "/generate"
        self._play_endpoint = "/play"

        # Populate server URLs from config
        required_keys = {"piper_a", "piper_b", "piper_human"}
        configured_keys = set()
        for server_id_config, config_entry in tts_config.items():
            tts_key = config_entry.get("tts_server_pref")
            tts_url = config_entry.get("tts_url")
            if tts_key in required_keys:
                self.tts_servers[tts_key] = tts_url
                configured_keys.add(tts_key)
                if not tts_url:
                    colored_print(Colors.ERROR, f"Config warning: TTS key '{tts_key}' for '{server_id_config}' missing URL.")

        missing_keys = required_keys - configured_keys
        if missing_keys:
            colored_print(Colors.ERROR, f"Config error: Missing TTS server configs for key(s): {', '.join(missing_keys)}")

        self._initialize()

    def _check_wsl_server(self, server_key: str, url: Optional[str]) -> bool:
        """Checks if a specific WSL TTS server endpoint is responsive."""
        if not url:
            colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"No URL for server '{server_key}'. Skip check.")
            return False
        generate_url = url.rstrip('/') + self._generate_endpoint
        colored_print(Colors.SYSTEM, f"Checking WSL server '{server_key}' at {generate_url}...")
        try:
            # Use OPTIONS or HEAD as a lightweight check
            response = requests.options(generate_url, timeout=3)
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
        """Checks all required TTS servers and sets the engine type."""
        all_servers_ok = True
        required_keys_present = {"piper_a", "piper_b", "piper_human"}.issubset(self.tts_servers.keys())

        if not required_keys_present:
            all_servers_ok = False
            colored_print(Colors.ERROR, "TTS server configuration incomplete (missing required keys).")
        else:
             for key, url in self.tts_servers.items():
                 if key in {"piper_a", "piper_b", "piper_human"}:
                     if url:
                         if not self._check_wsl_server(key, url):
                             all_servers_ok = False
                     else: # URL is missing for a required key
                        all_servers_ok = False

        if all_servers_ok:
            self.engine_type = "piper_wsl_aplay"
            colored_print(Colors.SUCCESS + Colors.TTS_ENGINE, f"Initialized CUSTOM Piper/Aplay Servers.")
        else:
            self.engine_type = None
            colored_print(Colors.ERROR, "One or more required Piper/Aplay servers FAILED check/config. TTS disabled.")

    def is_available(self) -> bool:
        """Returns True if the TTS system is initialized and available."""
        return self.engine_type == "piper_wsl_aplay"

    # === Request generation in background ===
    def request_generation(self, text: str, server_key: str, result_queue: Queue) -> Optional[threading.Thread]:
        """Sends text to the appropriate TTS server to generate audio in a background thread."""
        target_url = self.tts_servers.get(server_key)
        if not target_url:
            colored_print(Colors.ERROR, f"No URL for TTS server key '{server_key}'. Cannot generate.")
            try:
                result_queue.put(None)
            except Full:
                 colored_print(Colors.ERROR, f"TTS generation result queue is full for {server_key} even on error!")
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
        if not audio_id:
            colored_print(Colors.ERROR, "request_playback called with no audio_id.")
            return
        target_url = self.tts_servers.get(server_key)
        if not target_url:
            colored_print(Colors.ERROR, f"No URL for TTS server key '{server_key}'. Cannot play.")
            return

        play_url = target_url.rstrip('/') + self._play_endpoint

        # Determine speaker name for logging
        speaker_name = server_key # Default
        if server_key == "piper_human": speaker_name = "Human (Initial)"
        elif server_key == "piper_a": speaker_name = SERVER_CONFIG['A'].get('name', 'Server A')
        elif server_key == "piper_b": speaker_name = SERVER_CONFIG['B'].get('name', 'Server B')

        colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Requesting playback for {speaker_name} via server '{server_key}' [ID: {audio_id[:8]}...])")
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
                colored_print(Colors.TTS_ENGINE + Colors.SUCCESS, f"WSL '{server_key}' playback complete.")
            else:
                # Handle cases like "not_found", "error", or unexpected JSON
                colored_print(Colors.TTS_ENGINE + Colors.ERROR, f"WSL '{server_key}' playback issue: {json_resp.get('message', 'Non-JSON OK response or unexpected status')}")

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
             colored_print(Colors.SYSTEM, f"(Playback request for {audio_id[:8]}... finished - Duration: {duration:.2f}s)")


    def shutdown(self):
        """Perform any cleanup needed for the TTS manager."""
        colored_print(Colors.SYSTEM, "TTS Manager shutdown.")
        self.engine_type = None # Mark as unavailable


# --- Network Request Function ---
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

# --- Main Conversation Simulation ---
def simulate_conversation(
    server_config: Dict[str, Dict[str, Any]],
    num_turns: int,
    initial_prompt: str,
    req_timeout: int,
    max_tokens: int,
    temperature: float,
    file_location: Optional[str],
    file_delimiter: str,
    prompts_file_path: str, # Keep for potential future use (e.g., reading files mid-convo)
    prompt_add_delimiter: str, # Keep for potential future use
    tts_timeout: int,
    force_human: bool,
    debug_prompts: bool
):
    """Simulates the conversation between the LLM servers."""
    tts_manager = TTSManager(tts_config=server_config, tts_timeout=tts_timeout)
    request_session = requests.Session() # Use a session for potential connection reuse

    initial_prompt_text_only = initial_prompt # Keep the original full prompt if needed elsewhere
    # Start history with the human's initial turn
    conversation_history: List[str] = [f"{server_config['Human']['name']}: {initial_prompt_text_only}"]

    colored_print(Colors.HEADER, "\n===== CONVERSATION START =====")
    colored_print(Colors.MESSAGE, f"Starting Prompt Context:\n{conversation_history[0]}\n--------------------")
    if force_human:
        colored_print(Colors.SYSTEM, "Force Human mode ENABLED (instructing models to roleplay).")
    if file_location:
        colored_print(Colors.SYSTEM, f"Block saving enabled: File='{file_location}', Delimiter='{file_delimiter}'")
    else:
        colored_print(Colors.SYSTEM, "Block saving disabled.")
    colored_print(Colors.SYSTEM, f"Prompt adding: File='{prompts_file_path}', Delim: '{prompt_add_delimiter}' (Functionality may be limited)")
    if tts_manager.is_available():
        colored_print(Colors.SYSTEM, f"TTS Engine: {tts_manager.engine_type} (Playback Timeout: {tts_timeout}s)")
    else:
        colored_print(Colors.SYSTEM, "TTS is disabled (server check failed or config missing).")
    if debug_prompts:
        colored_print(Colors.DEBUG, "DEBUG PROMPTS ENABLED.")

    # --- Queues and State Variables ---
    llm_result_queue = Queue(maxsize=1) # Only need space for the next response
    tts_generate_queue = Queue(maxsize=1) # Queue for getting the generated audio ID
    active_llm_thread: Optional[threading.Thread] = None
    active_tts_gen_thread: Optional[threading.Thread] = None
    pending_llm_server_id: Optional[str] = None # Which server's response are we waiting for?
    audio_id_to_play: Optional[str] = None # Audio ID generated in the *previous* step, ready for playback
    tts_key_to_play: Optional[str] = None # Which TTS server corresponds to audio_id_to_play

    # --- File/Prompt Interaction State (Placeholder) ---
    # These would need more logic if fully implemented
    file_write_state: Optional[str] = None
    prompt_add_state: Optional[str] = None
    agreed_file_delimiter = file_delimiter.lower()
    agreed_prompt_add_delimiter = prompt_add_delimiter.lower()

    # --- Initial Prompt Audio Handling ---
    initial_audio_id = None
    if tts_manager.is_available():
        colored_print(Colors.SYSTEM, "Pre-generating initial human prompt audio...")
        temp_q = Queue(maxsize=1) # Temporary queue for this specific generation
        init_gen_thread = tts_manager.request_generation(initial_prompt_text_only, "piper_human", temp_q)
        # *** CORRECTED INDENTATION BLOCK STARTS HERE ***
        try:
            initial_audio_id = temp_q.get(timeout=tts_manager.tts_generate_timeout + 5) # Wait for ID
        except Empty:
            colored_print(Colors.ERROR, "Timeout waiting for initial prompt audio ID.")
        # Wait for the generation thread to finish cleanly
        if init_gen_thread:
            init_gen_thread.join(timeout=1.0) # Brief join timeout
        # If we got an ID, play it back
        if initial_audio_id:
            colored_print(Colors.SYSTEM, "Speaking initial human prompt...")
            tts_manager.request_playback(initial_audio_id, "piper_human")
        else:
            colored_print(Colors.ERROR, "Failed to generate initial prompt audio (no ID received).")
        # *** CORRECTED INDENTATION BLOCK ENDS HERE ***
    else:
        colored_print(Colors.SYSTEM, "Skipping initial human prompt speech (TTS unavailable).")

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

        # Loop through turns (each loop iteration handles one LLM response and TTS)
        # Need num_turns * 2 iterations for A->B->A->B...
        for turn_index in range(num_turns * 2):
            actual_turn_num = (turn_index // 2) + 1
            is_last_llm_response = (turn_index == (num_turns * 2) - 1)

            # --- 1. Wait for the pending LLM response ---
            current_llm_key = pending_llm_server_id # This was set in the *previous* iteration (or before loop)
            if not current_llm_key:
                 colored_print(Colors.SYSTEM, "No pending LLM request. Ending loop.")
                 break # Should not happen in normal flow unless last turn
            current_llm_info = server_config[current_llm_key]
            colored_print(Colors.SYSTEM, f"Waiting for LLM response from {current_llm_info['name']}...")
            raw_response = None
            llm_timed_out = False
            try:
                 # Wait for result from the queue
                 retrieved_id, queue_response = llm_result_queue.get(timeout=req_timeout + 15) # Add buffer to timeout
                 if retrieved_id == current_llm_key:
                     raw_response = queue_response
                 else:
                     # This indicates a logic error - the queue should only have the expected response
                     colored_print(Colors.ERROR, f"LLM Queue Logic Error! Expected {current_llm_key}, got {retrieved_id}.")
                     raw_response = None # Treat as failure
            except Empty:
                 colored_print(Colors.ERROR, f"Timeout waiting for LLM response from {current_llm_info['name']}.")
                 llm_timed_out = True
            except Exception as e:
                 colored_print(Colors.ERROR, f"LLM Queue Error: {e}")
                 traceback.print_exc()

            # Ensure the LLM thread object is cleaned up
            if active_llm_thread and active_llm_thread.is_alive():
                active_llm_thread.join(timeout=1.0) # Brief join timeout
            active_llm_thread = None
            pending_llm_server_id = None # We've consumed the pending request

            # --- 2. Process the LLM response ---
            processed_response = None
            text_for_tts = None
            tts_key_for_current = None

            if raw_response: # If we received a non-empty string
                processed_response = raw_response # Start with the raw response

                # Optional: Clean human roleplay clause if forced (and if model included it)
                if force_human:
                    cleaned = processed_response.strip().replace(HUMAN_ROLEPLAY_CLAUSE.strip(), "").strip()
                    if cleaned != processed_response.strip(): # Only update if changed
                        processed_response = cleaned
                        # colored_print(Colors.DEBUG, "Cleaned roleplay clause from response.")

                if processed_response: # Check again after cleaning
                    colored_print(current_llm_info['color'], f"{current_llm_info['name']}: {processed_response}")
                    conversation_history.append(f"{current_llm_info['name']}: {processed_response}")
                    text_for_tts = processed_response # Use the final processed text for speech
                    tts_key_for_current = current_llm_info.get('tts_server_pref')

                    # --- Placeholder: Interaction & Action Logic ---
                    # This is where you'd parse processed_response for keywords
                    response_lower = processed_response.lower()
                    write_block_content = None
                    extracted_prompt_to_add = None
                    # Example (Needs full implementation):
                    # if agreed_file_delimiter in response_lower: ... set write_block_content ...
                    # if agreed_prompt_add_delimiter in response_lower: ... set extracted_prompt_to_add ...
                    # if DEFAULT_FILE_READ_KEYWORD in processed_response: ... read file logic ...

                    # Reset interaction states if action was taken/expected
                    # if write_block_content: file_write_state = None ... perform write ...
                    # if extracted_prompt_to_add: prompt_add_state = None ... add prompt ...

            # Handle failed/empty response
            if not processed_response:
                colored_print(Colors.ERROR, f"{current_llm_info['name']} failed or returned empty response.")
                # Reset any pending interaction states if the response failed
                if file_write_state in ['A_agreed', 'B_agreed']: file_write_state = None
                if prompt_add_state in ['A_agreed_add', 'B_agreed_add']: prompt_add_state = None
                if is_last_llm_response:
                    break # Don't proceed if the last response failed

            # --- 3. Start background TTS generation for the *current* response ---
            if tts_manager.is_available() and text_for_tts and tts_key_for_current:
                colored_print(Colors.SYSTEM, f"Starting background TTS generation for {current_llm_info['name']}...")
                active_tts_gen_thread = tts_manager.request_generation(text_for_tts, tts_key_for_current, tts_generate_queue)
            else:
                active_tts_gen_thread = None # TTS disabled or no text to speak

            # --- 4. Start the *next* LLM request in the background (if not the last turn) ---
            if not is_last_llm_response:
                next_llm_key = llm_server_keys[(turn_index + 1) % 2] # Alternate server
                next_llm_info = server_config[next_llm_key]
                next_turn_indicator = actual_turn_num + 1 if next_llm_key == 'A' else actual_turn_num # For logging
                colored_print(Colors.SYSTEM, f"Starting background LLM request for {next_llm_info['name']} (Turn {next_turn_indicator})...")

                active_llm_thread = threading.Thread(
                    target=request_worker,
                    args=(request_session, next_llm_info, conversation_history.copy(), llm_result_queue, next_llm_key,
                          max_tokens, temperature, req_timeout, force_human, debug_prompts, BASE_STOP_WORDS.copy()),
                    daemon=True, name=f"LLM_{next_llm_key}_T{next_turn_indicator}"
                )
                active_llm_thread.start()
                pending_llm_server_id = next_llm_key # Set the ID for the *next* loop iteration
            else:
                pending_llm_server_id = None # No more LLM requests needed

            # --- 5. Play back the audio from the *previous* turn (if available) ---
            # This happens *after* starting the next LLM request to maximize overlap
            if audio_id_to_play and tts_key_to_play:
                if tts_manager.is_available():
                    tts_manager.request_playback(audio_id_to_play, tts_key_to_play)
                else:
                     colored_print(Colors.SYSTEM, "Skipping playback (TTS unavailable).")
            # Clear the playback variables regardless of whether playback happened
            audio_id_to_play = None
            tts_key_to_play = None

            # --- 6. Wait for the *current* TTS generation to finish ---
            # We need the audio_id before the *next* loop iteration starts playback
            if active_tts_gen_thread:
                 colored_print(Colors.SYSTEM, f"Waiting for TTS generation ID from {current_llm_info['name']}...")
                 next_audio_id = None
                 try:
                     next_audio_id = tts_generate_queue.get(timeout=tts_manager.tts_generate_timeout + 10) # Timeout for getting ID
                 except Empty:
                      colored_print(Colors.ERROR, f"Timeout waiting for generated audio ID for {current_llm_info['name']}.")
                 except Exception as e:
                      colored_print(Colors.ERROR, f"TTS Queue Error: {e}")
                      traceback.print_exc()

                 # If successful, store the ID and key for playback in the *next* iteration
                 if next_audio_id:
                     audio_id_to_play = next_audio_id
                     tts_key_to_play = tts_key_for_current # The key corresponding to the *current* server
                 else:
                      colored_print(Colors.ERROR, f"TTS generation failed for {current_llm_info['name']}. No audio will be played.")
                      audio_id_to_play = None # Ensure it's None if generation failed
                      tts_key_to_play = None

                 # Clean up the TTS generation thread
                 active_tts_gen_thread.join(timeout=1.0)
                 active_tts_gen_thread = None

            # --- End of Loop Iteration ---
            if is_last_llm_response:
                 colored_print(Colors.SYSTEM, "Last LLM response processed.")
                 break # Exit loop after handling the final response

            # Print turn marker for the *next* turn if it's starting
            next_actual_turn = ((turn_index + 1) // 2) + 1
            if pending_llm_server_id and next_actual_turn > actual_turn_num: # Only print when the actual turn number increments
                colored_print(Colors.HEADER, f"\n--- Turn {next_actual_turn}/{num_turns} ---")

    except KeyboardInterrupt:
        colored_print(Colors.SYSTEM, "\nInterrupted by user.")
    except Exception as e:
        colored_print(Colors.ERROR, f"\nError during conversation simulation:")
        traceback.print_exc()
    finally:
        colored_print(Colors.HEADER, "\n===== CONVERSATION END =====")
        # Cleanup resources
        if active_llm_thread and active_llm_thread.is_alive():
            colored_print(Colors.SYSTEM, "Waiting for lingering LLM thread...")
            active_llm_thread.join(timeout=2.0)
        if active_tts_gen_thread and active_tts_gen_thread.is_alive():
            colored_print(Colors.SYSTEM, "Waiting for lingering TTS generation thread...")
            active_tts_gen_thread.join(timeout=2.0)
        if tts_manager:
            tts_manager.shutdown()
        if request_session:
            request_session.close()
        colored_print(Colors.SYSTEM, "Resources cleaned up.")

# ===========================================================
# Argparse Setup and Main Execution Block
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Conversation Simulation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Conversation Setup Args ---
    parser.add_argument('--prompts_file', type=str, default=DEFAULT_PROMPTS_FILE_PATH,
                        help="File containing initial prompts (one per line, '#' for comments).")
    parser.add_argument('-t', '--turns', type=int, default=DEFAULT_NUM_TURNS,
                        help="Number of turns PER AI (e.g., 5 means A->B->A->B->A->B->A->B->A->B).")
    parser.add_argument('--initial_prompt', type=str, default=None,
                        help="A single initial prompt to use instead of a prompts file.")

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
                        help="Timeout in seconds for TTS playback requests.")
    parser.add_argument('--no-tts', action='store_true', help="Disable TTS checks and usage entirely.")


    args = parser.parse_args()

    # --- Update Server Config from Args ---
    # Create a deep copy to avoid modifying the original constant
    current_server_config = json.loads(json.dumps(SERVER_CONFIG))

    # Update LLM URLs
    current_server_config['A']['llm_url'] = args.llm_url_a
    current_server_config['B']['llm_url'] = args.llm_url_b

    # Update TTS URLs (or disable if --no-tts)
    if args.no_tts:
        colored_print(Colors.SYSTEM, "TTS explicitly disabled via --no-tts argument.")
        current_server_config['A']['tts_url'] = None
        current_server_config['B']['tts_url'] = None
        current_server_config['Human']['tts_url'] = None
    else:
        current_server_config['A']['tts_url'] = args.tts_url_a
        current_server_config['B']['tts_url'] = args.tts_url_b
        current_server_config['Human']['tts_url'] = args.tts_url_human

    # --- Determine Prompts to Run ---
    prompts_to_run = []
    if args.initial_prompt:
        prompts_to_run.append(args.initial_prompt)
        colored_print(Colors.SYSTEM, "Using single prompt provided via --initial_prompt.")
    else:
        try:
            with open(args.prompts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    prompt_line = line.strip()
                    # Add line if it's not empty and not a comment
                    if prompt_line and not prompt_line.startswith('#'):
                        prompts_to_run.append(prompt_line)
            if not prompts_to_run:
                colored_print(Colors.ERROR, f"No valid prompts found in '{args.prompts_file}'. Exiting.")
                sys.exit(1)
            colored_print(Colors.SYSTEM, f"Loaded {len(prompts_to_run)} prompts from '{args.prompts_file}'.")
        except FileNotFoundError:
            colored_print(Colors.ERROR, f"Prompts file not found: '{args.prompts_file}'. Exiting.")
            sys.exit(1)
        except Exception as e:
            colored_print(Colors.ERROR, f"Error reading prompts file '{args.prompts_file}': {e}")
            sys.exit(1)

    # --- Run Simulation for Each Prompt ---
    for i, current_initial_prompt in enumerate(prompts_to_run):
        colored_print(Colors.HEADER, f"\n===== PROCESSING PROMPT {i+1}/{len(prompts_to_run)} =====")
        colored_print(Colors.MESSAGE, f"Base Prompt: {current_initial_prompt}")
        time.sleep(1) # Small delay for readability

        final_initial_prompt = current_initial_prompt

        # Modify initial prompt based on flags
        if args.aware:
             # Add info about potential keywords the AI might use
            awareness_clause = (f" (System Note: You might be able to use '{DEFAULT_FILE_READ_KEYWORD} <filename>' to read files, "
                                f"'{args.file_delimiter}' to save conversation blocks, "
                                f"or '{args.prompt_add_delimiter}' to add prompts.)")
            final_initial_prompt += awareness_clause
        if args.human:
             # Append roleplay clause if not already present (simple check)
             if not final_initial_prompt.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()):
                 final_initial_prompt += HUMAN_ROLEPLAY_CLAUSE

        # Determine file location (None if not specified, enabling/disabling feature)
        file_loc = args.file_location if args.file_location else None

        # --- Start the simulation ---
        simulate_conversation(
            server_config=current_server_config,
            num_turns=args.turns,
            initial_prompt=final_initial_prompt,
            req_timeout=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            file_location=file_loc, # Pass None to disable saving
            file_delimiter=args.file_delimiter,
            prompts_file_path=args.prompts_file, # Pass for reference (e.g., adding prompts)
            prompt_add_delimiter=args.prompt_add_delimiter,
            tts_timeout=args.tts_timeout,
            force_human=args.force_human,
            debug_prompts=args.debug_prompts
        )
        colored_print(Colors.HEADER, f"===== FINISHED PROMPT {i+1}/{len(prompts_to_run)} =====")
        # Add a small delay between runs if processing multiple prompts
        if len(prompts_to_run) > 1 and i < len(prompts_to_run) - 1:
            time.sleep(2)

    colored_print(Colors.SUCCESS, "\nAll prompts processed.")
