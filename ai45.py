# ai45.py - Complete Code (Seamless Looping, Delayed Print, TTS & LLM Lookahead)

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
# Colors class and SERVER_CONFIG remain the same...
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
# colored_print remains the same...
def colored_print(color: str, message: str):
    """Prints a message to the console with the specified ANSI color code."""
    try: print(f"{color}{message}{Colors.RESET}"); sys.stdout.flush()
    except Exception as e: print(f"(ColorPrint Error: {e}) {message}") # Basic fallback

# === TTS Manager ===
# TTSManager class remains the same as the previous version...
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
        check_url = url.rstrip('/') + self._generate_endpoint
        colored_print(Colors.SYSTEM, f"Checking WSL server '{server_key}' at {check_url}...")
        try:
            response = requests.options(check_url, timeout=3)
            if response.status_code < 500:
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
            self.engine_type = None
            return

        all_servers_ok = True
        for key, url in self.tts_servers.items():
             if key in {"piper_a", "piper_b", "piper_human"}:
                 if url:
                     if not self._check_wsl_server(key, url):
                         all_servers_ok = False
                 else:
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

    def request_generation(self, text: str, server_key: str, result_queue: Queue) -> Optional[threading.Thread]:
        """Sends text to the appropriate TTS server to generate audio in a background thread."""
        if not self.is_available():
            # colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, "TTS is unavailable. Skipping generation request.")
            try: result_queue.put(None)
            except Full: pass
            return None

        target_url = self.tts_servers.get(server_key)
        if not target_url:
            colored_print(Colors.ERROR, f"No URL for TTS server key '{server_key}'. Cannot generate.")
            try: result_queue.put(None)
            except Full: pass
            return None

        generate_url = target_url.rstrip('/') + self._generate_endpoint

        def worker():
            audio_id = None
            response = None
            try:
                headers = {'Content-Type': 'text/plain'}
                response = requests.post(generate_url, data=text.encode('utf-8'), headers=headers, timeout=self.tts_generate_timeout)
                response.raise_for_status()
                try:
                     json_resp = response.json()
                     if json_resp.get("status") == "generated":
                         audio_id = json_resp.get("audio_id")
                         if not audio_id: colored_print(Colors.ERROR, f"WSL '{server_key}' generated ok but returned no audio_id.")
                     else: colored_print(Colors.ERROR, f"WSL '{server_key}' generation issue: {json_resp.get('message', 'Unknown status')}")
                except json.JSONDecodeError: colored_print(Colors.ERROR, f"WSL '{server_key}' generation OK but received non-JSON response: {response.text[:100]}")
            except requests.exceptions.Timeout: colored_print(Colors.ERROR, f"Generation request to WSL server '{server_key}' timed out ({self.tts_generate_timeout}s).")
            except requests.exceptions.HTTPError as e:
                colored_print(Colors.ERROR, f"WSL '{server_key}' generation HTTP Error {e.response.status_code}.")
                try: colored_print(Colors.DEBUG, f"Error Body ({server_key}): {e.response.text[:200]}")
                except Exception: pass
            except requests.exceptions.RequestException as e: colored_print(Colors.ERROR, f"Error requesting WSL generation '{server_key}': {e}")
            except Exception as e: colored_print(Colors.ERROR, f"Unexpected error during WSL generation '{server_key}': {e}"); traceback.print_exc()
            finally:
                try: result_queue.put(audio_id)
                except Full: colored_print(Colors.ERROR, f"TTS generation result queue is full for {server_key}!")

        thread = threading.Thread(target=worker, daemon=True, name=f"TTSGen_{server_key}")
        thread.start()
        return thread


    def request_playback(self, audio_id: str, server_key: str):
        """Requests playback of a previously generated audio ID from the appropriate server."""
        if not self.is_available():
            # colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, "TTS is unavailable. Skipping playback request.")
            return
        if not audio_id:
            colored_print(Colors.ERROR, "request_playback called with no audio_id.")
            return

        target_url = self.tts_servers.get(server_key)
        if not target_url:
            colored_print(Colors.ERROR, f"No URL for TTS server key '{server_key}'. Cannot play.")
            return

        play_url = target_url.rstrip('/') + self._play_endpoint
        start_time = time.time()
        response = None
        try:
            headers = {'Content-Type': 'application/json'}
            payload = json.dumps({"audio_id": audio_id})
            response = requests.post(play_url, data=payload, headers=headers, timeout=self.tts_timeout)
            response.raise_for_status()
            try:
                json_resp = response.json()
                status = json_resp.get("status")
            except json.JSONDecodeError: status = "unknown_response"
            if status != "played":
                message = "Unknown error"
                try: message = json_resp.get('message', 'Non-JSON OK response or unexpected status')
                except NameError: message = f"Non-JSON OK response or unexpected status: {response.text[:100]}" if response else "Unknown status"
                colored_print(Colors.TTS_ENGINE + Colors.ERROR, f"WSL '{server_key}' playback issue: {message}")
        except requests.exceptions.Timeout: colored_print(Colors.ERROR, f"Playback request to WSL server '{server_key}' timed out ({self.tts_timeout}s).")
        except requests.exceptions.ConnectionError: colored_print(Colors.ERROR, f"Could not connect to WSL playback server '{server_key}' at {play_url}.")
        except requests.exceptions.HTTPError as e:
            error_msg = f"WSL playback server '{server_key}' HTTP Error {e.response.status_code}."
            try: json_resp = e.response.json(); error_msg += f" Msg: {json_resp.get('message', 'N/A')}."
            except json.JSONDecodeError: error_msg += f" Response: {e.response.text[:200]}"
            colored_print(Colors.TTS_ENGINE + Colors.ERROR, error_msg)
        except requests.exceptions.RequestException as e: colored_print(Colors.ERROR, f"Error requesting WSL playback '{server_key}': {e}")
        except Exception as e: colored_print(Colors.ERROR, f"Unexpected error during WSL playback '{server_key}': {e}"); traceback.print_exc()
        finally:
             end_time = time.time()
             duration = max(0, end_time - start_time)
             colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Playback request completed - Duration: {duration:.2f}s)")


    def shutdown(self):
        colored_print(Colors.SYSTEM, "TTS Manager shutdown.")
        self.engine_type = None


# --- Network Request Function ---
# send_llm_request remains the same...
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

    prompt_base = "\n".join(message_history).strip()

    if force_human:
        separator = "\n" if prompt_base else ""
        if not prompt_base.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()):
             prompt_base += separator + HUMAN_ROLEPLAY_CLAUSE.strip()

    final_prompt = prompt_base + f"\n{server_name}:"

    if debug_prompts:
        colored_print(Colors.DEBUG, f"--- DEBUG PROMPT SENT TO {server_name} ---\n{final_prompt}\n--- END DEBUG PROMPT ---")

    payload = {
        'prompt': final_prompt, 'temperature': temperature, 'n_predict': max_tokens,
        'stop': stop_words, 'stream': False
    }
    headers = {'Content-Type': 'application/json'}
    response_content: Optional[str] = None
    response = None

    try:
        colored_print(Colors.SYSTEM, f"Sending request to {server_name} (Timeout={timeout}s, MaxTokens={max_tokens}, Temp={temperature})...")
        response = session.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()

        response_json = response.json()
        content = response_json.get('content', '').strip()

        prefix_to_remove = f"{server_name}:"
        content_stripped = content.strip()
        if content_stripped.lower().startswith(prefix_to_remove.lower()):
            response_content = content_stripped[len(prefix_to_remove):].lstrip()
        else:
            response_content = content_stripped

        if not response_content or response_content in stop_words: return None

    except requests.exceptions.Timeout: colored_print(Colors.ERROR, f"Timeout ({timeout}s) requesting {server_name}")
    except requests.exceptions.HTTPError as e:
        colored_print(Colors.ERROR, f"HTTP Error ({server_name}): {e.response.status_code} {e.response.reason}")
        try: colored_print(Colors.DEBUG, f"Error Body ({server_name}): {e.response.text[:500]}")
        except Exception: pass
    except requests.exceptions.RequestException as e: colored_print(Colors.ERROR, f"Request Error ({server_name}): {type(e).__name__}: {e}")
    except json.JSONDecodeError:
        err_resp = response.text[:500] if response else 'N/A'
        colored_print(Colors.ERROR, f"JSON Decode Error from {server_name}. Response: {err_resp}")
    except Exception as e: colored_print(Colors.ERROR, f"Unexpected request error ({server_name}): {type(e).__name__}"); traceback.print_exc()

    return response_content


# --- Worker Thread Function ---
# request_worker remains the same...
def request_worker(
    session: requests.Session,
    server_config: Dict[str, Any],
    history: List[str],
    result_queue: Queue,
    server_id: str, # Can include _Lookahead suffix
    max_tokens: int,
    temperature: float,
    timeout: int,
    force_human: bool,
    debug_prompts: bool,
    stop_words: List[str]
):
    """Worker function to run send_llm_request in a thread and put result in queue."""
    # Use a clean copy of stop words for each request
    current_stop_words = stop_words[:]
    # Add the other server's name as a potential stop word (prevents self-reply loops)
    other_server_name = None
    if server_id == 'A' or server_id.startswith('A_'): other_server_name = SERVER_CONFIG['B']['name']
    elif server_id == 'B' or server_id.startswith('B_'): other_server_name = SERVER_CONFIG['A']['name']
    if other_server_name and f"{other_server_name}:" not in current_stop_words:
        current_stop_words.append(f"{other_server_name}:")


    result = send_llm_request(
        session=session,
        server_config=server_config,
        message_history=history,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        force_human=force_human,
        debug_prompts=debug_prompts,
        stop_words=current_stop_words # Use the potentially modified list
    )
    try:
        # Only put actual results, not lookahead placeholders
        if not server_id.endswith("_Lookahead"):
             result_queue.put((server_id, result))
    except Full:
         # Avoid printing error for dummy queue if used
         if not server_id.endswith("_Lookahead"):
              colored_print(Colors.ERROR, f"LLM result queue is full for {server_id}!")
    except Exception as e:
         colored_print(Colors.ERROR, f"Error putting result in queue for {server_id}: {e}")


# --- Main Conversation Simulation ---
# simulate_conversation remains the same as the previous version
# (It doesn't need to know about the LLM lookahead directly)
def simulate_conversation(
    tts_manager: TTSManager,
    request_session: requests.Session,
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
    tts_timeout: int,
    force_human: bool,
    debug_prompts: bool,
    pregen_initial_audio_id: Optional[str] = None
):
    """
    Simulates the conversation for ONE initial prompt.
    Uses the provided TTSManager and requests.Session.
    Prints LLM responses just before TTS playback.
    Can use a pre-generated audio ID for the initial prompt.
    """
    initial_prompt_text_only = initial_prompt
    conversation_history: List[str] = [f"{server_config['Human']['name']}: {initial_prompt_text_only}"]

    colored_print(Colors.HEADER, "\n===== CONVERSATION START =====")
    colored_print(Colors.HUMAN_SPEAKER, f"{server_config['Human']['name']}: {initial_prompt_text_only}")
    colored_print(Colors.MESSAGE, "--------------------")

    llm_result_queue = Queue(maxsize=1)
    tts_generate_queue = Queue(maxsize=1)
    active_llm_thread: Optional[threading.Thread] = None
    active_tts_gen_thread: Optional[threading.Thread] = None
    pending_llm_server_id: Optional[str] = None
    playback_info_pending: Optional[Tuple[str, Optional[str], Optional[str], str, str]] = None # Audio ID/Key can be None

    file_write_state: Optional[str] = None
    prompt_add_state: Optional[str] = None
    agreed_file_delimiter = file_delimiter.lower()
    agreed_prompt_add_delimiter = prompt_add_delimiter.lower()

    initial_audio_id = pregen_initial_audio_id
    init_gen_thread = None

    if tts_manager.is_available():
        if not initial_audio_id:
            colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, "Generating initial human prompt audio...")
            temp_q = Queue(maxsize=1)
            init_gen_thread = tts_manager.request_generation(initial_prompt_text_only, "piper_human", temp_q)
            if init_gen_thread:
                try: initial_audio_id = temp_q.get(timeout=tts_manager.tts_generate_timeout + 5)
                except Empty: colored_print(Colors.ERROR, "Timeout waiting for initial prompt audio ID.")
            else: colored_print(Colors.ERROR,"Failed to start initial prompt generation thread.")

        if initial_audio_id:
            colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Speaking initial prompt [ID: {initial_audio_id[:8]}...])")
            tts_manager.request_playback(initial_audio_id, "piper_human")
        else: colored_print(Colors.ERROR, "Failed to generate or use pre-generated initial prompt audio.")
    else: colored_print(Colors.SYSTEM, "Skipping initial human prompt speech (TTS unavailable).")

    if init_gen_thread and init_gen_thread.is_alive(): init_gen_thread.join(timeout=0.5)

    try:
        llm_server_keys = ['A', 'B']
        current_llm_key = llm_server_keys[0]
        current_llm_info = server_config[current_llm_key]

        # Start the first LLM request normally here
        colored_print(Colors.HEADER, f"\n--- Turn 1/{num_turns} ---")
        active_llm_thread = threading.Thread(
            target=request_worker,
            args=(request_session, current_llm_info, conversation_history.copy(), llm_result_queue, current_llm_key,
                  max_tokens, temperature, int(req_timeout * 1.5), force_human, debug_prompts, BASE_STOP_WORDS.copy()),
            daemon=True, name=f"LLM_{current_llm_key}_T1"
        )
        active_llm_thread.start()
        pending_llm_server_id = current_llm_key

        for turn_index in range(num_turns * 2):
            actual_turn_num = (turn_index // 2) + 1
            is_last_llm_response = (turn_index == (num_turns * 2) - 1)

            # --- 1. Wait for the pending LLM response ---
            current_llm_key = pending_llm_server_id
            if not current_llm_key: break
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
            current_playback_data = None

            if raw_response:
                processed_response = raw_response
                if force_human:
                    cleaned = processed_response.strip().replace(HUMAN_ROLEPLAY_CLAUSE.strip(), "").strip()
                    if cleaned != processed_response.strip(): processed_response = cleaned

                if processed_response:
                    conversation_history.append(f"{current_server_name}: {processed_response}")
                    text_for_tts = processed_response
                    tts_key_for_current = current_llm_info.get('tts_server_pref')
                    current_playback_data = (processed_response, tts_key_for_current, current_server_name, current_server_color)
                else: colored_print(Colors.ERROR, f"{current_server_name} returned empty/cleaned response.")
            else: colored_print(Colors.ERROR, f"{current_server_name} failed or returned empty response.")

            if not processed_response:
                if file_write_state in ['A_agreed', 'B_agreed']: file_write_state = None
                if prompt_add_state in ['A_agreed_add', 'B_agreed_add']: prompt_add_state = None
                if is_last_llm_response: break

            # --- 3. Start background TTS generation for the *current* response ---
            if tts_manager.is_available() and text_for_tts and tts_key_for_current:
                # Reducing verbosity of this message
                # colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Starting background TTS generation for {current_server_name}...")
                active_tts_gen_thread = tts_manager.request_generation(text_for_tts, tts_key_for_current, tts_generate_queue)
            else: active_tts_gen_thread = None

            # --- 4. Start the *next* LLM request (if applicable) ---
            if not is_last_llm_response:
                next_llm_key = llm_server_keys[(turn_index + 1) % 2]
                next_llm_info = server_config[next_llm_key]
                next_turn_indicator = actual_turn_num + 1 if next_llm_key == 'A' else actual_turn_num
                # Reducing verbosity
                # colored_print(Colors.SYSTEM, f"Starting background LLM request for {next_llm_info['name']} (Turn {next_turn_indicator})...")
                active_llm_thread = threading.Thread(
                    target=request_worker,
                    args=(request_session, next_llm_info, conversation_history.copy(), llm_result_queue, next_llm_key,
                          max_tokens, temperature, req_timeout, force_human, debug_prompts, BASE_STOP_WORDS.copy()),
                    daemon=True, name=f"LLM_{next_llm_key}_T{next_turn_indicator}"
                )
                active_llm_thread.start()
                pending_llm_server_id = next_llm_key
            else: pending_llm_server_id = None

            # --- 5. Play back audio from the *previous* turn (Print just before) ---
            if playback_info_pending:
                prev_text, prev_audio_id, prev_tts_key, prev_name, prev_color = playback_info_pending
                # **PRINT NOW**
                colored_print(prev_color, f"{prev_name}: {prev_text}")
                if prev_audio_id and prev_tts_key and tts_manager.is_available():
                     colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Speaking for {prev_name} [ID: {prev_audio_id[:8]}...])")
                     tts_manager.request_playback(prev_audio_id, prev_tts_key)
                # else: # Don't print skip message if TTS was never available or generated
                #     if tts_manager.is_available(): # Only print skip if TTS is generally on but failed for this part
                #         colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, "(Skipping playback - audio ID/key missing)")

            playback_info_pending = None # Clear regardless

            # --- 6. Wait for *current* TTS generation & Prepare for *next* playback ---
            if active_tts_gen_thread:
                 # Reducing verbosity
                 # colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Waiting for TTS generation ID from {current_server_name}...")
                 next_audio_id = None
                 try: next_audio_id = tts_generate_queue.get(timeout=tts_manager.tts_generate_timeout + 10)
                 except Empty: colored_print(Colors.ERROR, f"Timeout waiting for generated audio ID for {current_server_name}.")
                 except Exception as e: colored_print(Colors.ERROR, f"TTS Queue Error: {e}"); traceback.print_exc()

                 if next_audio_id and current_playback_data:
                     text, key, name, color = current_playback_data
                     playback_info_pending = (text, next_audio_id, key, name, color)
                 elif current_playback_data: # TTS failed, but have text
                     text, _, name, color = current_playback_data
                     playback_info_pending = (text, None, None, name, color) # Prepare to print text only
                     colored_print(Colors.ERROR, f"TTS generation failed for {current_server_name}. Will print text only.")
                 else: # Failed and no data? Should not happen if raw_response was valid.
                      playback_info_pending = None

                 if active_tts_gen_thread.is_alive(): active_tts_gen_thread.join(timeout=1.0)
                 active_tts_gen_thread = None
            elif current_playback_data: # TTS gen wasn't started, but have text
                text, _, name, color = current_playback_data
                playback_info_pending = (text, None, None, name, color) # Prepare to print text only

            # --- End of Loop Iteration ---
            if is_last_llm_response:
                 colored_print(Colors.SYSTEM, "Last LLM response processed.")
                 break

            # Print turn marker for the *next* turn
            next_actual_turn = ((turn_index + 1) // 2) + 1
            if pending_llm_server_id and next_actual_turn > actual_turn_num:
                colored_print(Colors.HEADER, f"\n--- Turn {next_actual_turn}/{num_turns} ---")

    except KeyboardInterrupt:
        colored_print(Colors.SYSTEM, "\nInterrupted by user during conversation.")
        raise
    except Exception as e:
        colored_print(Colors.ERROR, f"\nError during conversation simulation:")
        traceback.print_exc()
    finally:
        # Play final pending audio/Print final text
        if playback_info_pending:
            final_text, final_audio_id, final_tts_key, final_name, final_color = playback_info_pending
            colored_print(final_color, f"{final_name}: {final_text}") # Print text always
            if final_audio_id and final_tts_key and tts_manager.is_available():
                colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Speaking final response for {final_name} [ID: {final_audio_id[:8]}...])")
                tts_manager.request_playback(final_audio_id, final_tts_key)

        colored_print(Colors.HEADER, "\n===== CONVERSATION END =====")
        # Daemon threads clean themselves up, no explicit join needed here

# ===========================================================
# Argparse Setup and Main Execution Block (Modified for LLM Lookahead)
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Conversation Simulation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- Add Arguments (no changes here vs previous) ---
    # Conversation Setup Args
    parser.add_argument('--prompts_file', type=str, default=DEFAULT_PROMPTS_FILE_PATH, help="File containing initial prompts (one per line, '#' for comments). Re-read if looping infinitely.")
    parser.add_argument('-t', '--turns', type=int, default=DEFAULT_NUM_TURNS, help="Number of turns PER AI (e.g., 5 means 10 total LLM responses).")
    parser.add_argument('--initial_prompt', type=str, default=None, help="A single initial prompt to use instead of a prompts file. Overrides --prompts_file.")
    parser.add_argument('--loop', type=int, default=DEFAULT_LOOP_COUNT, help="Number of times to loop through all prompts. Set to 0 or negative for infinite looping.")
    # Behavior Args
    parser.add_argument('--human', action='store_true', help=f"Append the clause '{HUMAN_ROLEPLAY_CLAUSE}' to the initial prompt.")
    parser.add_argument('--force-human', action='store_true', help="Append the human roleplay clause to EVERY prompt sent to the LLMs.")
    parser.add_argument('--aware', action='store_true', help="Append system note about file/prompt keywords to the initial prompt.")
    parser.add_argument('--debug-prompts', action='store_true', help="Print the full prompt being sent to the LLM.")
    # LLM & Request Args
    parser.add_argument('--timeout', type=int, default=DEFAULT_REQUEST_TIMEOUT, help="Timeout in seconds for LLM requests.")
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS, help="Maximum number of tokens for the LLM to generate.")
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature for the LLM.")
    parser.add_argument('--llm-url-a', type=str, default=SERVER_CONFIG['A']['llm_url'], help="URL for LLM Server A.")
    parser.add_argument('--llm-url-b', type=str, default=SERVER_CONFIG['B']['llm_url'], help="URL for LLM Server B.")
    # File Interaction Args
    parser.add_argument('--file_location', type=str, default=None, help="Base filename for saving conversation blocks (if keyword detected).")
    parser.add_argument('--file_delimiter', type=str, default=DEFAULT_FILE_DELIMITER, help="Keyword AI should use to trigger saving a block.")
    parser.add_argument('--prompt_add_delimiter', type=str, default=DEFAULT_PROMPT_ADD_DELIMITER, help="Keyword AI should use to trigger adding a prompt from the file.")
    # TTS Args
    parser.add_argument('--tts-url-a', type=str, default=SERVER_CONFIG['A']['tts_url'], help="URL for Piper/Aplay TTS Server A (e.g., http://localhost:5001).")
    parser.add_argument('--tts-url-b', type=str, default=SERVER_CONFIG['B']['tts_url'], help="URL for Piper/Aplay TTS Server B (e.g., http://localhost:5002).")
    parser.add_argument('--tts-url-human', type=str, default=SERVER_CONFIG['Human']['tts_url'], help="URL for Piper/Aplay TTS Server for the initial prompt (e.g., http://localhost:5003).")
    parser.add_argument('--tts_timeout', type=int, default=DEFAULT_TTS_TIMEOUT, help="Timeout in seconds for TTS *playback* requests (generation has separate timeout).")
    parser.add_argument('--no-tts', action='store_true', help="Disable TTS checks and usage entirely.")

    args = parser.parse_args()

    # --- Update Server Config from Args ---
    current_server_config = json.loads(json.dumps(SERVER_CONFIG))
    current_server_config['A']['llm_url'] = args.llm_url_a
    current_server_config['B']['llm_url'] = args.llm_url_b
    if args.no_tts:
        current_server_config['A']['tts_url'] = None
        current_server_config['B']['tts_url'] = None
        current_server_config['Human']['tts_url'] = None
    else:
        current_server_config['A']['tts_url'] = args.tts_url_a
        current_server_config['B']['tts_url'] = args.tts_url_b
        current_server_config['Human']['tts_url'] = args.tts_url_human

    # --- Initialize Shared Resources ---
    colored_print(Colors.SYSTEM, "Initializing shared resources...")
    tts_manager = TTSManager(tts_config=current_server_config, tts_timeout=args.tts_timeout, no_tts=args.no_tts)
    request_session = requests.Session()
    colored_print(Colors.SYSTEM, "Shared resources initialized.")

    # --- Main Loop ---
    loop_iteration = 0
    run_infinitely = args.loop <= 0

    # Variables for Lookahead
    next_prompt_tts_audio_q = Queue(maxsize=1)
    next_prompt_tts_audio_thread: Optional[threading.Thread] = None
    pregenerated_tts_audio_id: Optional[str] = None
    # ** NEW ** For LLM Lookahead
    llm_lookahead_thread: Optional[threading.Thread] = None
    dummy_llm_lookahead_queue = Queue(maxsize=1) # Needed for worker, but we don't use result

    try:
        while run_infinitely or loop_iteration < args.loop:
            loop_iteration += 1
            if run_infinitely: colored_print(Colors.HEADER, f"\n===== STARTING INFINITE LOOP ITERATION {loop_iteration} ====="); time.sleep(0.5)
            else: colored_print(Colors.HEADER, f"\n===== STARTING LOOP ITERATION {loop_iteration}/{args.loop} =====")

            # --- Load Prompts ---
            prompts_to_run = []
            # (Prompt loading logic remains the same)
            if args.initial_prompt:
                if loop_iteration == 1 or not run_infinitely : colored_print(Colors.SYSTEM, f"Using single prompt provided via --initial_prompt for iteration {loop_iteration}.")
                prompts_to_run.append(args.initial_prompt)
            else:
                try:
                    with open(args.prompts_file, 'r', encoding='utf-8') as f: prompts_to_run = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    if not prompts_to_run:
                        colored_print(Colors.ERROR, f"No valid prompts found in '{args.prompts_file}' for iteration {loop_iteration}. Skipping iteration.")
                        if run_infinitely: time.sleep(30); continue
                        else: break
                    colored_print(Colors.SYSTEM, f"Loaded {len(prompts_to_run)} prompts from '{args.prompts_file}' for iteration {loop_iteration}.")
                except FileNotFoundError:
                    colored_print(Colors.ERROR, f"Prompts file not found: '{args.prompts_file}'. Skipping iteration {loop_iteration}.")
                    if run_infinitely: time.sleep(30); continue
                    else: break
                except Exception as e:
                    colored_print(Colors.ERROR, f"Error reading prompts file '{args.prompts_file}' in iteration {loop_iteration}: {e}"); traceback.print_exc()
                    if run_infinitely: time.sleep(30); continue
                    else: break

            # --- Run Simulation for Each Prompt ---
            num_prompts_in_iter = len(prompts_to_run)
            for i, current_initial_prompt in enumerate(prompts_to_run):
                is_last_prompt_in_iter = (i == num_prompts_in_iter - 1)
                global_prompt_num = (loop_iteration -1) * num_prompts_in_iter + i + 1 # Just for logging clarity
                colored_print(Colors.HEADER, f"\n--- Processing Prompt {i+1}/{num_prompts_in_iter} (Global #{global_prompt_num}, Loop Iteration {loop_iteration}) ---")

                # --- Wait for previous LLM Lookahead thread to finish (if any) ---
                # Ensures we don't have multiple lookaheads running if sims are very fast
                if llm_lookahead_thread and llm_lookahead_thread.is_alive():
                    colored_print(Colors.SYSTEM, f"Waiting for previous LLM lookahead thread ({llm_lookahead_thread.name}) to complete...")
                    llm_lookahead_thread.join(timeout=args.timeout) # Wait up to the LLM timeout
                    if llm_lookahead_thread.is_alive():
                         colored_print(Colors.WARNING, "Previous LLM lookahead thread did not finish in time.")
                llm_lookahead_thread = None # Clear it

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
                        tts_manager=tts_manager, request_session=request_session,
                        server_config=current_server_config, num_turns=args.turns,
                        initial_prompt=final_initial_prompt, req_timeout=args.timeout,
                        max_tokens=args.max_tokens, temperature=args.temperature,
                        file_location=file_loc, file_delimiter=args.file_delimiter,
                        prompts_file_path=args.prompts_file, prompt_add_delimiter=args.prompt_add_delimiter,
                        tts_timeout=args.tts_timeout, force_human=args.force_human,
                        debug_prompts=args.debug_prompts,
                        pregen_initial_audio_id=pregenerated_tts_audio_id # Use TTS lookahead result
                    )
                except KeyboardInterrupt: raise # Re-raise to exit the outer loop
                except Exception as sim_error:
                     colored_print(Colors.ERROR, f"\nError during simulation for prompt {i+1}. Continuing loop if possible.")
                     traceback.print_exc()

                colored_print(Colors.HEADER, f"--- FINISHED PROMPT {i+1}/{num_prompts_in_iter} (Global #{global_prompt_num}, Loop Iteration {loop_iteration}) ---")
                pregenerated_tts_audio_id = None # Clear after use

                # --- Start Lookahead for the *next* prompt (TTS and LLM) ---
                if not is_last_prompt_in_iter:
                    next_prompt_index = i + 1
                    next_prompt_text = prompts_to_run[next_prompt_index]
                    next_global_prompt_num = global_prompt_num + 1

                    # Prepare next prompt's initial text (with flags)
                    next_final_initial_prompt = next_prompt_text
                    if args.aware: next_final_initial_prompt += f" (...awareness clause...)" # Simplified for TTS
                    if args.human and not next_final_initial_prompt.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()):
                         next_final_initial_prompt += " " + HUMAN_ROLEPLAY_CLAUSE

                    # 1. TTS Lookahead
                    if tts_manager.is_available():
                        colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Lookahead: Starting TTS gen for next prompt (#{next_global_prompt_num})...")
                        next_prompt_tts_audio_thread = tts_manager.request_generation(
                            next_final_initial_prompt, "piper_human", next_prompt_tts_audio_q
                        )
                    else:
                        next_prompt_tts_audio_thread = None

                    # 2. LLM Lookahead (Turn 1, Server A)
                    colored_print(Colors.SYSTEM, f"LLM Lookahead: Starting background request for Server A Turn 1 of next prompt (#{next_global_prompt_num})...")
                    next_initial_history = [f"{server_config['Human']['name']}: {next_final_initial_prompt}"]
                    server_a_info = current_server_config['A']
                    llm_lookahead_thread = threading.Thread(
                        target=request_worker,
                        args=(request_session, server_a_info, next_initial_history,
                              dummy_llm_lookahead_queue, # Use dummy queue
                              'A_Lookahead', # Indicate it's a lookahead request
                              args.max_tokens, args.temperature, int(args.timeout * 1.5), # Longer timeout
                              args.force_human, args.debug_prompts, BASE_STOP_WORDS.copy()),
                        daemon=True, name=f"LLMLookahead_A_P{next_global_prompt_num}"
                    )
                    llm_lookahead_thread.start()

                    # 3. Retrieve TTS Lookahead Result (already started)
                    if next_prompt_tts_audio_thread:
                        colored_print(Colors.SYSTEM + Colors.TTS_ENGINE,"Waiting for lookahead TTS result...")
                        try:
                            pregenerated_tts_audio_id = next_prompt_tts_audio_q.get(timeout=tts_manager.tts_generate_timeout + 5)
                            if pregenerated_tts_audio_id: colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Lookahead TTS complete (ID: {pregenerated_tts_audio_id[:8]}...). Ready for next prompt.")
                            else: colored_print(Colors.ERROR, "Lookahead TTS generation failed (returned None).")
                        except Empty: colored_print(Colors.ERROR, "Timeout waiting for lookahead TTS result.")
                        except Exception as q_err: colored_print(Colors.ERROR, f"Error getting lookahead TTS result: {q_err}")
                        if next_prompt_tts_audio_thread.is_alive(): next_prompt_tts_audio_thread.join(timeout=0.5)
                    next_prompt_tts_audio_thread = None # Clear thread variable

                # Short delay between prompts
                if not is_last_prompt_in_iter: time.sleep(0.5) # Very short delay now


            # Longer delay between full loop iterations
            if run_infinitely or loop_iteration < args.loop:
                 colored_print(Colors.SYSTEM, f"Finished loop iteration {loop_iteration}. Pausing before next...")
                 time.sleep(2) # Reduced delay

    except KeyboardInterrupt:
        colored_print(Colors.SYSTEM, "\nLoop interrupted by user. Exiting.")
    except Exception as main_loop_error:
         colored_print(Colors.ERROR, f"\nUnhandled error in main execution loop:")
         traceback.print_exc()
    finally:
        # --- Cleanup Shared Resources ---
        colored_print(Colors.SYSTEM, "Cleaning up shared resources...")
        # Wait for the *last* lookahead thread if it's still running
        if llm_lookahead_thread and llm_lookahead_thread.is_alive():
             colored_print(Colors.SYSTEM, "Waiting for final LLM lookahead thread...")
             # No join needed - daemon thread will exit
        if tts_manager: tts_manager.shutdown()
        if request_session: request_session.close()
        colored_print(Colors.SUCCESS, "\nScript execution finished.")
