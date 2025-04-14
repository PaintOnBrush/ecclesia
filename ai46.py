# ai46.py - Final Code (Simplified Player Shutdown Synchronization)

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
from queue import Queue, Empty, Full # Use FIFO Queue
from typing import List, Optional, Tuple, Dict, Any, Callable
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
DEFAULT_REQUEST_TIMEOUT = 120; DEFAULT_MAX_TOKENS = 512; DEFAULT_TTS_TIMEOUT = 600
DEFAULT_TEMPERATURE = 0.7; DEFAULT_NUM_TURNS = 5; DEFAULT_FILE_LOCATION = "conversation_output.md"
DEFAULT_FILE_DELIMITER = "SAVE_BLOCK"; DEFAULT_PROMPTS_FILE_PATH = "aiprompts.txt"
DEFAULT_PROMPT_ADD_DELIMITER = "AGREE_ADD_PROMPT"; DEFAULT_INITIAL_PROMPT = "Hi there! Let's have a conversation."
DEFAULT_FILE_READ_KEYWORD = "READ_FILE_CONTENT"; HUMAN_ROLEPLAY_CLAUSE = " Remember, you are roleplaying as a singular human and should not reveal you are an AI."
BASE_STOP_WORDS = [ "\n<human>:", "</assistant>", "<|user|>", "<|im_end|>", "</s>", "[INST]", "[/INST]", "Human:", "Assistant:", f"{SERVER_CONFIG['A']['name']}:", f"{SERVER_CONFIG['B']['name']}:" ]
DEFAULT_LOOP_COUNT = 1
PLAYBACK_QUEUE_SIZE = 10

# --- Utility Functions ---
def colored_print(color: str, message: str):
    try: print(f"{color}{message}{Colors.RESET}"); sys.stdout.flush()
    except Exception as e: print(f"(ColorPrint Error: {e}) {message}")

# === TTS Manager ===
class TTSManager:
    # (No changes needed)
    def __init__(self, tts_config: Dict[str, Dict[str, Any]], tts_timeout: int = DEFAULT_TTS_TIMEOUT, no_tts: bool = False):
        self.engine_type: Optional[str] = None; self.tts_servers: Dict[str, Optional[str]] = {}
        self.tts_timeout = tts_timeout; self.tts_generate_timeout = 60
        self._generate_endpoint = "/generate"; self._play_endpoint = "/play"
        self.is_disabled = no_tts
        if self.is_disabled: self.engine_type = None; return
        required_keys = {"piper_a", "piper_b", "piper_human"}; configured_keys = set(); self.config_valid = True
        for server_id_config, config_entry in tts_config.items():
            tts_key = config_entry.get("tts_server_pref"); tts_url = config_entry.get("tts_url")
            if tts_key in required_keys:
                self.tts_servers[tts_key] = tts_url; configured_keys.add(tts_key)
                if not tts_url: colored_print(Colors.ERROR, f"Config warning: TTS key '{tts_key}' missing URL."); self.config_valid = False
        missing_keys = required_keys - configured_keys
        if missing_keys: colored_print(Colors.ERROR, f"Config error: Missing TTS configs: {', '.join(missing_keys)}"); self.config_valid = False
        if self.config_valid: self._initialize()
        else: self.engine_type = None
    def _check_wsl_server(self, server_key: str, url: Optional[str]) -> bool:
        if not url: return False
        check_url = url.rstrip('/') + self._generate_endpoint
        try: response = requests.options(check_url, timeout=2); return response.status_code < 500
        except Exception: return False
    def _initialize(self):
        if self.is_disabled or not self.config_valid: self.engine_type = None; return
        all_servers_ok = True; checked_servers = 0
        for key, url in self.tts_servers.items():
             if key in {"piper_a", "piper_b", "piper_human"}:
                 checked_servers+=1
                 if url:
                     if not self._check_wsl_server(key, url): all_servers_ok = False
                 else: all_servers_ok = False
        if checked_servers < 3: all_servers_ok = False
        if all_servers_ok: self.engine_type = "piper_wsl_aplay"; colored_print(Colors.SUCCESS + Colors.TTS_ENGINE, f"TTS Servers OK.")
        else: self.engine_type = None; colored_print(Colors.ERROR, "One or more TTS servers FAILED check/config. TTS disabled.")
    def is_available(self) -> bool:
        return self.engine_type == "piper_wsl_aplay" and not self.is_disabled
    def request_generation(self, text: str, server_key: str, result_queue: Queue) -> Optional[threading.Thread]:
        if not self.is_available():
            try: result_queue.put(None);
            except Full: pass
            return None
        target_url = self.tts_servers.get(server_key)
        if not target_url:
            colored_print(Colors.ERROR, f"No URL for TTS key '{server_key}'. Cannot generate.")
            try: result_queue.put(None);
            except Full: pass
            return None
        generate_url = target_url.rstrip('/') + self._generate_endpoint
        def worker():
            audio_id = None; response = None
            try:
                headers = {'Content-Type': 'text/plain'}; response = requests.post(generate_url, data=text.encode('utf-8'), headers=headers, timeout=self.tts_generate_timeout); response.raise_for_status()
                try: json_resp = response.json(); audio_id = json_resp.get("audio_id") if json_resp.get("status") == "generated" else None
                except json.JSONDecodeError: colored_print(Colors.ERROR, f"WSL '{server_key}' gen OK non-JSON: {response.text[:100]}")
            except requests.exceptions.Timeout: colored_print(Colors.ERROR, f"TTS Gen Timeout '{server_key}' ({self.tts_generate_timeout}s).")
            except requests.exceptions.HTTPError as e: colored_print(Colors.ERROR, f"TTS Gen HTTP Error '{server_key}' {e.response.status_code}.")
            except requests.exceptions.RequestException as e: colored_print(Colors.ERROR, f"TTS Gen Req Error '{server_key}': {e}")
            except Exception as e: colored_print(Colors.ERROR, f"TTS Gen Unexpected error '{server_key}': {e}"); traceback.print_exc()
            finally:
                try: result_queue.put(audio_id)
                except Full: colored_print(Colors.ERROR, f"TTS gen result Q full for {server_key}!")
        thread = threading.Thread(target=worker, daemon=True, name=f"TTSGen_{server_key}"); thread.start(); return thread
    def request_playback(self, audio_id: str, server_key: str):
        if not self.is_available(): return
        if not audio_id: colored_print(Colors.ERROR, "request_playback called with no audio_id."); return
        target_url = self.tts_servers.get(server_key)
        if not target_url: colored_print(Colors.ERROR, f"No URL for TTS key '{server_key}'. Cannot play."); return
        play_url = target_url.rstrip('/') + self._play_endpoint; start_time = time.time()
        try:
            headers = {'Content-Type': 'application/json'}; payload = json.dumps({"audio_id": audio_id}); response = requests.post(play_url, data=payload, headers=headers, timeout=self.tts_timeout); response.raise_for_status()
            try: json_resp = response.json(); status = json_resp.get("status")
            except json.JSONDecodeError: status = "unknown_response"
            if status != "played": colored_print(Colors.TTS_ENGINE + Colors.ERROR, f"WSL '{server_key}' playback issue: {json_resp.get('message', 'Non-JSON/Unknown')}")
        except requests.exceptions.Timeout: colored_print(Colors.ERROR, f"TTS Play Timeout '{server_key}' ({self.tts_timeout}s).")
        except requests.exceptions.ConnectionError: colored_print(Colors.ERROR, f"TTS Play Connect Error '{server_key}' {play_url}.")
        except requests.exceptions.HTTPError as e: colored_print(Colors.TTS_ENGINE + Colors.ERROR, f"TTS Play HTTP Error '{server_key}' {e.response.status_code}.")
        except requests.exceptions.RequestException as e: colored_print(Colors.ERROR, f"TTS Play Req Error '{server_key}': {e}")
        except Exception as e: colored_print(Colors.ERROR, f"TTS Play Unexpected error '{server_key}': {e}"); traceback.print_exc()
        finally: duration = max(0, time.time() - start_time); colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Playback request completed - Duration: {duration:.2f}s)")
    def shutdown(self): colored_print(Colors.SYSTEM, "TTS Manager shutdown."); self.engine_type = None

# --- Network Request Function ---
def send_llm_request(
    session: requests.Session, server_config: Dict[str, Any], message_history: List[str],
    max_tokens: int, temperature: float, timeout: int, force_human: bool, debug_prompts: bool,
    stop_words: List[str] = BASE_STOP_WORDS
) -> Optional[str]:
    url = server_config.get('llm_url'); server_name = server_config.get('name', 'UnknownServer')
    if not url: colored_print(Colors.ERROR, f"LLM URL missing for server '{server_name}'."); return None
    prompt_base = "\n".join(message_history).strip()
    if force_human:
        separator = "\n" if prompt_base else ""
        if not prompt_base.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()): prompt_base += separator + HUMAN_ROLEPLAY_CLAUSE.strip()
    final_prompt = prompt_base + f"\n{server_name}:"
    if debug_prompts: colored_print(Colors.DEBUG, f"--- DEBUG PROMPT SENT TO {server_name} ---\n{final_prompt}\n--- END DEBUG PROMPT ---")
    payload = {'prompt': final_prompt, 'temperature': temperature, 'n_predict': max_tokens, 'stop': stop_words, 'stream': False }
    headers = {'Content-Type': 'application/json'}; response_content: Optional[str] = None; response = None
    try:
        request_start_time = time.monotonic(); response = session.post(url, json=payload, headers=headers, timeout=timeout); request_duration = time.monotonic() - request_start_time
        if debug_prompts: colored_print(Colors.DEBUG, f"LLM Request {server_name} duration: {request_duration:.2f}s");
        response.raise_for_status()
        response_json = response.json(); content = response_json.get('content', '').strip(); prefix_to_remove = f"{server_name}:"; content_stripped = content.strip()
        if content_stripped.lower().startswith(prefix_to_remove.lower()): response_content = content_stripped[len(prefix_to_remove):].lstrip()
        else: response_content = content_stripped
        if not response_content or response_content in stop_words:
             if debug_prompts and response_content: colored_print(Colors.DEBUG, f"LLM response for {server_name} was only a stop word.")
             return None
    except requests.exceptions.Timeout: colored_print(Colors.ERROR, f"Timeout ({timeout}s) requesting {server_name}")
    except requests.exceptions.HTTPError as e: colored_print(Colors.ERROR, f"HTTP Error ({server_name}): {e.response.status_code} {e.response.reason}")
    except requests.exceptions.RequestException as e: colored_print(Colors.ERROR, f"Request Error ({server_name}): {type(e).__name__}: {e}")
    except json.JSONDecodeError: colored_print(Colors.ERROR, f"JSON Decode Error from {server_name}. Response: {response.text[:500] if response else 'N/A'}")
    except Exception as e: colored_print(Colors.ERROR, f"Unexpected request error ({server_name}): {type(e).__name__}"); traceback.print_exc()
    return response_content

# --- Worker Thread Functions ---
def request_worker(
    session: requests.Session, server_config: Dict[str, Any], history: List[str],
    result_queue: Queue, server_id: str, max_tokens: int, temperature: float, timeout: int,
    force_human: bool, debug_prompts: bool, stop_words: List[str]
):
    current_stop_words = stop_words[:]; other_server_name = None; server_name = server_config.get('name', 'UnknownServer')
    if server_id == 'A' or server_id.startswith('A_'): other_server_name = SERVER_CONFIG['B']['name']
    elif server_id == 'B' or server_id.startswith('B_'): other_server_name = SERVER_CONFIG['A']['name']
    if other_server_name and f"{other_server_name}:" not in current_stop_words: current_stop_words.append(f"{other_server_name}:")
    if server_name and f"{server_name}:" not in current_stop_words: current_stop_words.append(f"{server_name}:")
    result = send_llm_request(session=session, server_config=server_config, message_history=history, max_tokens=max_tokens, temperature=temperature, timeout=timeout, force_human=force_human, debug_prompts=debug_prompts, stop_words=current_stop_words)
    try:
        result_queue.put((server_id, result))
    except Full:
        colored_print(Colors.ERROR, f"LLM result queue is full for {server_id}!")
    except Exception as e: colored_print(Colors.ERROR, f"Error putting result in queue for {server_id}: {e}")

# --- tts_player_worker ---
def tts_player_worker(playback_queue: Queue, tts_manager: TTSManager, stop_event: threading.Event, total_turns: int):
    colored_print(Colors.SYSTEM, "TTS Player Thread started.")
    turn_header_printed_for = -1
    while not stop_event.is_set():
        try:
            item = playback_queue.get(timeout=1.0)
            if item is None: colored_print(Colors.DEBUG, "Player thread received sentinel."); playback_queue.task_done(); break # Exit loop
            turn_num, text, audio_id, tts_key, name, color = item
            try:
                if turn_num > 0 and turn_num != turn_header_printed_for:
                    colored_print(Colors.HEADER, f"\n--- Turn {turn_num}/{total_turns} ---")
                    turn_header_printed_for = turn_num
                colored_print(color, f"{name}: {text}")
                if audio_id and tts_key and tts_manager.is_available():
                    colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"(Speaking for {name} [ID: {audio_id[:8]}...])")
                    tts_manager.request_playback(audio_id, tts_key)
            except Exception as play_err: colored_print(Colors.ERROR, f"Error during playback processing for {name} (Turn {turn_num}): {play_err}"); traceback.print_exc()
            finally:
                 # Ensure task_done is called even if playback errors
                 playback_queue.task_done()
        except Empty:
            if stop_event.is_set() and playback_queue.empty():
                 colored_print(Colors.DEBUG, "Player queue empty and stop event set. Exiting.")
                 break
            continue
        except Exception as q_err: colored_print(Colors.ERROR, f"Unexpected error in player thread loop: {q_err}"); traceback.print_exc(); time.sleep(1)
    colored_print(Colors.SYSTEM, "TTS Player Thread finished.")

# --- Main Conversation Simulation ---
def simulate_conversation(
    tts_manager: TTSManager, request_session: requests.Session, server_config: Dict[str, Dict[str, Any]],
    num_turns: int, initial_prompt: str, req_timeout: int, max_tokens: int, temperature: float,
    file_location: Optional[str], file_delimiter: str, prompts_file_path: str, prompt_add_delimiter: str,
    tts_timeout: int, force_human: bool, debug_prompts: bool, debug_logic: bool,
    pregen_initial_audio_id: Optional[str] = None,
    initial_llm_thread: Optional[threading.Thread] = None,
    initial_llm_queue: Optional[Queue] = None,
    prompt_global_num: int = 0
):
    if debug_logic: colored_print(Colors.DEBUG, f"SimulateConversation called with num_turns={num_turns}")
    total_ai_turns = num_turns
    if total_ai_turns <= 0:
        colored_print(Colors.ERROR, f"Invalid num_turns ({total_ai_turns}) received for prompt {prompt_global_num}. Setting to 1.")
        total_ai_turns = 1

    initial_prompt_text_only = initial_prompt
    conversation_history: List[str] = [f"{server_config['Human']['name']}: {initial_prompt_text_only}"]
    colored_print(Colors.HEADER, f"\n===== CONVERSATION START (Prompt #{prompt_global_num}, Turns: {total_ai_turns}) =====")

    llm_result_queue = initial_llm_queue if initial_llm_queue else Queue(maxsize=1)
    playback_queue = Queue(maxsize=PLAYBACK_QUEUE_SIZE)
    player_stop_event = threading.Event()
    active_llm_thread = initial_llm_thread
    pending_llm_server_id: Optional[str] = 'A' if initial_llm_thread else None
    file_write_state: Optional[str] = None; prompt_add_state: Optional[str] = None
    agreed_file_delimiter = file_delimiter.lower(); agreed_prompt_add_delimiter = prompt_add_delimiter.lower()

    player_thread = threading.Thread(target=tts_player_worker, args=(playback_queue, tts_manager, player_stop_event, total_ai_turns), name=f"TTSPlayer_P{prompt_global_num}", daemon=False)
    player_thread.start()

    initial_audio_id = pregen_initial_audio_id; initial_tts_key = server_config['Human']['tts_server_pref']
    initial_name = server_config['Human']['name']; initial_color = Colors.HUMAN_SPEAKER
    initial_gen_thread : Optional[threading.Thread] = None; initial_generation_q = Queue(maxsize=1)
    if initial_audio_id is None and tts_manager.is_available():
         if debug_logic: colored_print(Colors.DEBUG, "Starting initial prompt TTS generation...")
         initial_gen_thread = tts_manager.request_generation(initial_prompt_text_only, initial_tts_key, initial_generation_q)
         if not initial_gen_thread: colored_print(Colors.ERROR,"Failed to start initial prompt generation.")

    llm_server_keys = ['A', 'B']
    if not active_llm_thread:
        current_llm_key = llm_server_keys[0]
        colored_print(Colors.SYSTEM, f"Starting LLM request for Turn 1 ({current_llm_key})...")
        current_llm_info = server_config[current_llm_key]
        active_llm_thread = threading.Thread(target=request_worker, args=(request_session, current_llm_info, conversation_history.copy(), llm_result_queue, current_llm_key, max_tokens, temperature, int(req_timeout * 1.5), force_human, debug_prompts, BASE_STOP_WORDS.copy()), daemon=True, name=f"LLM_{current_llm_key}_T1")
        active_llm_thread.start()
        pending_llm_server_id = current_llm_key
    else:
         if debug_logic: colored_print(Colors.DEBUG, "Using pre-started LLM thread for Turn 1 (A)...")

    if initial_gen_thread:
         if debug_logic: colored_print(Colors.DEBUG, "Waiting for initial prompt TTS generation result...")
         try: initial_audio_id = initial_generation_q.get(timeout=tts_manager.tts_generate_timeout + 10)
         except Empty: colored_print(Colors.ERROR, "Timeout waiting for initial prompt TTS generation.")
         except Exception as e: colored_print(Colors.ERROR, f"Error getting initial TTS result: {e}")
    initial_playback_item = (0, initial_prompt_text_only, initial_audio_id, initial_tts_key, initial_name, initial_color)
    try:
        if debug_logic: colored_print(Colors.DEBUG, "Queueing initial prompt for playback...")
        playback_queue.put(initial_playback_item)
    except Full: colored_print(Colors.ERROR, "Playback queue full, cannot add initial prompt!")

    try:
        target_iterations = total_ai_turns * 2
        if debug_logic: colored_print(Colors.DEBUG, f"Starting conversation loop: target_iterations={target_iterations}, num_turns={total_ai_turns}")

        for turn_index in range(target_iterations):
            actual_turn_num = (turn_index // 2) + 1
            is_last_iteration = (turn_index == target_iterations - 1)
            current_llm_key = pending_llm_server_id

            if debug_logic: colored_print(Colors.DEBUG, f"--- Loop Start: turn_index={turn_index}, actual_turn={actual_turn_num}, pending_llm={current_llm_key}, last_iter={is_last_iteration} ---")

            if not current_llm_key: colored_print(Colors.SYSTEM,"No pending LLM server, ending turn loop early."); break

            current_llm_info = server_config[current_llm_key]; current_server_name = current_llm_info['name']
            current_server_color = current_llm_info['color']; current_tts_key = current_llm_info.get('tts_server_pref')

            # 1. Wait LLM
            if debug_logic: colored_print(Colors.DEBUG, f"Waiting for LLM response from {current_server_name}...")
            raw_response = None; llm_request_failed = False; retrieved_id = None; queue_response = "NO_RESPONSE_YET"
            try:
                current_llm_result_queue = initial_llm_queue if turn_index == 0 and initial_llm_queue else llm_result_queue
                retrieved_id, queue_response = current_llm_result_queue.get(timeout=req_timeout + 15)
                while retrieved_id is not None and retrieved_id.endswith("_Lookahead"):
                     colored_print(Colors.DEBUG, f"Ignoring lookahead result for {retrieved_id} in main queue.")
                     retrieved_id, queue_response = current_llm_result_queue.get(timeout=req_timeout + 5)
                expected_id = 'A' if turn_index == 0 and initial_llm_queue else current_llm_key
                if retrieved_id == expected_id:
                     raw_response = queue_response
                     if raw_response is None: colored_print(Colors.ERROR, f"LLM worker for {current_server_name} (Turn {actual_turn_num}) returned None."); llm_request_failed = True
                else: colored_print(Colors.ERROR, f"LLM Q Logic Error! Expected {expected_id}, got {retrieved_id}."); llm_request_failed = True
            except Empty: colored_print(Colors.ERROR, f"Timeout waiting LLM response for {current_server_name} (Turn {actual_turn_num})."); llm_request_failed = True; queue_response = "TIMEOUT"
            except Exception as e: colored_print(Colors.ERROR, f"LLM Q Error: {e}"); traceback.print_exc(); llm_request_failed = True; queue_response = f"ERROR: {e}"

            if debug_logic: colored_print(Colors.DEBUG, f"LLM Result Get (T{actual_turn_num} {current_server_name}): ID={retrieved_id}, Failed={llm_request_failed}, Raw Type={type(queue_response)}")

            active_llm_thread = None; pending_llm_server_id = None

            if llm_request_failed:
                colored_print(Colors.ERROR, f"Conversation {prompt_global_num} ending early due to LLM failure for {current_server_name} (Turn {actual_turn_num}).")
                break

            # 3. Process LLM Response
            processed_response = None
            if raw_response:
                processed_response = raw_response
                if force_human: cleaned = processed_response.strip().replace(HUMAN_ROLEPLAY_CLAUSE.strip(), "").strip(); processed_response = cleaned if cleaned != processed_response.strip() else processed_response
                if not processed_response: colored_print(Colors.ERROR, f"{current_server_name} (Turn {actual_turn_num}) returned empty/cleaned response."); processed_response = None
            if not processed_response:
                colored_print(Colors.ERROR, f"No valid response from {current_server_name} (Turn {actual_turn_num}) after processing. Ending conversation.")
                break

            # --- Valid response ---
            # 4. Start TTS Generation
            current_gen_thread : Optional[threading.Thread] = None; current_generation_q = Queue(maxsize=1)
            current_audio_id : Optional[str] = None
            if tts_manager.is_available() and current_tts_key:
                if debug_logic: colored_print(Colors.DEBUG, f"Starting TTS generation for {current_server_name} (Turn {actual_turn_num})...")
                current_gen_thread = tts_manager.request_generation(processed_response, current_tts_key, current_generation_q)
                if not current_gen_thread: colored_print(Colors.ERROR,f"Failed to start TTS generation for {current_server_name}.")

            # 5. Start Next LLM Request
            if not is_last_iteration:
                next_llm_key = llm_server_keys[(turn_index + 1) % 2]; next_llm_info = server_config[next_llm_key]
                next_actual_turn_num = ((turn_index + 1) // 2) + 1
                if debug_logic: colored_print(Colors.DEBUG, f"Starting next LLM request for {next_llm_info['name']} (Turn {next_actual_turn_num})...")
                active_llm_thread = threading.Thread(target=request_worker, args=(request_session, next_llm_info, conversation_history.copy(), llm_result_queue, next_llm_key, max_tokens, temperature, req_timeout, force_human, debug_prompts, BASE_STOP_WORDS.copy()), daemon=True, name=f"LLM_{next_llm_key}_T{next_actual_turn_num}")
                active_llm_thread.start()
                pending_llm_server_id = next_llm_key
            else:
                 if debug_logic: colored_print(Colors.DEBUG, "Not starting next LLM request, was last iteration.")
                 pending_llm_server_id = None

            # 6. Wait TTS Gen
            if current_gen_thread:
                if debug_logic: colored_print(Colors.DEBUG, f"Waiting for TTS generation result for {current_server_name} (Turn {actual_turn_num})...")
                try: current_audio_id = current_generation_q.get(timeout=tts_manager.tts_generate_timeout + 10)
                except Empty: colored_print(Colors.ERROR, f"Timeout waiting TTS gen result for {current_server_name}.")
                except Exception as e: colored_print(Colors.ERROR, f"Error getting TTS result for {current_server_name}: {e}")

            # 7. Queue Playback Item
            if debug_logic: colored_print(Colors.DEBUG, f"Queueing playback item for {current_server_name} (Turn {actual_turn_num})...")
            playback_item = (actual_turn_num, processed_response, current_audio_id, current_tts_key, current_server_name, current_server_color)
            try: playback_queue.put(playback_item)
            except Full: colored_print(Colors.ERROR, f"Playback queue full! Cannot add item for {current_server_name}.")
            except Exception as e: colored_print(Colors.ERROR, f"Error putting item on playback queue for {current_server_name}: {e}")
            conversation_history.append(f"{current_server_name}: {processed_response}")

            # --- Loop End Check ---
            if debug_logic: colored_print(Colors.DEBUG, f"--- Loop End Check: turn_index={turn_index}, target_iterations={target_iterations}, is_last_iteration={is_last_iteration} ---")
            # Loop terminates naturally after last iteration

    except KeyboardInterrupt: colored_print(Colors.SYSTEM, "\nInterrupted by user during conversation."); raise
    except Exception as e: colored_print(Colors.ERROR, f"\nError during conversation simulation:"); traceback.print_exc()
    finally:
        colored_print(Colors.SYSTEM, "Conversation loop finished. Waiting for player to complete...")
        try:
             playback_queue.put(None) # Send sentinel *BEFORE* joining queue
             if debug_logic: colored_print(Colors.DEBUG, f"Waiting for playback queue to empty (qsize ~{playback_queue.qsize()})...")
             playback_queue.join() # **** WAIT FOR QUEUE TO BE EMPTY ****
             # Now that queue is empty, we know player received sentinel and should exit
             if debug_logic: colored_print(Colors.DEBUG,"Playback queue empty. Setting stop event and joining player thread...")
             player_stop_event.set() # Ensure stop is set
             player_thread.join(timeout=15) # Join with a reasonable timeout

             if player_thread.is_alive(): # Check if it finished
                 colored_print(Colors.WARNING, f"Player thread did not stop cleanly!")
        except Exception as final_e:
            colored_print(Colors.ERROR, f"Error during player thread shutdown: {final_e}"); traceback.print_exc()
        colored_print(Colors.HEADER, f"\n===== CONVERSATION END (Prompt #{prompt_global_num}) =====")


# ===========================================================
# Argparse Setup and Main Execution Block
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Conversation Simulation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # (Arguments remain the same)
    parser.add_argument('--prompts_file', type=str, default=DEFAULT_PROMPTS_FILE_PATH, help="File containing initial prompts.")
    parser.add_argument('-t', '--turns', type=int, default=DEFAULT_NUM_TURNS, help="Number of turns PER AI.")
    parser.add_argument('--initial_prompt', type=str, default=None, help="Single initial prompt instead of file.")
    parser.add_argument('--loop', type=int, default=DEFAULT_LOOP_COUNT, help="Loops (0=infinite).")
    parser.add_argument('--human', action='store_true', help="Append human clause to initial prompt.")
    parser.add_argument('--force-human', action='store_true', help="Append human clause to EVERY prompt.")
    parser.add_argument('--aware', action='store_true', help="Append system note to initial prompt.")
    parser.add_argument('--debug-prompts', action='store_true', help="Print full LLM prompts and basic request timing.")
    parser.add_argument('--debug-logic', action='store_true', help="Print detailed step-by-step logic flow.")
    parser.add_argument('--timeout', type=int, default=DEFAULT_REQUEST_TIMEOUT, help="LLM request timeout (s).")
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS, help="LLM max tokens.")
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature.")
    parser.add_argument('--llm-url-a', type=str, default=SERVER_CONFIG['A']['llm_url'], help="URL for LLM Server A.")
    parser.add_argument('--llm-url-b', type=str, default=SERVER_CONFIG['B']['llm_url'], help="URL for LLM Server B.")
    parser.add_argument('--file_location', type=str, default=None, help="Base filename for saving blocks.")
    parser.add_argument('--file_delimiter', type=str, default=DEFAULT_FILE_DELIMITER, help="Keyword to trigger saving.")
    parser.add_argument('--prompt_add_delimiter', type=str, default=DEFAULT_PROMPT_ADD_DELIMITER, help="Keyword to trigger adding prompt.")
    parser.add_argument('--tts-url-a', type=str, default=SERVER_CONFIG['A']['tts_url'], help="URL for TTS Server A.")
    parser.add_argument('--tts-url-b', type=str, default=SERVER_CONFIG['B']['tts_url'], help="URL for TTS Server B.")
    parser.add_argument('--tts-url-human', type=str, default=SERVER_CONFIG['Human']['tts_url'], help="URL for TTS Server Human.")
    parser.add_argument('--tts_timeout', type=int, default=DEFAULT_TTS_TIMEOUT, help="TTS playback request timeout (s).")
    parser.add_argument('--no-tts', action='store_true', help="Disable TTS.")
    args = parser.parse_args()

    current_server_config = json.loads(json.dumps(SERVER_CONFIG))
    current_server_config['A']['llm_url'] = args.llm_url_a; current_server_config['B']['llm_url'] = args.llm_url_b
    if args.no_tts: current_server_config['A']['tts_url'] = None; current_server_config['B']['tts_url'] = None; current_server_config['Human']['tts_url'] = None
    else: current_server_config['A']['tts_url'] = args.tts_url_a; current_server_config['B']['tts_url'] = args.tts_url_b; current_server_config['Human']['tts_url'] = args.tts_url_human

    colored_print(Colors.SYSTEM, "Initializing shared resources...")
    tts_manager = TTSManager(tts_config=current_server_config, tts_timeout=args.tts_timeout, no_tts=args.no_tts)
    request_session = requests.Session()
    colored_print(Colors.SYSTEM, "Shared resources initialized.")

    loop_iteration = 0; run_infinitely = args.loop <= 0
    next_prompt_tts_audio_q = Queue(maxsize=1); next_prompt_tts_audio_thread: Optional[threading.Thread] = None
    pregenerated_tts_audio_id: Optional[str] = None
    next_prompt_llm_thread: Optional[threading.Thread] = None
    next_prompt_llm_queue: Queue = Queue(maxsize=1)

    try:
        while run_infinitely or loop_iteration < args.loop:
            loop_iteration += 1
            if run_infinitely: colored_print(Colors.HEADER, f"\n===== STARTING INFINITE LOOP ITERATION {loop_iteration} ====="); time.sleep(0.2)
            else: colored_print(Colors.HEADER, f"\n===== STARTING LOOP ITERATION {loop_iteration}/{args.loop} =====")

            prompts_to_run = []
            if args.initial_prompt:
                if loop_iteration == 1 or not run_infinitely : colored_print(Colors.SYSTEM, f"Using single prompt for iteration {loop_iteration}.")
                prompts_to_run.append(args.initial_prompt)
            else:
                try:
                    with open(args.prompts_file, 'r', encoding='utf-8') as f: prompts_to_run = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    if not prompts_to_run: colored_print(Colors.ERROR, f"No valid prompts in '{args.prompts_file}'. Skipping."); time.sleep(5); continue
                    colored_print(Colors.SYSTEM, f"Loaded {len(prompts_to_run)} prompts from '{args.prompts_file}'.")
                except FileNotFoundError: colored_print(Colors.ERROR, f"Prompts file not found: '{args.prompts_file}'. Skipping."); time.sleep(5); continue
                except Exception as e: colored_print(Colors.ERROR, f"Error reading prompts file '{args.prompts_file}': {e}"); time.sleep(5); continue

            num_prompts_in_iter = len(prompts_to_run)
            for i, current_initial_prompt in enumerate(prompts_to_run):
                is_last_prompt_in_iter = (i == num_prompts_in_iter - 1); global_prompt_num = (loop_iteration -1) * num_prompts_in_iter + i + 1
                final_initial_prompt = current_initial_prompt
                if args.aware: final_initial_prompt += f" (...awareness...)"
                if args.human and not final_initial_prompt.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()): final_initial_prompt += " " + HUMAN_ROLEPLAY_CLAUSE
                file_loc = args.file_location if args.file_location else None

                current_initial_llm_thread = next_prompt_llm_thread
                current_initial_llm_queue = next_prompt_llm_queue
                next_prompt_llm_thread = None
                next_prompt_llm_queue = Queue(maxsize=1)

                try:
                    simulate_conversation(
                        tts_manager=tts_manager, request_session=request_session, server_config=current_server_config,
                        num_turns=args.turns,
                        initial_prompt=final_initial_prompt, req_timeout=args.timeout, max_tokens=args.max_tokens, temperature=args.temperature,
                        file_location=file_loc, file_delimiter=args.file_delimiter, prompts_file_path=args.prompts_file, prompt_add_delimiter=args.prompt_add_delimiter,
                        tts_timeout=args.tts_timeout, force_human=args.force_human, debug_prompts=args.debug_prompts,
                        debug_logic=args.debug_logic,
                        pregen_initial_audio_id=pregenerated_tts_audio_id,
                        initial_llm_thread=current_initial_llm_thread,
                        initial_llm_queue=current_initial_llm_queue,
                        prompt_global_num=global_prompt_num
                    )
                except KeyboardInterrupt: raise
                except Exception as sim_error: colored_print(Colors.ERROR, f"\nError during simulation for prompt {global_prompt_num}. Continuing."); traceback.print_exc()

                pregenerated_tts_audio_id = None
                if next_prompt_tts_audio_thread:
                    try:
                        pregenerated_tts_audio_id = next_prompt_tts_audio_q.get(timeout=tts_manager.tts_generate_timeout + 5 if tts_manager.is_available() else 1)
                        if pregenerated_tts_audio_id: colored_print(Colors.SYSTEM + Colors.TTS_ENGINE, f"Lookahead TTS complete (ID: {pregenerated_tts_audio_id[:8]}...). Ready for next prompt.")
                    except Empty: colored_print(Colors.ERROR, "Timeout waiting for lookahead TTS result.")
                    except Exception as q_err: colored_print(Colors.ERROR, f"Error getting lookahead TTS result: {q_err}")
                    if next_prompt_tts_audio_thread.is_alive(): next_prompt_tts_audio_thread.join(timeout=0.5)
                next_prompt_tts_audio_thread = None

                if not is_last_prompt_in_iter:
                    next_prompt_index = i + 1; next_prompt_text = prompts_to_run[next_prompt_index]; next_global_prompt_num = global_prompt_num + 1
                    next_final_initial_prompt = next_prompt_text
                    if args.aware: next_final_initial_prompt += f" (...awareness...)"
                    if args.human and not next_final_initial_prompt.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()): next_final_initial_prompt += " " + HUMAN_ROLEPLAY_CLAUSE

                    if tts_manager.is_available(): next_prompt_tts_audio_thread = tts_manager.request_generation(next_final_initial_prompt, "piper_human", next_prompt_tts_audio_q)
                    else: next_prompt_tts_audio_thread = None

                    colored_print(Colors.SYSTEM, f"Starting LLM Lookahead for prompt #{next_global_prompt_num} Turn 1...")
                    next_initial_history = [f"{current_server_config['Human']['name']}: {next_final_initial_prompt}"]
                    server_a_info = current_server_config['A']
                    next_prompt_llm_thread = threading.Thread(
                        target=request_worker,
                        args=(request_session, server_a_info, next_initial_history, next_prompt_llm_queue, 'A',
                              args.max_tokens, args.temperature, int(args.timeout * 1.5),
                              args.force_human, args.debug_prompts, BASE_STOP_WORDS.copy()),
                        daemon=True, name=f"LLMLookahead_A_P{next_global_prompt_num}"
                    )
                    next_prompt_llm_thread.start()

                if not is_last_prompt_in_iter: time.sleep(0.2)

            if run_infinitely or loop_iteration < args.loop: time.sleep(1)

    except KeyboardInterrupt: colored_print(Colors.SYSTEM, "\nLoop interrupted by user. Exiting.")
    except Exception as main_loop_error: colored_print(Colors.ERROR, f"\nUnhandled error in main execution loop:"); traceback.print_exc()
    finally:
        colored_print(Colors.SYSTEM, "Cleaning up shared resources...")
        if tts_manager: tts_manager.shutdown()
        if request_session: request_session.close()
        colored_print(Colors.SUCCESS, "\nScript execution finished.")
