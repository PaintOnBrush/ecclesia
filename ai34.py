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
from queue import Queue, Empty
from typing import List, Optional, Tuple, Dict, Any
import argparse

# --- Configuration (Constants and Defaults) ---
class Colors:
    RESET = "\033[0m"
    SERVER_A = "\033[38;5;39m"  # Blue
    SERVER_B = "\033[38;5;208m" # Orange
    SYSTEM = "\033[38;5;245m"  # Grey
    HEADER = "\033[38;5;105m" # Purple
    MESSAGE = "\033[38;5;252m" # Light Grey
    ERROR = "\033[38;5;196m"   # Red
    SUCCESS = "\033[38;5;46m"  # Green
    FILE_IO = "\033[38;5;220m" # Yellow for file operations
    PROMPT_ADD = "\033[38;5;154m" # Light Green for prompt adding

SERVER_CONFIG = {
    "A": { "name": "David", "url": "http://127.0.0.1:8080/completion", "color": Colors.SERVER_A, "voice_pref": "pyttsx3_0" },
    "B": { "name": "Zira", "url": "http://127.0.0.1:8081/completion", "color": Colors.SERVER_B, "voice_pref": "pyttsx3_1" }
}

# --- Default values for argparse arguments ---
DEFAULT_REQUEST_TIMEOUT = 45        # Request timeout in seconds
DEFAULT_MAX_TOKENS = 512            # Max tokens per LLM response
DEFAULT_TEMPERATURE = 0.7           # LLM temperature
DEFAULT_NUM_TURNS = 5               # Default conversation turns
DEFAULT_FILE_LOCATION = "conversation_output.md" # For saving ```blocks```
DEFAULT_FILE_DELIMITER = "SAVE_BLOCK"          # Word to agree on saving blocks
DEFAULT_PROMPTS_FILE_PATH = "aiprompts.txt"    # <<< NEW: Default name for the prompts list file
DEFAULT_PROMPT_ADD_DELIMITER = "AGREE_ADD_PROMPT" # <<< NEW: Word to agree on adding a prompt
DEFAULT_INITIAL_PROMPT = "Hi there! Let's have a conversation."

# --- Stop Words ---
BASE_STOP_WORDS = [
    "\n<human>:", "</assistant>", "<|user|>", "<|im_end|>", "</s>",
    "[INST]", "[/INST]", "Human:", "Assistant:",
    f"{SERVER_CONFIG['A']['name']}:", f"{SERVER_CONFIG['B']['name']}:",
    "David:", "Zira:"
]

# --- Utility Functions ---
def colored_print(color: str, message: str):
    """Prints a message to stdout with ANSI colors."""
    try:
        print(f"{color}{message}{Colors.RESET}")
        sys.stdout.flush()
    except Exception as e:
        print(f"(ColorPrint Error: {e}) {message}") # Fallback

# --- TTS Manager ---
# (Keep TTSManager class exactly as before)
class TTSManager:
    # ... (TTSManager code remains unchanged) ...
    """Handles Text-to-Speech initialization, speaking, and cleanup."""
    def __init__(self):
        self.engine = None
        self.engine_type: Optional[str] = None
        self.voices: Dict[str, Optional[str]] = {'A': None, 'B': None}
        self.pygame_initialized: bool = False
        self._initialize()

    def _initialize_pyttsx3(self) -> bool:
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            available_voices = self.engine.getProperty('voices')
            if not available_voices:
                 colored_print(Colors.ERROR,"pyttsx3: No voices found.")
                 return False
            if len(available_voices) > 0: self.voices['A'] = available_voices[0].id
            if len(available_voices) > 1: self.voices['B'] = available_voices[1].id
            else: self.voices['B'] = self.voices['A']
            self.engine_type = "pyttsx3"
            colored_print(Colors.SUCCESS, f"Initialized pyttsx3 TTS. Voice A ID: {self.voices['A']}, Voice B ID: {self.voices['B']}")
            if self.voices['A'] == self.voices['B'] and len(available_voices) < 2:
                 colored_print(Colors.SYSTEM, "Warning: Only one unique pyttsx3 voice found.")
            return True
        except ImportError: return False
        except Exception as e:
            colored_print(Colors.ERROR, f"Error initializing pyttsx3: {e}")
            traceback.print_exc()
            return False

    def _initialize_gtts(self) -> bool:
        try:
            from gtts import gTTS
            import pygame
            if not self.pygame_initialized:
                pygame.mixer.init()
                self.pygame_initialized = True
            self.engine_type = "gtts"
            self.voices['A'] = 'en-us'
            self.voices['B'] = 'en-uk'
            colored_print(Colors.SUCCESS, f"Initialized gTTS + Pygame TTS. Voice A: '{self.voices['A']}', Voice B: '{self.voices['B']}'.")
            return True
        except ImportError: return False
        except Exception as e:
            colored_print(Colors.ERROR, f"Error initializing gTTS/pygame: {e}")
            traceback.print_exc()
            return False

    def _initialize(self):
        if not self._initialize_pyttsx3():
            if not self._initialize_gtts():
                self.engine_type = None
                colored_print(Colors.SYSTEM, "No suitable TTS engine found. TTS disabled.")

    def is_available(self) -> bool:
        return self.engine_type is not None

    def speak(self, text: str, server_id: str):
        if not self.is_available() or not text: return
        voice_id = self.voices.get(server_id)
        # Fallback logic if specific voice ID wasn't found/set
        if not voice_id:
            if self.engine_type == 'gtts': voice_id = self.voices.get('A') or self.voices.get('B') or 'en'
            elif self.engine_type == 'pyttsx3': voice_id = self.voices.get('A') or self.voices.get('B')
            if not voice_id:
                 colored_print(Colors.ERROR, f"No voice usable for server {server_id}. Skipping speech.")
                 return

        start_time = time.time()
        colored_print(Colors.SYSTEM, f"(Speaking as {SERVER_CONFIG[server_id]['name']} using {self.engine_type}...)")
        try:
            if self.engine_type == "pyttsx3": self._speak_pyttsx3(text, voice_id)
            elif self.engine_type == "gtts": self._speak_gtts(text, voice_id)
        except Exception as e:
            colored_print(Colors.ERROR, f"TTS ({self.engine_type}) error during speech: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            end_time = time.time()
            duration = max(0, end_time - start_time)
            colored_print(Colors.SYSTEM, f"(Finished speaking - Duration: {duration:.2f}s)")

    def _speak_pyttsx3(self, text: str, voice_id: str):
        try: self.engine.setProperty('voice', voice_id)
        except Exception as voice_err: colored_print(Colors.ERROR, f"pyttsx3 error setting voice ID '{voice_id}': {voice_err}.")
        self.engine.say(text)
        self.engine.runAndWait()

    def _speak_gtts(self, text: str, lang_code: str):
        import pygame
        from gtts import gTTS
        try:
            tts_obj = gTTS(text=text, lang=lang_code, slow=False)
            with io.BytesIO() as fp:
                tts_obj.write_to_fp(fp)
                fp.seek(0)
                # Wait if music is busy
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.music.load(fp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
        except pygame.error as pg_err: colored_print(Colors.ERROR, f"Pygame mixer error: {pg_err}")
        except Exception as e: colored_print(Colors.ERROR, f"gTTS/Pygame error: {e}")
        finally: # Ensure music stops if error occurred mid-play
             try:
                  if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                       pygame.mixer.music.stop()
                       pygame.mixer.music.unload()
             except Exception: pass

    def shutdown(self):
        if self.engine_type == "gtts" and self.pygame_initialized:
            try:
                import pygame
                if pygame.mixer.get_init():
                     pygame.mixer.music.stop()
                     pygame.mixer.quit()
                     self.pygame_initialized = False
                     colored_print(Colors.SYSTEM, "Pygame mixer shut down.")
            except Exception as e: colored_print(Colors.ERROR, f"Error shutting down pygame mixer: {e}")
        self.engine = None
        self.engine_type = None


# --- Network Request Function ---
# (Keep send_llm_request exactly as before)
def send_llm_request(
    session: requests.Session, server_info: Dict[str, Any], message_history: List[str],
    max_tokens: int, temperature: float, timeout: int,
    stop_words: List[str] = BASE_STOP_WORDS
) -> Optional[str]:
    # ... (send_llm_request code remains unchanged) ...
    url = server_info['url']
    server_name = server_info['name']
    prompt = "\n".join(message_history).strip() + f"\n{server_name}:"

    payload = {
        'prompt': prompt, 'temperature': temperature, 'n_predict': max_tokens,
        'stop': stop_words, 'stream': False,
    }
    headers = {'Content-Type': 'application/json'}
    response_content: Optional[str] = None

    try:
        colored_print(Colors.SYSTEM, f"Sending request to {server_name} (Timeout: {timeout}s, Max Tokens: {max_tokens}, Temp: {temperature})...")
        response = session.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        response_json = response.json()
        content = response_json.get('content', '').strip()
        response_content = content.strip()
        if response_content in stop_words: return None
        if not response_content: return None
    except requests.exceptions.Timeout: colored_print(Colors.ERROR, f"Timeout error ({timeout}s) requesting {server_name}")
    except requests.exceptions.HTTPError as e: colored_print(Colors.ERROR, f"HTTP Error ({server_name}): {e.response.status_code} {e.response.reason}\n{e.response.text[:500]}")
    except requests.exceptions.RequestException as e: colored_print(Colors.ERROR, f"Request Error ({server_name}): {type(e).__name__}: {e}")
    except json.JSONDecodeError: colored_print(Colors.ERROR, f"JSON Decode Error from {server_name}. Response: {response.text[:500] if 'response' in locals() else 'N/A'}")
    except Exception as e: colored_print(Colors.ERROR, f"Unexpected request error ({server_name}): {type(e).__name__}"); traceback.print_exc()
    return response_content


# --- Worker Thread Function ---
# (Keep request_worker exactly as before)
def request_worker(
    session: requests.Session, server_info: Dict[str, Any], history: List[str],
    result_queue: Queue, server_id: str, max_tokens: int, temperature: float,
    timeout: int, stop_words: List[str]
    ):
    # ... (request_worker code remains unchanged) ...
    result = send_llm_request(session, server_info, history, max_tokens, temperature, timeout, stop_words)
    result_queue.put((server_id, result))


# --- Main Conversation Simulation ---
# (Modify signature and add prompt add logic)
def simulate_conversation(
    num_turns: int,
    initial_prompt: str,
    req_timeout: int,
    max_tokens: int,
    temperature: float,
    file_location: Optional[str], # For saving ```blocks```
    file_delimiter: str,          # For saving ```blocks```
    prompts_file_path: str,       # <<< NEW: Path to the aiprompts.txt file
    prompt_add_delimiter: str     # <<< NEW: Delimiter for adding prompts
    ):
    """Runs the conversation simulation with specified parameters."""

    tts_manager = TTSManager()
    request_session = requests.Session()

    conversation_history: List[str] = [f"Human: {initial_prompt}"]

    colored_print(Colors.HEADER, "\n===== CONVERSATION START =====\n")
    colored_print(Colors.MESSAGE, f"Initial Prompt: {conversation_history[0]}")
    if file_location: colored_print(Colors.SYSTEM, f"Block saving enabled: '{file_location}', Delimiter: '{file_delimiter}'")
    else: colored_print(Colors.SYSTEM, "Block saving disabled.")
    colored_print(Colors.SYSTEM, f"Prompt adding enabled: File='{prompts_file_path}', Delimiter: '{prompt_add_delimiter}'")

    result_queue = Queue()
    active_thread: Optional[threading.Thread] = None
    pending_server_id: Optional[str] = None

    # --- State Variables ---
    file_write_state: Optional[str] = None      # For saving blocks
    prompt_add_state: Optional[str] = None      # <<< NEW: For adding prompts
    agreed_file_delimiter = file_delimiter.lower()
    agreed_prompt_add_delimiter = prompt_add_delimiter.lower() # <<< NEW
    # --- End State Variables ---

    server_ids = ['A', 'B']

    try:
        # --- Initial Request (Server A) ---
        first_server_id = server_ids[0]
        first_server_info = SERVER_CONFIG[first_server_id]
        colored_print(Colors.HEADER, f"\n--- Turn 1/{num_turns} ---")
        response = send_llm_request(request_session, first_server_info, conversation_history,
                                    max_tokens, temperature, int(req_timeout * 1.5), BASE_STOP_WORDS)

        # --- Main Loop ---
        for turn in range(num_turns * 2): # Loop through A and B responses
            actual_turn = (turn // 2) + 1
            if actual_turn > num_turns and turn % 2 == 0: break # Check turn limit

            current_speaker_index = turn % 2
            current_server_id = server_ids[current_speaker_index]
            current_server_info = SERVER_CONFIG[current_server_id]
            next_speaker_index = (turn + 1) % 2
            next_server_id = server_ids[next_speaker_index]
            next_server_info = SERVER_CONFIG[next_server_id]

            # --- Process Response for Current Speaker ---
            if response:
                colored_print(current_server_info['color'], f"{current_server_info['name']}: {response}")
                response_lower = response.lower() # For case-insensitive checks

                # =====================================================
                # Interaction Logic (File Write & Prompt Add)
                # =====================================================

                # --- Check for Block Saving ---
                write_block_content = None
                # State: Looking for block from A to write
                if file_write_state == 'A_needs_block' and current_server_id == 'A':
                    match = re.search(r"```(.*?)```", response, re.DOTALL | re.MULTILINE)
                    if match:
                        write_block_content = match.group(1).strip()
                        colored_print(Colors.FILE_IO, f"Server A provided block for saving.")
                    else:
                        colored_print(Colors.SYSTEM, f"Server A agreed to save block but didn't provide ```block```.")
                    file_write_state = None # Reset save state
                # State: Looking for block from B to write
                elif file_write_state == 'B_needs_block' and current_server_id == 'B':
                    match = re.search(r"```(.*?)```", response, re.DOTALL | re.MULTILINE)
                    if match:
                        write_block_content = match.group(1).strip()
                        colored_print(Colors.FILE_IO, f"Server B provided block for saving.")
                    else:
                        colored_print(Colors.SYSTEM, f"Server B agreed to save block but didn't provide ```block```.")
                    file_write_state = None # Reset save state
                # Check for block saving agreement word
                elif agreed_file_delimiter in response_lower:
                    colored_print(Colors.FILE_IO, f"{current_server_info['name']} used the block save delimiter '{file_delimiter}'.")
                    if current_server_id == 'A':
                        if file_write_state == 'B_agreed': file_write_state = 'B_needs_block' # B provides next
                        else: file_write_state = 'A_agreed'
                    elif current_server_id == 'B':
                        if file_write_state == 'A_agreed': file_write_state = 'A_needs_block' # A provides next
                        else: file_write_state = 'B_agreed'
                    # Reset prompt add state if save delimiter used
                    if prompt_add_state: prompt_add_state = None
                # If no save delimiter, reset partial agreement if needed
                elif file_write_state in ['A_agreed', 'B_agreed']:
                     colored_print(Colors.SYSTEM, "Block saving agreement sequence broken.")
                     file_write_state = None

                # --- <<< NEW: Check for Prompt Adding >>> ---
                extracted_prompt_to_add = None
                # State: Looking for prompt text from A
                if prompt_add_state == 'A_needs_prompt' and current_server_id == 'A':
                    # Simple check for "Prompt to add:\n[THE PROMPT]"
                    prompt_marker = "prompt to add:"
                    marker_pos = response_lower.find(prompt_marker)
                    if marker_pos != -1:
                        # Extract everything after the marker and newline
                        start_extract = marker_pos + len(prompt_marker)
                        # Find the first newline after the marker
                        newline_after_marker = response.find('\n', start_extract)
                        if newline_after_marker != -1:
                             extracted_prompt_to_add = response[newline_after_marker:].strip()
                             if extracted_prompt_to_add:
                                 colored_print(Colors.PROMPT_ADD, f"Server A provided prompt to add: '{extracted_prompt_to_add[:50]}...'")
                             else:
                                 colored_print(Colors.ERROR, "Server A used 'Prompt to add:' but provided no text after.")
                        else: # Marker found, but no newline after it
                             extracted_prompt_to_add = response[start_extract:].strip()
                             if extracted_prompt_to_add:
                                 colored_print(Colors.PROMPT_ADD, f"Server A provided prompt (no newline after marker): '{extracted_prompt_to_add[:50]}...'")
                             else:
                                 colored_print(Colors.ERROR, "Server A used 'Prompt to add:' but provided no text after.")
                    else:
                        colored_print(Colors.SYSTEM, f"Server A agreed to add prompt but didn't provide it correctly (missing 'Prompt to add:').")
                    prompt_add_state = None # Reset add state

                # State: Looking for prompt text from B
                elif prompt_add_state == 'B_needs_prompt' and current_server_id == 'B':
                    # Simple check for "Prompt to add:\n[THE PROMPT]"
                    prompt_marker = "prompt to add:"
                    marker_pos = response_lower.find(prompt_marker)
                    if marker_pos != -1:
                        start_extract = marker_pos + len(prompt_marker)
                        newline_after_marker = response.find('\n', start_extract)
                        if newline_after_marker != -1:
                             extracted_prompt_to_add = response[newline_after_marker:].strip()
                             if extracted_prompt_to_add:
                                 colored_print(Colors.PROMPT_ADD, f"Server B provided prompt to add: '{extracted_prompt_to_add[:50]}...'")
                             else:
                                 colored_print(Colors.ERROR, "Server B used 'Prompt to add:' but provided no text after.")
                        else:
                             extracted_prompt_to_add = response[start_extract:].strip()
                             if extracted_prompt_to_add:
                                 colored_print(Colors.PROMPT_ADD, f"Server B provided prompt (no newline after marker): '{extracted_prompt_to_add[:50]}...'")
                             else:
                                 colored_print(Colors.ERROR, "Server B used 'Prompt to add:' but provided no text after.")
                    else:
                        colored_print(Colors.SYSTEM, f"Server B agreed to add prompt but didn't provide it correctly (missing 'Prompt to add:').")
                    prompt_add_state = None # Reset add state

                # Check for prompt adding agreement word (only if not currently agreeing to save block)
                elif agreed_prompt_add_delimiter in response_lower and not file_write_state:
                    colored_print(Colors.PROMPT_ADD, f"{current_server_info['name']} used the prompt add delimiter '{prompt_add_delimiter}'.")
                    if current_server_id == 'A':
                        if prompt_add_state == 'B_agreed_add':
                            colored_print(Colors.PROMPT_ADD, "Prompt add agreement reached! Waiting for prompt from Server B.")
                            prompt_add_state = 'B_needs_prompt' # B provides next
                        else:
                            prompt_add_state = 'A_agreed_add'
                    elif current_server_id == 'B':
                        if prompt_add_state == 'A_agreed_add':
                            colored_print(Colors.PROMPT_ADD, "Prompt add agreement reached! Waiting for prompt from Server A.")
                            prompt_add_state = 'A_needs_prompt' # A provides next
                        else:
                            prompt_add_state = 'B_agreed_add'
                # If no add delimiter, reset partial agreement if needed
                elif prompt_add_state in ['A_agreed_add', 'B_agreed_add']:
                     colored_print(Colors.SYSTEM, "Prompt adding agreement sequence broken.")
                     prompt_add_state = None
                # --- <<< End Prompt Adding Check >>> ---

                # =====================================================
                # Perform Actions (File Write & Prompt Add)
                # =====================================================

                # --- Write Block to File ---
                if write_block_content is not None and file_location:
                    try:
                        with open(file_location, 'a', encoding='utf-8') as f:
                            f.write(f"\n\n---\n\n```\n{write_block_content}\n```\n")
                            f.write(f"\n__{current_server_info['name']} at Turn {actual_turn}__\n")
                        colored_print(Colors.SUCCESS, f"Successfully wrote block to '{file_location}'")
                    except Exception as e:
                        colored_print(Colors.ERROR, f"Error writing block to file '{file_location}': {e}")
                    file_write_state = None # Ensure reset after attempt

                elif write_block_content is not None and not file_location:
                     colored_print(Colors.ERROR, f"Block provided but no file location specified. Block ignored.")
                     file_write_state = None

                # --- <<< NEW: Add Prompt to File >>> ---
                if extracted_prompt_to_add:
                    try:
                        # Make sure prompts_file_path exists before appending
                        if not os.path.exists(prompts_file_path):
                            colored_print(Colors.ERROR, f"Prompts file '{prompts_file_path}' not found. Cannot add prompt.")
                        else:
                            with open(prompts_file_path, 'a', encoding='utf-8') as f:
                                f.write(f"{extracted_prompt_to_add}\n") # Add prompt and newline
                            colored_print(Colors.SUCCESS, f"Successfully added prompt to '{prompts_file_path}'")
                    except Exception as e:
                        colored_print(Colors.ERROR, f"Error adding prompt to file '{prompts_file_path}': {e}")
                    prompt_add_state = None # Ensure reset after attempt
                # --- <<< End Add Prompt >>> ---

                # =====================================================
                # Continue Conversation Flow
                # =====================================================
                conversation_history.append(f"{current_server_info['name']}: {response}")

                # Start next request thread
                if turn < (num_turns * 2) - 1:
                    # ... (thread starting logic remains the same) ...
                    colored_print(Colors.SYSTEM, f"Starting background request for {next_server_info['name']}...")
                    active_thread = threading.Thread(
                        target=request_worker,
                        args=(request_session, next_server_info, conversation_history.copy(), result_queue, next_server_id,
                              max_tokens, temperature, req_timeout, BASE_STOP_WORDS), daemon=True)
                    active_thread.start()
                    pending_server_id = next_server_id
                else:
                     active_thread = None; pending_server_id = None

                # Speak current response
                tts_manager.speak(response, current_server_id)

            else: # Current speaker failed
                colored_print(Colors.ERROR, f"{current_server_info['name']} failed to respond or returned empty.")
                # Reset any partial agreements on failure
                if file_write_state in ['A_agreed', 'B_agreed']: file_write_state = None
                if prompt_add_state in ['A_agreed_add', 'B_agreed_add']: prompt_add_state = None

                # Still try to start next request
                if turn < (num_turns * 2) - 1:
                    # ... (thread starting logic after failure remains the same) ...
                    colored_print(Colors.SYSTEM, f"Starting background request for {next_server_info['name']} (after failure)...")
                    active_thread = threading.Thread(
                         target=request_worker,
                         args=(request_session, next_server_info, conversation_history.copy(), result_queue, next_server_id,
                               max_tokens, temperature, req_timeout, BASE_STOP_WORDS), daemon=True)
                    active_thread.start()
                    pending_server_id = next_server_id
                else:
                    active_thread = None; pending_server_id = None

            # --- Wait for Next Result ---
            if active_thread and pending_server_id:
                # ... (queue waiting logic remains the same) ...
                next_server_name_wait = SERVER_CONFIG[pending_server_id]['name']
                colored_print(Colors.SYSTEM, f"Waiting for {next_server_name_wait}'s result...")
                try:
                    q_timeout = req_timeout + 15
                    retrieved_id, response = result_queue.get(timeout=q_timeout)
                    if retrieved_id != pending_server_id:
                        colored_print(Colors.ERROR, f"Queue Logic Error: Expected {pending_server_id}, got {retrieved_id}.")
                        response = None; # Discard
                        try: retrieved_id, response = result_queue.get_nowait() # Quick check
                        except Empty: pass
                        if retrieved_id != pending_server_id: response = None
                    active_thread.join(timeout=1.0)
                    if active_thread.is_alive(): colored_print(Colors.SYSTEM, f"Warning: Thread {active_thread.name} did not exit.")
                    active_thread = None; pending_server_id = None
                except Empty:
                    colored_print(Colors.ERROR, f"Timeout ({q_timeout}s) waiting for {next_server_name_wait} queue. Assuming failure.")
                    response = None; active_thread = None; pending_server_id = None
                except Exception as e:
                    colored_print(Colors.ERROR, f"Queue error: {e}"); traceback.print_exc()
                    response = None; active_thread = None; pending_server_id = None


            # End of loop processing
            if turn >= (num_turns * 2) - 1: break # Reached max turns

            # Print header for next logical turn
            next_actual_turn = ((turn + 1) // 2) + 1
            if response is not None and next_actual_turn > actual_turn:
                 colored_print(Colors.HEADER, f"\n--- Turn {next_actual_turn}/{num_turns} ---")


    except KeyboardInterrupt: colored_print(Colors.SYSTEM, "\nInterrupted.")
    except Exception as e: colored_print(Colors.ERROR, f"\nUnexpected simulation error:"); traceback.print_exc()
    finally:
        colored_print(Colors.HEADER, "\n===== CONVERSATION END =====\n")
        if active_thread and active_thread.is_alive(): active_thread.join(timeout=2.0)
        tts_manager.shutdown()
        request_session.close()
        colored_print(Colors.SYSTEM, "Resources cleaned up.")


# ===========================================================
# Argparse Setup and Main Execution Block
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate an LLM conversation with TTS, block saving, and prompt adding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter )

    # Core Conversation Args
    parser.add_argument('-p', '--prompt', type=str, default=None, help="Initial conversation prompt. Overrides default.")
    parser.add_argument('-t', '--turns', type=int, default=DEFAULT_NUM_TURNS, help="Number of conversation turns (A & B).")
    parser.add_argument('--human', action='store_true', help="Append human roleplaying instruction.")

    # LLM Parameter Args
    parser.add_argument('--timeout', type=int, default=DEFAULT_REQUEST_TIMEOUT, help="LLM request timeout (seconds).")
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per LLM response.")
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE, help="LLM sampling temperature.")

    # Block File Saving Args
    parser.add_argument('--file_location', type=str, default=None, help=f"Path for saving ```blocks```. Default when enabled: '{DEFAULT_FILE_LOCATION}'")
    parser.add_argument('--file_delimiter', type=str, default=DEFAULT_FILE_DELIMITER, help="Word to agree on saving blocks.")

    # <<< NEW: Prompt Adding Args >>>
    parser.add_argument('--prompts_file', type=str, default=DEFAULT_PROMPTS_FILE_PATH,
                        help="Path to the file containing prompts (for reading by batch and potential writing by AI).")
    parser.add_argument('--prompt_add_delimiter', type=str, default=DEFAULT_PROMPT_ADD_DELIMITER,
                        help="Word for AIs to agree on adding a new prompt to the prompts file.")
    # <<< END NEW >>>

    args = parser.parse_args()

    # --- Post-process arguments ---
    final_initial_prompt = args.prompt if args.prompt is not None else DEFAULT_INITIAL_PROMPT
    if args.human:
        human_clause = " Remember, you are roleplaying as a human and should not reveal you are an AI."
        if human_clause.lower() not in final_initial_prompt.lower(): final_initial_prompt += human_clause

    file_loc = args.file_location # Keep None unless explicitly set

    # --- Call the main function ---
    simulate_conversation(
        num_turns=args.turns,
        initial_prompt=final_initial_prompt,
        req_timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        file_location=file_loc,
        file_delimiter=args.file_delimiter,
        prompts_file_path=args.prompts_file,           # <<< Pass prompts file path
        prompt_add_delimiter=args.prompt_add_delimiter # <<< Pass prompt add delimiter
    )
