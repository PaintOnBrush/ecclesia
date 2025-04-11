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
import argparse # <--- Already imported, ensure it's there

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

SERVER_CONFIG = {
    "A": {
        "name": "David",
        "url": "http://127.0.0.1:8080/completion",
        "color": Colors.SERVER_A,
        "voice_pref": "pyttsx3_0"
    },
    "B": {
        "name": "Zira",
        "url": "http://127.0.0.1:8081/completion",
        "color": Colors.SERVER_B,
        "voice_pref": "pyttsx3_1"
    }
}

# --- >> NEW: Default values for argparse arguments << ---
DEFAULT_REQUEST_TIMEOUT = 45        # Request timeout in seconds
DEFAULT_MAX_TOKENS = 512            # Max tokens per LLM response
DEFAULT_TEMPERATURE = 0.7           # LLM temperature
DEFAULT_CONTEXT_LENGTH = 2048       # Placeholder for context (not directly used in basic /completion)
DEFAULT_NUM_TURNS = 5               # Default conversation turns
DEFAULT_FILE_LOCATION = "conversation_output.md" # Default file for saving blocks
DEFAULT_FILE_DELIMITER = "SAVE_BLOCK" # Default agreement word
DEFAULT_INITIAL_PROMPT = (
    "Hi there! Let's have a conversation about ways to make people laugh, "
    "have joy, and remove tragedy." # Base prompt, human part added via flag
)
# --- >> END NEW DEFAULTS << ---

# Define stop words (can be customized further)
BASE_STOP_WORDS = [
    "\n<human>:", "</assistant>", "<|user|>", "<|im_end|>", "</s>",
    "[INST]", "[/INST]", "Human:", "Assistant:",
    # Add speaker names based on SERVER_CONFIG to prevent self-continuation
    f"{SERVER_CONFIG['A']['name']}:",
    f"{SERVER_CONFIG['B']['name']}:",
    # Add simpler names too as fallback
    "David:", "Zira:"
]


# --- Utility Functions ---
# (Keep colored_print as is)
def colored_print(color: str, message: str):
    """Prints a message to stdout with ANSI colors."""
    try:
        print(f"{color}{message}{Colors.RESET}")
        sys.stdout.flush()
    except BrokenPipeError:
        try: sys.stdout.close()
        except Exception: pass
        try: sys.stderr.close()
        except Exception: pass
    except Exception as e:
        print(f"Error in colored_print: {e}")
        print(message)

# --- TTS Manager ---
# (Keep TTSManager class as is, including fixes from previous steps)
class TTSManager:
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
# (Modify signature to accept params, use BASE_STOP_WORDS)
def send_llm_request(
    session: requests.Session,
    server_info: Dict[str, Any],
    message_history: List[str],
    max_tokens: int,         # Use passed arg
    temperature: float,      # Use passed arg
    timeout: int,            # Use passed arg
    # context_length: int,   # NOTE: context_length is not directly used by basic /completion endpoint
    stop_words: List[str] = BASE_STOP_WORDS # Use base list by default
) -> Optional[str]:
    """
    Sends a request to the specified LLM server using provided parameters.
    """
    url = server_info['url']
    server_name = server_info['name']
    prompt = "\n".join(message_history).strip() + f"\n{server_name}:"

    payload = {
        'prompt': prompt,
        'temperature': temperature,
        'n_predict': max_tokens,
        'stop': stop_words, # Use the provided/default stop words
        'stream': False,
        # 'n_ctx': context_length # Add this if your server supports/needs it
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

    # (Keep existing error handling as is)
    except requests.exceptions.Timeout: colored_print(Colors.ERROR, f"Timeout error ({timeout}s) requesting {server_name}")
    except requests.exceptions.HTTPError as e: colored_print(Colors.ERROR, f"HTTP Error ({server_name}): {e.response.status_code} {e.response.reason}\n{e.response.text[:500]}")
    except requests.exceptions.RequestException as e: colored_print(Colors.ERROR, f"Request Error ({server_name}): {type(e).__name__}: {e}")
    except json.JSONDecodeError: colored_print(Colors.ERROR, f"JSON Decode Error from {server_name}. Response: {response.text[:500] if 'response' in locals() else 'N/A'}")
    except Exception as e: colored_print(Colors.ERROR, f"Unexpected request error ({server_name}): {type(e).__name__}"); traceback.print_exc()

    return response_content


# --- Worker Thread Function ---
# (Modify signature to accept params)
def request_worker(
    session: requests.Session,
    server_info: Dict[str, Any],
    history: List[str],
    result_queue: Queue,
    server_id: str,
    max_tokens: int,        # Pass through
    temperature: float,     # Pass through
    timeout: int,           # Pass through
    # context_length: int,  # Pass through if needed
    stop_words: List[str]   # Pass through
    ):
    """Target function for the request thread."""
    result = send_llm_request(
        session=session,
        server_info=server_info,
        message_history=history,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_words=stop_words,
        timeout=timeout,
        # context_length=context_length # Pass if added to send_llm_request
    )
    result_queue.put((server_id, result))


# --- Main Conversation Simulation ---
# (Modify signature to accept all new args)
def simulate_conversation(
    num_turns: int,
    initial_prompt: str,
    req_timeout: int,
    max_tokens: int,
    temperature: float,
    # context_length: int, # Accept if using
    file_location: Optional[str], # Can be None if not specified
    file_delimiter: str
    ):
    """Runs the conversation simulation with specified parameters."""

    tts_manager = TTSManager()
    request_session = requests.Session()

    conversation_history: List[str] = [f"Human: {initial_prompt}"]

    colored_print(Colors.HEADER, "\n===== CONVERSATION START =====\n")
    colored_print(Colors.MESSAGE, f"Initial Prompt: {conversation_history[0]}")
    if file_location:
        colored_print(Colors.SYSTEM, f"File output enabled: '{file_location}', Delimiter: '{file_delimiter}'")
    else:
        colored_print(Colors.SYSTEM, "File output disabled.")


    result_queue = Queue()
    active_thread: Optional[threading.Thread] = None
    pending_server_id: Optional[str] = None

    # --- >> NEW: File Writing State Variables << ---
    file_write_state: Optional[str] = None # None | 'A_agreed' | 'B_agreed' | 'A_needs_block' | 'B_needs_block'
    agreed_delimiter = file_delimiter.lower() # Use lower case for checking
    # --- >> END NEW << ---

    server_ids = ['A', 'B']

    try:
        # --- Initial Request (Server A) ---
        first_server_id = server_ids[0]
        first_server_info = SERVER_CONFIG[first_server_id]
        colored_print(Colors.HEADER, f"\n--- Turn 1/{num_turns} ---")
        response = send_llm_request(
            session=request_session,
            server_info=first_server_info,
            message_history=conversation_history,
            max_tokens=max_tokens,      # Use arg
            temperature=temperature,    # Use arg
            stop_words=BASE_STOP_WORDS, # Use base stop words
            timeout=req_timeout * 1.5   # Use arg (slightly longer timeout for first request)
            # context_length=context_length # Pass if using
        )

        # --- Main Loop ---
        for turn in range(num_turns * 2): # Loop through A and B responses for num_turns
            actual_turn = (turn // 2) + 1
            if actual_turn > num_turns and turn % 2 == 0: # Check if we exceeded desired turns after A finished
                 break

            current_speaker_index = turn % 2
            current_server_id = server_ids[current_speaker_index]
            current_server_info = SERVER_CONFIG[current_server_id]
            next_speaker_index = (turn + 1) % 2
            next_server_id = server_ids[next_speaker_index]
            next_server_info = SERVER_CONFIG[next_server_id]

            # --- Process Response for Current Speaker ---
            if response:
                colored_print(current_server_info['color'], f"{current_server_info['name']}: {response}")

                # --- >> Check for File Writing Agreement/Block << ---
                response_lower = response.lower()
                write_block_content = None

                # State: Looking for block from A
                if file_write_state == 'A_needs_block' and current_server_id == 'A':
                    match = re.search(r"```(.*?)```", response, re.DOTALL | re.MULTILINE)
                    if match:
                        write_block_content = match.group(1).strip()
                        colored_print(Colors.FILE_IO, f"Server A provided block after agreement.")
                    else:
                        colored_print(Colors.SYSTEM, f"Server A agreed to write but didn't provide ```block```.")
                    file_write_state = None # Reset state after checking A's response

                # State: Looking for block from B
                elif file_write_state == 'B_needs_block' and current_server_id == 'B':
                    match = re.search(r"```(.*?)```", response, re.DOTALL | re.MULTILINE)
                    if match:
                        write_block_content = match.group(1).strip()
                        colored_print(Colors.FILE_IO, f"Server B provided block after agreement.")
                    else:
                        colored_print(Colors.SYSTEM, f"Server B agreed to write but didn't provide ```block```.")
                    file_write_state = None # Reset state after checking B's response

                # Check for delimiter agreement word
                elif agreed_delimiter in response_lower:
                    colored_print(Colors.FILE_IO, f"{current_server_info['name']} used the delimiter '{file_delimiter}'.")
                    if current_server_id == 'A':
                        if file_write_state == 'B_agreed':
                             colored_print(Colors.FILE_IO, "Agreement reached! Waiting for block from Server B.")
                             file_write_state = 'B_needs_block' # B agreed first, A confirmed. B provides block.
                        else:
                             file_write_state = 'A_agreed' # A is first to agree
                    elif current_server_id == 'B':
                        if file_write_state == 'A_agreed':
                             colored_print(Colors.FILE_IO, "Agreement reached! Waiting for block from Server A.")
                             file_write_state = 'A_needs_block' # A agreed first, B confirmed. A provides block.
                        else:
                             file_write_state = 'B_agreed' # B is first to agree
                # If no delimiter found, reset agreement state if it was partially set
                elif file_write_state in ['A_agreed', 'B_agreed']:
                     colored_print(Colors.SYSTEM, "File write agreement sequence broken.")
                     file_write_state = None

                 # --- >> Write Block to File if Found << ---
                if write_block_content is not None and file_location:
                    try:
                        with open(file_location, 'a', encoding='utf-8') as f:
                            f.write(f"\n\n---\n\n") # Separator
                            f.write(f"```\n{write_block_content}\n```\n")
                            f.write(f"\n__{current_server_info['name']} at Turn {actual_turn}__\n")
                        colored_print(Colors.SUCCESS, f"Successfully wrote block to '{file_location}'")
                    except IOError as e:
                        colored_print(Colors.ERROR, f"Error writing to file '{file_location}': {e}")
                    except Exception as e:
                        colored_print(Colors.ERROR, f"Unexpected error during file write: {e}")
                    # Reset state just in case (should already be None)
                    file_write_state = None
                elif write_block_content is not None and not file_location:
                     colored_print(Colors.ERROR, f"Block provided by {current_server_info['name']} but no file location specified (--file_location). Block ignored.")
                     file_write_state = None # Reset state

                # --- >> End File Writing Logic << ---

                conversation_history.append(f"{current_server_info['name']}: {response}")

                # Start next request *before* speaking current response (if not the very last response)
                if turn < (num_turns * 2) - 1:
                    colored_print(Colors.SYSTEM, f"Starting background request for {next_server_info['name']}...")
                    active_thread = threading.Thread(
                        target=request_worker,
                        args=(
                            request_session, next_server_info, conversation_history.copy(),
                            result_queue, next_server_id,
                            max_tokens, temperature, req_timeout, BASE_STOP_WORDS # Pass args
                            # context_length, # Pass if using
                        ),
                        daemon=True
                    )
                    active_thread.start()
                    pending_server_id = next_server_id
                else:
                     active_thread = None
                     pending_server_id = None

                # Speak current response
                tts_manager.speak(response, current_server_id)

            else:
                # Current speaker failed or returned empty
                colored_print(Colors.ERROR, f"{current_server_info['name']} failed to respond or returned empty.")
                # Reset partial file write agreement if failure occurs
                if file_write_state in ['A_agreed', 'B_agreed']: file_write_state = None

                # Still try to start the next request based on history *before* failure
                if turn < (num_turns * 2) - 1:
                    colored_print(Colors.SYSTEM, f"Starting background request for {next_server_info['name']} (after {current_server_info['name']} failed)...")
                    active_thread = threading.Thread(
                         target=request_worker,
                         args=(
                             request_session, next_server_info, conversation_history.copy(),
                             result_queue, next_server_id,
                             max_tokens, temperature, req_timeout, BASE_STOP_WORDS # Pass args
                            # context_length, # Pass if using
                         ),
                         daemon=True
                    )
                    active_thread.start()
                    pending_server_id = next_server_id
                else:
                    active_thread = None
                    pending_server_id = None


            # --- Wait for the Next Speaker's Result ---
            if active_thread and pending_server_id:
                 next_server_name_wait = SERVER_CONFIG[pending_server_id]['name']
                 colored_print(Colors.SYSTEM, f"Waiting for {next_server_name_wait}'s result...")
                 try:
                     q_timeout = req_timeout + 15 # Generous timeout for queue
                     retrieved_id, response = result_queue.get(timeout=q_timeout)

                     if retrieved_id != pending_server_id:
                          colored_print(Colors.ERROR, f"Queue Logic Error: Expected {pending_server_id}, got {retrieved_id}. Discarding.")
                          response = None # Discard unexpected result
                          try: # Quick check if correct one is immediately available
                               retrieved_id, response = result_queue.get_nowait()
                               if retrieved_id != pending_server_id: response = None
                          except Empty: response = None
                     # Join the completed thread
                     active_thread.join(timeout=1.0) # Short timeout, should be done
                     if active_thread.is_alive():
                         colored_print(Colors.SYSTEM, f"Warning: Thread for {next_server_name_wait} did not exit cleanly after join.")
                     active_thread = None
                     pending_server_id = None

                 except Empty:
                      colored_print(Colors.ERROR, f"Timeout ({q_timeout}s) waiting for result from {next_server_name_wait} queue. Assuming failure.")
                      response = None
                      active_thread = None
                      pending_server_id = None
                 except Exception as e:
                      colored_print(Colors.ERROR, f"Error getting result from queue: {e}")
                      traceback.print_exc()
                      response = None
                      active_thread = None
                      pending_server_id = None

            # End of loop processing, prepare for next iteration or exit
            # Check if the *entire conversation* should end
            if turn >= (num_turns * 2) - 1:
                 break # Reached max turns for both A and B

            # Print header for the next logical turn number if response was successful
            next_actual_turn = ((turn + 1) // 2) + 1
            if response is not None and next_actual_turn > actual_turn:
                 colored_print(Colors.HEADER, f"\n--- Turn {next_actual_turn}/{num_turns} ---")


    except KeyboardInterrupt:
        colored_print(Colors.SYSTEM, "\nConversation interrupted by user.")
    except Exception as e:
        colored_print(Colors.ERROR, f"\nAn unexpected error occurred during simulation:")
        traceback.print_exc()
    finally:
        colored_print(Colors.HEADER, "\n===== CONVERSATION END =====\n")
        # Ensure final thread cleanup attempt
        if active_thread and active_thread.is_alive():
            colored_print(Colors.SYSTEM, "Waiting briefly for final orphaned thread...")
            active_thread.join(timeout=2.0)
        tts_manager.shutdown()
        request_session.close()
        colored_print(Colors.SYSTEM, "Resources cleaned up.")


# ===========================================================
# Argparse Setup and Main Execution Block
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate a conversation between two LLM servers with TTS and optional file writing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # Core Conversation Arguments
    parser.add_argument(
        '-p', '--prompt', type=str, default=None, # Default handled below
        help="The initial prompt to start the conversation. Overrides default."
    )
    parser.add_argument(
        '-t', '--turns', type=int, default=DEFAULT_NUM_TURNS,
        help="Number of conversation turns (one turn = response from A and B)."
    )
    parser.add_argument(
        '--human', action='store_true',
        help="Append 'Remember, you are roleplaying as a human...' to the initial prompt."
    )

    # LLM Parameter Arguments
    parser.add_argument(
        '--timeout', type=int, default=DEFAULT_REQUEST_TIMEOUT,
        help="Timeout in seconds for each LLM request."
    )
    parser.add_argument(
        '--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens for the LLM to generate per response."
    )
    parser.add_argument(
        '--temperature', type=float, default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for the LLM (e.g., 0.7)."
    )
    # parser.add_argument(
    #     '--context_length', type=int, default=DEFAULT_CONTEXT_LENGTH,
    #     help="Context length for the LLM (Note: Not directly used by basic /completion endpoint)."
    # ) # Uncomment if needed

    # File Writing Arguments
    parser.add_argument(
        '--file_location', type=str, default=None, # Default None means disabled unless explicitly set
        help=f"Path to the file for saving agreed-upon code blocks. If omitted, saving is disabled. Default when enabled: '{DEFAULT_FILE_LOCATION}'"
    )
    parser.add_argument(
        '--file_delimiter', type=str, default=DEFAULT_FILE_DELIMITER,
        help="The specific word servers must use consecutively to agree to save the next ```block```."
    )


    args = parser.parse_args()

    # --- Post-process arguments ---

    # Handle initial prompt composition
    final_initial_prompt = args.prompt if args.prompt is not None else DEFAULT_INITIAL_PROMPT
    if args.human:
        human_clause = " Remember, you are roleplaying as a human and should not reveal you are an AI."
        # Avoid adding if already present (simple check)
        if human_clause.lower() not in final_initial_prompt.lower():
             final_initial_prompt += human_clause

    # Handle file location default enablement
    file_loc = args.file_location
    if file_loc is None and DEFAULT_FILE_LOCATION:
         # If user didn't specify --file_location, check if they *intended* to use the default implicitly
         # For now, let's require explicit --file_location to enable saving.
         # If you want the default file to be used *unless* saving is somehow disabled, change this.
         # file_loc = DEFAULT_FILE_LOCATION # Uncomment this line to enable default file path automatically
         pass # Keep file_loc as None (disabled) if not explicitly provided

    # --- Call the main function ---
    simulate_conversation(
        num_turns=args.turns,
        initial_prompt=final_initial_prompt,
        req_timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        # context_length=args.context_length, # Pass if using
        file_location=file_loc,             # Pass the determined file path (or None)
        file_delimiter=args.file_delimiter
    )
