import requests
import json
import time
import sys
import os
import traceback
import re
import threading
import io  # Needed for gTTS in-memory audio
from queue import Queue, Empty # Use Queue for thread-safe result passing
from typing import List, Optional, Tuple, Dict, Any

# --- Configuration ---

class Colors:
    RESET = "\033[0m"
    SERVER_A = "\033[38;5;39m"  # Blue
    SERVER_B = "\033[38;5;208m" # Orange
    SYSTEM = "\033[38;5;245m"  # Grey
    HEADER = "\033[38;5;105m" # Purple
    MESSAGE = "\033[38;5;252m" # Light Grey
    ERROR = "\033[38;5;196m"   # Red
    SUCCESS = "\033[38;5;46m"  # Green

# Consider moving to a config file or command-line args for more flexibility
SERVER_CONFIG = {
    "A": {
        "name": "David (8080)",
        "url": "http://127.0.0.1:8080/completion",
        "color": Colors.SERVER_A,
        "voice_pref": "pyttsx3_0" # Placeholder, actual voice ID assigned by TTSManager
    },
    "B": {
        "name": "Zira (8081)",
        "url": "http://127.0.0.1:8081/completion",
        "color": Colors.SERVER_B,
        "voice_pref": "pyttsx3_1" # Placeholder
    }
}

# LLM Request Defaults
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_STOP_WORDS = [
    "\n<human>:", "</assistant>", "<|user|>", "<|im_end|>", "</s>",
    "[INST]", "[/INST]", "Human:", "Assistant:",
    # Make sure these EXACT strings are included and match the speaker names
    # used in the prompt and likely generated by the model.
    "David (8080):",  # Or maybe just "David:" if the model doesn't add the port
    "Zira (8081):"   # Or maybe just "Zira:" # Added speaker names
]
REQUEST_TIMEOUT = 45 # Seconds

# Conversation Defaults
DEFAULT_NUM_TURNS = 5
DEFAULT_INITIAL_PROMPT = (
    "Lets have a conversational tournament about which plant is the healthiest plant towards mankind. When it's over start another one you have 250 messages ,excluding this one in total. "
    #"let\'s have a conversation on growing chacruna. Start by discussing toxicity of the chacruna plant or whether it is safe."
    #"let's have a conversation about stephen king."
    #"let's talk about how to make my girlfriend laugh."
    #"let's talk about how to have a good time at work flipping burgers"rfc
    #" Remember, you are roleplaying as a human and should not reveal you are an AI. "
 #   "Start by discussing a specific technique." # Adjusted for a better start
)

# --- Utility Functions ---

def colored_print(color: str, message: str):
    """Prints a message to stdout with ANSI colors."""
    try:
        print(f"{color}{message}{Colors.RESET}")
        sys.stdout.flush()
    except BrokenPipeError:
        # Handle cases where the output pipe is closed (e.g., piping to `head`)
        try: sys.stdout.close()
        except Exception: pass
        try: sys.stderr.close()
        except Exception: pass
    except Exception as e:
        # Fallback if coloring fails for some reason
        print(f"Error in colored_print: {e}")
        print(message) # Print without color

# --- TTS Manager ---

class TTSManager:
    """Handles Text-to-Speech initialization, speaking, and cleanup."""
    def __init__(self):
        self.engine = None
        self.engine_type: Optional[str] = None
        self.voices: Dict[str, Optional[str]] = {'A': None, 'B': None}
        self.pygame_initialized: bool = False
        self._initialize()

    def _initialize_pyttsx3(self) -> bool:
        """Attempts to initialize pyttsx3."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            available_voices = self.engine.getProperty('voices')
            if len(available_voices) > 0:
                self.voices['A'] = available_voices[0].id
            if len(available_voices) > 1:
                self.voices['B'] = available_voices[1].id
            else:
                self.voices['B'] = self.voices['A'] # Fallback to same voice

            self.engine_type = "pyttsx3"
            colored_print(Colors.SUCCESS, f"Initialized pyttsx3 TTS. Voice A: {self.voices['A']}, Voice B: {self.voices['B']}")
            if self.voices['A'] == self.voices['B'] and len(available_voices) < 2:
                 colored_print(Colors.SYSTEM, "Warning: Only one pyttsx3 voice found or available.")
            return True
        except ImportError:
            colored_print(Colors.SYSTEM, "pyttsx3 not found. Trying gTTS...")
            return False
        except Exception as e:
            colored_print(Colors.ERROR, f"Error initializing pyttsx3: {e}")
            traceback.print_exc()
            return False

    def _initialize_gtts(self) -> bool:
        """Attempts to initialize gTTS with Pygame."""
        try:
            from gtts import gTTS
            import pygame
            if not self.pygame_initialized:
                pygame.mixer.init()
                self.pygame_initialized = True
            self.engine_type = "gtts"
            # Define desired accents/languages for gTTS
            self.voices['A'] = 'en-us' # Example: American English for A
            self.voices['B'] = 'en-uk' # Example: British English for B
            colored_print(Colors.SUCCESS, f"Initialized gTTS + Pygame TTS. Voice A: '{self.voices['A']}', Voice B: '{self.voices['B']}'.")
            return True
        except ImportError:
            colored_print(Colors.SYSTEM, "gTTS or Pygame not found.")
            return False
        except Exception as e:
            colored_print(Colors.ERROR, f"Error initializing gTTS/pygame: {e}")
            traceback.print_exc()
            return False

    def _initialize(self):
        """Initializes the first available TTS engine."""
        if self._initialize_pyttsx3():
            return
        if self._initialize_gtts():
            return

        self.engine_type = None
        colored_print(Colors.SYSTEM, "No suitable TTS engine found. TTS disabled.")

    def is_available(self) -> bool:
        """Checks if TTS is initialized."""
        return self.engine_type is not None

    def speak(self, text: str, server_id: str):
        """ Speaks the given text using the appropriate voice for the server_id ('A' or 'B'). Runs synchronously."""
        if not self.is_available() or not text:
            return

        voice_id = self.voices.get(server_id)
        if not voice_id:
            colored_print(Colors.ERROR, f"No voice configured for server {server_id}. Skipping speech.")
            return

        start_time = time.time()
        colored_print(Colors.SYSTEM, f"(Speaking as {SERVER_CONFIG[server_id]['name']} using {self.engine_type}...)")

        try:
            if self.engine_type == "pyttsx3":
                self._speak_pyttsx3(text, voice_id)
            elif self.engine_type == "gtts":
                self._speak_gtts(text, voice_id)

        except Exception as e:
            colored_print(Colors.ERROR, f"TTS ({self.engine_type}) error during speech: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            end_time = time.time()
            # Guard against division by zero if speech failed instantly
            duration = end_time - start_time if end_time > start_time else 0
            colored_print(Colors.SYSTEM, f"(Finished speaking - Duration: {duration:.2f}s)")

    def _speak_pyttsx3(self, text: str, voice_id: str):
        """Internal method for pyttsx3 speech."""
        try:
            self.engine.setProperty('voice', voice_id)
        except Exception as voice_err:
            colored_print(Colors.ERROR, f"pyttsx3 error setting voice '{voice_id}': {voice_err}. Using default.")
        self.engine.say(text)
        self.engine.runAndWait() # This blocks

    def _speak_gtts(self, text: str, lang_code: str):
        """Internal method for gTTS speech using in-memory data."""
        import pygame # Ensure pygame is accessible
        from gtts import gTTS
        try:
            tts_obj = gTTS(text=text, lang=lang_code, slow=False)
            with io.BytesIO() as fp:
                tts_obj.write_to_fp(fp)
                fp.seek(0)
                pygame.mixer.music.load(fp) # Load from file-like object
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10) # Keep CPU low, prevents busy-waiting
            # Ensure music stops and unloads if loop finishes prematurely
            if pygame.mixer.music.get_busy():
                 pygame.mixer.music.stop()
                 pygame.mixer.music.unload() # Necessary? Let's be safe.
        except Exception as e:
             colored_print(Colors.ERROR, f"gTTS/Pygame speaking error: {e}")
             # Attempt to stop music if it was playing
             try:
                 if pygame.mixer.music.get_busy():
                      pygame.mixer.music.stop()
                      pygame.mixer.music.unload()
             except Exception:
                 pass # Ignore errors during cleanup attempt


    def shutdown(self):
        """Cleans up TTS resources."""
        if self.engine_type == "pyttsx3" and self.engine:
            try:
                # Optional: Stop any ongoing speech if possible (depends on driver)
                # self.engine.stop()
                pass # pyttsx3 doesn't have an explicit shutdown
            except Exception as e:
                colored_print(Colors.ERROR, f"Error during pyttsx3 shutdown: {e}")
        elif self.engine_type == "gtts" and self.pygame_initialized:
            try:
                import pygame
                pygame.mixer.quit()
                self.pygame_initialized = False
                colored_print(Colors.SYSTEM, "Pygame mixer shut down.")
            except Exception as e:
                colored_print(Colors.ERROR, f"Error shutting down pygame mixer: {e}")
        self.engine = None
        self.engine_type = None


# --- Network Request Function ---

def send_llm_request(
    session: requests.Session,
    server_info: Dict[str, Any],
    message_history: List[str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    stop_words: List[str] = DEFAULT_STOP_WORDS,
    timeout: int = REQUEST_TIMEOUT
) -> Optional[str]:
    """
    Sends a request to the specified LLM server and returns the response content.

    Args:
        session: The requests.Session object to use.
        server_info: Dictionary containing server 'url' and 'name'.
        message_history: List of strings representing the conversation history.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        stop_words: List of strings to stop generation at.
        timeout: Request timeout in seconds.

    Returns:
        The cleaned response content string, or None if an error occurred.
    """
    url = server_info['url']
    server_name = server_info['name']
    # Construct the prompt (adjust formatting as needed for your specific LLM)
    # Example: Simple concatenation. Consider more structured formats if required.
    prompt = "\n".join(message_history) + f"\n{server_info['name']}:" # Add speaker name for context

    payload = {
        'prompt': prompt,
        'temperature': temperature,
        'n_predict': max_tokens,
        'stop': stop_words,
        'stream': False, # Keep stream=False for simplicity here
    }
    headers = {'Content-Type': 'application/json'}
    response_content: Optional[str] = None

    try:
        colored_print(Colors.SYSTEM, f"Sending request to {server_name}...")
        response = session.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        response_json = response.json()
        content = response_json.get('content', '').strip()

        # Clean response: remove potential boilerplate/stop words at start/end
        # Example simple cleaning: remove the prompt/speaker tag if the model echoes it
        prompt_end_tag = f"{server_info['name']}:"
        if content.lower().startswith(prompt_end_tag.lower()):
            content = content[len(prompt_end_tag):].lstrip()

        # Remove common assistant tags (adjust regex if needed)
        content = re.sub(r"^\s*<assistant>[:\s]*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*</assistant>\s*$", "", content, flags=re.IGNORECASE)

        response_content = content.strip()
        if not response_content:
             colored_print(Colors.SYSTEM, f"{server_name} returned an empty response.")
             return None # Treat empty response as failure for conversation flow

    except requests.exceptions.Timeout:
        colored_print(Colors.ERROR, f"Timeout error requesting {server_name} (limit: {timeout}s)")
    except requests.exceptions.HTTPError as e:
        colored_print(Colors.ERROR, f"HTTP Error ({server_name}): {e.response.status_code} {e.response.reason}")
        try:
             colored_print(Colors.SYSTEM, f"Response text: {e.response.text[:500]}") # Show error details from server
        except Exception: pass
    except requests.exceptions.RequestException as e:
        colored_print(Colors.ERROR, f"Request Error ({server_name}): {type(e).__name__}: {e}")
    except json.JSONDecodeError:
        colored_print(Colors.ERROR, f"Error decoding JSON from {server_name}")
        try:
             # Access response potentially set before error
             resp_text = response.text if 'response' in locals() else "N/A"
             colored_print(Colors.SYSTEM, f"Response text: {resp_text[:500]}")
        except Exception: pass
    except Exception as e:
        colored_print(Colors.ERROR, f"Unexpected error during request to {server_name}: {type(e).__name__}")
        traceback.print_exc()

    return response_content

# --- Worker Thread Function ---

def request_worker(
    session: requests.Session,
    server_info: Dict[str, Any],
    history: List[str],
    result_queue: Queue,
    server_id: str,
    max_tokens: int,
    temperature: float,
    stop_words: List[str],
    timeout: int
    ):
    """Target function for the request thread."""
    result = send_llm_request(
        session=session,
        server_info=server_info,
        message_history=history,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_words=stop_words,
        timeout=timeout
    )
    result_queue.put((server_id, result))

# --- Main Conversation Simulation ---

def simulate_conversation(
    num_turns: int = DEFAULT_NUM_TURNS,
    initial_prompt: str = DEFAULT_INITIAL_PROMPT
    ):
    """Runs the conversation simulation between two LLM servers."""

    tts_manager = TTSManager()
    request_session = requests.Session() # Use a session

    conversation_history: List[str] = [f"Human: {initial_prompt}"] # Start with the human prompt

    colored_print(Colors.HEADER, "\n===== CONVERSATION START =====\n")
    colored_print(Colors.MESSAGE, f"Initial Prompt: {conversation_history[0]}")

    result_queue = Queue() # Queue for receiving results from background threads
    active_thread: Optional[threading.Thread] = None
    pending_server_id: Optional[str] = None # Which server's result are we expecting next?

    # Define server sequence for turns
    server_ids = ['A', 'B']

    try:
        # --- Initial Request (Server A) ---
        # Send the first request synchronously to kick things off
        first_server_id = server_ids[0]
        first_server_info = SERVER_CONFIG[first_server_id]
        colored_print(Colors.HEADER, f"\n--- Turn 1/{num_turns} ---")
        response = send_llm_request(
            session=request_session,
            server_info=first_server_info,
            message_history=conversation_history,
            # Use slightly fewer tokens for the first response potentially
            max_tokens=DEFAULT_MAX_TOKENS // 2,
            temperature=DEFAULT_TEMPERATURE,
            stop_words=DEFAULT_STOP_WORDS,
            timeout=REQUEST_TIMEOUT * 2 # Longer timeout for first request maybe?
        )

        # --- Main Loop ---
        for turn in range(num_turns):
            current_turn = turn + 1

            # Determine current and next speaker
            current_speaker_index = turn % 2 # 0 for A, 1 for B in the first loop iteration etc.
            current_server_id = server_ids[current_speaker_index]
            current_server_info = SERVER_CONFIG[current_server_id]
            next_speaker_index = (turn + 1) % 2
            next_server_id = server_ids[next_speaker_index]
            next_server_info = SERVER_CONFIG[next_server_id]

            # --- Process Response for Current Speaker ---
            # For turn 0, 'response' is from the initial sync call above.
            # For subsequent turns, 'response' comes from the queue (retrieved at end of last loop)
            if response:
                colored_print(current_server_info['color'], f"{current_server_info['name']}: {response}")
                conversation_history.append(f"{current_server_info['name']}: {response}")

                # Start background request for the *next* speaker *before* speaking
                if current_turn < num_turns: # Don't request after the last planned response
                    colored_print(Colors.SYSTEM, f"Starting background request for {next_server_info['name']} (Turn {current_turn+1 if next_speaker_index == 0 else current_turn})...")
                    active_thread = threading.Thread(
                        target=request_worker,
                        args=(
                            request_session, next_server_info, conversation_history.copy(),
                            result_queue, next_server_id, DEFAULT_MAX_TOKENS,
                            DEFAULT_TEMPERATURE, DEFAULT_STOP_WORDS, REQUEST_TIMEOUT
                        ),
                        daemon=True
                    )
                    active_thread.start()
                    pending_server_id = next_server_id
                else:
                     active_thread = None # No more requests needed
                     pending_server_id = None

                # Speak the current speaker's response *after* starting next request
                tts_manager.speak(response, current_server_id)

            else:
                # Current speaker failed to respond
                colored_print(Colors.ERROR, f"{current_server_info['name']} failed to respond or returned empty.")
                # Need to decide how to proceed. Options:
                # 1. Stop the conversation.
                # 2. Let the *next* speaker respond based on history *before* the failure.
                # Let's try option 2: Start the next speaker's request anyway.
                if current_turn < num_turns:
                    colored_print(Colors.SYSTEM, f"Starting background request for {next_server_info['name']} (after {current_server_info['name']} failed)...")
                    active_thread = threading.Thread(
                         target=request_worker,
                         args=(
                             request_session, next_server_info, conversation_history.copy(), # History *without* failed response
                             result_queue, next_server_id, DEFAULT_MAX_TOKENS,
                             DEFAULT_TEMPERATURE, DEFAULT_STOP_WORDS, REQUEST_TIMEOUT
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
                 colored_print(Colors.SYSTEM, f"Waiting for {SERVER_CONFIG[pending_server_id]['name']}'s result...")
                 try:
                     # Block until the result for the expected server is ready
                     # Add a timeout to queue.get to prevent indefinite hangs if a thread dies
                     q_timeout = REQUEST_TIMEOUT + 10 # Slightly longer than request timeout
                     retrieved_id, response = result_queue.get(timeout=q_timeout)

                     if retrieved_id != pending_server_id:
                          # This shouldn't happen with one worker thread at a time, but good practice
                          colored_print(Colors.ERROR, f"Logic Error: Expected result for {pending_server_id}, got {retrieved_id}. Discarding.")
                          response = None # Treat as failure
                          # Try to retrieve the correct one if it's somehow stuck? Unlikely.
                          try:
                              retrieved_id, response = result_queue.get(timeout=1)
                              if retrieved_id != pending_server_id: response = None
                          except Empty:
                              response = None

                     # Result retrieved, thread finished its work for this stage
                     active_thread.join(timeout=1.0) # Ensure thread object is cleaned up
                     active_thread = None
                     pending_server_id = None

                 except Empty:
                      colored_print(Colors.ERROR, f"Timeout waiting for result from {SERVER_CONFIG[pending_server_id]['name']} queue (limit: {q_timeout}s). Assuming failure.")
                      response = None
                      if active_thread.is_alive():
                          colored_print(Colors.SYSTEM, f"Thread for {SERVER_CONFIG[pending_server_id]['name']} still seems alive but didn't produce result.")
                      active_thread = None # Stop waiting for this thread
                      pending_server_id = None
                 except Exception as e:
                      colored_print(Colors.ERROR, f"Error getting result from queue: {e}")
                      traceback.print_exc()
                      response = None
                      active_thread = None
                      pending_server_id = None

            elif current_turn >= num_turns:
                 # End of conversation, no more results expected
                 response = None
            else:
                 # This case might happen if the previous turn failed to start a thread
                 colored_print(Colors.ERROR, "Logic Error: No active thread or pending server ID found when expected.")
                 response = None # Cannot continue this turn effectively


            # Prepare for the next iteration of the loop (or exit if done)
            if current_turn >= num_turns:
                break
            else:
                 colored_print(Colors.HEADER, f"\n--- Turn {current_turn + 1}/{num_turns} ---")


    except KeyboardInterrupt:
        colored_print(Colors.SYSTEM, "\nConversation interrupted by user.")
    except Exception as e:
        colored_print(Colors.ERROR, f"\nAn unexpected error occurred during simulation:")
        traceback.print_exc()
    finally:
        colored_print(Colors.HEADER, "\n===== CONVERSATION END =====\n")
        # Final history dump (optional)
        # print("\n--- Final Conversation History ---")
        # for line in conversation_history:
        #     print(line)
        #     print("-" * 20)

        # Clean up resources
        if active_thread and active_thread.is_alive():
            colored_print(Colors.SYSTEM, "Attempting to wait briefly for the final active thread...")
            active_thread.join(timeout=2.0) # Short wait
        tts_manager.shutdown()
        request_session.close()
        colored_print(Colors.SYSTEM, "Resources cleaned up.")


if __name__ == "__main__":
    simulate_conversation(num_turns=250)
    # You can adjust the number of turns here
