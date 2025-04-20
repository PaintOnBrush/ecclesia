#!/usr/bin/env python3
"""
AI Conversation Simulator with TTS and Dynamic Prompting
Refactored version with improved structure, type safety, and error handling
"""

# Standard library imports
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import traceback
from queue import Queue, Empty, Full
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime

# Third-party imports
import requests

# === Configuration ===
class Colors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    SERVER_A = "\033[38;5;39m"  # Blue
    SERVER_B = "\033[38;5;208m"  # Orange
    SYSTEM = "\033[38;5;245m"
    HEADER = "\033[38;5;105m"
    MESSAGE = "\033[38;5;252m"
    ERROR = "\033[38;5;196m"
    SUCCESS = "\033[38;5;46m"
    FILE_IO = "\033[38;5;220m"
    PROMPT_ADD = "\033[38;5;154m"
    TTS_ENGINE = "\033[38;5;81m"
    HUMAN_SPEAKER = "\033[38;5;228m"
    DEBUG = "\033[38;5;240m"
    SUMMARIZER = "\033[38;5;123m"
    WARNING = "\033[38;5;214m"
    PROMPT_GEN = "\033[38;5;117m"
    GENDER_LOG = "\033[38;5;207m"

# Server configuration
SERVER_CONFIG = {
    "A": {
        "name": "David",
        "llm_url": "http://127.0.0.1:8080/completion",
        "tts_server_key": "piper_a",
        "tts_url": "http://127.0.0.1:5001",
        "color": Colors.SERVER_A,
        "preferred_gender": "male"
    },
    "B": {
        "name": "Zira",
        "llm_url": "http://127.0.0.1:8080/completion",
        "tts_server_key": "piper_b",
        "tts_url": "http://127.0.0.1:5001",
        "color": Colors.SERVER_B,
        "preferred_gender": "female"
    },
    "Human": {
        "name": "Human",
        "llm_url": None,
        "tts_server_key": "piper_human",
        "tts_url": "http://127.0.0.1:5001",
        "color": Colors.HUMAN_SPEAKER,
        "preferred_gender": "male"
    },
    "Summarizer": {
        "name": "SummarizerLLM",
        "llm_url": "http://127.0.0.1:8080/completion",
        "color": Colors.SUMMARIZER
    }
}

# Constants
DEFAULT_REQUEST_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 512
DEFAULT_TTS_TIMEOUT = 600
DEFAULT_TTS_GENERATE_TIMEOUT = 60
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_TURNS = 5
DEFAULT_SUMMARY_RETRIES = 2
DEFAULT_PROMPTS_FILE = "aiprompts.txt"
SECONDARY_PROMPTS_FILE = "aiprompts2.txt"
DEFAULT_PROMPT_GEN_MAX_TOKENS = 100
DEFAULT_SUMMARY_MAX_TOKENS = 150
PLAYBACK_QUEUE_SIZE = 10
DEFAULT_LOOP_COUNT = 1
DEFAULT_VOICES_DIR = "../models/piper-voices/"
PERSISTENT_SUMMARY_FILE = "persistent_summary.txt"
SUMMARY_LOG_FILE = "summary_log.txt"
IDLE_WAIT_SECONDS = 2

HUMAN_ROLEPLAY_CLAUSE = (
    " Remember, you are roleplaying as a singular human and "
    "should not reveal you are an AI."
)

BASE_STOP_WORDS = [
    "\n<human>:", "</assistant>", "<|user|>", "<|im_end|>", "</s>", "[INST]",
    "[/INST]", "Human:", "Assistant:",
    f"{SERVER_CONFIG['A']['name']}:",
    f"{SERVER_CONFIG['B']['name']}:"
]

SUMMARIZER_PROMPT_INSTRUCTION = (
    'You are roleplaying as a single human analyzing a diverse set of input messages from various sources (e.g., chats, forums, or other platforms). Do not reveal you are an AI. Provide a concise summary capturing the core idea of each message, emphasizing relationships between ideas and their underlying causes, not just solutions. Process messages sequentially, noting if a message modifies or builds on a previous one. Avoid using specific names, emojis, or repetitive phrases like "happy to hear," "excited to hear," "final answer," or "correct." Structure the output as a single paragraph, like: "The discussion explored transparency in systems, identifying challenges like lack of trust due to cultural resistance and proposing community engagement, while highlighting causes such as unclear policies." Ensure the summary supports building a hierarchy or network of ideas by clearly linking themes, causes, and relationships. If relationships or causes are unclear, summarize the main themes.'
)

SUMMARIZER_STOP_WORDS = [
    "<|im_end|>", "</s>", "\nHuman:", "\nDavid:", "\nZira:", "\nSummary:",
    "Concise Summary:", "Prompt:"
]

PROMPT_GENERATION_STOP_WORDS = [
    "<|im_end|>", "</s>", "\nHuman:", "\nDavid:", "\nZira:", "\nSummary:",
    "\nNotes:", "Prompt:", "[INST]", "[/INST]"
]

# Global variables
VOICE_NAME_TO_PATH: Dict[str, str] = {}
VOICE_GENDER_MAP: Dict[str, Literal["male", "female", "unknown"]] = {}
VOICE_TEAMS: List[Dict[str, str]] = []

# === Utility Functions ===
def colored_print(color: str, message: str) -> None:
    """Print message with specified ANSI color, thread-safe."""
    try:
        print(f"{color}{message}{Colors.RESET}", flush=True)
    except Exception as e:
        print(f"(ColorPrint Error: {e}) {message}", flush=True)

def validate_directory(path: str) -> bool:
    """Validate if directory exists and is accessible."""
    if not os.path.isdir(path):
        colored_print(Colors.ERROR, f"Directory not found: {path}")
        return False
    return True

def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        colored_print(Colors.ERROR, f"Error loading JSON file {path}: {e}")
        return None

# === Voice Management ===
def get_voice_gender(voice_name: str) -> Literal["male", "female", "unknown"]:
    """Infer gender from voice name."""
    name_lower = voice_name.lower()
    
    if 'female' in name_lower or 'girl' in name_lower:
        return "female"
    if 'male' in name_lower or 'boy' in name_lower:
        return "male"

    female_names = [
        'alba', 'aru', 'cori', 'jenny_dioco', 'semaine', 'amy',
        'hfc_female', 'kristin', 'ljspeech', 'libritts'
    ]
    male_names = [
        'alan', 'northern_english_male', 'arctic', 'l2arctic', 'bryce',
        'hfc_male', 'joe', 'john', 'kusal', 'lessac', 'norman', 'ryan'
    ]

    if any(n in name_lower for n in female_names):
        return "female"
    if any(n in name_lower for n in male_names):
        return "male"
    return "unknown"

def load_voice_data(voices_dir: str) -> bool:
    """Load voice model paths and create voice teams."""
    global VOICE_NAME_TO_PATH, VOICE_GENDER_MAP, VOICE_TEAMS
    if not validate_directory(voices_dir):
        return False

    VOICE_NAME_TO_PATH.clear()
    VOICE_GENDER_MAP.clear()
    VOICE_TEAMS.clear()

    try:
        for filename in os.listdir(voices_dir):
            if filename.endswith(".onnx"):
                base_name = filename[:-5]
                json_path = os.path.join(voices_dir, f"{base_name}.onnx.json")
                if os.path.exists(json_path):
                    VOICE_NAME_TO_PATH[base_name] = os.path.join(voices_dir, filename)
                    gender = get_voice_gender(base_name)
                    VOICE_GENDER_MAP[base_name] = gender
                    colored_print(
                        Colors.DEBUG + Colors.GENDER_LOG,
                        f"Voice: {base_name} -> Gender: {gender}"
                    )
    except Exception as e:
        colored_print(Colors.ERROR, f"Error scanning voices dir '{voices_dir}': {e}")
        return False

    if not VOICE_NAME_TO_PATH:
        colored_print(Colors.ERROR, f"No valid voice models found in '{voices_dir}'")
        return False

    colored_print(Colors.SYSTEM, f"Found {len(VOICE_NAME_TO_PATH)} voices")

    teams_config = [
        {'A': 'en_US-ryan-high', 'B': 'en_US-ljspeech-high', 'Human': 'en_US-joe-medium'},
        {'A': 'en_GB-alan-medium', 'B': 'en_GB-alba-medium', 'Human': 'en_GB-northern_english_male-medium'},
        {'A': 'en_US-joe-medium', 'B': 'en_US-amy-medium', 'Human': 'en_US-bryce-medium'},
        {'A': 'en_US-kusal-medium', 'B': 'en_GB-aru-medium', 'Human': 'en_US-john-medium'},
        {'A': 'en_US-lessac-high', 'B': 'en_US-libritts-high', 'Human': 'en_US-norman-medium'},
        {'A': 'en_US-hfc_male-medium', 'B': 'en_US-hfc_female-medium', 'Human': 'en_US-joe-medium'},
        {'A': 'en_US-arctic-medium', 'B': 'en_GB-cori-high', 'Human': 'en_US-ryan-high'},
        {'A': 'en_US-ryan-medium', 'B': 'en_US-kusal-medium', 'Human': 'en_US-joe-medium'},
        {'A': 'en_GB-aru-medium', 'B': 'en_US-ljspeech-high', 'Human': 'en_US-joe-medium'},
    ]

    valid_teams = [
        team for team in teams_config
        if all(key in VOICE_NAME_TO_PATH for key in team.values())
    ]

    if not valid_teams:
        colored_print(Colors.ERROR, "No valid voice teams created")
        return False

    VOICE_TEAMS = valid_teams
    colored_print(Colors.SYSTEM, f"Created {len(VOICE_TEAMS)} voice teams")
    return True

# === TTS Manager ===
class TTSManager:
    """Manages Text-to-Speech generation and playback."""
    
    def __init__(
        self,
        tts_config: Dict[str, Dict[str, Any]],
        tts_timeout: int = DEFAULT_TTS_TIMEOUT,
        tts_generate_timeout: int = DEFAULT_TTS_GENERATE_TIMEOUT,
        no_tts: bool = False
    ):
        self.engine_type: Optional[str] = None
        self.tts_servers: Dict[str, Optional[str]] = {}
        self.tts_timeout = tts_timeout
        self.tts_generate_timeout = tts_generate_timeout
        self.is_disabled = no_tts
        self.config_valid = True
        self._generate_endpoint = "/generate"
        self._play_endpoint = "/play"

        if self.is_disabled:
            colored_print(Colors.SYSTEM, "TTS disabled by user")
            return

        required_keys = {"piper_a", "piper_b", "piper_human"}
        configured_keys = set()

        for server_id, config in tts_config.items():
            tts_key = config.get("tts_server_key")
            tts_url = config.get("tts_url")
            if tts_key in required_keys:
                self.tts_servers[tts_key] = tts_url
                configured_keys.add(tts_key)
                if not tts_url:
                    colored_print(Colors.ERROR, f"TTS key '{tts_key}' missing URL")
                    self.config_valid = False

        if missing_keys := required_keys - configured_keys:
            colored_print(Colors.ERROR, f"Missing TTS configs: {', '.join(missing_keys)}")
            self.config_valid = False

        if self.config_valid:
            self._initialize()
        else:
            colored_print(Colors.ERROR, "TTS config invalid. TTS disabled")

    def _check_server(self, server_key: str, url: Optional[str]) -> bool:
        """Check if TTS server is reachable."""
        if not url:
            return False
        check_url = f"{url.rstrip('/')}{self._generate_endpoint}"
        try:
            response = requests.options(check_url, timeout=2)
            return response.status_code < 500
        except requests.RequestException as e:
            colored_print(Colors.DEBUG, f"TTS server check error {server_key}: {type(e).__name__}")
            return False

    def _initialize(self) -> None:
        """Initialize TTS servers."""
        if self.is_disabled or not self.config_valid:
            return

        all_servers_ok = True
        unique_urls = set(self.tts_servers.values()) - {None}

        for url in unique_urls:
            if not self._check_server("TTS Server", url):
                all_servers_ok = False
                colored_print(Colors.ERROR, f"TTS Server check FAILED: {url}")
            else:
                colored_print(Colors.DEBUG, f"TTS Server check OK: {url}")

        self.engine_type = "piper_wsl_aplay" if all_servers_ok else None
        status = "OK" if all_servers_ok else "FAILED"
        colored_print(
            Colors.SUCCESS + Colors.TTS_ENGINE if all_servers_ok else Colors.ERROR,
            f"TTS Server(s) {status}: {', '.join(unique_urls)}. Engine: {self.engine_type or 'None'}"
        )

    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self.engine_type == "piper_wsl_aplay" and not self.is_disabled

    def request_generation(
        self,
        text: str,
        server_key: str,
        model_path: str,
        result_queue: Queue
    ) -> Optional[threading.Thread]:
        """Request TTS generation in background thread."""
        if not self.is_available():
            result_queue.put_nowait(None)
            return None

        target_url = self.tts_servers.get(server_key)
        if not target_url or not model_path or not os.path.exists(model_path):
            colored_print(Colors.ERROR, f"Invalid TTS config: key={server_key}, path={model_path}")
            result_queue.put_nowait(None)
            return None

        generate_url = f"{target_url.rstrip('/')}{self._generate_endpoint}"
        model_path_relative = os.path.basename(model_path)

        def worker():
            try:
                response = requests.post(
                    generate_url,
                    json={"text": text, "model_path": model_path_relative},
                    headers={'Content-Type': 'application/json'},
                    timeout=self.tts_generate_timeout
                )
                response.raise_for_status()
                json_resp = response.json()
                audio_id = json_resp.get("audio_id") if json_resp.get("status") == "generated" else None
                result_queue.put_nowait(audio_id)
            except Exception as e:
                colored_print(Colors.ERROR, f"TTS Gen Error '{server_key}': {e}")
                result_queue.put_nowait(None)

        thread = threading.Thread(
            target=worker,
            daemon=True,
            name=f"TTSGen_{server_key}_{model_path_relative}"
        )
        thread.start()
        return thread

    def request_playback(self, audio_id: str, server_key: str) -> None:
        """Request playback of generated audio."""
        if not self.is_available() or not audio_id:
            return

        target_url = self.tts_servers.get(server_key)
        if not target_url:
            colored_print(Colors.ERROR, f"No URL for TTS key '{server_key}'")
            return

        play_url = f"{target_url.rstrip('/')}{self._play_endpoint}"
        start_time = time.time()

        try:
            response = requests.post(
                play_url,
                json={"audio_id": audio_id},
                headers={'Content-Type': 'application/json'},
                timeout=self.tts_timeout
            )
            response.raise_for_status()
            colored_print(
                Colors.SYSTEM + Colors.TTS_ENGINE,
                f"Playback {audio_id[:8]} done - Dur: {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            colored_print(Colors.ERROR, f"TTS Playback Error '{server_key}': {e}")

    def shutdown(self) -> None:
        """Shutdown TTS manager."""
        colored_print(Colors.SYSTEM, "TTS Manager shutdown")
        self.engine_type = None

# === LLM Request Handler ===
def send_llm_request(
    session: requests.Session,
    server_config: Dict[str, Any],
    message_history: Optional[List[str]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
    force_human: bool = False,
    debug_prompts: bool = False,
    stop_words: List[str] = BASE_STOP_WORDS,
    is_summarizer: bool = False,
    prompt_instruction: Optional[str] = None,
    input_text: Optional[str] = None,
    human_summarizer: bool = False
) -> Optional[str]:
    """Send request to LLM server."""
    server_name = server_config.get('name', 'UnknownServer')
    url = server_config.get('llm_url')
    if not url:
        colored_print(Colors.ERROR, f"LLM URL missing for '{server_name}'")
        return None

    prompt_str = ""
    current_stop_words = stop_words[:]
    task_type = "Unknown"

    if prompt_instruction and input_text:
        prompt_str = f"{prompt_instruction}\n\nInput Text:\n{input_text}\n\nGenerated Prompt:"
        current_stop_words = PROMPT_GENERATION_STOP_WORDS
        task_type = "Prompt Generation"
    elif is_summarizer and message_history:
        prompt_str = f"{SUMMARIZER_PROMPT_INSTRUCTION}\n\n" + "\n".join(message_history) + "\n\nConcise Summary:"
        if human_summarizer:
            prompt_str += f"\n{HUMAN_ROLEPLAY_CLAUSE.strip()}"
        current_stop_words = SUMMARIZER_STOP_WORDS
        task_type = "Summarization"
    elif message_history:
        prompt_base = "\n".join(message_history).strip()
        if force_human and not prompt_base.rstrip().endswith(HUMAN_ROLEPLAY_CLAUSE.strip()):
            prompt_base += f"\n{HUMAN_ROLEPLAY_CLAUSE.strip()}"
        prompt_str = f"{prompt_base}\n{server_name}:"
        task_type = "Conversation"
    else:
        colored_print(Colors.ERROR, f"Invalid params for '{server_name}'")
        return None

    if debug_prompts:
        colored_print(Colors.DEBUG, f"--- DEBUG PROMPT ({server_name} - {task_type}) ---\n{prompt_str}\n---")

    try:
        response = session.post(
            url,
            json={
                'prompt': prompt_str,
                'temperature': temperature,
                'n_predict': max_tokens,
                'stop': current_stop_words,
                'stream': False
            },
            headers={'Content-Type': 'application/json'},
            timeout=timeout
        )
        response.raise_for_status()
        content = response.json().get('content', '').strip()

        if debug_prompts:
            colored_print(Colors.DEBUG, f"--- RAW LLM RESPONSE ({server_name} - {task_type}) ---\n{content or 'Empty'}\n---")

        if not content or content in current_stop_words:
            colored_print(Colors.ERROR, f"LLM returned empty or invalid content for {server_name} ({task_type})")
            return None

        if is_summarizer or prompt_instruction:
            return content

        prefix = f"{server_name}:"
        return content[len(prefix):].lstrip() if content.lower().startswith(prefix.lower()) else content

    except Exception as e:
        colored_print(Colors.ERROR, f"LLM Request Error '{server_name}' ({task_type}): {e}")
        return None

# === Worker Functions ===
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
) -> None:
    """Handle LLM conversation request."""
    server_name = server_config.get('name', 'UnknownServer')
    other_server_name = SERVER_CONFIG['B']['name'] if server_id == 'A' else SERVER_CONFIG['A']['name']
    
    current_stop_words = stop_words + [f"{server_name}:", f"{other_server_name}:"]
    
    result = send_llm_request(
        session=session,
        server_config=server_config,
        message_history=history,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        force_human=force_human,
        debug_prompts=debug_prompts,
        stop_words=current_stop_words
    )
    
    try:
        result_queue.put_nowait((server_id, result))
    except Full:
        colored_print(Colors.ERROR, f"LLM result queue full for {server_id}")

def request_summary_worker(
    session: requests.Session,
    summarizer_config: Dict[str, Any],
    history: List[str],
    result_queue: Queue,
    max_tokens: int,
    temperature: float,
    timeout: int,
    debug_prompts: bool,
    debug_context: bool,
    previous_summary: Optional[str] = None,
    human_summarizer: bool = False,
    retries: int = DEFAULT_SUMMARY_RETRIES
) -> None:
    """Handle summary generation request with retries."""
    if not summarizer_config.get("llm_url"):
        colored_print(Colors.ERROR, "Summarizer LLM config missing")
        result_queue.put_nowait(None)
        return

    current_history = history[:]
    if previous_summary:
        current_history.insert(0, f"Previous Summary:\n{previous_summary}\n--- Conversation ---")
        if debug_context:
            colored_print(Colors.SUMMARIZER + Colors.DEBUG, "Prepended previous summary")

    # Warn if history is very long
    history_token_estimate = sum(len(line.split()) for line in current_history) // 0.75
    if history_token_estimate > 2048:
        colored_print(
            Colors.WARNING,
            f"Conversation history is large (~{history_token_estimate} tokens). May cause LLM issues."
        )

    summary = None
    for attempt in range(retries + 1):
        colored_print(Colors.SUMMARIZER, f"Generating summary (Attempt {attempt + 1}/{retries + 1})")
        summary = send_llm_request(
            session=session,
            server_config=summarizer_config,
            message_history=current_history,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            debug_prompts=debug_prompts,
            stop_words=SUMMARIZER_STOP_WORDS,
            is_summarizer=True,
            human_summarizer=human_summarizer
        )
        if summary:
            break
        colored_print(Colors.WARNING, f"Summary attempt {attempt + 1} failed. Retrying...")
        time.sleep(1)  # Brief delay between retries

    if summary is None:
        colored_print(Colors.ERROR, "Summary generation failed: No valid response from LLM after retries")
    
    result_queue.put_nowait(summary)

def generate_prompt_worker(
    session: requests.Session,
    generator_config: Dict[str, Any],
    input_text: str,
    result_queue: Queue,
    max_tokens: int,
    temperature: float,
    timeout: int,
    debug_prompts: bool
) -> None:
    """Generate conversation prompt from text."""
    if not generator_config.get("llm_url"):
        colored_print(Colors.ERROR + Colors.PROMPT_GEN, "Prompt Gen LLM config missing")
        result_queue.put_nowait(None)
        return

    instruction = (
        "Analyze the following text and generate a single, concise, and "
        "engaging conversation starter prompt suitable for initiating a "
        "discussion between two AI chatbots. The prompt should be neutral "
        "and open-ended."
    )

    prompt = send_llm_request(
        session=session,
        server_config=generator_config,
        input_text=input_text,
        prompt_instruction=instruction,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        debug_prompts=debug_prompts,
        stop_words=PROMPT_GENERATION_STOP_WORDS
    )

    if prompt:
        prompt = prompt.strip('"`\'')
        if prompt.lower().startswith("prompt:"):
            prompt = prompt[len("prompt:"):].strip()
    
    result_queue.put_nowait(prompt if prompt else None)

# === TTS Player ===
def tts_player_worker(
    playback_queue: Queue,
    tts_manager: TTSManager,
    stop_event: threading.Event,
    total_turns: int
) -> None:
    """Handle TTS playback queue."""
    colored_print(Colors.SYSTEM, "TTS Player Thread started")
    last_header_turn = -99

    while not stop_event.is_set():
        try:
            item = playback_queue.get(timeout=1.0)
            if item is None:
                colored_print(Colors.DEBUG, "Player received sentinel")
                playback_queue.task_done()  # Task done for sentinel
                break

            turn_num, text, audio_id, tts_key, name, color = item
            try:
                if turn_num > 0 and turn_num != last_header_turn:
                    colored_print(Colors.HEADER, f"\n--- Turn {turn_num}/{total_turns} ---")
                    last_header_turn = turn_num
                elif turn_num == -1:
                    colored_print(Colors.HEADER, "\n--- CONVERSATION SUMMARY ---")
                    last_header_turn = -1

                colored_print(color, f"{name}: {text}")
                if audio_id and tts_key and tts_manager.is_available():
                    colored_print(
                        Colors.SYSTEM + Colors.TTS_ENGINE,
                        f"{'(Summary) ' if turn_num == -1 else ''}(Speaking {name} [ID:{audio_id[:8]}...])"
                    )
                    tts_manager.request_playback(audio_id, tts_key)
                elif tts_manager.is_available() and not audio_id:
                    colored_print(
                        Colors.WARNING + Colors.TTS_ENGINE,
                        f"{'(Summary) ' if turn_num == -1 else ''}No audio ID for {name} (T{turn_num})"
                    )
            finally:
                playback_queue.task_done()  # Task done only after processing valid item
        except Empty:
            continue
        except Exception as e:
            colored_print(Colors.ERROR, f"Player loop error: {e}")
            traceback.print_exc()
            # Do not call task_done() for unexpected exceptions

    colored_print(Colors.SYSTEM, "TTS Player Thread finished")

# === Conversation Simulation ===
def simulate_conversation(
    tts_manager: TTSManager,
    request_session: requests.Session,
    server_config: Dict[str, Dict[str, Any]],
    num_turns: int,
    initial_prompt: str,
    req_timeout: int,
    max_tokens: int,
    temperature: float,
    summary_max_tokens: int,
    tts_timeout: int,
    force_human: bool,
    debug_prompts: bool,
    debug_logic: bool,
    debug_context: bool,
    voice_name_a: str,
    voice_name_b: str,
    voice_name_human: str,
    prompt_global_num: int = 0,
    log_summaries: bool = False,
    summary_log_format: str = "detailed",
    human_summarizer: bool = False
) -> None:
    """Simulate conversation between AI agents."""
    total_turns = max(1, num_turns)
    conversation_history = [f"{server_config['Human']['name']}: {initial_prompt}"]
    
    colored_print(
        Colors.HEADER,
        f"\n===== CONVERSATION START (Prompt #{prompt_global_num}, Turns: {total_turns}) ====="
    )
    colored_print(Colors.HUMAN_SPEAKER, f"{server_config['Human']['name']}: {initial_prompt}")
    colored_print(Colors.MESSAGE, "--------------------")

    llm_result_queue = Queue(maxsize=1)
    playback_queue = Queue(maxsize=PLAYBACK_QUEUE_SIZE)
    player_stop_event = threading.Event()
    
    player_thread = threading.Thread(
        target=tts_player_worker,
        args=(playback_queue, tts_manager, player_stop_event, total_turns),
        name=f"TTSPlayer_P{prompt_global_num}",
        daemon=False
    )
    player_thread.start()

    try:
        # Handle initial prompt TTS
        initial_audio_id = None
        initial_tts_key = server_config['Human']['tts_server_key']
        initial_voice_path = VOICE_NAME_TO_PATH.get(voice_name_human)
        initial_generation_q = Queue(maxsize=1)

        if tts_manager.is_available() and initial_voice_path:
            initial_gen_thread = tts_manager.request_generation(
                initial_prompt, initial_tts_key, initial_voice_path, initial_generation_q
            )
            if not initial_gen_thread:
                colored_print(Colors.ERROR, "Failed to start initial TTS thread")

        # Conversation loop
        llm_server_keys = ['A', 'B']
        current_llm_key = llm_server_keys[0]
        first_turn_processed = False

        for turn_index in range(total_turns * 2):
            actual_turn_num = (turn_index // 2) + 1
            is_last_iteration = turn_index == total_turns * 2 - 1
            current_llm_info = server_config[current_llm_key]
            current_tts_key = current_llm_info.get('tts_server_key')
            current_voice_name = voice_name_a if current_llm_key == 'A' else voice_name_b
            current_voice_path = VOICE_NAME_TO_PATH.get(current_voice_name)

            # Start LLM request
            active_llm_thread = threading.Thread(
                target=request_worker,
                args=(
                    request_session, server_config[current_llm_key], conversation_history.copy(),
                    llm_result_queue, current_llm_key, max_tokens, temperature,
                    req_timeout, force_human, debug_prompts, BASE_STOP_WORDS.copy()
                ),
                daemon=True,
                name=f"LLM_{current_llm_key}_T{actual_turn_num}"
            )
            active_llm_thread.start()

            # Wait for LLM response
            try:
                server_id, response = llm_result_queue.get(timeout=req_timeout + 15)
                if server_id != current_llm_key or not response:
                    colored_print(Colors.ERROR, f"LLM failure for {current_llm_info['name']}")
                    break
            except Empty:
                colored_print(Colors.ERROR, f"LLM timeout for {current_llm_info['name']}")
                break

            # Generate TTS for current response
            current_audio_id = None
            current_generation_q = Queue(maxsize=1)
            if tts_manager.is_available() and current_tts_key and current_voice_path:
                current_gen_thread = tts_manager.request_generation(
                    response, current_tts_key, current_voice_path, current_generation_q
                )
                if current_gen_thread:
                    try:
                        current_audio_id = current_generation_q.get(
                            timeout=tts_manager.tts_generate_timeout + 10
                        )
                    except Empty:
                        colored_print(Colors.ERROR, f"TTS timeout for {current_llm_info['name']}")

            # Queue initial playback if needed
            if not first_turn_processed and initial_gen_thread:
                try:
                    initial_audio_id = initial_generation_q.get(
                        timeout=tts_manager.tts_generate_timeout + 10
                    )
                except Empty:
                    colored_print(Colors.ERROR, "Initial TTS timeout")
                playback_queue.put_nowait((
                    0, initial_prompt, initial_audio_id, initial_tts_key,
                    server_config['Human']['name'], Colors.HUMAN_SPEAKER
                ))
                first_turn_processed = True

            # Queue current playback
            playback_queue.put_nowait((
                actual_turn_num, response, current_audio_id, current_tts_key,
                current_llm_info['name'], current_llm_info['color']
            ))
            conversation_history.append(f"{current_llm_info['name']}: {response}")

            # Prepare for next turn
            if not is_last_iteration:
                current_llm_key = llm_server_keys[(turn_index + 1) % 2]

    finally:
        # Generate and play summary
        summary_text = "Summary could not be generated"
        summary_audio_id = None
        summary_tts_key = server_config['Human']['tts_server_key']
        summarizer_config = server_config.get("Summarizer", server_config['A'])

        previous_summary = None
        if os.path.exists(PERSISTENT_SUMMARY_FILE):
            with open(PERSISTENT_SUMMARY_FILE, 'r', encoding='utf-8') as f:
                previous_summary = f.read().strip()

        summary_q = Queue(maxsize=1)
        summary_thread = threading.Thread(
            target=request_summary_worker,
            args=(
                request_session, summarizer_config, conversation_history, summary_q,
                summary_max_tokens, DEFAULT_TEMPERATURE, req_timeout, debug_prompts,
                debug_context, previous_summary, human_summarizer
            ),
            daemon=True,
            name=f"Summarizer_P{prompt_global_num}"
        )
        summary_thread.start()

        try:
            # Increase timeout for larger summary_max_tokens
            summary_timeout = req_timeout + 30 if summary_max_tokens > 512 else req_timeout + 10
            summary_text = summary_q.get(timeout=summary_timeout) or summary_text
            with open(PERSISTENT_SUMMARY_FILE, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            if log_summaries:
                try:
                    with open(SUMMARY_LOG_FILE, 'a', encoding='utf-8') as f:
                        if summary_log_format == "simple":
                            f.write(f"{summary_text}\n\n")
                            format_msg = "simple format"
                        else:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S PDT")
                            f.write(f"Conversation #{prompt_global_num} ({timestamp})\n{summary_text}\n\n")
                            format_msg = "detailed format"
                    colored_print(Colors.FILE_IO, f"Appended summary to {SUMMARY_LOG_FILE} ({format_msg})")
                except Exception as e:
                    colored_print(Colors.ERROR, f"Error appending to {SUMMARY_LOG_FILE}: {e}")
        except Empty:
            colored_print(Colors.ERROR, f"Summary timeout after {summary_timeout} seconds")
            summary_text = "Summary could not be generated due to timeout"

        if tts_manager.is_available() and initial_voice_path:
            summary_gen_q = Queue(maxsize=1)
            summary_gen_thread = tts_manager.request_generation(
                summary_text, summary_tts_key, initial_voice_path, summary_gen_q
            )
            if summary_gen_thread:
                try:
                    summary_audio_id = summary_gen_q.get(
                        timeout=tts_manager.tts_generate_timeout + 10
                    )
                except Empty:
                    colored_print(Colors.ERROR, "Summary TTS timeout")

        playback_queue.put_nowait((
            -1, summary_text, summary_audio_id, summary_tts_key,
            server_config['Human']['name'], Colors.SUMMARIZER
        ))
        playback_queue.put_nowait(None)
        playback_queue.join()
        player_stop_event.set()
        player_thread.join(timeout=15)

        colored_print(
            Colors.HEADER,
            f"===== CONVERSATION END (Prompt #{prompt_global_num}) ====="
        )

# === Main Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Conversation Simulator with TTS and Dynamic Prompting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    prompt_group = parser.add_argument_group('Input Prompts')
    prompt_group.add_argument('--prompts_file', default=DEFAULT_PROMPTS_FILE, help="Primary prompts file")
    prompt_group.add_argument('--initial_prompt', help="Single initial prompt")
    prompt_group.add_argument('--secondary_prompts_file', default=SECONDARY_PROMPTS_FILE, help="Secondary prompts file")

    convo_group = parser.add_argument_group('Conversation Control')
    convo_group.add_argument('-t', '--turns', type=int, default=DEFAULT_NUM_TURNS, help="Turns per AI agent")
    convo_group.add_argument('--loop', type=int, default=DEFAULT_LOOP_COUNT, help="Loop count (0=infinite)")
    convo_group.add_argument('--log-summaries', action='store_true', help="Append summaries to summary_log.txt")
    convo_group.add_argument(
        '--summary-log-format',
        choices=['detailed', 'simple'],
        default='detailed',
        help="Format for summary_log.txt: 'detailed' (with conversation number and timestamp) or 'simple' (summaries only)"
    )

    llm_group = parser.add_argument_group('LLM Configuration')
    llm_group.add_argument('--llm-url', default=SERVER_CONFIG['A']['llm_url'], help="Conversational LLM URL")
    llm_group.add_argument('--llm-url-summarizer', default=SERVER_CONFIG['Summarizer']['llm_url'], help="Summarizer LLM URL")
    llm_group.add_argument('--timeout', type=int, default=DEFAULT_REQUEST_TIMEOUT, help="LLM request timeout")
    llm_group.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for convo")
    llm_group.add_argument('--summary-max-tokens', type=int, default=DEFAULT_SUMMARY_MAX_TOKENS, help="Max tokens for summary")
    llm_group.add_argument('--prompt-gen-max-tokens', type=int, default=DEFAULT_PROMPT_GEN_MAX_TOKENS, help="Max tokens for prompt gen")
    llm_group.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature")
    llm_group.add_argument('--force-human', action='store_true', help="Force human roleplay for conversation")
    llm_group.add_argument('--human-summarizer', action='store_true', help="Force human roleplay for summarizer")

    tts_group = parser.add_argument_group('TTS Configuration')
    tts_group.add_argument('--voices-dir', default=DEFAULT_VOICES_DIR, help="Piper voice models directory")
    tts_group.add_argument('--tts-url', default=SERVER_CONFIG['A']['tts_url'], help="TTS server URL")
    tts_group.add_argument('--tts_timeout', type=int, default=DEFAULT_TTS_TIMEOUT, help="TTS playback timeout")
    tts_group.add_argument('--tts_generate_timeout', type=int, default=DEFAULT_TTS_GENERATE_TIMEOUT, help="TTS generation timeout")
    tts_group.add_argument('--no-tts', action='store_true', help="Disable TTS")

    debug_group = parser.add_argument_group('Debugging')
    debug_group.add_argument('--debug-prompts', action='store_true', help="Print LLM prompts")
    debug_group.add_argument('--debug-logic', action='store_true', help="Print logic flow")
    debug_group.add_argument('--debug-context', action='store_true', help="Print summary context")

    args = parser.parse_args()

    colored_print(Colors.SYSTEM, "--- Starting Conversation Simulator ---")

    # Initialize resources
    if not args.no_tts and not load_voice_data(args.voices_dir):
        colored_print(Colors.ERROR, "Voice data load failed")
        sys.exit(1)

    server_config = json.loads(json.dumps(SERVER_CONFIG))
    server_config['A']['llm_url'] = args.llm_url
    server_config['B']['llm_url'] = args.llm_url
    server_config['Summarizer']['llm_url'] = args.llm_url_summarizer
    server_config['A']['tts_url'] = args.tts_url
    server_config['B']['tts_url'] = args.tts_url
    server_config['Human']['tts_url'] = args.tts_url

    tts_manager = TTSManager(
        tts_config=server_config,
        tts_timeout=args.tts_timeout,
        tts_generate_timeout=args.tts_generate_timeout,
        no_tts=args.no_tts
    )
    request_session = requests.Session()

    # Main loop
    loop_iteration = 0
    global_prompt_num = 0
    current_team_index = 0
    primary_prompts = []
    paused = False
    has_idled = False

    if args.initial_prompt:
        primary_prompts = [args.initial_prompt]
    elif os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            primary_prompts = [
                line.strip() for line in f
                if line.strip() and not line.startswith('#')
            ]

    try:
        while args.loop <= 0 or loop_iteration < args.loop:
            loop_iteration += 1

            # Check secondary prompts for control commands and content
            initial_prompt = None
            prompt_source = "None"
            has_command = False
            if os.path.exists(args.secondary_prompts_file) and os.path.getsize(args.secondary_prompts_file) > 0:
                with open(args.secondary_prompts_file, 'r', encoding='utf-8') as f:
                    secondary_lines = [line.strip() for line in f if line.strip()]

                # Process commands
                for line in secondary_lines:
                    if line.lower() == ":pause":
                        paused = True
                        colored_print(Colors.SYSTEM, "Pausing primary prompt processing due to :pause command")
                        has_command = True
                    elif line.lower() in [":continue", ":resume"]:
                        paused = False
                        colored_print(Colors.SYSTEM, f"Resuming primary prompt processing due to {line} command")
                        has_command = True

                # Filter out commands for prompt generation
                valid_content = [
                    line for line in secondary_lines
                    if line.lower() not in [":pause", ":continue", ":resume"]
                ]
                secondary_content = "\n".join(valid_content).strip()

                if secondary_content:
                    prompt_q = Queue(maxsize=1)
                    gen_thread = threading.Thread(
                        target=generate_prompt_worker,
                        args=(
                            request_session, server_config['Summarizer'], secondary_content,
                            prompt_q, args.prompt_gen_max_tokens, args.temperature + 0.1,
                            args.timeout, args.debug_prompts
                        ),
                        daemon=True
                    )
                    gen_thread.start()
                    try:
                        initial_prompt = prompt_q.get(timeout=args.timeout + 10)
                        prompt_source = "Secondary File (Generated)"
                    except Empty:
                        colored_print(Colors.ERROR, "Prompt generation timeout")

                # Clear secondary prompts file
                with open(args.secondary_prompts_file, 'w', encoding='utf-8') as f:
                    f.write("")

                # Reset idling state if any content or command was processed
                if secondary_content or has_command:
                    has_idled = False

            # If paused, skip primary prompts as if aiprompts.txt is empty
            if not paused and not initial_prompt and primary_prompts:
                initial_prompt = primary_prompts[global_prompt_num % len(primary_prompts)]
                prompt_source = f"Primary ('{args.prompts_file}')"

            # If no prompt is available, idle silently after first message
            if not initial_prompt:
                if not has_idled:
                    colored_print(
                        Colors.HEADER,
                        f"\n===== START LOOP {loop_iteration}{' (Infinite)' if args.loop <= 0 else f'/{args.loop}'} ====="
                    )
                    colored_print(Colors.SYSTEM, f"No prompts found. Idling {IDLE_WAIT_SECONDS}s")
                    has_idled = True
                time.sleep(IDLE_WAIT_SECONDS)
                loop_iteration -= 1
                continue

            # Reset idling state and print loop header for new prompt
            has_idled = False
            colored_print(
                Colors.HEADER,
                f"\n===== START LOOP {loop_iteration}{' (Infinite)' if args.loop <= 0 else f'/{args.loop}'} ====="
            )

            global_prompt_num += 1
            voice_names = {"A": "default", "B": "default", "Human": "default"}
            if not args.no_tts and VOICE_TEAMS:
                team = VOICE_TEAMS[current_team_index % len(VOICE_TEAMS)]
                voice_names = {
                    'A': team['A'],
                    'B': team['B'],
                    'Human': team['Human']
                }
                current_team_index += 1

            simulate_conversation(
                tts_manager=tts_manager,
                request_session=request_session,
                server_config=server_config,
                num_turns=args.turns,
                initial_prompt=initial_prompt,
                req_timeout=args.timeout,
                max_tokens=args.max_tokens,
                summary_max_tokens=args.summary_max_tokens,
                temperature=args.temperature,
                tts_timeout=args.tts_timeout,
                force_human=args.force_human,
                debug_prompts=args.debug_prompts,
                debug_logic=args.debug_logic,
                debug_context=args.debug_context,
                voice_name_a=voice_names['A'],
                voice_name_b=voice_names['B'],
                voice_name_human=voice_names['Human'],
                prompt_global_num=global_prompt_num,
                log_summaries=args.log_summaries,
                summary_log_format=args.summary_log_format,
                human_summarizer=args.human_summarizer
            )

    except KeyboardInterrupt:
        colored_print(Colors.SYSTEM, "Interrupted. Exiting")
    finally:
        colored_print(Colors.SYSTEM, "Cleaning up")
        tts_manager.shutdown()
        request_session.close()
        colored_print(Colors.SUCCESS, "--- Script finished ---")
