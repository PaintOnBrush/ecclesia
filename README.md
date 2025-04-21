```markdown
# AI Conversation Simulator (ai67.py)

## Overview

`ai67.py` simulates multi-turn conversations between two AI agents (configurable names, e.g., "David" and "Zira"), initiated by a "Human" prompt. It features dynamic voice selection using Piper TTS, persistent conversation summaries via a third LLM, and flexible prompt input methods.

The script leverages Python threading for concurrent LLM requests and TTS operations (generation and playback), aiming for a smooth, interactive experience. Output is color-coded in the terminal, and various debugging flags provide insight into the script's logic and external server interactions.

## Key Features

*   **Multi-Agent Conversation:** Simulates dialogue between two configurable LLM endpoints (can be the same model/URL).
*   **Dynamic TTS Voices:**
    *   Loads available Piper voices from a specified directory.
    *   Uses pre-defined "voice teams" (assigning voices to Human, Agent A, Agent B).
    *   Rotates through voice teams for each new conversation prompt.
    *   Requires a single, consolidated `wsl_piper_aplay_server.py` instance capable of handling requests for different voices.
*   **Persistent Summarization:**
    *   After each conversation, calls a dedicated (or fallback) Summarizer LLM.
    *   The Summarizer LLM generates a concise summary of the just-completed conversation, potentially using the summary from the *previous* conversation as context.
    *   The newly generated summary is saved to `persistent_summary.txt`.
    *   The "Human" voice speaks this summary at the end of the conversation turn.
*   **Dynamic Prompting & Control:**
    *   Reads initial prompts from a primary file (`aiprompts.txt` by default).
    *   Can optionally read prompts from a secondary file (`aiprompts2.txt`).
    *   Content in the secondary file is used to *generate* a new prompt via the Summarizer LLM (acting as a prompt generator).
    *   Supports `:pause` and `:continue`/`:resume` commands in the secondary file to control processing of the primary prompts.
*   **Concurrency:** Uses threading to overlap LLM requests and TTS generation/playback where possible.
*   **Ordered Playback:** A dedicated player thread ensures conversational turns (including initial Human prompt and final summary) are spoken in the correct sequence.
*   **Configurable:** Rich command-line arguments control LLM/TTS endpoints, behavior (roleplay, awareness), conversation length, looping, voice directory, timeouts, summarization, logging, and debugging.
*   **Error Handling:** Includes checks for server availability, handles request timeouts and errors, attempts summary retries, and manages thread shutdowns.

## Prerequisites

1.  **Python 3.x:** Python 3.7+ recommended.
2.  **Python Libraries:**
    *   `requests`: Install via `pip install requests`.
    *   `waitress` (Optional, Recommended for `wsl_piper_aplay_server.py`): Install via `pip install waitress`.
3.  **LLM Servers:**
    *   **Conversational LLM:** At least one running LLM server (e.g., `llama.cpp` server) compatible with the `/completion` endpoint. This server will handle requests for both Agent A and Agent B. (Default: `http://127.0.0.1:8080/completion`)
    *   **Summarizer LLM (Optional but Recommended):** Another running LLM server, ideally suited for summarization or good instruction following. If not specified or fails, the script falls back to using the conversational LLM for summaries. (Default: `http://127.0.0.1:8082/completion`)
4.  **Consolidated Piper TTS Server:**
    *   **One instance** of the modified `wsl_piper_aplay_server.py` (the version that accepts JSON with `model_path`) running, typically on WSL.
    *   This single server instance must be started with the `--voices-dir` argument pointing to the directory containing *all* your Piper `.onnx` and `.onnx.json` voice files.
    *   (Default URL expected by `ai67.py`: `http://127.0.0.1:5001`)

## Setup

1.  **Install Python & Libraries:** See Prerequisites.
2.  **Set up LLM Servers:**
    *   Start your conversational LLM server (e.g., on port 8080).
    *   Start your summarizer LLM server (e.g., on port 8082).
    *   Verify they are running and accessible.
3.  **Set up Consolidated TTS Server:**
    *   Ensure you have the **correct version** of `wsl_piper_aplay_server.py` (the one modified to accept `model_path` via JSON and using `--voices-dir`).
    *   Run it on WSL, pointing to your voices directory:
        ```bash
        # Example:
        python wsl_piper_aplay_server.py --voices-dir /path/to/your/piper-voices/ --port 5001 --cuda
        ```
    *   Verify it starts without errors and reports the correct "Allowed Voices Dir".
4.  **Prepare Voices Directory:** Ensure the directory specified by `--voices-dir` contains matching pairs of `.onnx` and `.onnx.json` files for the voices you want to use.
5.  **Configure Voice Teams:** Edit the `teams_config` list within the `load_voice_data` function in `ai67.py` to define your desired voice pairings for 'A', 'B', and 'Human', using the base filenames (without extensions) found in your voices directory.
6.  **Prepare Prompts Files:**
    *   Create `aiprompts.txt` (or use `--prompts_file`) with one conversation starter prompt per line.
    *   Optionally create `aiprompts2.txt` (or use `--secondary_prompts_file`) for dynamic prompt generation or control commands (`:pause`, `:continue`).

## Running the Script

Ensure all required servers (LLMs, TTS) are running before executing `ai67.py`.

**Basic Usage (Defaults):**
Uses default servers, 5 turns, default voices dir, rotates default teams, reads `aiprompts.txt`.

```bash
python ai67.py
```

**Infinite Loop with Forced Human Roleplay & Debugging:**

```bash
python ai67.py --force-human --loop 0 --debug-logic --debug-prompts --debug-context
```

**Specify Custom URLs and Different Number of Turns:**

```bash
python ai67.py -t 3 --llm-url http://192.168.1.100:8080/completion --llm-url-summarizer http://192.168.1.100:8082/completion --tts-url http://192.168.1.105:5001
```

**Disable TTS:**

```bash
python ai67.py --no-tts
```

**Use a Single Initial Prompt (Ignoring Files):**

```bash
python ai67.py --initial_prompt "Discuss the pros and cons of universal basic income."
```

**Log Summaries to File:**

```bash
python ai67.py --log-summaries --summary-log-format simple # Or 'detailed'
```

## Command-Line Arguments

**(Run `python ai67.py -h` for the full, up-to-date list)**

*   **Input Prompts:**
    *   `--prompts_file`: Primary prompts file.
    *   `--initial_prompt`: Use this single prompt instead of files.
    *   `--secondary_prompts_file`: Secondary file for dynamic prompts/commands.
*   **Conversation Control:**
    *   `-t`, `--turns`: Number of turns *per AI agent*.
    *   `--loop`: Number of times to loop through prompts (0=infinite).
    *   `--log-summaries`: Enable logging summaries to `summary_log.txt`.
    *   `--summary-log-format`: Format for summary log ('detailed' or 'simple').
*   **LLM Configuration:**
    *   `--llm-url`: URL for the main conversational LLM (A/B).
    *   `--llm-url-summarizer`: URL for the Summarizer LLM.
    *   `--timeout`: LLM request timeout (seconds).
    *   `--max_tokens`: Max generation tokens for A/B.
    *   `--summary-max-tokens`: Max generation tokens for Summarizer.
    *   `--prompt-gen-max-tokens`: Max tokens for generating prompts from secondary file.
    *   `--temperature`: LLM sampling temperature.
    *   `--force-human`: Append roleplay clause to A/B prompts.
    *   `--human-summarizer`: Append roleplay clause to Summarizer prompts.
*   **TTS Configuration:**
    *   `--voices-dir`: Path to Piper voices directory.
    *   `--tts-url`: URL of the single, consolidated TTS server.
    *   `--tts_timeout`: TTS playback request timeout (seconds).
    *   `--tts_generate_timeout`: TTS generation request timeout (seconds).
    *   `--no-tts`: Disable all TTS functionality.
*   **Debugging:**
    *   `--debug-prompts`: Show LLM prompts and request durations.
    *   `--debug-logic`: Show detailed script execution flow.
    *   `--debug-context`: Show summary content being read/written/used.
*   **Other:**
    *   `--aware`: Append system note about keywords to initial prompt.
    *   `--file_location`, `--file_delimiter`, `--prompt_add_delimiter`: Arguments for (currently placeholder) file interaction features.

## How it Works Internally

1.  **Initialization:** Parses args, loads voice data (`load_voice_data`), initializes `TTSManager` and `requests.Session`.
2.  **Main Loop:** Iterates based on `--loop`.
    *   **Prompt Selection:** Checks secondary file for commands/content. If content exists, generates a prompt using the summarizer LLM. If paused or no secondary content, tries to get a prompt from the primary file list. If no prompt found, idles.
    *   **Voice Team Selection:** Selects the next voice team using round-robin.
    *   **`simulate_conversation` Call:** Passes the selected prompt, voice names, configuration, and previous summary (if any) to the function.
3.  **`simulate_conversation`:**
    *   Prints the initial Human prompt text.
    *   Starts the TTS player thread.
    *   Generates TTS for the initial Human prompt (using the selected Human voice) and queues it for playback.
    *   Starts the LLM request for Turn 1 (Agent A).
    *   Enters the main conversation loop (`for turn_index in range(...)`):
        *   Waits for the pending LLM response (e.g., A's Turn 1).
        *   If the response is valid, starts TTS generation for it *in the background* using the correct voice for that agent (A or B).
        *   Starts the *next* LLM request (e.g., B's Turn 1) *in the background*.
        *   Waits for the *current* turn's TTS generation to complete.
        *   Queues the completed item (text, audio ID, speaker info) onto the `playback_queue`.
        *   Adds the AI's response to `conversation_history`.
        *   Repeats for the configured number of turns, alternating A and B.
    *   **Finally Block:**
        *   Reads the `persistent_summary.txt` file (if it exists).
        *   Calls the Summarizer LLM (using `request_summary_worker`) with the full conversation history and the previously read summary.
        *   Waits for the new summary text.
        *   Writes the new summary text to `persistent_summary.txt`.
        *   Optionally appends to `summary_log.txt`.
        *   Generates TTS for the new summary text (using the selected Human voice).
        *   Waits for summary TTS generation.
        *   Queues the summary playback item.
        *   Sends a `None` sentinel to the playback queue.
        *   Waits for the `playback_queue` to become empty using `join()`.
        *   Signals the player thread to stop and joins it.
4.  **Loop Continuation/Exit:** The main loop continues to the next prompt or exits based on the `--loop` argument.
5.  **Cleanup:** Closes the requests session.

## Troubleshooting

*   **Connection Errors:** Verify all three server URLs (Convo LLM, Summarizer LLM, TTS) are correct and the servers are running and accessible. Check firewalls and WSL networking if applicable.
*   **TTS Failures:** Check `wsl_piper_aplay_server.py` logs. Ensure `--voices-dir` is correct and contains valid `.onnx`/`.onnx.json` pairs for voices defined in `VOICE_TEAMS`. Use `--no-tts` to isolate.
*   **LLM Errors / Early Exit:** Check `llama.cpp` server logs. Use `--debug-prompts` and `--debug-logic` to see the request flow and content. Increase `--timeout` or `--max_tokens` if needed. Check if models are appropriate for the task (chat vs. summarization).
*   **Summarizer Fails:** Check the Summarizer LLM server logs. Use `--debug-prompts` and `--debug-context`. Ensure the model is suitable. Try simplifying the summarizer prompt instruction in `send_llm_request`. Verify `--llm-url-summarizer` is correct.
*   **Voice Issues:** Ensure `load_voice_data` reports finding voices and creating teams. Double-check voice names in the `teams_config` list match filenames exactly.
```
