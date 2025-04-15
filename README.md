      
# Conversation Simulator

## Overview

`ai47.py` simulates a multi-turn conversation between two Language Learning Models (LLMs), referred to as "David" (Server A) and "Zira" (Server B). The conversation is initiated by a "Human" prompt. The script integrates with external Piper TTS (Text-to-Speech) servers to provide spoken output for each participant's turn, including the initial Human prompt.

It utilizes Python's threading capabilities to perform LLM requests and TTS generation concurrently, aiming for a more seamless and natural-sounding interaction where one AI can process its response while the previous AI's response is being spoken. It also features TTS lookahead, pre-generating the audio for the *next* initial prompt while the current conversation is concluding.

Output is displayed in the console with color-coding for different speakers and system messages. LLM responses are printed *just before* their corresponding TTS playback begins, synchronizing the visual and auditory output.

## Features

*   **Dual LLM Conversation:** Simulates back-and-forth dialogue between two configurable LLM endpoints.
*   **TTS Integration:** Uses external Piper TTS servers for voice output for each AI and the initial prompt.
*   **Concurrency:**
    *   Overlaps LLM inference and TTS generation/playback using threads.
    *   Pre-generates TTS for the next conversation's initial prompt (lookahead).
*   **Seamless Looping:** Initializes network resources (TTS manager, HTTP session) once for smoother transitions when looping through multiple prompts or iterations.
*   **Delayed Printing:** Synchronizes console text output with TTS playback.
*   **Configurable:** Extensive command-line arguments to control behavior, server endpoints, LLM parameters, TTS, and looping.
*   **Prompt Management:** Reads initial prompts from a file or command line.
*   **Roleplaying Options:** Can instruct LLMs to maintain a specific persona (e.g., `--force-human`).

## Prerequisites

1.  **Python 3.x:** The script is written for Python 3.
2.  **Python Libraries:**
    *   `requests`: For making HTTP requests to LLM and TTS servers. Install using pip:
        ```bash
        pip install requests
        ```
3.  **LLM Servers (x2):** Two running LLM inference servers compatible with the `llama.cpp` server's `/completion` endpoint format.
    *   **Default URLs:** `http://127.0.0.1:8080` (Server A) and `http://127.0.0.1:8081` (Server B).
    *   These need to be started independently (e.g., using `llama.cpp`'s `server` command with appropriate models).
4.  **Piper TTS Servers (x3):** Three running Piper TTS HTTP servers. A common way to achieve this is using a wrapper script (like those designed for WSL) that exposes Piper CLI functionality over HTTP. **Crucially, you need three separate instances running on different ports**, one for each voice:
    *   **Server A (David):** Default URL `http://127.0.0.1:5001`
    *   **Server B (Zira):** Default URL `http://127.0.0.1:5002`
    *   **Server Human (Initial Prompt):** Default URL `http://127.0.0.1:5003`
    *   These servers need to be configured with the desired Piper voice models and started independently. They must expose `/generate` (POST, takes raw text, returns JSON with `audio_id`) and `/play` (POST, takes JSON `{"audio_id": "..."}`, plays audio) endpoints.

## Setup

1.  **Install Python and Pip:** Ensure you have a working Python 3 installation.
2.  **Install Requests Library:**
    ```bash
    pip install requests
    ```
3.  **Set up LLM Servers:**
    *   Download or build `llama.cpp`.
    *   Obtain suitable GGUF model files for Server A and Server B.
    *   Start Server A:
        ```bash
        ./server -m path/to/model_a.gguf -c 2048 --port 8080 # Add other options as needed
        ```
    *   Start Server B:
        ```bash
        ./server -m path/to/model_b.gguf -c 2048 --port 8081 # Add other options as needed
        ```
    *   Verify they are running by accessing `http://127.0.0.1:8080` and `http://127.0.0.1:8081` in your browser or using `curl`.
4.  **Set up Piper TTS Servers:**
    *   Install Piper TTS.
    *   Download desired Piper voice models (`.onnx` and `.onnx.json`).
    *   Set up and run three instances of a Piper HTTP wrapper script (ensure the wrapper matches the required `/generate` and `/play` endpoints). Configure each instance with a different voice model and run it on the designated port (5001, 5002, 5003 by default).
    *   Example (conceptual command for a hypothetical wrapper):
        ```bash
        # Terminal 1 (Server A - David)
        python piper_http_wrapper.py --model path/to/david_voice.onnx --port 5001

        # Terminal 2 (Server B - Zira)
        python piper_http_wrapper.py --model path/to/zira_voice.onnx --port 5002

        # Terminal 3 (Server Human)
        python piper_http_wrapper.py --model path/to/human_voice.onnx --port 5003
        ```
    *   Verify they are running and responsive (e.g., check the wrapper's logs or try sending test requests if possible). Check connectivity, especially if using WSL (network configuration might be needed).

5.  **Prepare Prompts File:**
    *   Create a text file named `aiprompts.txt` (or use the `--prompts_file` argument to specify a different name).
    *   Add one initial conversation prompt per line.
    *   Lines starting with `#` are treated as comments and ignored.
    *   Example `aiprompts.txt`:
        ```
        # Space Prompts
        Tell me about the challenges of colonizing Mars.
        Discuss the possibility of finding extraterrestrial life.

        # Creative Prompts
        Write a short story about a time-traveling librarian.
        ```
    *   Alternatively, use the `--initial_prompt` argument to provide a single prompt directly.

## Running the Script

Once all servers are running and the prompts file is ready, you can run the script from your terminal.

**Basic Usage (Defaults):**
Uses `aiprompts.txt`, default server URLs, 5 turns per AI, default LLM parameters.

```bash
python ai47.py

    
