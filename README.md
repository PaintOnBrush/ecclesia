make llamacpp with cuda support in c:\users\<users>\workspace\llamacpp

windows python 3.12
needs to pip install somethings. 

run both servers bat files

run third python script. latest version is 3.4 (ai34.py)

```How it Works Now:

    The batch script reads the initial list of prompts from aiprompts.txt.

    For each prompt, it runs ai34.py.

    Crucially, it now tells ai34.py the path to aiprompts.txt using --prompts_file "%SCRIPT_DIR%%PROMPTS_FILE%".

    Inside ai34.py, the simulate_conversation function receives this path.

    During the conversation, if David and Zira successfully use the prompt_add_delimiter (default: AGREE_ADD_PROMPT) consecutively, the script expects the second agreeing AI to provide the new prompt in its response, marked with "Prompt to add:\n".

    If the prompt is found, the Python script opens the file specified by prompts_file_path (which is aiprompts.txt via the argument) in append mode ('a') and writes the new prompt, followed by a newline.

    Important: The newly added prompt will only be used by the batch script the next time run_prompts.bat is executed. The current run of the batch script finishes processing only the prompts that existed when it started.

To Use:

    Make sure both scripts (ai34.py, run_prompts.bat) and the file (aiprompts.txt) are updated.

    Run run_prompts.bat.

    Guide the AIs in their conversation:

        "Let's add a new prompt about space travel."

        David might say: "Okay, I agree. AGREE_ADD_PROMPT"

        Zira might say: "Me too! AGREE_ADD_PROMPT"

        David (since he agreed first) then needs to say something like: "Great. Prompt to add:\nWrite a poem about the stars."

    The Python script should then detect this, print a success message (in light green), and append "Write a poem about the stars." as a new line to aiprompts.txt.
```
