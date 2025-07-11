from openai import OpenAI
import os
from pathlib import Path
import datetime
import json



def llm_openai(prompt, llm_model, temperature, max_tokens):
    """Call OpenAI ChatCompletion API and return output text."""

    # Set your OpenAI API key as an environment variable before running
    api_key = Path("./openai.key").read_text().strip()
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user","content": prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        seed=42,
    )   
    return completion.choices[0].message.content



def llm_with_logging(prompt, llm_model, llm_func, temperature, max_tokens):
    """
    Executes an LLM prompt using a provided function, saves the prompt and logs the interaction.
    Args:
        prompt (str): The prompt to send to the LLM.
        llm_model (str): The name of the LLM model to use.
        llm_func (callable): The function to execute the LLM call. Defaults to llm_execute.
        temperature (float): The temperature setting for the LLM.
        max_tokens (int): The maximum number of tokens to generate in the response.
    Returns:
        str: The output from the LLM.
    """

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    now = datetime.datetime.now()
    date_str = now.date().isoformat()
    
    output_file = os.path.join(log_dir, f"logfile_{date_str}.jsonl")

    output = llm_func(prompt, llm_model, temperature, max_tokens)

    log_entry = {
        "timestamp": now.isoformat(),
        "model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "output": output,
    }

    with open(output_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"... Adding logs to {output_file}")
    return output



if __name__ == "__main__":

    prompt = """Write a compelling headline for an email marketing campaign promoting a new rewards credit card for young professionals. Focus on travel perks, no annual fee, and cash back."""

    # # read promt from a file
    # file_prompt = "prompt.txt"
    # with open(file_prompt, "r") as f:
    #     prompt = f.read().strip()

    llm_model_list = [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        # "gpt-4o",
    ]

    for llm_model in llm_model_list:
        temperature = 1.0
        max_tokens = 500
        output = llm_with_logging(prompt, llm_model, llm_openai, temperature, max_tokens)
    