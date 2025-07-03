from openai import OpenAI
import os
from pathlib import Path
from typing import List
import datetime
import json
from pydantic import BaseModel
import pandas as pd

#########
# Schema
#########

class Severance(BaseModel):
    exec_name: str
    date: str
    severance_amount: str
    other_perks: str


def llm_openai(prompt, llm_model, temperature, max_tokens):
    """Call OpenAI ChatCompletion API and return output text."""

    # Set your OpenAI API key as an environment variable before running
    api_key = Path("./openai.key").read_text().strip()
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.parse(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user","content": prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        seed=42,
        response_format=Severance,
    )   

    # print(completion.choices[0].message.content)
    return completion.choices[0].message.parsed



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
    timestamp_str = now.strftime("%Y%m%d-%H%M%S")
    
    output_file = os.path.join(log_dir, f"logfile_{date_str}.jsonl")

    output = llm_func(prompt, llm_model, temperature, max_tokens)

    log_entry = {
        "timestamp": now.isoformat(),
        "model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "output": str(output),
    }

    with open(output_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"... Adding logs to {output_file}")
    return output



if __name__ == "__main__":

    data_path = Path("./data")
    files_list = list(data_path.glob("*.txt"))

    prompt_list = []
    for file in files_list:
        with open(file, "r") as f:
            prompt_list.append(f.read().strip())

    llm_model = "gpt-4.1"
    # llm_model = "gpt-4o"

    df = pd.DataFrame()
    for prompt in prompt_list[:1]:
        temperature = 1.0
        max_tokens = 500
        output = llm_with_logging(prompt, llm_model, llm_openai, temperature, max_tokens)

        new_row = pd.DataFrame([output.model_dump()])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv("output.csv", index=False)
    