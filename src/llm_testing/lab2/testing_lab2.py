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
    severance_amount_text_description: str
    other_perks: str


def llm_openai_schema(prompt, llm_model, temperature, max_tokens):
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


def fuzzy_string_match(str1, str2):
    """
    Compares two strings by converting them to lowercase and checking for substring presence.
    Returns True if one string is a substring of the other, case-insensitively.
    """
    s1_lower = str(str1).lower()
    s2_lower = str(str2).lower()
    return s1_lower in s2_lower or s2_lower in s1_lower

def extract_int(val):
        val = str(val).replace("$", "").replace(",", "").replace("nan", "0").replace("N/A", "0")
        val_int = int(val)  
        return val_int

def test_severance(df, df_ref):

    for idx, row in df.iterrows():
        exec_name = row["exec_name"]
        severance_amount = extract_int(row["severance_amount"])
        
        match_found = False
        for idx_ref, row_ref in df_ref.iterrows():
            if fuzzy_string_match(exec_name, row_ref["exec_name"]):
                match_found = True
                ref_severance_amount = extract_int(row_ref["severance_amount"])
                
                if severance_amount == ref_severance_amount:
                    print(f"=== Match found for {exec_name}: Severance amounts agree: {severance_amount}")
                else:
                    print(f"=== Match found for {exec_name}: Severance amounts disagree: {severance_amount} vs {ref_severance_amount}")
                break
        
        if not match_found:
            print(f"=== No match found for {exec_name}.")



if __name__ == "__main__":

    data_path = Path("./data")

    prompt_file = "prompt.txt"
    with open(prompt_file, "r") as f:
        prompt_main = f.read().strip()
    # prompt_main = ""

    # Read data text files
    prompt_list = []
    files_list = list(data_path.glob("*.txt"))
    for file in files_list:
        with open(file, "r") as f:
            prompt_list.append(f"{prompt_main}{f.read().strip()}")

    llm_model = "gpt-4.1"

    output_file = Path("./output-v6.csv")

    if output_file.exists():
        print(f"Output file {output_file} already exists. Skip LLM model. Start testing...")
        df = pd.read_csv(output_file)
    else:
        print(f"Output file {output_file} does not exist. Running LLM query...")
        # df = pd.DataFrame()

        # Generate outputs for each prompt
        for prompt in prompt_list:
            temperature = 1.0
            max_tokens = 500
            output = llm_with_logging(prompt, llm_model, llm_openai_schema, temperature, max_tokens)

            new_row = pd.DataFrame([output.model_dump()])
            if output_file.exists():
                new_row.to_csv(output_file, mode='a', header=False, index=False)
            else:
                new_row.to_csv(output_file, mode='w', header=True, index=False)
        # df = pd.DataFrame(prompt_list, columns=["prompt"])
            # df = pd.concat([df, new_row], ignore_index=True)

        # df.to_csv(output_file, index=False)
        df = pd.read_csv(output_file)
    
    ref_file = Path("./ref.csv")
    df_ref = pd.read_csv(ref_file)

    test_severance(df, df_ref)
