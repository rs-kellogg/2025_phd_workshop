from openai import OpenAI
import os
from pathlib import Path
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


def extract_int(val):
        val = str(val).replace("$", "").replace(",", "").replace("nan", "0").replace("N/A", "0")
        val_int = int(val)  
        return val_int


def test_severance(df, df_ref):

    df_merge = pd.merge(df, df_ref, on="filename", how="inner", suffixes=("", "_ref"))

    # compare the value of severance_amount and severance_amount_ref, create a new column "severance_amount_match"
    df_merge["severance_amount_match"] = df_merge.apply(
        lambda row: extract_int(row["severance_amount"]) == extract_int(row["severance_amount_ref"]),
        axis=1
    )

    df_merge.drop(columns=["text", "severance_amount_text_description", "other_perks"], inplace=True)

    return df_merge



if __name__ == "__main__":

    data_path = Path("./data")

    prompt_file = "prompt.txt"
    with open(prompt_file, "r") as f:
        prompt_main = f.read().strip()
    # prompt_main = ""

    # Read data text files
    prompt_list = []
    files_list = list(data_path.glob("*.txt"))
    df = pd.DataFrame()
    for file in files_list:
        row = dict()
        row["filename"] = file.name
        with open(file, "r") as f:
            row["text"]= f"{prompt_main}{f.read().strip()}"
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    llm_model = "gpt-4.1"

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    output_folder = Path(f"./output_{timestamp}")
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / "output.csv"
    output_testing_file = output_folder / "output_testing.csv"

    if output_file.exists():
        print(f"Output file {output_file} already exists. Skip LLM model. Start testing...")
    else:
        print(f"Output file {output_file} does not exist. Running LLM query...")

        # Generate outputs for each prompt
        for idx, row in df.iterrows():
            temperature = 1.0
            max_tokens = 500
            output = llm_with_logging(row["text"], llm_model, llm_openai_schema, temperature, max_tokens)

            output_dict = output.model_dump()
            new_row = pd.DataFrame([{**row.to_dict(), **output_dict}])

            if output_file.exists():
                new_row.to_csv(output_file, mode='a', header=False, index=False)
            else:
                new_row.to_csv(output_file, mode='w', header=True, index=False)

    # Testing
    df = pd.read_csv(output_file)
    
    ref_file = Path("./data/ref.csv")
    df_ref = pd.read_csv(ref_file)

    df_testing = test_severance(df, df_ref)
    df_testing.to_csv(output_testing_file, index=False)
