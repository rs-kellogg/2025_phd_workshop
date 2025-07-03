from openai import OpenAI
import os
from pathlib import Path
from typing import List
import datetime
import json
import pandas as pd
import re
import numpy as np


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
    timestamp_str = now.strftime("%Y%m%d-%H%M%S")
    
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



def get_embedding_openai(text):
    """
    Get embedding vector for a category string using OpenAI embeddings API.
    """
    api_key = Path("./openai.key").read_text().strip()
    client = OpenAI(api_key=api_key)

    # Use a small model for speed/cost; adjust as needed
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def fuzzy_string_match(str1, str2):
    """
    Compares two strings by converting them to lowercase and checking for substring presence.
    Returns True if one string is a substring of the other, case-insensitively.
    """
    s1_lower = str(str1).lower()
    s2_lower = str(str2).lower()
    return s1_lower in s2_lower or s2_lower in s1_lower


#  Evaluation Function
def evaluate_product_classification(df):
    """
    Compare output category to golden label.
    """
    results = []

    for _, row in df.iterrows():

        product = row["product_name"]
        gold_category = row["ground_truth_category"]
        llm_output = row["llm_response"]

        # Try to extract category from LLM output
        category_match = re.search(r"Category:\s*(.*)", llm_output)
        predicted_category = category_match.group(1).strip().lower() if category_match else "N/A"

        match = fuzzy_string_match(predicted_category, gold_category)

        # Get embeddings and similarity
        pred_emb = get_embedding_openai(predicted_category)
        gold_emb = get_embedding_openai(gold_category)
        similarity = cosine_similarity(pred_emb, gold_emb)

        results.append({
            "product_name": product,
            "gold_category": gold_category,
            # "llm_response": llm_output,
            "llm_predicted_category": predicted_category,
            "match_string": match,
            "cosine_similarity": similarity
        })

    return pd.DataFrame(results)



if __name__ == "__main__":

    df_products = pd.DataFrame({
        "product_name": [
            "Dyson V11 Torque Drive",
            "Keurig K-Supreme Plus SMART",
            "iPhone 15 Pro Max",
            "YETI Rambler 20 oz Tumbler",
            "Sony WH-1000XM5",
            "Blue Diamond Almonds - Lightly Salted",
            "Peloton Bike+",
            "Nest Thermostat (3rd Gen)",
            "L'Oréal Paris Revitalift Serum",
            "Kindle Paperwhite Signature Edition"
        ],
        "ground_truth_category": [
            "vacuum cleaner",
            "coffee maker",
            "smartphone",
            "drinkware",
            "headphones",
            "snack",
            "exercise equipment",
            "smart home device",
            "skincare",
            "e-reader"
        ]
    })

    # prompt = """Classify the following product into a specific consumer product category similar to how Amazon categorizes products. 
    prompt = """Classify the following product into a specific consumer product category. 
    Reply in the format: 
    Category: <your category>
    Reason: <your reasoning based on the product name>
    Here is the product: """

    # llm_model = "gpt-4.1"
    # llm_model = "gpt-4o"
    llm_model = "gpt-4.1-nano"
    temperature = 0.1
    max_tokens = 1000

    output_file = "product_classification_results.csv"

    # LLM evaluation
    df_results = df_products.copy()
    for _, row in df_results.iterrows():

        product = row["product_name"]
        user_prompt = f"{prompt} {product}\n"

        llm_output = llm_with_logging(user_prompt, llm_model, llm_openai, temperature, max_tokens)

        df_results.loc[_, "llm_response"] = llm_output

    df_results.to_csv(output_file, index=False)

    # Testing
    # df_results = pd.read_csv(output_file)
    df_testing = evaluate_product_classification(df_results)

    testing_output_file = "product_classification_testing_results.csv"
    df_testing.to_csv(testing_output_file, index=False)
        





