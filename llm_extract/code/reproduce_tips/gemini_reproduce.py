################################
# Gemini Reproducibility Example
################################

# libraries
import os
import csv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError 
from typing import List
from datetime import datetime
import pandas as pd
import time
import json

#########
# Inputs
#########

# API Token
api_file = "/home/<your_net_id>/keys/gemini-key.txt"
with open(api_file, 'r') as file:
    api_key = file.read().strip()
client = genai.Client(api_key=api_key)

# settings
model_select = 'gemini-1.5-flash' 
temp_set = 0
top_pset = 0.0

# Prompt Text
prompt1 = "For New York City in 2025, name ONE significant mayoral candidate and their political party. Provide the response as a JSON object with 'candidate_name' and 'political_party' fields."
prompt2 = "Tell me about one prominent person running for mayor of New York City in 2025, and their party affiliation. Return the data as a JSON object with 'candidate_name' and 'political_party'."
prompt3 = "Identify ONE mayoral candidate for the New York City election in 2025 and their political party. Output in JSON format with 'candidate_name' and 'political_party'."

# outputs
output_dir = "/home/<your_net_id>/llm_extract/reproducible_output"
results_track = os.path.join(output_dir, "mayor_results_track.csv")

#########
# Schema
#########

# New Schema for a single object with two fields
class CandidateResponse(BaseModel):
    candidate_name: str
    political_party : str


#########
# Functions
#########

# api call
def call_gem(prompt_text, model_select, temp_set, top_pset, response_schema):
    try:
        response = client.models.generate_content(
            model=model_select,
            contents=[
                prompt_text
            ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    "temperature": temp_set,
                    "top_p": top_pset
                }
        )

        json_output = response.text
        return json_output

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None


# Create tracker file if doesn't exist
def init_tracker(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Updated header to include the two separate fields
            writer.writerow(["timestamp", "prompt", "model", "temperature", "top_p", "candidate_name", "political_party"])
        print(f"Created new tracker at {csv_path}")

# Append results to tracker
def log_response(csv_path, prompt, model, temperature, top_p, candidate_name, political_party):
    now = datetime.now().isoformat()
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the two separate fields directly
        writer.writerow([now, prompt, model, temperature, top_p, candidate_name, political_party])

# Run all prompts
def run_prompts(prompt_list, model_select, temperature, top_p, response_schema, csv_path):
    for prompt in prompt_list:
        print(f"Processing prompt: {prompt}...")
        
        raw_json_response = call_gem(prompt, model_select, temperature, top_p, response_schema)
        
        candidate_name = ""
        political_party = ""

        if raw_json_response:
            try:
                # Parse the JSON string into the Pydantic model
                parsed_data = response_schema.model_validate_json(raw_json_response)
                candidate_name = parsed_data.candidate_name
                political_party = parsed_data.political_party
            except ValidationError as e:
                print(f"Schema validation error for prompt '{prompt}': {e}")
                print(f"Raw JSON response: {raw_json_response}")
            except json.JSONDecodeError as e:
                print(f"JSON decoding error for prompt '{prompt}': {e}")
                print(f"Raw response: {raw_json_response}")
            except Exception as e:
                print(f"An unexpected error occurred during parsing for prompt '{prompt}': {e}")
                print(f"Raw response: {raw_json_response}")
        else:
            print(f"No response or error from call_gem for prompt: {prompt}")

        # Pass the extracted fields to log_response
        log_response(csv_path, prompt, model_select, temperature, top_p, candidate_name, political_party)
        time.sleep(5)

#####
# Run
#####

def main():
    prompt_list = [prompt1, prompt2, prompt3]
    init_tracker(results_track)
    # Pass the single-object schema: CandidateResponse
    run_prompts(prompt_list, model_select, temp_set, top_pset, CandidateResponse, results_track)
    
    print(f"\nAll prompts processed. Results logged to {results_track}")
    try:
        df = pd.read_csv(results_track)
        print("\nFirst 5 rows of mayor_results_track.csv:")
        print(df.head())
    except Exception as e:
        print(f"Could not read CSV file: {e}")

if __name__ == "__main__":
    main()
