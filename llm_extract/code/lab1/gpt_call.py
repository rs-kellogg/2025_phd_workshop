# libraries
from openai import OpenAI
import os

# 1. Read API token from file 
api_file = "/home/<your_net_id>/keys/api_key.txt"
with open(api_file, 'r') as file:
    api_key = file.read().strip()
client = OpenAI(api_key=api_key)

# 2. Function for API call
def call_gpt(prompt_text, model_select):
    response = client.chat.completions.create( 
	model= model_select,
	messages=[ 
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt_text} ], 
             temperature=0
    )
    response.choices[0].message.content
    return response.choices[0].message.content          

# 3. Run:
question = "What colors do you mix to make purple?"
answer = call_gpt(question, "gpt-4o")
print(f"GPT's answer: {answer}")

