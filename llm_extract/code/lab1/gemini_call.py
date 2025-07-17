# libraries
import os
import google.generativeai as genai

# 1. Read API token from file 
# Configure the Gemini API key
gem_path = "/home/<your_net_id/keys/gemini-key.txt"  # Adjust path as needed
with open(gem_path, 'r') as file:
    api_key = file.read().strip()
genai.configure(api_key=api_key)

# 2. Function for API call
def call_gemini(prompt_text, model_select):
    model = genai.GenerativeModel(model_select)
    response = model.generate_content(
        prompt_text,
        generation_config=genai.types.GenerationConfig(temperature=0))
    return response.text

# 3. Run:
question = "What colors do you mix to make purple?"
answer = call_gemini(question, 'gemini-1.5-flash')
print(f"Gemini's answer: {answer}")
