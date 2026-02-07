import json
import os
import requests
import time
from openai import OpenAI
from tqdm import tqdm

# --- CONFIGURATION ---

# 1. FILES
INPUT_FILE = "data/nyc_benchmark_data_duke.json" 
OUTPUT_FILE = "results/gpt-5.1_gpt-5-mini_llama3.json"

# 2. DUKE AI GATEWAY (The "Smart" Contestants)
DUKE_API_KEY = "REDACTED_FOR_SECURITY" 
DUKE_BASE_URL = "https://litellm.oit.duke.edu/v1"

# Define all the Duke models you want to test here:
DUKE_MODELS = [
    "gpt-5.1",      # The Heavyweight Champion
    "gpt-5-mini"    # The Lightweight Challenger
]

# 3. LOCAL OLLAMA (The "Privacy" Contestant)
OLLAMA_URL = "http://localhost:11434/api/chat"
LOCAL_MODEL = "llama3"

# --- HELPER FUNCTIONS ---

def get_duke_response(question, model_name):
    """Hits Duke's Gateway with Rate Limit Protection."""
    client = OpenAI(api_key=DUKE_API_KEY, base_url=DUKE_BASE_URL)
    
    # Retry logic for 429 Errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful housing law assistant. Answer accurately based on NYC law."},
                    {"role": "user", "content": question}
                ],
                temperature=0,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            if "429" in str(e) or "Rate limit" in str(e):
                print(f"\n⚠️ Rate Limit on {model_name}! Sleeping 60s...")
                time.sleep(60)
            else:
                return f"[ERROR] {e}"
    return "[ERROR] Failed after retries."

def get_ollama_response(question, model_name):
    """Hits Local Ollama."""
    try:
        response = requests.post(
            OLLAMA_URL, 
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": question}],
                "stream": False,
                "temperature": 0
            }
        )
        if response.status_code == 200:
            return response.json()['message']['content']
        return f"[ERROR] Status {response.status_code}"
    except Exception as e:
        return f"[ERROR] Connection Failed: {e}"

# --- MAIN ENGINE ---

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"CRITICAL ERROR: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r") as f:
        questions = json.load(f)

    results = []

    print(f"Starting Multi-Model Benchmark")
    print(f"Contestants: {DUKE_MODELS} vs. {LOCAL_MODEL}")
    print(f"Speed Limit: 1 request every 4s per model")
    
    # Main Loop
    for i, item in tqdm(enumerate(questions), total=len(questions)):
        q_text = item["question"]
        
        # Dictionary to store answers for this specific question
        current_responses = {}

        # 1. Loop through all DUKE models
        for model_name in DUKE_MODELS:
            ans = get_duke_response(q_text, model_name)
            current_responses[model_name] = ans
            time.sleep(4) # Sleep after EACH call to respect rate limit
        
        # 2. Call LOCAL model (No sleep needed)
        ans_local = get_ollama_response(q_text, LOCAL_MODEL)
        current_responses[LOCAL_MODEL] = ans_local

        # 3. Save Row
        results.append({
            "question_id": i,
            "category": item.get("category", "General"),
            "question": q_text,
            "ground_truth": item["ground_truth_answer"],
            "citation": item.get("citation", "N/A"),
            "responses": current_responses
        })

    # Dump to file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBenchmark Complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()