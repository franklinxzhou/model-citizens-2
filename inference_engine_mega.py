import json
import os
import requests
import time
from openai import OpenAI
import google.generativeai as genai
from tqdm import tqdm

# --- CONFIGURATION ---

# 1. FILES
INPUT_FILE = "data/nyc_benchmark_data_duke.json" 
OUTPUT_FILE = "results/gpt-5.1_gpt-5-mini_gpt-5-nano_llama-4-scout_mistral_gemini-3.0-flash_gemini-2.5-flash.json"

# 2. DUKE AI GATEWAY (OpenAI-Compatible)
# Includes your new futuristic models
DUKE_API_KEY = "sk-LDJVJXbFywt0v3MZC-N1lw" 
DUKE_BASE_URL = "https://litellm.oit.duke.edu/v1"

DUKE_MODELS = [
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-5-nano",     # New
    "llama-4-scout",  # New
    "mistral"         # New
]

# 3. GOOGLE GEMINI API (Native Google SDK)
# Get key from: aistudio.google.com
GEMINI_API_KEY = "AIzaSyBns8-lTR7x9b7ub8XBtViIcBa-YpA7uwE"

GEMINI_MODELS = [
    "gemini-3.0-flash", 
    "gemini-2.5-flash"
]

# 4. LOCAL OLLAMA
OLLAMA_URL = "http://localhost:11434/api/chat"
LOCAL_MODELS = ["llama3"]

# --- RATE LIMIT CONFIG ---
# Duke: ~20 RPM -> 3s delay (Safe: 4s)
# Google: 5 RPM -> 12s delay (Safe: 15s)
DELAY_DUKE = 4
DELAY_GEMINI = 15 

# --- HELPER FUNCTIONS ---

def get_duke_response(question, model_name):
    """Hits Duke's LiteLLM Gateway."""
    client = OpenAI(api_key=DUKE_API_KEY, base_url=DUKE_BASE_URL)
    
    for attempt in range(3):
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
                print(f"\n⚠️ Duke Rate Limit ({model_name}). Sleeping 60s...")
                time.sleep(60)
            else:
                return f"[ERROR] Duke Failed: {e}"
    return "[ERROR] Failed after 3 retries"

def get_gemini_response(question, model_name):
    """Hits Google's Generative AI API (Native SDK)."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)
        
        # Safety settings to prevent 'None' responses on legal topics
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]

        response = model.generate_content(
            f"You are a housing law assistant. Answer accurately based on NYC law: {question}",
            generation_config=genai.types.GenerationConfig(temperature=0.0),
            safety_settings=safety_settings
        )
        
        if response.text:
            return response.text
        else:
            return "[ERROR] Gemini Safety Filter Triggered"
            
    except Exception as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            print(f"\n⚠️ Gemini 5 RPM Limit Hit! Sleeping 30s...")
            time.sleep(30)
            return get_gemini_response(question, model_name) # One retry
        return f"[ERROR] Gemini Failed: {e}"

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
        return f"[ERROR] Ollama Connect Failed: {e}"

# --- MAIN ENGINE ---

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"CRITICAL ERROR: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r") as f:
        questions = json.load(f)

    results = []
    
    # Time Estimation
    q_count = len(questions)
    d_time = q_count * len(DUKE_MODELS) * DELAY_DUKE
    g_time = q_count * len(GEMINI_MODELS) * DELAY_GEMINI
    total_time_min = (d_time + g_time) / 60
    
    print(f"🚀 Starting Mega-Benchmark")
    print(f"📊 Total Questions: {q_count}")
    print(f"🤖 Models: {len(DUKE_MODELS)} Duke + {len(GEMINI_MODELS)} Gemini + {len(LOCAL_MODELS)} Local")
    print(f"⏳ Est. Runtime: ~{total_time_min:.1f} minutes (due to strict rate limits)\n")
    
    for i, item in tqdm(enumerate(questions), total=len(questions)):
        q_text = item["question"]
        current_responses = {}

        # 1. DUKE LOOP
        for model in DUKE_MODELS:
            ans = get_duke_response(q_text, model)
            current_responses[model] = ans
            time.sleep(DELAY_DUKE) 

        # 2. GEMINI LOOP (Strict 15s delay)
        for model in GEMINI_MODELS:
            ans = get_gemini_response(q_text, model)
            current_responses[model] = ans
            time.sleep(DELAY_GEMINI) # 15s sleep > 12s required by 5 RPM

        # 3. OLLAMA LOOP
        for model in LOCAL_MODELS:
            ans = get_ollama_response(q_text, model)
            current_responses[model] = ans

        # Save Row
        results.append({
            "question_id": i,
            "category": item.get("category", "General"),
            "question": q_text,
            "ground_truth": item["ground_truth_answer"],
            "citation": item.get("citation", "N/A"),
            "responses": current_responses
        })

        # Periodic Save
        if i % 2 == 0: # Save often because this script is slow
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=4)

    # Final Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Benchmark Complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()