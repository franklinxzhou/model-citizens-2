import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_FILE = "results/gpt-5.1_gpt-5-mini_llama3.json"
RESULTS_FILE = "results/final_scorecard.json"
# Load a small, fast, local embedding model (runs on CPU)
print("Loading Embedding Model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_key_entities(text):
    """
    Extracts dates, dollar amounts, and specific numbers for rigorous matching.
    """
    if not text: return set()
    # Regex for dates (Oct 1, October 1st), Money ($2,500), Numbers (1/60th, 30 days)
    patterns = [
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s\d{1,2}(st|nd|rd|th)?\b', # Dates
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?', # Money
        r'\b\d+[\/-]\d+(?:th)?\b', # Fractions like 1/60th
        r'\b\d+\s(days|years|months)\b' # Durations
    ]
    entities = set()
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        # Flatten and clean matches
        for m in matches:
            if isinstance(m, tuple): m = " ".join(m)
            entities.add(m.lower().strip())
    return entities

def calculate_scores(ground_truth, model_response):
    scores = {}
    
    # 1. SEMANTIC SIMILARITY (0.0 - 1.0)
    # Encodes both texts into vectors and measures cosine distance
    embeddings = embedder.encode([ground_truth, model_response])
    scores['semantic_score'] = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    
    # 2. KEY ENTITY RECALL (0.0 - 1.0)
    # Did the model include the specific numbers/dates from Ground Truth?
    gt_entities = extract_key_entities(ground_truth)
    if not gt_entities:
        scores['entity_recall'] = 1.0 # No entities to miss
    else:
        model_text_lower = model_response.lower()
        # Check how many GT entities appear in Model Text
        hits = sum(1 for e in gt_entities if e in model_text_lower) # Simple string match for speed
        scores['entity_recall'] = hits / len(gt_entities)
        
    return scores

def main():
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        
    final_report = defaultdict(lambda: {'semantic': [], 'recall': []})
    
    print(f"Grading {len(data)} questions across models...")
    
    # Iterate through questions
    for item in data:
        ground_truth = item['ground_truth']
        
        # Iterate through models (gpt-5.1, llama3, etc.)
        for model_name, response_text in item['responses'].items():
            metrics = calculate_scores(ground_truth, response_text)
            
            # Aggregate scores
            final_report[model_name]['semantic'].append(metrics['semantic_score'])
            final_report[model_name]['recall'].append(metrics['entity_recall'])

    # PRINT LEADERBOARD
    print("\n" + "="*40)
    print("FINAL BENCHMARK LEADERBOARD 🏆")
    print("="*40)
    print(f"{'MODEL':<15} | {'SEMANTIC':<10} | {'FACT RECALL':<10}")
    print("-" * 40)
    
    for model, scores in final_report.items():
        avg_semantic = np.mean(scores['semantic'])
        avg_recall = np.mean(scores['recall'])
        print(f"{model:<15} | {avg_semantic:.4f}     | {avg_recall:.4f}")
        
    print("="*40)
    print("\nNOTE: Low 'Fact Recall' on highly accurate models (like GPT-5.1)")
    print("    may indicate the Model knows NEWER laws than your Old PDF.")

if __name__ == "__main__":
    main()