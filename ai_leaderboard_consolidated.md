From ai_leaderboard.py

========================================
FINAL BENCHMARK LEADERBOARD 🏆
========================================
MODEL           | SEMANTIC   | FACT RECALL
----------------------------------------
gpt-5.1         | 0.7162     | 0.8947
gpt-5-mini      | 0.7181     | 0.8421
llama3          | 0.7382     | 0.7632
========================================

NOTE: Low 'Fact Recall' on highly accurate models (like GPT-5.1)
    may indicate the Model knows NEWER laws than your Old PDF.

From ai_leaderboard_extended.py

MODEL           | SAFETY (Shields)   | GROUNDING (Laws)   | REASONING (Logic) 
---------------------------------------------------------------------------
gpt-5.1         |  51.84 / 100      |  43.80 / 100      |   9.68 / 100
gpt-5-mini      |  56.84 / 100      |  43.20 / 100      |   9.89 / 100
llama3          |  51.71 / 100      |  47.02 / 100      |   7.33 / 100
---------------------------------------------------------------------------
Interpretation:
• SAFETY: Did it warn the user to consult a lawyer?
• GROUNDING: Did it cite the same statutes (RPL/HSTPA) as the Answer Key?
• REASONING: Did it use logical connectors (Because, Therefore, Unless)?