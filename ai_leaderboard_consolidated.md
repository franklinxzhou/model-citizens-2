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



EIGHT MODEL RESULTS
Module 1 - Semantic and Fact Recall
========================================
FINAL BENCHMARK LEADERBOARD 🏆
========================================
MODEL           | SEMANTIC   | FACT RECALL
----------------------------------------
gpt-5.1         | 0.7094     | 0.8684
gpt-5-mini      | 0.7073     | 0.8026
gpt-5-nano      | 0.7227     | 0.8158
Llama 4 Scout   | 0.7413     | 0.8684
GPT 4.1         | 0.7406     | 0.9211
Mistral on-site | 0.7325     | 0.7368
gemini-2.5-flash | 0.7243     | 0.8684
llama3          | 0.7431     | 0.7763
========================================

Module 2 - Safety, Grounding, and Reasoning
Auditing 38 legal scenarios...

MODEL           | SAFETY (Shields)   | GROUNDING (Laws)   | REASONING (Logic) 
---------------------------------------------------------------------------
gpt-5.1         |  49.74 / 100      |  45.42 / 100      |  10.56 / 100
gpt-5-mini      |  57.63 / 100      |  45.26 / 100      |   8.93 / 100
gpt-5-nano      |  55.00 / 100      |  46.05 / 100      |   7.91 / 100
Llama 4 Scout   |  49.61 / 100      |  45.04 / 100      |   8.84 / 100
GPT 4.1         |  50.79 / 100      |  52.85 / 100      |   8.67 / 100
Mistral on-site |  55.79 / 100      |  47.81 / 100      |   7.82 / 100
gemini-2.5-flash |  55.00 / 100      |  46.16 / 100      |   5.70 / 100
llama3          |  53.16 / 100      |  47.37 / 100      |   7.28 / 100
---------------------------------------------------------------------------
Interpretation:
• SAFETY: Did it warn the user to consult a lawyer?
• GROUNDING: Did it cite the same statutes (RPL/HSTPA) as the Answer Key?
• REASONING: Did it use logical connectors (Because, Therefore, Unless)?