# Benchmarking Reliability in AI-generated Legal Advice
By Team BenchLAWk: \
Wenjie Gong, Simeng Wu, Carol Zhou, Franklin Zhou

## Step 1:
Build the automated pipeline to generate the test dataset. \
*Approach*: Scrape a few official "Tenant Rights" PDFs (e.g., New York City, California, etc - in `resource` folder). Generate synthetic user questions with "Grouth Truth" answers based only on the text. Output into a JSON file.

## Step 2:
Run the models.\
*Approach*: Through an automated pipeline, loop through the test set and send the questions to different models. Return the answer, latency, and refusal rate (e.g., `I am not a lawyer` argument).

## Step 3:
Evaluate the model response. \
*Approach*: Implement some metrics evaluating the AI response.

## Step 4:
Visualize the results at front-end.\
*Approach*: Use `Streamlit` to build a dashboard where you can paste a legal question, select a model, and see the real-time evaluation results.

## Further questions:
This is benchmarking AI for tenant-landlord law. How about AI for traffic law? AI for family law? AI for immigration law? This project is all about scalability - today is tenants' right, tomorrow is immigration law!

## Link to our slides: 
https://docs.google.com/presentation/d/1nJkU3DylH-KSh4aq5t62-vs3PI1ns4BNiFfNyZXu5Pg/edit?usp=sharing
