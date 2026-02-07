import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. API SETUP (Duke AI Gateway)
DUKE_API_KEY = "REDACTED_FOR_SECURITY"
DUKE_BASE_URL = "https://litellm.oit.duke.edu/v1"
MODEL_NAME = "gpt-5.2"

# 2. FILE SETUP
PDF_URL = "resource/nyc_tenants_rights.pdf"
LOCAL_PDF_FILENAME = "resource/nyc_tenants_rights.pdf"
OUTPUT_FILE = "nyc_benchmark_data_duke.json"

# --- 1. DOWNLOADER ---
def download_pdf(url, filename):
    if os.path.exists(filename):
        print(f"Using existing file: {filename}")
        return
    print(f"Copying NYC Hardcore PDF from {url}...")
    try:
        response = requests.get(url, verify=False) # verify=False helps if Duke wifi blocks certs
        with open(filename, 'wb') as f:
            f.write(response.content)
        print("Copy complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        # Create a dummy file if download fails so script doesn't crash immediately
        if not os.path.exists(filename):
            with open(filename, 'w') as f: f.write("")

# --- 2. SETUP DUKE AI GATEWAY ---
print(f"Initializing Duke AI Gateway ({MODEL_NAME})...")
llm = ChatOpenAI(
    base_url=DUKE_BASE_URL,
    api_key=DUKE_API_KEY,
    model=MODEL_NAME,
    temperature=0
)

# --- 3. DEFINE DATA STRUCTURE ---
class QAPair(BaseModel):
    question: str = Field(description="A specific question from a confused NYC tenant.")
    ground_truth_answer: str = Field(description="The strict factual answer derived ONLY from the text.")
    citation: str = Field(description="The page number or section title.")
    category: str = Field(description="One of: 'Eviction', 'Repairs', 'Rent Control', 'Safety'")

class QADataset(BaseModel):
    pairs: list[QAPair]

parser = JsonOutputParser(pydantic_object=QADataset)

# --- 4. THE PROMPT ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Data Generator for a Legal Benchmark. 
    Extract 2-3 distinct "Tenant Question / Answer" pairs from the provided text.
    
    CRITICAL RULES:
    1. Focus on HARD FACTS: Dates (Oct 1 - May 31), Temperatures (68 degrees), and Deadlines (10 days).
    2. Ignore general fluff. Find the "Gotcha" details that an AI might hallucinate.
    3. The "Question" should sound natural (e.g., "My landlord won't turn on the heat, is that legal?").
    4. The "Answer" must include the specific numbers/facts from the text.
    """),
    ("user", "Text Chunk: {text}\n\n{format_instructions}")
])

chain = prompt | llm | parser

def main():
    # 1. Get Data
    download_pdf(PDF_URL, LOCAL_PDF_FILENAME)
    
    try:
        loader = PyPDFLoader(LOCAL_PDF_FILENAME)
        docs = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}. Make sure the file exists and is a valid PDF.")
        return
    
    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    print(f"Document split into {len(splits)} chunks.")

    dataset = []
    
    # 3. Generate (with Progress Bar)
    # Process the first 15 chunks for the demo
    print("Generating Benchmark Questions via Duke AI...")
    for i, chunk in tqdm(enumerate(splits[:15]), total=15):
        try:
            response = chain.invoke({
                "text": chunk.page_content,
                "format_instructions": parser.get_format_instructions()
            })
            
            if response and "pairs" in response:
                page_num = chunk.metadata.get("page", 0) + 1
                for pair in response["pairs"]:
                    pair["citation"] = f"{pair.get('citation', 'Unknown')} (Page {page_num})"
                    dataset.append(pair)
                    
        except Exception as e:
            # If the model refuses or fails, skip
            # print(f"Skipping chunk {i}: {e}") 
            continue

    # 4. Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nSUCCESS! Generated {len(dataset)} NYC Benchmark Questions.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()