import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from tqdm import tqdm

# --- CONFIGURATION ---
PDF_URL = "resource/tenants_rights.pdf"
LOCAL_PDF_FILENAME = "resource/nyc_tenants_rights.pdf"
OUTPUT_FILE = "data/nyc_benchmark_data.json"

# --- 1. DOWNLOADER ---
def download_pdf(url, filename):
    if os.path.exists(filename):
        print(f"Using existing file: {filename}")
        return
    print(f"Copying NYC Hardcore PDF from {url}...")
    with open(url, 'rb') as src, open(filename, 'wb') as dst:
        dst.write(src.read())
    print("Copy complete.")

# --- 2. SETUP LOCAL AI (Ollama) ---
# Ensure you have run `ollama run llama3` in your terminal!
print("Initializing Local Llama 3...")
local_llm = ChatOllama(model="llama3", format="json", temperature=0)

# --- 3. DEFINE DATA STRUCTURE ---
class QAPair(BaseModel):
    question: str = Field(description="A specific question from a confused NYC tenant (e.g., about heat, rats, or rent increases).")
    ground_truth_answer: str = Field(description="The strict factual answer derived ONLY from the text.")
    citation: str = Field(description="The page number or section title (e.g., 'Heat and Hot Water').")
    category: str = Field(description="One of: 'Eviction', 'Repairs', 'Rent Control', 'Safety'")

class QADataset(BaseModel):
    pairs: list[QAPair]

parser = JsonOutputParser(pydantic_object=QADataset)

# --- 4. THE PROMPT ---
# This is tuned for NYC Law's specific complexity
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

chain = prompt | local_llm | parser

def main():
    # 1. Get Data
    download_pdf(PDF_URL, LOCAL_PDF_FILENAME)
    
    loader = PyPDFLoader(LOCAL_PDF_FILENAME)
    docs = loader.load()
    
    # 2. Chunking
    # NYC docs are dense. We use smaller overlap to keep context clear.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    print(f"Document split into {len(splits)} chunks.")

    dataset = []
    
    # 3. Generate (with Progress Bar)
    # We process the first 15 chunks for the Hackathon Demo (to save time).
    # Remove [:15] to process the whole book (might take ~20 mins on a laptop).
    print("Generating Benchmark Questions...")
    for i, chunk in tqdm(enumerate(splits[:15]), total=15):
        try:
            response = chain.invoke({
                "text": chunk.page_content,
                "format_instructions": parser.get_format_instructions()
            })
            
            if response and "pairs" in response:
                # Add page metadata to the citation if missing
                page_num = chunk.metadata.get("page", 0) + 1
                for pair in response["pairs"]:
                    pair["citation"] = f"{pair.get('citation', 'Unknown')} (Page {page_num})"
                    dataset.append(pair)
                    
        except Exception as e:
            # Llama sometimes fails on empty chunks, just skip
            continue

    # 4. Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nSUCCESS! Generated {len(dataset)} NYC Benchmark Questions.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Sample Question:", dataset[0]['question'] if dataset else "None")

if __name__ == "__main__":
    main()