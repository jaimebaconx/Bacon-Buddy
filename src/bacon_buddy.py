import sys
sys.path.insert(0, r"E:\BaconBuddy\packages")
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ────────────────────────────────────────────────
#               EASY SETTINGS - CHANGE THESE IF NEEDED
# ────────────────────────────────────────────────
KNOWLEDGE_FOLDER = r"E:\BaconBuddy\knowledge\medical"
CHROMA_FOLDER    = r"E:\BaconBuddy\data\medical_chroma"
MODEL_NAME = "llama3.2:3b"
BATCH_SIZE       = 100

# ────────────────────────────────────────────────
#               DON'T CHANGE BELOW HERE
# ────────────────────────────────────────────────

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

print("┌───────────────────────────────────────────────┐")
print("│       Bacon-Buddy Medical - Offline AI        │")
print("└───────────────────────────────────────────────┘")
print(f"Model:            {MODEL_NAME}")
print(f"Knowledge folder: {KNOWLEDGE_FOLDER}")
print(f"Index folder:     {CHROMA_FOLDER}")
print()

# ── Load PDFs and TXTs
def load_all_documents(folder):
    docs = []
    folder_path = Path(folder)

    pdf_files = list(folder_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s)")
    for pdf in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf))
            docs.extend(loader.load())
            print(f"  ✓ {pdf.name}")
        except Exception as e:
            print(f"  ✗ {pdf.name}: {e}")

    txt_files = list(folder_path.glob("**/*.txt"))
    print(f"Found {len(txt_files)} TXT file(s)")
    for txt in txt_files:
        try:
            loader = TextLoader(str(txt), encoding="utf-8")
            docs.extend(loader.load())
            print(f"  ✓ {txt.name}")
        except Exception as e:
            print(f"  ✗ {txt.name}: {e}")

    return docs

# ── Medical-optimized chunking
def get_medical_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

# ── Embedding model - small, fast, CUDA accelerated
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ── Clear old index
if os.path.exists(CHROMA_FOLDER):
    shutil.rmtree(CHROMA_FOLDER)
    print("Cleared old index.\n")

# ── Load documents
print(f"Loading documents from: {KNOWLEDGE_FOLDER}")
docs = load_all_documents(KNOWLEDGE_FOLDER)

if not docs:
    print("ERROR: No documents found!")
    print(f"Check: {KNOWLEDGE_FOLDER}")
    exit(1)

print(f"\nLoaded {len(docs)} document section(s)")

# ── Chunk
splitter = get_medical_splitter()
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# ── Embed with progress bar
batches = [chunks[i:i+BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
print(f"\nEmbedding {len(chunks)} chunks in {len(batches)} batches...")

vectorstore = None
for batch in tqdm(batches, desc="Embedding", unit="batch"):
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            batch,
            embedding,
            persist_directory=CHROMA_FOLDER
        )
    else:
        vectorstore.add_documents(batch)

print(f"\nIndex saved. {vectorstore._collection.count()} chunks indexed.")

# ── Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ── LLM
llm = OllamaLLM(model=MODEL_NAME)

# ── Medical-specific prompt
prompt = ChatPromptTemplate.from_template("""
You are Bacon-Buddy, an offline medical reference assistant for use in remote and austere environments.
You help people with no access to professional medical care.

CRITICAL RULES:
- Answer ONLY from the context provided below
- If the context doesn't contain enough information, say: "I don't have enough information on that in my knowledge base."
- Always recommend professional medical care when available
- For dosages and procedures, quote the source material precisely
- Flag any situation that requires immediate evacuation or professional care
- Speak in plain language, not clinical jargon

Context from medical field guides:
{context}

Question: {question}

Answer:""")

chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

print("\n" + "═" * 50)
print("Bacon-Buddy Medical ready!")
print("Type 'quit' to exit")
print("═" * 50 + "\n")

while True:
    question = input("You: ").strip()

    if question.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    if not question:
        continue

    print("\nThinking...", end="", flush=True)
    try:
        answer = chain.invoke(question)
        print("\r" + " " * 20 + "\r", end="")
        print(f"Bacon-Buddy: {answer.strip()}\n")
    except Exception as e:
        print(f"\nError: {e}")
        print("→ Is Ollama running? Try: ollama serve")