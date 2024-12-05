import os
from dotenv import load_dotenv
from pathlib import Path
from agents.rag_agent import RAGAgent

def load_environment():
    current_dir = Path(__file__).parent
    env_path = current_dir / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
    else:
        raise FileNotFoundError("No .env file found!")
    
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not found in environment variables!")

def get_pdf_files():
    pdf_dir = Path("data/pdf_docs")
    if not pdf_dir.exists():
        raise FileNotFoundError("PDF documents directory not found!")
        
    return [str(pdf) for pdf in pdf_dir.glob("*.pdf")]

def main():
    try:
        print("Loading environment variables...")
        load_environment()
        
        print("Initializing RAG agent...")
        rag_agent = RAGAgent()
        
        print("Loading PDF documents...")
        pdf_files = get_pdf_files()
        print(f"Found {len(pdf_files)} PDF documents")
        
        print("Initializing knowledge base...")
        rag_agent.initialize_vector_store(pdf_files)
        
        print("✅ Knowledge base initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()