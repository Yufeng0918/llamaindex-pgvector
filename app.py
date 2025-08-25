import os
import logging
import uuid
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import MarkdownNodeParser # Import MarkdownNodeParser

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, DocumentChunk # Import our SQLAlchemy model

logging.basicConfig(level=logging.INFO)
load_dotenv() # Load environment variables from .env file

# Database connection details
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "mydb")
DB_USER = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypass")
DB_PORT = os.getenv("DB_PORT", "5432") # Default PostgreSQL port

# OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Construct the database URL
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Create a session class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def process_and_store_documents(docs_folder="docs"):
    """
    Reads markdown files from the 'docs' folder, splits them into chunks,
    generates embeddings, and stores them in the PostgreSQL database using SQLAlchemy.
    """
    if not os.path.exists(docs_folder):
        print(f"Error: The 'docs' folder does not exist at {docs_folder}")
        return

    # Configure Settings for embedding model and node parser
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = MarkdownNodeParser(chunk_size=1024, chunk_overlap=20)
    Settings.llm = None # No LLM needed for ingestion

    # Initialize components for ingestion
    reader = SimpleDirectoryReader(input_dir=docs_folder, recursive=True)
    documents = reader.load_data()

    # Parse documents into nodes using Settings.node_parser
    all_nodes = Settings.node_parser.get_nodes_from_documents(documents)

    # Create a new SQLAlchemy session
    db_session = SessionLocal()
    try:
        for doc in documents:
            doc_id = str(uuid.uuid4()) # Generate a unique ID for each document
            for i, node in enumerate(all_nodes):
                if node.metadata.get('file_path') == doc.metadata.get('file_path'): # Associate node with its original document
                    # Manually calculate embedding for each node
                    embedding = embed_model.get_text_embedding(node.text)

                    # Create a DocumentChunk instance
                    chunk = DocumentChunk(
                        document_id=doc_id,
                        chunk_sequence=i, # Use chunk_sequence
                        text=node.text,
                        embedding=embedding,
                        file_name=doc.metadata.get('file_name', 'unknown'),
                        metadata_=node.metadata, # Add metadata_
                        active_ind=True, # Default to true
                        # create_timestamp will be defaulted by the database
                    )
                    db_session.add(chunk)
        db_session.commit()
        print(f"Finished processing and storing chunks from {docs_folder} using SQLAlchemy.")
    except Exception as e:
        db_session.rollback()
        print(f"An error occurred: {e}")
    finally:
        db_session.close()



if __name__ == "__main__":
    print("Starting document processing...")
    process_and_store_documents()
    print("Document processing complete.")
