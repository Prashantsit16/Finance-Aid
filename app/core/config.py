from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Embedding model
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    # ChromaDB
    CHROMA_PERSIST_DIR: str = "data/processed/chroma_db"
    CHROMA_COLLECTION_NAME: str = "finance_aid"
    
    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    
    # Retrieval
    TOP_K_DENSE: int = 20      # candidates from dense search
    TOP_K_BM25: int = 20       # candidates from BM25
    TOP_K_RERANK: int = 5      # final chunks after reranking
    
    # LLM
    LLM_MODEL: str = "mistral"
    LLM_BASE_URL: str = "http://localhost:11434"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()
