"""
CHAMPIONSHIP Configuration - Optimized for <15s Response & 90%+ Accuracy
Date: 2025-08-01 13:09:06 UTC | User: vkhare2909
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
from typing import List

load_dotenv()

class Settings(BaseSettings):
    """Championship configuration for guaranteed performance"""
    
    # AI Model Configuration
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API key")
    
    # Multiple fallback keys for championship reliability
    GEMINI_FALLBACK_KEYS: str = Field(
        default="",
        description="Comma-separated fallback Gemini API keys"
    )
    
    @property
    def gemini_fallback_keys_list(self) -> List[str]:
        """Convert comma-separated fallback keys to list"""
        if not self.GEMINI_FALLBACK_KEYS:
            return []
        return [key.strip() for key in self.GEMINI_FALLBACK_KEYS.split(",") if key.strip()]
    
    GROQ_API_KEY: str = Field(..., description="Groq API key")
    
    # Optional HuggingFace API token for additional AI capabilities
    HUGGINGFACE_API_TOKEN: str = Field(default="", description="HuggingFace API token (optional)")
    
    # CHAMPIONSHIP Performance Settings (Optimized for <15s)
    MAX_RESPONSE_TIME: int = Field(default=15, description="Max response time in seconds")
    MAX_DOCUMENT_SIZE: int = Field(default=10_000_000, description="Max document size (10MB)")
    DOWNLOAD_TIMEOUT: int = Field(default=8, description="Download timeout (8s)")
    
    # Authentication
    BEARER_TOKEN: str = Field(
        default="fff018ce90c02cb0554e8968e827c7696a7304b28ba190b9cace6236579f7258",
        description="Bearer token for authentication"
    )
    
    # Environment Settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    ENVIRONMENT: str = Field(default="production", description="Environment")
    
    # AI Model Settings (Championship-optimized)
    GEMINI_MODEL: str = Field(default="gemini-2.0-flash", description="Gemini model")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq model")
    
    # Vector Search Settings (Ultra-optimized for speed)
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    CHUNK_SIZE: int = Field(default=300, description="Chunk size (reduced for speed)")
    CHUNK_OVERLAP: int = Field(default=30, description="Chunk overlap (minimal for speed)")
    TOP_K_CHUNKS: int = Field(default=2, description="Top chunks to retrieve (reduced for speed)")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings
settings = Settings()