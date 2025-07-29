"""
Configuration management using Pydantic Settings
Handles environment variables and validation
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv
import os
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application configuration with validation
    Uses environment variables or defaults
    """
    
    # AI Model Configuration
    GEMINI_API_KEY: str = Field(
        ..., 
        description="Google Gemini API key",
        min_length=20
    )
    GROQ_API_KEY: str = Field(
        ..., 
        description="Groq API key",
        min_length=20
    )
    
    # Redis Configuration
    REDIS_URL: str = Field(
        ..., 
        description="Redis connection URL"
    )
    REDIS_PASSWORD: str = Field(
        ..., 
        description="Redis password"
    )
    
    # Authentication
    BEARER_TOKEN: str = Field(
        default="fff018ce90c02cb0554e8968e827c7696a7304b28ba190b9cace6236579f7258",
        description="Expected Bearer token for authentication"
    )
    
    # Performance Settings
    MAX_RESPONSE_TIME: int = Field(
        default=15,
        description="Maximum response time in seconds",
        ge=5,
        le=30
    )
    CACHE_TTL: int = Field(
        default=3600,
        description="Cache time-to-live in seconds",
        ge=300,
        le=86400
    )
    MAX_DOCUMENT_SIZE: int = Field(
        default=10485760,  # 10MB
        description="Maximum document size in bytes",
        ge=1024,
        le=52428800  # 50MB
    )
    
    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    STRUCTURED_LOGGING: bool = Field(
        default=True,
        description="Enable structured logging"
    )
    
    # Environment Settings
    ENVIRONMENT: str = Field(
        default="development",
        description="Environment name"
    )
    
    # AI Model Settings
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model to use"
    )
    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model to use"
    )
    
    # Vector Search Settings
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    CHUNK_SIZE: int = Field(
        default=500,
        description="Document chunk size for processing"
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description="Overlap between document chunks"
    )
    TOP_K_CHUNKS: int = Field(
        default=3,
        description="Number of top chunks to retrieve"
    )
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        """Validate environment"""
        valid_envs = ['development', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        return v.lower()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"❌ Configuration Error: {e}")
    print("Please check your .env file and ensure all required variables are set")
    raise

# Configuration validation on startup
def validate_configuration():
    """Validate configuration on startup"""
    errors = []
    
    # Check API keys are not placeholder values
    if "your_" in settings.GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY appears to be a placeholder")
    
    if "your_" in settings.GROQ_API_KEY:
        errors.append("GROQ_API_KEY appears to be a placeholder")
    
    if "your_" in settings.REDIS_URL:
        errors.append("REDIS_URL appears to be a placeholder")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True

# Export settings for use in other modules
__all__ = ["settings", "validate_configuration"]