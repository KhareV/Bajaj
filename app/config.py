"""
FINAL ENHANCED CHAMPIONSHIP Configuration - Dual Semaphore API Management
Date: 2025-08-05 18:    # Vector Search Settings - Enhanced for championship accuracy
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L12-v2", description="Embedding model")
    CHUNK_SIZE: int = Field(default=1200, description="Chunk size for document processing")  # Increased from 800
    CHUNK_OVERLAP: int = Field(default=100, description="Chunk overlap for better context")  # Reduced from 150
    TOP_K_CHUNKS: int = Field(default=5, description="Top chunks to retrieve from vector search")  # Reduced from 6
    VECTOR_SEARCH_SIMILARITY_THRESHOLD: float = Field(default=0.3, description="Minimum similarity threshold")
    MAX_CONTEXT_LENGTH: int = Field(default=15000, description="Maximum context length for AI processing")  # Reduced from 25000
    ENABLE_SEMANTIC_SEARCH: bool = Field(default=True, description="Enable semantic search enhancement")
    MAX_CHUNKS_FOR_EMBEDDING: int = Field(default=50, description="Maximum chunks to process for embeddings")C | User: vkhare2909
Complete configuration for Gemini & Groq semaphore-based concurrent API usage
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
from typing import List
import asyncio

load_dotenv()

class Settings(BaseSettings):
    """Final enhanced championship configuration with dual API key management"""
    
    # AI Model Configuration - Enhanced for dual semaphore management
    GEMINI_API_KEY: str = Field(..., description="Primary Google Gemini API key")
    GEMINI_FALLBACK_KEYS: str = Field(
        default="",
        description="Comma-separated additional Gemini API keys for concurrent use"
    )
    GROQ_API_KEY: str = Field(..., description="Primary Groq API key")
    GROQ_FALLBACK_KEYS: str = Field(
        default="",
        description="Comma-separated additional Groq API keys for concurrent use"
    )
    HUGGINGFACE_API_TOKEN: str = Field(default="", description="HuggingFace API token (optional)")
    
    # Enhanced API Key Management - CONSERVATIVE for Rate Limits
    MAX_CONCURRENT_REQUESTS: int = Field(default=1, description="Max concurrent API requests - very conservative for rate limits")
    API_RETRY_ATTEMPTS: int = Field(default=2, description="API retry attempts - reduced to avoid quota exhaustion")
    API_RETRY_DELAY: float = Field(default=5.0, description="Initial retry delay in seconds - increased for rate limits")
    SEMAPHORE_TIMEOUT: float = Field(default=120.0, description="Semaphore acquisition timeout - increased for rate limits")
    
    # Key Health Management - More conservative for rate limits
    KEY_HEALTH_CHECK_INTERVAL: int = Field(default=600, description="Key health check interval in seconds")
    KEY_ERROR_THRESHOLD: int = Field(default=2, description="Max errors before marking key unhealthy - reduced for rate limits")
    KEY_COOLDOWN_PERIOD: int = Field(default=1800, description="Cooldown period for unhealthy keys in seconds - increased for rate limits")
    
    @property
    def all_gemini_keys(self) -> List[str]:
        """Get all Gemini API keys including primary and fallbacks"""
        keys = [self.GEMINI_API_KEY]
        if self.GEMINI_FALLBACK_KEYS:
            fallback_keys = [key.strip() for key in self.GEMINI_FALLBACK_KEYS.split(",") if key.strip()]
            keys.extend(fallback_keys)
        return keys
    
    @property
    def all_groq_keys(self) -> List[str]:
        """Get all Groq API keys including primary and fallbacks"""
        keys = [self.GROQ_API_KEY]
        if self.GROQ_FALLBACK_KEYS:
            fallback_keys = [key.strip() for key in self.GROQ_FALLBACK_KEYS.split(",") if key.strip()]
            keys.extend(fallback_keys)
        return keys
    
    @property
    def total_api_keys(self) -> int:
        """Get total number of API keys available"""
        return len(self.all_gemini_keys) + len(self.all_groq_keys)
    
    @property
    def api_key_summary(self) -> dict:
        """Get summary of API key configuration"""
        return {
            "gemini_keys": len(self.all_gemini_keys),
            "groq_keys": len(self.all_groq_keys),
            "total_keys": self.total_api_keys,
            "max_concurrent": self.MAX_CONCURRENT_REQUESTS,
            "semaphore_enabled": True
        }
    
    # CHAMPIONSHIP Performance Settings (Optimized for <15s)
    MAX_RESPONSE_TIME: int = Field(default=12, description="Max response time in seconds")
    MAX_DOCUMENT_SIZE: int = Field(default=8_000_000, description="Max document size (8MB)")
    DOWNLOAD_TIMEOUT: int = Field(default=8, description="Download timeout (8s)")
    
    # Authentication
    BEARER_TOKEN: str = Field(
        default="fff018ce90c02cb0554e8968e827c7696a7304b28ba190b9cace6236579f7258",
        description="Bearer token for authentication"
    )
    
    # Environment Settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    ENVIRONMENT: str = Field(default="production", description="Environment")
    
    # AI Model Settings (Championship-optimized for stability)
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Gemini model")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq model")
    
    # Enhanced Model Configuration
    GEMINI_TEMPERATURE: float = Field(default=0.0, description="Gemini temperature for consistency")
    GEMINI_MAX_TOKENS: int = Field(default=800, description="Gemini max output tokens")
    GROQ_TEMPERATURE: float = Field(default=0.0, description="Groq temperature for consistency")
    GROQ_MAX_TOKENS: int = Field(default=800, description="Groq max output tokens")
    
    # Vector Search Settings - CHAMPIONSHIP OPTIMIZED for 90%+ accuracy
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-base-en-v1.5", description="Embedding model")
    CHUNK_SIZE: int = Field(default=1500, description="Chunk size for document processing")  # Increased for more context
    CHUNK_OVERLAP: int = Field(default=200, description="Chunk overlap for better context")  # Increased for better continuity
    TOP_K_CHUNKS: int = Field(default=15, description="Top chunks to retrieve from vector search")  # Significantly increased for maximum coverage
    VECTOR_SEARCH_SIMILARITY_THRESHOLD: float = Field(default=0.03, description="Minimum similarity threshold")  # Very low for maximum recall
    MAX_CONTEXT_LENGTH: int = Field(default=35000, description="Maximum context length for AI processing")  # Maximum context for championship accuracy
    ENABLE_SEMANTIC_SEARCH: bool = Field(default=True, description="Enable semantic search enhancement")
    MAX_CHUNKS_FOR_EMBEDDING: int = Field(default=200, description="Maximum chunks to process for embeddings")  # No limits for championship accuracy
    
    # Performance Optimization Settings
    ENABLE_CACHING: bool = Field(default=True, description="Enable document and vector caching")
    CACHE_TTL: int = Field(default=7200, description="Cache time-to-live in seconds")  # Increased from 3600
    MAX_CACHE_SIZE: int = Field(default=3, description="Maximum number of items in cache")
    
    # Sequential Processing Settings - Optimized for Rate Limits
    MAX_PARALLEL_QUESTIONS: int = Field(default=50, description="Maximum questions allowed per request - using sequential processing")
    QUESTION_PROCESSING_THRESHOLD: int = Field(default=8, description="Threshold for parallel vs sequential processing - increased to favor sequential")
    QUESTION_SEMAPHORE_LIMIT: int = Field(default=2, description="Semaphore limit for question processing - reduced for rate limits")
    
    # Monitoring and Logging
    ENABLE_DETAILED_LOGGING: bool = Field(default=True, description="Enable detailed request logging")
    LOG_API_KEY_STATS: bool = Field(default=True, description="Log API key usage statistics")
    PERFORMANCE_MONITORING: bool = Field(default=True, description="Enable performance monitoring")
    
    # Error Handling and Resilience
    ENABLE_CIRCUIT_BREAKER: bool = Field(default=True, description="Enable circuit breaker for API calls")
    CIRCUIT_BREAKER_THRESHOLD: int = Field(default=5, description="Circuit breaker failure threshold")
    CIRCUIT_BREAKER_TIMEOUT: int = Field(default=60, description="Circuit breaker timeout in seconds")
    
    # Championship-specific settings
    ACCURACY_THRESHOLD: float = Field(default=0.05, description="Minimum confidence threshold for answers")
    ENABLE_CONSENSUS_BOOSTING: bool = Field(default=True, description="Enable confidence boosting when models agree")
    CONSENSUS_BOOST_VALUE: float = Field(default=0.15, description="Confidence boost when models agree")
    
    def validate_configuration(self) -> dict:
        """Validate the current configuration and return status"""
        issues = []
        warnings = []
        
        # Check API keys
        if len(self.all_gemini_keys) == 0:
            issues.append("No Gemini API keys configured")
        elif len(self.all_gemini_keys) == 1:
            warnings.append("Only one Gemini API key configured - consider adding more for better resilience")
        
        if len(self.all_groq_keys) == 0:
            issues.append("No Groq API keys configured")
        elif len(self.all_groq_keys) == 1:
            warnings.append("Only one Groq API key configured - consider adding more for better resilience")
        
        # Check concurrent limits
        if self.MAX_CONCURRENT_REQUESTS > self.total_api_keys * 10:
            warnings.append(f"MAX_CONCURRENT_REQUESTS ({self.MAX_CONCURRENT_REQUESTS}) might be too high for {self.total_api_keys} total keys")
        
        # Check performance settings
        if self.MAX_RESPONSE_TIME > 15:
            warnings.append("MAX_RESPONSE_TIME exceeds championship target of 15 seconds")
        
        return {
            "status": "valid" if len(issues) == 0 else "invalid",
            "issues": issues,
            "warnings": warnings,
            "api_key_summary": self.api_key_summary,
            "configuration_score": max(0, 100 - len(issues) * 30 - len(warnings) * 10)
        }
    
    def get_runtime_info(self) -> dict:
        """Get runtime configuration information"""
        return {
            "timestamp": "2025-08-05 18:16:51 UTC",
            "user": "vkhare2909",
            "version": "FINAL_ENHANCED_CHAMPIONSHIP_v1.0",
            "configuration": {
                "total_api_keys": self.total_api_keys,
                "gemini_keys": len(self.all_gemini_keys),
                "groq_keys": len(self.all_groq_keys),
                "max_concurrent": self.MAX_CONCURRENT_REQUESTS,
                "semaphore_enabled": True,
                "dual_model_support": True,
                "caching_enabled": self.ENABLE_CACHING,
                "monitoring_enabled": self.PERFORMANCE_MONITORING
            },
            "optimization_features": [
                "Dual semaphore API key management",
                "Load balancing across all keys",
                "Automatic quota handling",
                "Health monitoring and recovery",
                "Consensus-based confidence boosting",
                "Enhanced error handling and retries",
                "Parallel question processing",
                "Advanced caching strategies",
                "Real-time performance monitoring"
            ]
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
        # Enhanced validation
        validate_assignment = True
        use_enum_values = True

# Create global settings instance
settings = Settings()

# Optional: Log configuration summary on import
import logging
logger = logging.getLogger(__name__)

def log_configuration_summary():
    """Log a summary of the current configuration"""
    try:
        validation_result = settings.validate_configuration()
        runtime_info = settings.get_runtime_info()
        
        logger.info("🏆 FINAL ENHANCED CHAMPIONSHIP Configuration Loaded")
        logger.info(f"📊 Configuration Score: {validation_result['configuration_score']}/100")
        logger.info(f"🔑 API Keys: {runtime_info['configuration']['gemini_keys']} Gemini + {runtime_info['configuration']['groq_keys']} Groq = {runtime_info['configuration']['total_api_keys']} total")
        logger.info(f"⚡ Max Concurrent: {runtime_info['configuration']['max_concurrent']} requests")
        logger.info(f"✅ Status: {validation_result['status'].upper()}")
        
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                logger.warning(f"⚠️ {warning}")
                
        if validation_result['issues']:
            for issue in validation_result['issues']:
                logger.error(f"❌ {issue}")
                
    except Exception as e:
        logger.error(f"❌ Failed to log configuration summary: {e}")

# Log configuration on import
log_configuration_summary()