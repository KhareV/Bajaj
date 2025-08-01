"""
FINAL Pydantic schemas - Fixed for Azure Blob URLs with query parameters
Date: 2025-08-01 | User: vkhare2909
"""

from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Optional, Dict, Any, Union
import re
from urllib.parse import urlparse, parse_qs

class HackRxRequest(BaseModel):
    """Enhanced request schema with Azure blob URL support"""
    documents: HttpUrl = Field(
        ..., 
        description="URL to the document (PDF/DOCX) - supports Azure blob URLs",
        example="https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03..."
    )
    questions: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=50,  # Increased limit to handle up to 50 questions
        description="List of questions to answer",
        example=[
            "What is the grace period for premium payment?", 
            "What are the waiting periods for pre-existing conditions?"
        ]
    )
    
    @validator('questions')
    def validate_questions(cls, v):
        """Enhanced question validation"""
        if not v:
            raise ValueError("At least one question is required")
        
        for i, question in enumerate(v):
            question = question.strip()
            if not question:
                raise ValueError(f"Question {i+1} cannot be empty")
            if len(question) < 5:
                raise ValueError(f"Question {i+1} is too short (minimum 5 characters)")
            if len(question) > 500:
                raise ValueError(f"Question {i+1} is too long (maximum 500 characters)")
            
            # Check for potentially harmful content
            if any(word in question.lower() for word in ['<script', 'javascript:', 'eval(', 'exec(']):
                raise ValueError(f"Question {i+1} contains potentially harmful content")
        
        return [q.strip() for q in v]
    
    @validator('documents')
    def validate_document_url(cls, v):
        """Enhanced document URL validation with Azure blob support"""
        url_str = str(v).lower()
        
        # Parse URL to extract path without query parameters
        parsed_url = urlparse(str(v))
        url_path = parsed_url.path.lower()
        
        # Check file extension from path (ignore query parameters)
        valid_extensions = ['.pdf', '.docx']
        if not any(url_path.endswith(ext) for ext in valid_extensions):
            raise ValueError("Document URL must point to a PDF or DOCX file")
        
        # Check URL scheme
        if not url_str.startswith(('http://', 'https://')):
            raise ValueError("Document URL must use HTTP or HTTPS protocol")
        
        # Basic URL format validation
        if len(str(v)) > 3000:  # Increased limit for Azure blob URLs with SAS tokens
            raise ValueError("Document URL is too long")
        
        # Additional validation for Azure blob URLs
        if 'blob.core.windows.net' in url_str:
            # Azure blob URLs are valid
            return v
        elif any(domain in url_str for domain in ['hackrx.in', 'hackrx.blob.core.windows.net']):
            # Known hackathon domains are valid
            return v
        else:
            # For other domains, do basic validation
            if len(parsed_url.netloc) < 3:
                raise ValueError("Invalid domain name in URL")
        
        return v

class HackRxResponse(BaseModel):
    """Enhanced response schema with metadata"""
    answers: List[str] = Field(
        ..., 
        description="Answers corresponding to input questions",
        example=[
            "The grace period for premium payment is 30 days from the due date.",
            "Pre-existing diseases have a waiting period of 36 months from policy inception."
        ]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response metadata for analytics and debugging"
    )
    
    @validator('answers')
    def validate_answers(cls, v):
        """Enhanced answer validation"""
        if not v:
            raise ValueError("At least one answer is required")
        
        for i, answer in enumerate(v):
            if not isinstance(answer, str):
                raise ValueError(f"Answer {i+1} must be a string")
            if len(answer.strip()) == 0:
                raise ValueError(f"Answer {i+1} cannot be empty")
        
        return v

class HealthResponse(BaseModel):
    """Enhanced health check response"""
    status: str = Field(description="Health status: healthy, degraded, unhealthy")
    timestamp: float = Field(description="Response timestamp")
    version: str = Field(default="3.0.0-FINAL", description="Application version")
    environment: str = Field(description="Environment (development/production)")
    ai_models_status: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="AI models availability status"
    )
    cache_status: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cache system status"
    )
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")

class MetricsResponse(BaseModel):
    """Enhanced metrics response"""
    total_requests: int = Field(description="Total requests processed")
    cache_hits: int = Field(description="Number of cache hits")
    cache_misses: int = Field(description="Number of cache misses")
    cache_hit_rate: float = Field(description="Cache hit rate (0.0 to 1.0)")
    average_response_time: float = Field(description="Average response time in seconds")
    uptime_seconds: float = Field(description="Application uptime in seconds")
    requests_by_type: Dict[str, int] = Field(description="Breakdown of request types")
    performance_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed performance metrics"
    )
    ai_status: Optional[Dict[str, Any]] = Field(
        default=None,
        description="AI processing status"
    )
    cache_status: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cache system status"
    )
    error: Optional[str] = Field(default=None, description="Error message if metrics failed")

class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str = Field(description="Error type")
    detail: str = Field(description="Detailed error message")
    timestamp: float = Field(description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggestions for resolving the error"
    )

class AccuracyTestRequest(BaseModel):
    """Request schema for accuracy testing"""
    document_url: HttpUrl = Field(description="Document URL for testing")
    test_cases: List[Dict[str, str]] = Field(
        description="Test cases with questions and expected answers",
        example=[
            {
                "question": "What is the grace period?",
                "expected_answer": "30 days",
                "category": "grace_period"
            }
        ]
    )

class AccuracyTestResponse(BaseModel):
    """Response schema for accuracy testing"""
    overall_accuracy: float = Field(description="Overall accuracy score (0.0 to 1.0)")
    test_results: List[Dict[str, Any]] = Field(description="Individual test results")
    category_accuracy: Dict[str, float] = Field(description="Accuracy by category")
    performance_metrics: Dict[str, Any] = Field(description="Performance statistics")

# Legacy compatibility models
class ProcessingResult(BaseModel):
    """Legacy processing result model"""
    answer: str
    confidence: float
    source_chunks: List[str]
    reasoning: str

class APIResponse(BaseModel):
    """Legacy API response model"""
    answers: List[str]
    metadata: Optional[Dict[str, Any]] = None