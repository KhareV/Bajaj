"""
Basic tests for the main FastAPI application
These will be expanded by Role 5 (Testing Developer)
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

class TestBasicEndpoints:
    """Test basic application endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns basic info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "uptime_seconds" in data

class TestAuthentication:
    """Test authentication functionality"""
    
    def test_hackrx_run_no_auth(self):
        """Test /hackrx/run without authentication"""
        response = client.post(
            "/hackrx/run",
            json={
                "documents": "https://www.hackrx.in/policies/BAJHLIP23020V012223.pdf",
                "questions": ["What is this document about?"]
            }
        )
        assert response.status_code == 403  # No authorization header
    
    def test_hackrx_run_invalid_token(self):
        """Test /hackrx/run with invalid token"""
        response = client.post(
            "/hackrx/run",
            headers={"Authorization": "Bearer invalid_token"},
            json={
                "documents": "https://www.hackrx.in/policies/BAJHLIP23020V012223.pdf", 
                "questions": ["What is this document about?"]
            }
        )
        assert response.status_code == 401
    
    def test_hackrx_run_valid_token(self):
        """Test /hackrx/run with valid token (mock response)"""
        response = client.post(
            "/hackrx/run",
            headers={"Authorization": f"Bearer {settings.BEARER_TOKEN}"},
            json={
                "documents": "https://www.hackrx.in/policies/BAJHLIP23020V012223.pdf",
                "questions": ["What is this document about?"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "answers" in data
        assert len(data["answers"]) == 1

class TestRequestValidation:
    """Test request validation"""
    
    def test_empty_questions(self):
        """Test request with empty questions list"""
        response = client.post(
            "/hackrx/run",
            headers={"Authorization": f"Bearer {settings.BEARER_TOKEN}"},
            json={
                "documents": "https://www.hackrx.in/policies/BAJHLIP23020V012223.pdf",
                "questions": []
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_invalid_document_url(self):
        """Test request with invalid document URL"""
        response = client.post(
            "/hackrx/run",
            headers={"Authorization": f"Bearer {settings.BEARER_TOKEN}"},
            json={
                "documents": "not_a_valid_url",
                "questions": ["What is this document about?"]
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_too_many_questions(self):
        """Test request with too many questions"""
        response = client.post(
            "/hackrx/run",
            headers={"Authorization": f"Bearer {settings.BEARER_TOKEN}"},
            json={
                "documents": "https://www.hackrx.in/policies/BAJHLIP23020V012223.pdf",
                "questions": [f"Question {i}" for i in range(25)]  # More than limit
            }
        )
        assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async functionality"""
    
    async def test_async_processing(self):
        """Test that async processing works correctly"""
        # This will be expanded when actual AI processing is implemented
        assert True  # Placeholder test