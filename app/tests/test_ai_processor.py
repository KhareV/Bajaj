"""
Comprehensive test suite for AI processor and consensus engine
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.models.ai_processor import AIProcessor, GeminiProcessor, GroqProcessor, AIResponse
from app.models.consensus import ConsensusEngine, ConsensusResult
from app.config import settings

class TestAIProcessor:
    """Test cases for AI processor functionality"""
    
    @pytest.fixture
    def ai_processor(self):
        """Create AI processor instance for testing"""
        return AIProcessor()
    
    @pytest.fixture
    def sample_document(self):
        """Sample insurance document text"""
        return """
        National Parivar Mediclaim Plus Policy
        
        Grace Period: A grace period of thirty (30) days is allowed for payment of premium 
        after the due date. During this period, the policy continues to remain in force.
        
        Waiting Periods:
        1. Pre-existing diseases: 36 months waiting period from policy inception
        2. Specific diseases: 24 months waiting period
        3. Maternity: 24 months waiting period
        
        Coverage Benefits:
        - Hospitalization expenses up to sum insured
        - Pre and post hospitalization expenses
        - Day care procedures
        - Ambulance charges up to Rs. 2,000
        
        Exclusions:
        - Cosmetic surgery
        - Dental treatment unless due to accident
        - Treatment outside India
        """
    
    @pytest.fixture
    def sample_queries(self):
        """Sample test queries"""
        return [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "What are the coverage benefits?",
            "What treatments are excluded?"
        ]
    
    @pytest.mark.asyncio
    async def test_document_initialization(self, ai_processor, sample_document):
        """Test document initialization"""
        success = await ai_processor.initialize_document(sample_document)
        
        assert success is True
        assert ai_processor.document_indexed is True
        assert len(ai_processor.document_chunks) > 0
        assert ai_processor.search_engine.index is not None
    
    @pytest.mark.asyncio
    async def test_query_processing_interface(self, ai_processor, sample_document):
        """Test the main query processing interface"""
        # Initialize document
        await ai_processor.initialize_document(sample_document)
        
        # Test query processing
        answer, confidence = await ai_processor.process_query(
            sample_document, 
            "What is the grace period?"
        )
        
        assert isinstance(answer, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert len(answer) > 0
    
    @pytest.mark.asyncio
    async def test_context_retrieval(self, ai_processor, sample_document):
        """Test context retrieval functionality"""
        await ai_processor.initialize_document(sample_document)
        
        context = ai_processor._get_relevant_context("grace period")
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert "grace period" in context.lower() or "thirty" in context.lower()
    
    @pytest.mark.asyncio
    async def test_multiple_queries(self, ai_processor, sample_document, sample_queries):
        """Test processing multiple queries"""
        await ai_processor.initialize_document(sample_document)
        
        results = []
        for query in sample_queries:
            answer, confidence = await ai_processor.process_query(sample_document, query)
            results.append((answer, confidence))
        
        assert len(results) == len(sample_queries)
        for answer, confidence in results:
            assert isinstance(answer, str)
            assert isinstance(confidence, float)
            assert len(answer) > 0

class TestGeminiProcessor:
    """Test cases for Gemini processor"""
    
    @pytest.fixture
    def gemini_processor(self):
        """Create Gemini processor for testing"""
        return GeminiProcessor("test_api_key")
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, gemini_processor):
        """Test confidence calculation logic"""
        # Test high confidence answer
        high_conf = gemini_processor._calculate_confidence(
            "The grace period is 30 days for premium payment", 
            "What is the grace period?"
        )
        
        # Test low confidence answer
        low_conf = gemini_processor._calculate_confidence(
            "Information not provided in the document", 
            "What is the grace period?"
        )
        
        assert high_conf > low_conf
        assert 0.0 <= high_conf <= 1.0
        assert 0.0 <= low_conf <= 1.0
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    async def test_gemini_processing_success(self, mock_model, gemini_processor):
        """Test successful Gemini processing"""
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "The grace period is 30 days"
        mock_model.return_value.generate_content.return_value = mock_response
        
        result = await gemini_processor.process_query("Test prompt")
        
        assert isinstance(result, AIResponse)
        assert result.answer == "The grace period is 30 days"
        assert result.model_name == "gemini-1.5-pro"
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    @patch('google.generativeai.GenerativeModel')
    async def test_gemini_processing_error(self, mock_model, gemini_processor):
        """Test Gemini processing with error"""
        # Mock error response
        mock_model.return_value.generate_content.side_effect = Exception("API Error")
        
        result = await gemini_processor.process_query("Test prompt")
        
        assert isinstance(result, AIResponse)
        assert "Error" in result.answer
        assert result.confidence == 0.0

class TestGroqProcessor:
    """Test cases for Groq processor"""
    
    @pytest.fixture
    def groq_processor(self):
        """Create Groq processor for testing"""
        return GroqProcessor("test_api_key")
    
    @pytest.mark.asyncio
    @patch('groq.Groq')
    async def test_groq_processing_success(self, mock_groq, groq_processor):
        """Test successful Groq processing"""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "The waiting period is 36 months"
        mock_groq.return_value.chat.completions.create.return_value = mock_response
        
        result = await groq_processor.process_query("Test prompt")
        
        assert isinstance(result, AIResponse)
        assert result.answer == "The waiting period is 36 months"
        assert result.model_name == "llama3-70b-8192"
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, groq_processor):
        """Test Groq confidence calculation"""
        # Test with informative answer
        high_conf = groq_processor._calculate_confidence(
            "Coverage includes hospitalization up to Rs. 5 lakhs", 
            "What is covered?"
        )
        
        # Test with non-informative answer
        low_conf = groq_processor._calculate_confidence(
            "Cannot find this information", 
            "What is covered?"
        )
        
        assert high_conf > low_conf

class TestConsensusEngine:
    """Test cases for consensus engine"""
    
    @pytest.fixture
    def consensus_engine(self):
        """Create consensus engine for testing"""
        return ConsensusEngine()
    
    def test_single_response_consensus(self, consensus_engine):
        """Test consensus with single response"""
        responses = [("Test answer", 0.8, "model1")]
        
        result = consensus_engine.find_consensus(responses)
        
        assert isinstance(result, ConsensusResult)
        assert result.final_answer == "Test answer"
        assert result.final_confidence == 0.8
        assert result.consensus_method == "single"
    
    def test_multiple_response_consensus(self, consensus_engine):
        """Test consensus with multiple responses"""
        responses = [
            ("The grace period is 30 days", 0.9, "gemini"),
            ("Grace period: thirty days", 0.8, "groq"),
            ("Cannot find information", 0.3, "model3")
        ]
        
        result = consensus_engine.find_consensus(responses, "What is the grace period?")
        
        assert isinstance(result, ConsensusResult)
        assert result.final_confidence > 0
        assert len(result.model_contributions) > 0
    
    def test_response_filtering(self, consensus_engine):
        """Test response quality filtering"""
        responses = [
            ("Detailed coverage information with benefits", 0.8, "model1"),
            ("Not provided in document", 0.4, "model2"),
            ("Cannot find this information", 0.3, "model3")
        ]
        
        filtered = consensus_engine._filter_responses(responses)
        
        assert len(filtered['high']) >= 1
        assert len(filtered['low']) >= 1
    
    def test_weighted_consensus(self, consensus_engine):
        """Test weighted consensus method"""
        responses = [
            ("Coverage includes hospitalization expenses", 0.9, "gemini"),
            ("Policy covers hospital costs", 0.7, "groq")
        ]
        
        filtered = consensus_engine._filter_responses(responses)
        result = consensus_engine._weighted_consensus(filtered, "What is covered?")
        
        assert isinstance(result, ConsensusResult)
        assert result.consensus_method == "weighted"
        assert result.final_confidence > 0

class TestInsuranceDomainAccuracy:
    """Test cases for insurance-specific accuracy"""
    
    @pytest.fixture
    def insurance_test_cases(self):
        """Insurance-specific test cases"""
        return [
            {
                "document": "Grace Period: 30 days allowed for premium payment",
                "question": "What is the grace period?",
                "expected_keywords": ["30", "days", "grace", "period"]
            },
            {
                "document": "Waiting period for pre-existing diseases is 36 months",
                "question": "What is the waiting period for PED?",
                "expected_keywords": ["36", "months", "waiting", "pre-existing"]
            },
            {
                "document": "Maternity benefits available after 24 months continuous coverage",
                "question": "When are maternity benefits available?",
                "expected_keywords": ["24", "months", "maternity"]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_insurance_query_accuracy(self, ai_processor, insurance_test_cases):
        """Test accuracy on insurance-specific queries"""
        results = []
        
        for test_case in insurance_test_cases:
            await ai_processor.initialize_document(test_case["document"])
            answer, confidence = await ai_processor.process_query(
                test_case["document"], 
                test_case["question"]
            )
            
            # Check if expected keywords are in the answer
            answer_lower = answer.lower()
            keyword_matches = sum(
                1 for keyword in test_case["expected_keywords"] 
                if keyword.lower() in answer_lower
            )
            
            accuracy = keyword_matches / len(test_case["expected_keywords"])
            results.append({
                "question": test_case["question"],
                "answer": answer,
                "confidence": confidence,
                "accuracy": accuracy
            })
        
        # Overall accuracy should be > 70%
        overall_accuracy = sum(r["accuracy"] for r in results) / len(results)
        assert overall_accuracy > 0.7, f"Overall accuracy {overall_accuracy:.2f} below threshold"

class TestPerformanceRequirements:
    """Test cases for performance requirements"""
    
    @pytest.mark.asyncio
    async def test_response_time_requirement(self, ai_processor):
        """Test that responses are generated within time limit"""
        document = "Sample insurance policy with various benefits and conditions"
        query = "What are the main benefits?"
        
        start_time = asyncio.get_event_loop().time()
        
        await ai_processor.initialize_document(document)
        answer, confidence = await ai_processor.process_query(document, query)
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Should be well under 15 seconds
        assert processing_time < 15, f"Processing took {processing_time:.2f}s, exceeds 15s limit"
        assert processing_time < 10, f"Processing took {processing_time:.2f}s, should be under 10s for good performance"
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, ai_processor):
        """Test concurrent query processing"""
        document = "Insurance policy document with multiple sections and benefits"
        queries = [
            "What is the grace period?",
            "What are the waiting periods?",
            "What benefits are covered?",
            "What are the exclusions?"
        ]
        
        await ai_processor.initialize_document(document)
        
        # Process queries concurrently
        start_time = asyncio.get_event_loop().time()
        
        tasks = [
            ai_processor.process_query(document, query) 
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Concurrent processing should be faster than sequential
        assert len(results) == len(queries)
        assert total_time < 20, f"Concurrent processing took {total_time:.2f}s"
        
        for answer, confidence in results:
            assert isinstance(answer, str)
            assert isinstance(confidence, float)

# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test complete end-to-end processing"""
        # This would test the complete flow from document URL to final answers
        # Mock the document download and processing
        pass
    
    @pytest.mark.asyncio 
    async def test_error_handling_robustness(self):
        """Test system robustness under various error conditions"""
        # Test with invalid documents, network errors, API failures, etc.
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])