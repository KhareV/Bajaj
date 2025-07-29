"""
Comprehensive test suite for enhanced document processor and vector store
"""

import pytest
import asyncio
import re
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from app.services.document_processor import InsuranceDocumentProcessor, process_document
from app.services.vector_store import EnhancedVectorStore, VectorStore

class TestInsuranceDocumentProcessor:
    """Test cases for enhanced document processor"""
    
    @pytest.fixture
    def processor(self):
        return InsuranceDocumentProcessor()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF-like content for testing"""
        return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nGrace Period: 30 days allowed for premium payment.'
    
    @pytest.fixture
    def sample_docx_content(self):
        """Sample DOCX-like content for testing"""
        return b'PK\x03\x04\x14\x00\x00\x00\x08\x00word/document.xml\nGrace Period: 30 days\nWaiting Period: 36 months'
    
    @pytest.fixture
    def sample_insurance_text(self):
        """Sample insurance policy text"""
        return """
        ## GRACE PERIOD
        A grace period of thirty (30) days is allowed for payment of premium after due date.
        
        ## WAITING PERIODS
        **Pre-existing Diseases:** 36 months waiting period from policy inception.
        **Maternity:** 24 months continuous coverage required.
        
        ## COVERAGE
        - Hospitalization expenses up to sum insured
        - Ambulance charges up to ₹2,000
        
        ## EXCLUSIONS
        - Cosmetic surgery
        - Treatment outside India
        """
    
    @pytest.mark.asyncio
    async def test_document_processing_basic(self, processor):
        """Test basic document processing functionality"""
        
        # Mock successful document download and processing
        with patch.object(processor, '_download_document') as mock_download:
            with patch.object(processor, '_extract_pdf_text_enhanced') as mock_extract:
                
                mock_download.return_value = (b'%PDF content', 'application/pdf')
                mock_extract.return_value = ("Sample policy text", Mock())
                
                result = await processor.process_document("https://www.hackrx.in/policies/BAJHLIP23020V012223.pdf")
                
                assert isinstance(result, str)
                assert len(result) > 0
                mock_download.assert_called_once()
                mock_extract.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_text_cleaning_and_structure(self, processor, sample_insurance_text):
        """Test enhanced text cleaning and structure preservation"""
        
        cleaned_text = processor._clean_and_structure_text(sample_insurance_text)
        
        # Check structure preservation
        assert '##' in cleaned_text  # Headers preserved
        assert '**' in cleaned_text  # Bold sections preserved
        assert '₹' in cleaned_text   # Currency symbols preserved
        assert '30' in cleaned_text and 'days' in cleaned_text  # Numbers preserved
        
        # Check cleaning effectiveness
        assert not re.search(r'\n\s*\n\s*\n+', cleaned_text)  # No excessive newlines
        assert not re.search(r'  +', cleaned_text)  # No multiple spaces
    
    def test_insurance_specific_cleaning(self, processor):
        """Test insurance-specific text normalization"""
        
        test_text = "Rs. 5000 rupees 10000 INR 15000 pre existing 24 month 30 day 5 percent"
        cleaned = processor._insurance_specific_cleaning(test_text)
        
        # Check currency standardization
        assert '₹5000' in cleaned
        assert '₹10000' in cleaned
        assert '₹15000' in cleaned
        
        # Check terminology standardization
        assert 'pre-existing' in cleaned
        assert '24 months' in cleaned
        assert '30 days' in cleaned
        assert '5%' in cleaned
    
    def test_quality_assessment(self, processor, sample_insurance_text):
        """Test text quality assessment"""
        
        quality_score = processor._assess_text_quality(sample_insurance_text)
        
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.5  # Should be decent quality for insurance text
    
    def test_structure_detection(self, processor):
        """Test section header detection"""
        
        headers = [
            "GRACE PERIOD",
            "1. WAITING PERIODS", 
            "Coverage Benefits:",
            "**Exclusions**",
            "## Definitions"
        ]
        
        for header in headers:
            assert processor._is_section_header(header)
        
        non_headers = [
            "This is regular text",
            "payment of premium after due date",
            "123 some number"
        ]
        
        for non_header in non_headers:
            assert not processor._is_section_header(non_header)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, processor):
        """Test error handling in document processing"""
        
        # Test invalid URL
        with pytest.raises(Exception):
            await processor.process_document("invalid-url")
        
        # Test unsupported format
        with patch.object(processor, '_download_document') as mock_download:
            mock_download.return_value = (b'unsupported content', 'text/plain')
            
            with pytest.raises(Exception):
                await processor.process_document("https://www.hackrx.in/policies/BAJHLIP23020V012223.pdf")

class TestEnhancedVectorStore:
    """Test cases for enhanced vector store"""
    
    @pytest.fixture
    def vector_store(self):
        return EnhancedVectorStore("all-MiniLM-L6-v2")
    
    @pytest.fixture
    def sample_policy_text(self):
        return """
        Grace Period: A grace period of 30 days is allowed for premium payment after due date.
        
        Waiting Periods: Pre-existing diseases have 36 months waiting period. Maternity has 24 months waiting period.
        
        Coverage: Hospitalization expenses, ambulance charges up to ₹2,000, organ transplant expenses are covered.
        
        Exclusions: Cosmetic surgery, dental treatment, treatment outside India are excluded.
        """
    
    def test_vector_store_initialization(self, vector_store):
        """Test vector store initialization"""
        
        assert vector_store.model_name == "all-MiniLM-L6-v2"
        assert not vector_store.is_built
        assert vector_store.dimension == 384
        assert len(vector_store.chunks) == 0
    
    def test_index_building(self, vector_store, sample_policy_text):
        """Test enhanced index building"""
        
        vector_store.build_index(sample_policy_text)
        
        assert vector_store.is_built
        assert len(vector_store.chunks) > 0
        assert vector_store.index is not None
        assert vector_store.tfidf_matrix is not None
        assert len(vector_store.chunk_metadata) == len(vector_store.chunks)
    
    def test_semantic_search_accuracy(self, vector_store, sample_policy_text):
        """Test semantic search accuracy"""
        
        vector_store.build_index(sample_policy_text)
        
        # Test grace period query
        results = vector_store.search("What is the grace period?", k=2)
        
        assert len(results) > 0
        assert any('grace period' in result.lower() for result in results)
        assert any('30 days' in result.lower() for result in results)
    
    def test_hybrid_search_performance(self, vector_store, sample_policy_text):
        """Test hybrid search performance"""
        
        vector_store.build_index(sample_policy_text)
        
        test_queries = [
            "grace period",
            "waiting period pre-existing",
            "ambulance coverage",
            "exclusions cosmetic surgery"
        ]
        
        for query in test_queries:
            results = vector_store.search(query, k=3)
            assert len(results) > 0
            
            # Check relevance
            combined_text = ' '.join(results).lower()
            query_words = query.lower().split()
            matches = sum(1 for word in query_words if word in combined_text)
            assert matches > 0  # At least some query terms should match
    
    def test_search_with_metadata(self, vector_store, sample_policy_text):
        """Test search with metadata functionality"""
        
        vector_store.build_index(sample_policy_text)
        
        results = vector_store.search_with_metadata("grace period", k=2)
        
        assert len(results) > 0
        
        for result in results:
            assert hasattr(result, 'text')
            assert hasattr(result, 'score')
            assert hasattr(result, 'chunk_id')
            assert hasattr(result, 'relevance_factors')
            assert hasattr(result, 'snippet')
            
            assert isinstance(result.score, float)
            assert result.score >= 0
            assert isinstance(result.chunk_id, int)
    
    def test_query_expansion(self, vector_store):
        """Test query expansion functionality"""
        
        expanded = vector_store._expand_query("grace period")
        
        assert 'grace period' in expanded
        assert 'premium payment' in expanded
        assert 'policy' in expanded or 'insurance' in expanded
    
    def test_chunk_quality_assessment(self, vector_store):
        """Test chunk quality assessment"""
        
        chunker = vector_store.chunker
        
        high_quality_text = "Grace period of 30 days for premium payment. Coverage includes hospitalization expenses up to ₹5,00,000."
        low_quality_text = "Some random text without insurance terms."
        
        high_quality_score = chunker._calculate_chunk_quality(high_quality_text)
        low_quality_score = chunker._calculate_chunk_quality(low_quality_text)
        
        assert high_quality_score > low_quality_score
        assert high_quality_score > 0.5
    
    def test_error_handling_vector_store(self, vector_store):
        """Test error handling in vector store"""
        
        # Test search without building index
        with pytest.raises(Exception):
            vector_store.search("test query")
        
        # Test building index with empty text
        with pytest.raises(Exception):
            vector_store.build_index("")
    
    def test_legacy_compatibility(self, sample_policy_text):
        """Test legacy VectorStore class compatibility"""
        
        legacy_store = VectorStore()
        
        # Test interface compatibility
        legacy_store.build_index(sample_policy_text)
        results = legacy_store.search("grace period", k=3)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(result, str) for result in results)

class TestIntegration:
    """Integration tests for document processor and vector store"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test complete pipeline from document to search"""
        
        # Sample document content
        sample_text = """
        ## GRACE PERIOD
        A grace period of 30 days is provided for premium payment.
        
        ## WAITING PERIODS  
        Pre-existing diseases: 36 months waiting period.
        Maternity benefits: 24 months continuous coverage required.
        """
        
        # Initialize components
        processor = InsuranceDocumentProcessor()
        vector_store = EnhancedVectorStore()
        
        # Mock document processing
        with patch.object(processor, 'process_document') as mock_process:
            mock_process.return_value = sample_text
            
            # Process and index
            processed_text = await processor.process_document("https://www.hackrx.in/policies/BAJHLIP23020V012223.pdf")
            vector_store.build_index(processed_text)
            
            # Test searches
            grace_results = vector_store.search("grace period", k=2)
            waiting_results = vector_store.search("waiting period", k=2)
            
            # Verify results
            assert len(grace_results) > 0
            assert len(waiting_results) > 0
            assert any('30 days' in result for result in grace_results)
            assert any('36 months' in result for result in waiting_results)
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self):
        """Test performance requirements are met"""
        
        sample_text = "Grace period: 30 days. " * 1000  # Large text
        
        vector_store = EnhancedVectorStore()
        
        # Test index building time
        start_time = asyncio.get_event_loop().time()
        vector_store.build_index(sample_text)
        build_time = asyncio.get_event_loop().time() - start_time
        
        # Should build index reasonably quickly
        assert build_time < 30, f"Index building took {build_time:.2f}s, should be under 30s"
        
        # Test search time
        start_time = asyncio.get_event_loop().time()
        results = vector_store.search("grace period", k=5)
        search_time = asyncio.get_event_loop().time() - start_time
        
        # Should search very quickly
        assert search_time < 1, f"Search took {search_time:.2f}s, should be under 1s"
        assert len(results) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])