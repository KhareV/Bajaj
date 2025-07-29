"""
Enhanced Vector Store with Advanced Semantic Search
Optimized for maximum accuracy on insurance document queries
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple, Dict, Any, Union
import logging
import re
import json
from dataclasses import dataclass
import time
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    text: str
    score: float
    chunk_id: int
    relevance_factors: Dict[str, float]
    snippet: str

@dataclass
class IndexStats:
    """Vector store statistics"""
    total_chunks: int
    index_size: int
    model_name: str
    dimension: int
    build_time: float
    last_updated: float

class VectorStoreError(Exception):
    """Custom exception for vector store operations"""
    pass

class AdvancedChunker:
    """Advanced text chunking with insurance document awareness"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Insurance-specific section markers
        self.section_markers = [
            r'#{2,}\s+([^\n]+)',  # Markdown headers
            r'^\d+\.\s+[A-Z][^.]*$',  # 1. SECTION
            r'^\d+\.\d+\s+[A-Z]',     # 1.1 Subsection
            r'^\*\*([^*]+)\*\*',      # **Bold headers**
            r'^[A-Z\s]{5,}:?\s*$',    # ALL CAPS HEADERS
        ]
        
        # Content quality indicators
        self.quality_indicators = [
            r'grace\s+period', r'waiting\s+period', r'coverage', r'exclusion',
            r'premium', r'deductible', r'maternity', r'pre[-\s]existing',
            r'₹[\d,]+', r'\d+\s*(?:days?|months?|years?)', r'\d+\s*%'
        ]
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Create intelligent chunks with metadata
        
        Args:
            text: Input document text
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        logger.info(f"🔪 Creating chunks from {len(text)} characters")
        
        # Step 1: Split by sections if possible
        sections = self._split_by_sections(text)
        
        # Step 2: Create chunks from sections
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)
        
        # Step 3: Add overlap and enhance metadata
        enhanced_chunks = self._add_overlap_and_metadata(chunks)
        
        logger.info(f"✅ Created {len(enhanced_chunks)} chunks")
        return enhanced_chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by logical sections"""
        
        # Find section boundaries
        boundaries = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if self._is_section_boundary(line_stripped):
                boundaries.append(i)
        
        if not boundaries:
            # Fallback to paragraph-based splitting
            return self._split_by_paragraphs(text)
        
        # Create sections
        sections = []
        boundaries.append(len(lines))  # Add end boundary
        
        for i in range(len(boundaries) - 1):
            start_line = boundaries[i]
            end_line = boundaries[i + 1]
            section_lines = lines[start_line:end_line]
            section_text = '\n'.join(section_lines).strip()
            
            if section_text and len(section_text) > 50:
                sections.append(section_text)
        
        return sections if sections else [text]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Fallback paragraph-based splitting"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _is_section_boundary(self, line: str) -> bool:
        """Check if line is a section boundary"""
        if not line:
            return False
        
        for pattern in self.section_markers:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _chunk_section(self, section: str) -> List[Dict[str, Any]]:
        """Create chunks from a single section"""
        words = section.split()
        
        if len(words) <= self.chunk_size:
            return [{
                'text': section,
                'word_count': len(words),
                'is_complete_section': True
            }]
        
        # Split large sections
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'word_count': len(chunk_words),
                'is_complete_section': False,
                'chunk_position': i // (self.chunk_size - self.overlap)
            })
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def _add_overlap_and_metadata(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Add overlap between chunks and enhance metadata"""
        
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk['text']
            
            # Add context from adjacent chunks
            if i > 0 and not chunk.get('is_complete_section', False):
                prev_text = chunks[i-1]['text']
                prev_sentences = re.split(r'[.!?]+', prev_text)
                if len(prev_sentences) > 1:
                    context_prefix = '. '.join(prev_sentences[-2:]).strip()
                    if context_prefix:
                        text = context_prefix + '. ' + text
            
            # Calculate quality score
            quality_score = self._calculate_chunk_quality(text)
            
            # Extract key terms
            key_terms = self._extract_key_terms(text)
            
            enhanced_chunk = {
                'text': text,
                'original_text': chunk['text'],
                'chunk_id': i,
                'word_count': len(text.split()),
                'quality_score': quality_score,
                'key_terms': key_terms,
                'has_numbers': bool(re.search(r'\d+', text)),
                'has_currency': bool(re.search(r'₹[\d,]+', text)),
                'has_periods': bool(re.search(r'\d+\s*(?:days?|months?|years?)', text)),
                'is_complete_section': chunk.get('is_complete_section', False)
            }
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _calculate_chunk_quality(self, text: str) -> float:
        """Calculate quality score for a chunk"""
        score = 0.0
        text_lower = text.lower()
        
        # Check for quality indicators
        for indicator in self.quality_indicators:
            if re.search(indicator, text_lower):
                score += 0.1
        
        # Bonus for good length
        word_count = len(text.split())
        if 100 <= word_count <= 600:
            score += 0.2
        elif 50 <= word_count < 100:
            score += 0.1
        
        # Bonus for sentence completeness
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) >= 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from chunk"""
        key_terms = []
        text_lower = text.lower()
        
        # Insurance-specific terms
        insurance_keywords = [
            'grace period', 'waiting period', 'coverage', 'exclusion', 'premium',
            'deductible', 'maternity', 'pre-existing', 'policy', 'benefit',
            'treatment', 'hospital', 'surgery', 'claim', 'reimbursement'
        ]
        
        for term in insurance_keywords:
            if term in text_lower:
                key_terms.append(term)
        
        # Extract numerical terms
        numbers = re.findall(r'\d+\s*(?:days?|months?|years?|%)', text_lower)
        key_terms.extend(numbers[:3])  # Limit to 3 most relevant
        
        # Extract currency amounts
        amounts = re.findall(r'₹[\d,]+', text)
        key_terms.extend(amounts[:2])  # Limit to 2 amounts
        
        return key_terms[:10]  # Limit total key terms

class EnhancedVectorStore:
    """
    Advanced vector store with hybrid search and reranking
    Optimized for maximum accuracy on insurance queries
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize enhanced vector store
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.is_built = False
        self.build_time = 0
        self.chunker = AdvancedChunker()
        
        # Query enhancement patterns
        self.query_expansions = {
            'grace period': ['premium payment', 'renewal period', 'due date'],
            'waiting period': ['coverage begins', 'eligibility period', 'wait time'],
            'coverage': ['benefits', 'covered expenses', 'sum insured'],
            'exclusions': ['not covered', 'excluded items', 'limitations'],
            'maternity': ['pregnancy', 'childbirth', 'delivery', 'newborn'],
            'surgery': ['surgical procedure', 'operation', 'treatment'],
        }
        
        logger.info(f"🔧 Initialized EnhancedVectorStore with model: {model_name}")
    
    def build_index(self, text: str, chunk_size: int = 800, overlap: int = 150) -> None:
        """
        Build enhanced vector index with hybrid search capabilities
        
        Args:
            text: Input document text
            chunk_size: Words per chunk
            overlap: Overlap between chunks
            
        Raises:
            VectorStoreError: If index building fails
        """
        if not text or not text.strip():
            raise VectorStoreError("Empty text provided for indexing")
        
        start_time = time.time()
        logger.info(f"🏗️ Building enhanced vector index from {len(text)} characters")
        
        try:
            # Step 1: Create intelligent chunks
            self.chunker.chunk_size = chunk_size
            self.chunker.overlap = overlap
            chunk_data = self.chunker.create_chunks(text)
            
            if not chunk_data:
                raise VectorStoreError("No chunks created from text")
            
            # Step 2: Extract text and metadata
            self.chunks = [chunk['text'] for chunk in chunk_data]
            self.chunk_metadata = chunk_data
            
            logger.info(f"📝 Created {len(self.chunks)} chunks")
            
            # Step 3: Load model and create embeddings
            self._load_model()
            embeddings = self._create_embeddings(self.chunks)
            
            # Step 4: Build FAISS index
            self._build_faiss_index(embeddings)
            
            # Step 5: Build TF-IDF index for hybrid search
            self._build_tfidf_index()
            
            self.build_time = time.time() - start_time
            self.is_built = True
            
            logger.info(f"✅ Enhanced vector index built successfully in {self.build_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Index building failed: {str(e)}")
            raise VectorStoreError(f"Failed to build index: {str(e)}")
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """
        Enhanced hybrid search with query expansion and reranking
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of most relevant text chunks
            
        Raises:
            VectorStoreError: If search fails
        """
        if not self.is_built:
            raise VectorStoreError("Index not built. Call build_index() first.")
        
        if not query or not query.strip():
            raise VectorStoreError("Empty query provided")
        
        try:
            logger.info(f"🔍 Enhanced search for: '{query[:50]}...' (k={k})")
            
            # Step 1: Query expansion
            expanded_query = self._expand_query(query)
            
            # Step 2: Hybrid search (semantic + keyword)
            search_results = self._hybrid_search(expanded_query, k * 2)  # Get more candidates
            
            # Step 3: Rerank results
            reranked_results = self._rerank_results(search_results, query, k)
            
            # Step 4: Extract final texts
            final_results = [result.text for result in reranked_results]
            
            logger.info(f"✅ Found {len(final_results)} relevant chunks")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def search_with_metadata(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Enhanced search returning full metadata
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of SearchResult objects with metadata
        """
        if not self.is_built:
            raise VectorStoreError("Index not built. Call build_index() first.")
        
        try:
            expanded_query = self._expand_query(query)
            search_results = self._hybrid_search(expanded_query, k * 2)
            reranked_results = self._rerank_results(search_results, query, k)
            
            return reranked_results
            
        except Exception as e:
            raise VectorStoreError(f"Search with metadata failed: {str(e)}")
    
    def _load_model(self):
        """Load sentence transformer model"""
        if self.model is None:
            logger.info(f"📥 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts"""
        logger.info(f"🧠 Creating embeddings for {len(texts)} chunks...")
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f"✅ Created embeddings: {embeddings.shape}")
        
        return embeddings
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        logger.info("🏗️ Building FAISS index...")
        
        # Use IndexFlatIP for cosine similarity (normalized embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for keyword search"""
        logger.info("📊 Building TF-IDF index...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
        
        logger.info(f"✅ TF-IDF index built: {self.tfidf_matrix.shape}")
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        expanded_terms = [query]
        query_lower = query.lower()
        
        # Add expansion terms
        for key, expansions in self.query_expansions.items():
            if key in query_lower:
                expanded_terms.extend(expansions)
        
        # Add insurance context
        insurance_context = [
            'policy', 'insurance', 'coverage', 'benefit'
        ]
        
        return ' '.join(expanded_terms + insurance_context)
    
    def _hybrid_search(self, query: str, k: int) -> List[SearchResult]:
        """Perform hybrid semantic + keyword search"""
        
        # Semantic search
        semantic_results = self._semantic_search(query, k)
        
        # Keyword search  
        keyword_results = self._keyword_search(query, k)
        
        # Combine and deduplicate
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            combined_results[result.chunk_id] = result
            combined_results[result.chunk_id].relevance_factors['semantic'] = result.score
        
        # Add keyword results
        for result in keyword_results:
            if result.chunk_id in combined_results:
                # Combine scores
                combined_results[result.chunk_id].relevance_factors['keyword'] = result.score
                combined_results[result.chunk_id].score = (
                    combined_results[result.chunk_id].relevance_factors.get('semantic', 0) * 0.7 +
                    result.score * 0.3
                )
            else:
                combined_results[result.chunk_id] = result
                combined_results[result.chunk_id].relevance_factors['keyword'] = result.score
        
        # Sort by combined score
        final_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        
        return final_results[:k]
    
    def _semantic_search(self, query: str, k: int) -> List[SearchResult]:
        """Perform semantic search using FAISS"""
        
        # Create query embedding
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Search FAISS index
        scores, indices = self.index.search(
            query_embedding.astype(np.float32),
            min(k, len(self.chunks))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score > 0.1:  # Relevance threshold
                chunk_text = self.chunks[idx]
                snippet = self._create_snippet(chunk_text, query)
                
                result = SearchResult(
                    text=chunk_text,
                    score=float(score),
                    chunk_id=int(idx),
                    relevance_factors={'semantic': float(score)},
                    snippet=snippet
                )
                results.append(result)
        
        return results
    
    def _keyword_search(self, query: str, k: int) -> List[SearchResult]:
        """Perform keyword search using TF-IDF"""
        
        if self.tfidf_vectorizer is None:
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Relevance threshold
                chunk_text = self.chunks[idx]
                snippet = self._create_snippet(chunk_text, query)
                
                result = SearchResult(
                    text=chunk_text,
                    score=float(similarities[idx]),
                    chunk_id=int(idx),
                    relevance_factors={'keyword': float(similarities[idx])},
                    snippet=snippet
                )
                results.append(result)
        
        return results
    
    def _rerank_results(self, results: List[SearchResult], original_query: str, k: int) -> List[SearchResult]:
        """Rerank results using multiple factors"""
        
        for result in results:
            # Base score from search
            final_score = result.score
            
            # Quality boost
            if result.chunk_id < len(self.chunk_metadata):
                metadata = self.chunk_metadata[result.chunk_id]
                quality_score = metadata.get('quality_score', 0.5)
                final_score += quality_score * 0.1
                
                # Boost for key terms match
                key_terms = metadata.get('key_terms', [])
                query_lower = original_query.lower()
                matching_terms = sum(1 for term in key_terms if term.lower() in query_lower)
                if matching_terms > 0:
                    final_score += matching_terms * 0.05
                
                # Boost for numbers/currency if relevant
                if ('period' in query_lower or 'amount' in query_lower or 'limit' in query_lower):
                    if metadata.get('has_numbers') or metadata.get('has_currency'):
                        final_score += 0.1
            
            # Query-specific boosts
            text_lower = result.text.lower()
            if 'grace period' in original_query.lower() and 'grace period' in text_lower:
                final_score += 0.15
            elif 'waiting period' in original_query.lower() and 'waiting period' in text_lower:
                final_score += 0.15
            elif 'exclusion' in original_query.lower() and ('exclusion' in text_lower or 'excluded' in text_lower):
                final_score += 0.15
            
            result.score = final_score
        
        # Sort and return top k
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results[:k]
    
    def _create_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """Create a snippet highlighting query relevance"""
        
        query_words = query.lower().split()
        sentences = re.split(r'[.!?]+', text)
        
        # Find best sentence containing query terms
        best_sentence = ""
        max_matches = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for word in query_words if word in sentence_lower)
            
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence.strip()
        
        if not best_sentence:
            best_sentence = sentences[0].strip() if sentences else text[:max_length]
        
        # Truncate if too long
        if len(best_sentence) > max_length:
            best_sentence = best_sentence[:max_length] + "..."
        
        return best_sentence
    
    def get_stats(self) -> IndexStats:
        """Get comprehensive index statistics"""
        return IndexStats(
            total_chunks=len(self.chunks),
            index_size=self.index.ntotal if self.index else 0,
            model_name=self.model_name,
            dimension=self.dimension,
            build_time=self.build_time,
            last_updated=time.time()
        )

# Legacy compatibility class
class VectorStore:
    """
    Legacy VectorStore class for backward compatibility with Role 1
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.enhanced_store = EnhancedVectorStore(model_name)
    
    def build_index(self, text: str, chunk_size: int = 800, overlap: int = 150):
        """Build index from text - legacy interface"""
        self.enhanced_store.build_index(text, chunk_size, overlap)
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for relevant chunks - legacy interface"""
        return self.enhanced_store.search(query, k)