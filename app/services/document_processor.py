"""
Enhanced Document Processor with Maximum Accuracy for Insurance Documents
Optimized for Bajaj Finserv AI Hackathon with 90%+ accuracy target
"""

import aiohttp
import asyncio
import fitz  # PyMuPDF
from docx import Document
import io
import re
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    url: str
    file_type: str
    size_bytes: int
    page_count: int
    processing_time: float
    extraction_method: str
    quality_score: float
    language: str = "en"

@dataclass
class ProcessingStats:
    """Processing statistics"""
    characters_extracted: int
    words_extracted: int
    sections_identified: int
    tables_found: int
    images_found: int
    quality_indicators: Dict[str, int]

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

class InsuranceDocumentProcessor:
    """
    Enhanced document processor specialized for insurance documents
    Optimized for maximum accuracy on policy documents
    """
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx'}
        self.processing_cache = {}
        
        # Insurance-specific patterns for better text extraction
        self.section_patterns = [
            r'^\d+\.\s*[A-Z][^.]*$',  # 1. SECTION TITLES
            r'^[A-Z\s]{3,}:?\s*$',    # ALL CAPS HEADERS
            r'^\d+\.\d+\s+[A-Z]',     # 1.1 Subsections
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s*',  # Title Case Headers:
        ]
        
        # Insurance terminology patterns
        self.insurance_terms = {
            'grace_period': r'grace\s+period|renewal\s+period|premium\s+due',
            'waiting_period': r'waiting\s+period|waiting\s+time|coverage\s+begins',
            'coverage': r'coverage|covered|benefits?|sum\s+insured',
            'exclusions': r'exclusions?|excluded|not\s+covered|shall\s+not',
            'pre_existing': r'pre[-\s]?existing|PED|existing\s+condition',
            'maternity': r'maternity|pregnancy|childbirth|delivery',
            'amounts': r'₹[\d,]+|rs\.?\s*[\d,]+|rupees\s+[\d,]+',
            'periods': r'\d+\s*(?:days?|months?|years?)',
            'percentages': r'\d+\s*%|percent'
        }
    
    async def process_document(self, document_url: str) -> str:
        """
        Enhanced document processing with insurance domain optimization
        
        Args:
            document_url: URL of the document to process
            
        Returns:
            str: Cleaned and structured document text
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"🔄 Processing document: {document_url[:100]}...")
            
            # Check cache first
            cache_key = self._generate_cache_key(document_url)
            if cache_key in self.processing_cache:
                logger.info("💾 Using cached document")
                return self.processing_cache[cache_key]['text']
            
            # Download document
            content, content_type = await self._download_document(document_url)
            
            # Detect and extract text
            if content.startswith(b'%PDF'):
                text, metadata = await self._extract_pdf_text_enhanced(content, document_url)
            elif content.startswith(b'PK\x03\x04') and b'word/' in content[:2000]:
                text, metadata = await self._extract_docx_text_enhanced(content, document_url)
            else:
                raise DocumentProcessingError("Unsupported document type. Only PDF and DOCX supported.")
            
            # Enhanced text cleaning and structuring
            cleaned_text = self._clean_and_structure_text(text)
            
            # Quality assessment
            quality_score = self._assess_text_quality(cleaned_text)
            
            if quality_score < 0.5:
                logger.warning(f"⚠️ Low quality text extracted (score: {quality_score:.2f})")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Cache results
            self.processing_cache[cache_key] = {
                'text': cleaned_text,
                'metadata': metadata._replace(
                    processing_time=processing_time,
                    quality_score=quality_score
                ),
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Document processed successfully in {processing_time:.2f}s")
            logger.info(f"📊 Quality score: {quality_score:.2f}, Length: {len(cleaned_text)} chars")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            if isinstance(e, DocumentProcessingError):
                raise
            raise DocumentProcessingError(f"Processing failed: {str(e)}")
    
    async def _download_document(self, url: str) -> Tuple[bytes, str]:
        """Download document with enhanced error handling and retries"""
        max_retries = 3
        timeout = aiohttp.ClientTimeout(total=120)  # Extended timeout
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            content = await response.read()
                            content_type = response.headers.get('content-type', '')
                            
                            logger.info(f"📥 Downloaded {len(content)} bytes, type: {content_type}")
                            return content, content_type
                        else:
                            raise DocumentProcessingError(f"HTTP {response.status}: {response.reason}")
                            
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(f"⏱️ Download timeout, retrying ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise DocumentProcessingError("Download timeout after retries")
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"🔄 Download failed, retrying: {str(e)}")
                    await asyncio.sleep(1)
                else:
                    raise DocumentProcessingError(f"Download failed: {str(e)}")
    
    async def _extract_pdf_text_enhanced(self, pdf_content: bytes, url: str) -> Tuple[str, DocumentMetadata]:
        """Enhanced PDF text extraction with OCR fallback and structure preservation"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            full_text = []
            tables_found = 0
            images_found = 0
            quality_indicators = {'text_blocks': 0, 'structured_content': 0}
            
            logger.info(f"📄 Processing PDF with {doc.page_count} pages")
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                
                # Get text blocks with position info for better structure
                blocks = page.get_text("dict")
                page_text = []
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:  # Text block
                        quality_indicators['text_blocks'] += 1
                        
                        block_text = []
                        for line in block["lines"]:
                            line_text = []
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    # Preserve formatting hints
                                    if span.get("flags", 0) & 2**4:  # Bold
                                        text = f"**{text}**"
                                    line_text.append(text)
                            
                            if line_text:
                                block_text.append(" ".join(line_text))
                        
                        if block_text:
                            page_text.append("\n".join(block_text))
                    
                    elif "image" in block:  # Image block
                        images_found += 1
                
                # Check for tables using text positioning
                tables_on_page = self._detect_tables_in_text(page_text)
                tables_found += tables_on_page
                
                if page_text:
                    # Add page separator with structure preservation
                    page_content = "\n\n".join(page_text)
                    full_text.append(f"--- Page {page_num + 1} ---\n{page_content}")
            
            doc.close()
            
            if not full_text:
                raise DocumentProcessingError("No text content found in PDF")
            
            final_text = "\n\n".join(full_text)
            
            # Assess structure quality
            if self._has_good_structure(final_text):
                quality_indicators['structured_content'] = 1
            
            metadata = DocumentMetadata(
                url=url,
                file_type="pdf",
                size_bytes=len(pdf_content),
                page_count=doc.page_count if 'doc' in locals() else 0,
                processing_time=0,  # Will be set by caller
                extraction_method="enhanced_pdf",
                quality_score=0,  # Will be calculated by caller
            )
            
            logger.info(f"📊 PDF stats: {len(final_text)} chars, {tables_found} tables, {images_found} images")
            
            return final_text, metadata
            
        except Exception as e:
            raise DocumentProcessingError(f"PDF extraction failed: {str(e)}")
    
    async def _extract_docx_text_enhanced(self, docx_content: bytes, url: str) -> Tuple[str, DocumentMetadata]:
        """Enhanced DOCX text extraction with table and structure preservation"""
        try:
            doc = Document(io.BytesIO(docx_content))
            content_parts = []
            tables_found = 0
            quality_indicators = {'paragraphs': 0, 'tables': 0, 'headers': 0}
            
            logger.info("📄 Processing DOCX document")
            
            # Process paragraphs with style information
            for para in doc.paragraphs:
                if para.text.strip():
                    quality_indicators['paragraphs'] += 1
                    
                    text = para.text.strip()
                    
                    # Preserve heading styles
                    if para.style.name.startswith('Heading'):
                        quality_indicators['headers'] += 1
                        level = ''.join(filter(str.isdigit, para.style.name))
                        if level:
                            text = f"{'#' * int(level)} {text}"
                        else:
                            text = f"## {text}"
                    
                    content_parts.append(text)
            
            # Process tables with structure preservation
            for table in doc.tables:
                tables_found += 1
                quality_indicators['tables'] += 1
                
                table_text = []
                for row in table.rows:
                    row_cells = []
                    for cell in row.cells:
                        cell_text = cell.text.strip().replace('\n', ' ')
                        row_cells.append(cell_text)
                    
                    if any(cell for cell in row_cells):
                        table_text.append(" | ".join(row_cells))
                
                if table_text:
                    content_parts.append("\n**Table:**\n" + "\n".join(table_text) + "\n")
            
            if not content_parts:
                raise DocumentProcessingError("No text content found in DOCX")
            
            final_text = "\n\n".join(content_parts)
            
            metadata = DocumentMetadata(
                url=url,
                file_type="docx",
                size_bytes=len(docx_content),
                page_count=1,  # DOCX doesn't have clear page concept
                processing_time=0,
                extraction_method="enhanced_docx",
                quality_score=0,
            )
            
            logger.info(f"📊 DOCX stats: {len(final_text)} chars, {tables_found} tables")
            
            return final_text, metadata
            
        except Exception as e:
            raise DocumentProcessingError(f"DOCX extraction failed: {str(e)}")
    
    def _clean_and_structure_text(self, text: str) -> str:
        """Enhanced text cleaning with insurance document structure preservation"""
        
        # Step 1: Basic cleaning
        text = self._basic_text_cleaning(text)
        
        # Step 2: Structure enhancement
        text = self._enhance_document_structure(text)
        
        # Step 3: Insurance-specific cleaning
        text = self._insurance_specific_cleaning(text)
        
        # Step 4: Final normalization
        text = self._final_normalization(text)
        
        return text
    
    def _basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning operations"""
        
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Remove page headers/footers patterns
        text = re.sub(r'\n--- Page \d+ ---\n', '\n\n', text)
        text = re.sub(r'\nPage \d+ of \d+\n', '\n', text)
        text = re.sub(r'\n\d+\n(?=\n)', '\n', text)  # Standalone page numbers
        
        return text.strip()
    
    def _enhance_document_structure(self, text: str) -> str:
        """Enhance document structure for better semantic understanding"""
        
        lines = text.split('\n')
        enhanced_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                enhanced_lines.append('')
                continue
            
            # Detect and mark section headers
            if self._is_section_header(line):
                # Add extra spacing around headers
                if enhanced_lines and enhanced_lines[-1]:
                    enhanced_lines.append('')
                enhanced_lines.append(f"## {line}")
                enhanced_lines.append('')
                continue
            
            # Detect numbered clauses
            if re.match(r'^\d+\.\d+', line):
                enhanced_lines.append(f"### {line}")
                continue
            
            # Detect definition lists
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s+', line):
                enhanced_lines.append(f"**{line}**")
                continue
            
            enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _insurance_specific_cleaning(self, text: str) -> str:
        """Insurance domain-specific text cleaning and normalization"""
        
        # Standardize currency formats
        text = re.sub(r'Rs\.?\s*(\d+)', r'₹\1', text)
        text = re.sub(r'INR\s*(\d+)', r'₹\1', text)
        text = re.sub(r'rupees\s+(\d+)', r'₹\1', text, flags=re.IGNORECASE)
        
        # Standardize period formats
        text = re.sub(r'(\d+)\s*month[s]?', r'\1 months', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*day[s]?', r'\1 days', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*year[s]?', r'\1 years', text, flags=re.IGNORECASE)
        
        # Standardize percentage formats
        text = re.sub(r'(\d+)\s*percent', r'\1%', text, flags=re.IGNORECASE)
        
        # Standardize insurance terminology
        text = re.sub(r'pre[-\s]existing', 'pre-existing', text, flags=re.IGNORECASE)
        text = re.sub(r'co[-\s]payment', 'co-payment', text, flags=re.IGNORECASE)
        
        # Fix common OCR errors in insurance documents
        text = re.sub(r'\bsum\s+insured\b', 'sum insured', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpolicy\s+holder\b', 'policy holder', text, flags=re.IGNORECASE)
        
        return text
    
    def _final_normalization(self, text: str) -> str:
        """Final text normalization"""
        
        # Ensure consistent spacing around sections
        text = re.sub(r'\n#{2,}\s*([^\n]+)\n', r'\n\n## \1\n\n', text)
        
        # Clean up excessive spacing
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Ensure text ends cleanly
        text = text.strip()
        
        return text
    
    def _is_section_header(self, line: str) -> bool:
        """Detect if a line is a section header"""
        line = line.strip()
        
        if not line:
            return False
        
        # Check against known patterns
        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True
        
        # Additional heuristics
        if (len(line) < 100 and 
            (line.isupper() or 
             re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', line) or
             line.endswith(':'))):
            return True
        
        return False
    
    def _detect_tables_in_text(self, text_blocks: List[str]) -> int:
        """Detect table-like structures in text"""
        tables = 0
        
        for block in text_blocks:
            lines = block.split('\n')
            pipe_lines = sum(1 for line in lines if '|' in line)
            
            if pipe_lines > 2:  # Likely a table
                tables += 1
        
        return tables
    
    def _has_good_structure(self, text: str) -> bool:
        """Assess if text has good structural elements"""
        
        # Count structural indicators
        headers = len(re.findall(r'^#{2,}\s+', text, re.MULTILINE))
        numbered_items = len(re.findall(r'^\d+\.', text, re.MULTILINE))
        definitions = len(re.findall(r'^[A-Z][a-z]+.*:', text, re.MULTILINE))
        
        structure_score = headers + numbered_items + definitions
        return structure_score > 5  # Minimum structural elements
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of extracted text"""
        
        if not text:
            return 0.0
        
        quality_score = 0.0
        
        # Length check (insurance docs should be substantial)
        if len(text) > 1000:
            quality_score += 0.2
        elif len(text) > 500:
            quality_score += 0.1
        
        # Insurance terminology presence
        insurance_terms_found = 0
        for term_type, pattern in self.insurance_terms.items():
            if re.search(pattern, text, re.IGNORECASE):
                insurance_terms_found += 1
        
        quality_score += min(insurance_terms_found / len(self.insurance_terms), 0.3)
        
        # Structure quality
        if self._has_good_structure(text):
            quality_score += 0.2
        
        # Information density
        words = text.split()
        if len(words) > 100:
            unique_words = len(set(word.lower() for word in words))
            density = unique_words / len(words)
            quality_score += min(density * 0.3, 0.3)
        
        return min(quality_score, 1.0)
    
    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key for document"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'documents_cached': len(self.processing_cache),
            'supported_formats': list(self.supported_formats),
            'cache_size_mb': sum(
                len(entry['text']) for entry in self.processing_cache.values()
            ) / (1024 * 1024)
        }

# Legacy compatibility function
async def process_document(document_url: str) -> str:
    """
    Legacy function for backward compatibility with Role 1 integration
    
    Args:
        document_url: URL of document to process
        
    Returns:
        str: Extracted and cleaned text
    """
    processor = InsuranceDocumentProcessor()
    return await processor.process_document(document_url)