"""
HYPER-FAST Document Processor - Optimized for <8s processing
Date: 2025-08-01 17:01:45 UTC | User: vkhare2909
"""

import aiohttp
import asyncio
import fitz  # PyMuPDF
from docx import Document
import io
import re
from typing import Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

class HyperFastDocumentProcessor:
    """Hyper-optimized document processor for <8s total response"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=6)  # Increased workers
        self.session = None
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=6)  # Reduced timeout
            connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self.session
    
    async def process_document(self, document_url: str) -> str:
        """Hyper-fast document processing with extreme optimizations"""
        start_time = time.time()
        
        try:
            # Step 1: Hyper-fast download (max 5s)
            session = await self.get_session()
            
            logger.info(f"🚀 HYPER-FAST download starting: {document_url[:60]}...")
            
            async with session.get(document_url) as response:
                if response.status != 200:
                    raise DocumentProcessingError(f"Download failed: HTTP {response.status}")
                
                # Stream download for large files
                content = bytearray()
                async for chunk in response.content.iter_chunked(16384):  # 16KB chunks
                    content.extend(chunk)
                
                content = bytes(content)
            
            download_time = time.time() - start_time
            logger.info(f"⚡ Download completed in {download_time:.1f}s ({len(content)} bytes)")
            
            # Step 2: Hyper-fast text extraction (max 2s)
            extraction_start = time.time()
            
            if content.startswith(b'%PDF'):
                text = await self._extract_pdf_hyper_fast(content)
            elif content.startswith(b'PK\x03\x04') and b'word/' in content[:2000]:
                text = await self._extract_docx_hyper_fast(content)
            else:
                raise DocumentProcessingError("Unsupported format. Only PDF/DOCX supported.")
            
            extraction_time = time.time() - extraction_start
            logger.info(f"⚡ Extraction completed in {extraction_time:.1f}s")
            
            # Step 3: Hyper-fast cleaning (max 0.5s)
            clean_start = time.time()
            cleaned_text = self._hyper_clean_text(text)
            clean_time = time.time() - clean_start
            
            total_time = time.time() - start_time
            
            logger.info(f"✅ HYPER-FAST processing complete:")
            logger.info(f"   📊 Total time: {total_time:.1f}s")
            logger.info(f"   📄 Text length: {len(cleaned_text):,} chars")
            logger.info(f"   ⚡ Speed: {len(cleaned_text)/total_time:.0f} chars/sec")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"❌ HYPER-FAST processing failed: {e}")
            raise DocumentProcessingError(f"Processing failed: {str(e)}")
    
    async def _extract_pdf_hyper_fast(self, pdf_content: bytes) -> str:
        """Hyper-fast PDF extraction with extreme optimizations"""
        
        def extract_pdf_sync():
            try:
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                
                total_pages = len(doc)
                logger.info(f"📄 Processing {total_pages} PDF pages...")
                
                # EXTREME optimization for large documents (100-150 pages)
                if total_pages > 80:
                    # For very large docs, use strategic sampling
                    # First 20 pages (usually contain key policy info)
                    pages_to_process = list(range(0, min(20, total_pages)))
                    # Middle section sample (10 pages)
                    middle_start = max(20, total_pages//2 - 5)
                    middle_end = min(total_pages, total_pages//2 + 5)
                    pages_to_process.extend(range(middle_start, middle_end))
                    # Last 10 pages (usually contain important clauses)
                    last_start = max(middle_end, total_pages - 10)
                    pages_to_process.extend(range(last_start, total_pages))
                    pages_to_process = sorted(set(pages_to_process))[:50]  # Max 50 pages for speed
                    
                    logger.info(f"📄 Large document: processing {len(pages_to_process)} strategic pages")
                elif total_pages > 50:
                    # For medium docs, process every other page plus key sections
                    pages_to_process = list(range(0, min(25, total_pages))) + \
                                     list(range(25, total_pages, 2))  # Every 2nd page after 25
                    pages_to_process = sorted(set(pages_to_process))[:40]  # Max 40 pages
                    logger.info(f"📄 Medium document: processing {len(pages_to_process)} optimized pages")
                else:
                    pages_to_process = range(total_pages)
                
                text_parts = []
                for page_num in pages_to_process:
                    if page_num < total_pages:
                        page = doc.load_page(page_num)
                        page_text = page.get_text()
                        if page_text.strip():
                            text_parts.append(page_text)
                
                doc.close()
                
                full_text = "\n".join(text_parts)
                
                if not full_text.strip():
                    raise DocumentProcessingError("No text extracted from PDF")
                
                return full_text
                
            except Exception as e:
                raise DocumentProcessingError(f"PDF extraction failed: {str(e)}")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, extract_pdf_sync)
    
    async def _extract_docx_hyper_fast(self, docx_content: bytes) -> str:
        """Hyper-fast DOCX extraction"""
        
        def extract_docx_sync():
            try:
                doc = Document(io.BytesIO(docx_content))
                text_parts = []
                
                # Extract paragraphs (optimized limit)
                for i, paragraph in enumerate(doc.paragraphs):
                    if i > 800:  # Reduced limit for speed
                        break
                    if paragraph.text.strip():
                        text_parts.append(paragraph.text)
                
                # Extract tables (optimized limit)
                for i, table in enumerate(doc.tables):
                    if i > 30:  # Reduced limit for speed
                        break
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            text_parts.append(row_text)
                
                full_text = "\n".join(text_parts)
                
                if not full_text.strip():
                    raise DocumentProcessingError("No text extracted from DOCX")
                
                return full_text
                
            except Exception as e:
                raise DocumentProcessingError(f"DOCX extraction failed: {str(e)}")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, extract_docx_sync)
    
    def _hyper_clean_text(self, text: str) -> str:
        """Hyper-fast text cleaning optimized for insurance documents"""
        
        # Quick aggressive cleaning for speed
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' {3,}', '  ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Remove control characters but keep essential formatting
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
        
        # Preserve important insurance terms formatting
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()
        self.executor.shutdown(wait=False)

# Global instance for reuse
hyper_fast_processor = HyperFastDocumentProcessor()

async def process_document(document_url: str) -> str:
    """Main function for document processing"""
    return await hyper_fast_processor.process_document(document_url)