# app/models/ai_processor.py (OPTIMIZED VERSION)
"""
OPTIMIZED AI Processor - Enhanced for 90%+ Accuracy
Improved prompt engineering, context selection, and response validation
"""

import asyncio
import time
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# AI Model imports
import google.generativeai as genai
from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)

@dataclass
class AIResponse:
    answer: str
    confidence: float
    model_name: str
    processing_time: float
    reasoning: str

class OptimizedInsurancePrompts:
    """Enhanced Insurance AI Prompts for Maximum Accuracy"""
    
    SYSTEM_PROMPT = """You are an expert insurance policy analyst with 20+ years of experience in Indian insurance policies. Your task is to provide PRECISE, ACCURATE answers based ONLY on the policy document provided.

CRITICAL INSTRUCTIONS:
1. ONLY use information explicitly stated in the document
2. If information is not found, clearly state "The document does not specify..."
3. Extract EXACT numbers, dates, and terms from the document
4. Include ALL relevant conditions and limitations
5. Use the EXACT language from the policy document

ANSWER FORMAT GUIDELINES:
- Grace Period: "The policy provides a grace period of [X] days for premium payment after the due date."
- Waiting Period: "There is a waiting period of [X] months from policy inception for [specific condition]."
- Coverage: "Yes/No, the policy covers [specific benefit] with [conditions/limitations]."
- Exclusions: "The policy excludes [specific items] as stated in [section reference]."

Be concise but complete. Always include specific numbers, time periods, and conditions mentioned in the document."""

    QUERY_SPECIFIC_PROMPTS = {
        'grace_period': """Focus on finding information about:
- Premium payment deadlines
- Grace periods for late payments
- Policy continuity provisions
- Renewal terms and conditions
Extract the EXACT number of days and any conditions.""",
        
        'waiting_period': """Focus on finding information about:
- Pre-existing disease waiting periods
- Specific condition waiting periods
- Continuous coverage requirements
- Policy inception dates
Extract EXACT waiting periods in months/years and conditions.""",
        
        'maternity': """Focus on finding information about:
- Maternity benefits coverage
- Eligibility requirements
- Continuous coverage periods
- Benefit limitations
- Delivery/termination limits
Extract specific coverage terms and limitations.""",
        
        'coverage': """Focus on finding information about:
- What is covered under the policy
- Benefit limits and sub-limits
- Eligibility criteria
- Coverage conditions
Extract specific coverage details and amounts.""",
        
        'exclusions': """Focus on finding information about:
- What is specifically excluded
- Conditions not covered
- Limitations and restrictions
- Waiting period exclusions
List all exclusions mentioned in the document."""
    }
    
    @classmethod
    def get_optimized_prompt(cls, query: str, context: str) -> str:
        """Generate optimized prompt based on query type"""
        
        # Determine query type
        query_lower = query.lower()
        query_type = 'general'
        
        if any(term in query_lower for term in ['grace', 'premium', 'due', 'payment']):
            query_type = 'grace_period'
        elif any(term in query_lower for term in ['waiting', 'pre-existing', 'ped']):
            query_type = 'waiting_period'
        elif any(term in query_lower for term in ['maternity', 'pregnancy', 'childbirth']):
            query_type = 'maternity'
        elif any(term in query_lower for term in ['cover', 'benefit', 'include']):
            query_type = 'coverage'
        elif any(term in query_lower for term in ['exclude', 'not cover', 'exception']):
            query_type = 'exclusions'
        
        specific_instruction = cls.QUERY_SPECIFIC_PROMPTS.get(query_type, "")
        
        return f"""{cls.SYSTEM_PROMPT}

{specific_instruction}

POLICY DOCUMENT:
{context}

QUESTION: {query}

ANALYSIS AND ANSWER:"""

class OptimizedGeminiProcessor:
    """Enhanced Gemini processor with improved accuracy"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Slightly higher for better reasoning
                max_output_tokens=800,  # More tokens for detailed answers
                top_p=0.8,
                top_k=40
            )
        )
    
    async def process_query(self, prompt: str) -> AIResponse:
        start_time = time.time()
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            processing_time = time.time() - start_time
            
            if response and response.text:
                answer = response.text.strip()
                confidence = self._calculate_confidence(answer, prompt)
                
                return AIResponse(
                    answer=answer,
                    confidence=confidence,
                    model_name="gemini-2.0-flash",
                    processing_time=processing_time,
                    reasoning="Optimized Gemini with enhanced prompt engineering"
                )
            else:
                return AIResponse("No response generated", 0.0, "gemini-2.0-flash", processing_time, "Empty response")
                
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return AIResponse(f"Error: {str(e)[:100]}", 0.0, "gemini-2.0-flash", time.time() - start_time, f"Error: {e}")
    
    def _calculate_confidence(self, answer: str, prompt: str) -> float:
        """Enhanced confidence calculation"""
        if not answer or "error" in answer.lower():
            return 0.0
        
        answer_lower = answer.lower()
        
        # Check for uncertainty indicators
        uncertainty_phrases = [
            'not provided', 'not mentioned', 'cannot find', 'insufficient information',
            'no information', 'document does not specify', 'unclear', 'not stated',
            'may be', 'might be', 'possibly', 'appears to', 'seems to'
        ]
        
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            if 'document does not specify' in answer_lower:
                return 0.6  # Higher confidence for explicit acknowledgment
            return 0.3
        
        confidence = 0.7  # Base confidence
        
        # Boost for specific information patterns
        specific_patterns = [
            (r'grace period of \d+ days', 0.2),
            (r'waiting period of \d+ (?:months?|years?)', 0.2),
            (r'continuously covered for (?:at least )?\d+ months?', 0.15),
            (r'sum insured (?:of )?₹[\d,]+', 0.1),
            (r'limited to \d+ deliveries?', 0.1),
            (r'₹[\d,]+ (?:per|each)', 0.1),
            (r'as per (?:section|clause|table)', 0.1),
            (r'provided (?:that|the)', 0.05)
        ]
        
        for pattern, boost in specific_patterns:
            if re.search(pattern, answer_lower):
                confidence += boost
        
        # Boost for definitive language
        definitive_phrases = [
            'the policy covers', 'the policy excludes', 'the grace period is',
            'waiting period is', 'benefit is limited to', 'as specified in'
        ]
        
        for phrase in definitive_phrases:
            if phrase in answer_lower:
                confidence += 0.08
                break
        
        # Boost for structured answers
        if len(answer) > 50 and any(char in answer for char in ['.', ',']):
            confidence += 0.05
        
        # Penalty for very short answers (unless it's a clear "No" or "Yes")
        if len(answer) < 30 and not any(word in answer_lower for word in ['yes', 'no', 'not covered', 'excluded']):
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)

class OptimizedGroqProcessor:
    """Enhanced Groq processor with improved accuracy"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    async def process_query(self, prompt: str) -> AIResponse:
        start_time = time.time()
        
        try:
            # Split prompt for better Groq performance
            system_content = OptimizedInsurancePrompts.SYSTEM_PROMPT
            user_content = prompt.replace(OptimizedInsurancePrompts.SYSTEM_PROMPT, "").strip()
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            response = await asyncio.to_thread(
                self._create_completion, messages
            )
            
            processing_time = time.time() - start_time
            
            if response and response.choices:
                answer = response.choices[0].message.content.strip()
                confidence = self._calculate_confidence(answer, prompt)
                
                return AIResponse(
                    answer=answer,
                    confidence=confidence,
                    model_name="llama-3.3-70b-versatile",
                    processing_time=processing_time,
                    reasoning="Optimized Groq with enhanced system prompts"
                )
            else:
                return AIResponse("No response generated", 0.0, "llama-3.3-70b-versatile", processing_time, "Empty response")
                
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return AIResponse(f"Error: {str(e)[:100]}", 0.0, "llama-3.3-70b-versatile", time.time() - start_time, f"Error: {e}")
    
    def _create_completion(self, messages):
        return self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=800
        )
    
    def _calculate_confidence(self, answer: str, prompt: str) -> float:
        """Enhanced confidence calculation for Groq"""
        if not answer or "error" in answer.lower():
            return 0.0
        
        answer_lower = answer.lower()
        
        # Check for uncertainty
        if any(phrase in answer_lower for phrase in [
            'not provided', 'not mentioned', 'cannot find', 'insufficient information',
            'document does not specify'
        ]):
            if 'document does not specify' in answer_lower:
                return 0.65  # Higher for explicit acknowledgment
            return 0.25
        
        confidence = 0.65  # Base confidence for Groq
        
        # Boost for insurance-specific content
        insurance_terms = [
            'policy', 'coverage', 'premium', 'deductible', 'claim',
            'beneficiary', 'exclusion', 'waiting period', 'grace period'
        ]
        
        term_count = sum(1 for term in insurance_terms if term in answer_lower)
        confidence += min(term_count * 0.03, 0.15)
        
        # Boost for numerical information
        if re.search(r'\d+\s*(?:days?|months?|years?)', answer):
            confidence += 0.15
        
        if re.search(r'₹[\d,]+', answer):
            confidence += 0.1
        
        # Boost for structured response
        if len(answer) > 40:
            confidence += 0.08
        
        return min(max(confidence, 0.0), 1.0)

class EnhancedAIProcessor:
    """Enhanced AI Processor with improved accuracy algorithms"""
    
    def __init__(self):
        self.gemini = OptimizedGeminiProcessor(settings.GEMINI_API_KEY)
        self.groq = OptimizedGroqProcessor(settings.GROQ_API_KEY)
        self.document_chunks = []
        self.document_indexed = False
        
        # Enhanced keyword patterns for better context selection
        self.enhanced_patterns = {
            'grace_period': [
                r'grace\s+period', r'premium.*due', r'payment.*deadline',
                r'\d+\s*days?.*grace', r'continuity.*benefits', r'late.*payment'
            ],
            'waiting_period': [
                r'waiting\s+period', r'pre[-\s]?existing', r'PED', r'continuous.*coverage',
                r'\d+\s*months?.*waiting', r'policy.*inception', r'first.*policy'
            ],
            'maternity': [
                r'maternity', r'pregnancy', r'childbirth', r'delivery',
                r'termination.*pregnancy', r'female.*insured', r'continuously.*covered.*months'
            ],
            'coverage': [
                r'covered', r'benefits?', r'sum.*insured', r'expenses?',
                r'reimburs', r'indemnif', r'eligible', r'include'
            ],
            'exclusions': [
                r'exclud', r'not.*covered?', r'except', r'shall.*not',
                r'limitation', r'restriction', r'not.*eligible'
            ]
        }
    
    async def initialize_document(self, document_text: str) -> bool:
        """Enhanced document initialization"""
        try:
            logger.info("🔄 Initializing document with enhanced processing...")
            
            # Create smart chunks with overlap
            self.document_chunks = self._create_smart_chunks(document_text)
            self.document_indexed = True
            
            logger.info(f"✅ Document initialized with {len(self.document_chunks)} smart chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Document initialization failed: {e}")
            return False
    
    def _create_smart_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create intelligent chunks with metadata"""
        # Normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by logical sections
        section_patterns = [
            r'\n\s*(?:section|clause|article)\s*\d+',
            r'\n\s*[A-Z\s]{10,}:',
            r'\n\s*\d+\.\s*[A-Z]',
            r'\n\s*[IVX]+\.\s*[A-Z]'
        ]
        
        sections = [text]  # Start with full text
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section, flags=re.IGNORECASE)
                new_sections.extend([part.strip() for part in parts if len(part.strip()) > 100])
            if new_sections:
                sections = new_sections
        
        chunks = []
        for i, section in enumerate(sections):
            if len(section) <= 1000:
                # Small section - keep as is
                chunk_data = {
                    'text': section,
                    'index': i,
                    'keywords': self._extract_keywords(section),
                    'type': self._classify_section(section)
                }
                chunks.append(chunk_data)
            else:
                # Large section - split with overlap
                words = section.split()
                for j in range(0, len(words), 300):
                    chunk_words = words[max(0, j-50):j+400]  # 50 word overlap
                    chunk_text = ' '.join(chunk_words)
                    
                    if len(chunk_text.strip()) > 100:
                        chunk_data = {
                            'text': chunk_text,
                            'index': f"{i}_{j//300}",
                            'keywords': self._extract_keywords(chunk_text),
                            'type': self._classify_section(chunk_text)
                        }
                        chunks.append(chunk_data)
        
        return chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        text_lower = text.lower()
        keywords = []
        
        # Insurance-specific terms
        insurance_terms = [
            'grace period', 'waiting period', 'pre-existing', 'maternity',
            'coverage', 'exclusion', 'benefit', 'premium', 'deductible',
            'claim', 'hospital', 'treatment', 'surgery', 'diagnosis'
        ]
        
        for term in insurance_terms:
            if term in text_lower:
                keywords.append(term)
        
        # Extract numbers with units
        number_patterns = [
            r'\d+\s*days?', r'\d+\s*months?', r'\d+\s*years?',
            r'₹[\d,]+', r'\d+%', r'\d+\s*lakhs?'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        
        return list(set(keywords))
    
    def _classify_section(self, text: str) -> str:
        """Classify section type"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['definition', 'meaning', 'means']):
            return 'definitions'
        elif any(term in text_lower for term in ['exclusion', 'not covered', 'except']):
            return 'exclusions'
        elif any(term in text_lower for term in ['benefit', 'coverage', 'covered']):
            return 'benefits'
        elif any(term in text_lower for term in ['waiting period', 'grace period']):
            return 'periods'
        elif any(term in text_lower for term in ['claim', 'procedure', 'process']):
            return 'claims'
        else:
            return 'general'
    
    async def process_query(self, document: str, query: str) -> Tuple[str, float]:
        """Enhanced query processing with improved accuracy"""
        try:
            if not self.document_indexed:
                success = await self.initialize_document(document)
                if not success:
                    return "Failed to process document", 0.0
            
            # Get highly relevant context
            context = self._get_smart_context(query)
            
            if not context:
                return "No relevant information found in the document", 0.15
            
            # Generate optimized prompt
            prompt = OptimizedInsurancePrompts.get_optimized_prompt(query, context)
            
            # Process with both models
            logger.info(f"🔍 Processing query with enhanced dual AI models")
            
            gemini_task = self.gemini.process_query(prompt)
            groq_task = self.groq.process_query(prompt)
            
            gemini_response, groq_response = await asyncio.gather(
                gemini_task, groq_task, return_exceptions=True
            )
            
            # Enhanced consensus
            final_answer, final_confidence = self._enhanced_consensus(
                gemini_response, groq_response, query
            )
            
            logger.info(f"✅ Enhanced processing complete: confidence={final_confidence:.3f}")
            return final_answer, final_confidence
            
        except Exception as e:
            logger.error(f"❌ Enhanced processing failed: {e}")
            return f"Processing error: {str(e)[:100]}", 0.0
    
    def _get_smart_context(self, query: str) -> str:
        """Get highly relevant context using smart matching"""
        if not self.document_chunks:
            return ""
        
        query_lower = query.lower()
        
        # Determine query category
        query_category = 'general'
        for category, patterns in self.enhanced_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                query_category = category
                break
        
        # Score chunks
        chunk_scores = []
        for chunk in self.document_chunks:
            score = 0
            chunk_text_lower = chunk['text'].lower()
            
            # Category-specific pattern matching
            if query_category in self.enhanced_patterns:
                for pattern in self.enhanced_patterns[query_category]:
                    matches = len(re.findall(pattern, chunk_text_lower))
                    score += matches * 10
            
            # Keyword matching from extracted keywords
            query_words = set(query_lower.split())
            chunk_keywords = set([kw.lower() for kw in chunk.get('keywords', [])])
            keyword_overlap = len(query_words.intersection(chunk_keywords))
            score += keyword_overlap * 5
            
            # Section type relevance
            section_type = chunk.get('type', 'general')
            type_boosts = {
                'periods': ['grace', 'waiting', 'period'],
                'benefits': ['cover', 'benefit', 'include'],
                'exclusions': ['exclude', 'not cover', 'except'],
                'definitions': ['define', 'meaning', 'means']
            }
            
            for boost_type, boost_words in type_boosts.items():
                if section_type == boost_type and any(word in query_lower for word in boost_words):
                    score += 15
            
            # Word frequency matching
            for word in query_words:
                if len(word) > 3:
                    score += chunk_text_lower.count(word) * 2
            
            chunk_scores.append((chunk, score))
        
        # Select top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        selected_chunks = [chunk for chunk, score in chunk_scores[:4] if score > 0]
        
        if not selected_chunks:
            # Fallback to first chunks
            selected_chunks = self.document_chunks[:3]
        
        # Combine context
        context_parts = []
        for chunk in selected_chunks:
            context_parts.append(f"[Section {chunk['index']}]\n{chunk['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Limit context size
        if len(context) > 8000:
            context = context[:8000] + "\n\n[Context truncated for length]"
        
        return context
    
    def _enhanced_consensus(self, gemini_response, groq_response, query: str) -> Tuple[str, float]:
        """Enhanced consensus algorithm"""
        valid_responses = []
        
        if isinstance(gemini_response, AIResponse) and gemini_response.confidence > 0:
            valid_responses.append(gemini_response)
        
        if isinstance(groq_response, AIResponse) and groq_response.confidence > 0:
            valid_responses.append(groq_response)
        
        if not valid_responses:
            return "Unable to generate accurate answer from the provided document", 0.0
        
        if len(valid_responses) == 1:
            response = valid_responses[0]
            return response.answer, response.confidence
        
        # Enhanced dual response analysis
        gemini_resp = valid_responses[0] if valid_responses[0].model_name.startswith('gemini') else valid_responses[1]
        groq_resp = valid_responses[1] if valid_responses[0].model_name.startswith('gemini') else valid_responses[0]
        
        # Check for agreement indicators
        agreement_score = self._calculate_agreement(gemini_resp.answer, groq_resp.answer)
        
        logger.info(f"Agreement score: {agreement_score:.3f}")
        
        # Select best response based on multiple factors
        gemini_score = self._calculate_response_quality(gemini_resp.answer, gemini_resp.confidence, query)
        groq_score = self._calculate_response_quality(groq_resp.answer, groq_resp.confidence, query)
        
        if agreement_score > 0.6:
            # High agreement - boost confidence and select better response
            if gemini_score >= groq_score:
                final_confidence = min(gemini_resp.confidence + 0.15, 1.0)
                return gemini_resp.answer, final_confidence
            else:
                final_confidence = min(groq_resp.confidence + 0.15, 1.0)
                return groq_resp.answer, final_confidence
        
        elif gemini_score > groq_score * 1.2:
            # Gemini significantly better
            return gemini_resp.answer, gemini_resp.confidence
        
        elif groq_score > gemini_score * 1.2:
            # Groq significantly better
            return groq_resp.answer, groq_resp.confidence
        
        else:
            # Close call - select based on confidence with slight preference for more detailed answer
            if abs(gemini_resp.confidence - groq_resp.confidence) < 0.1:
                # Similar confidence - prefer longer, more detailed answer
                if len(gemini_resp.answer) > len(groq_resp.answer):
                    return gemini_resp.answer, gemini_resp.confidence
                else:
                    return groq_resp.answer, groq_resp.confidence
            else:
                # Different confidence - select higher confidence
                if gemini_resp.confidence > groq_resp.confidence:
                    return gemini_resp.answer, gemini_resp.confidence
                else:
                    return groq_resp.answer, groq_resp.confidence
    
    def _calculate_agreement(self, answer1: str, answer2: str) -> float:
        """Calculate agreement between two answers"""
        # Extract key information
        numbers1 = set(re.findall(r'\d+', answer1))
        numbers2 = set(re.findall(r'\d+', answer2))
        
        # Number agreement
        number_agreement = 0
        if numbers1 and numbers2:
            common_numbers = numbers1.intersection(numbers2)
            total_numbers = numbers1.union(numbers2)
            number_agreement = len(common_numbers) / len(total_numbers) if total_numbers else 0
        
        # Word overlap
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            word_overlap = 0
        
        # Key phrase agreement
        key_phrases = [
            'grace period', 'waiting period', 'covered', 'excluded', 'not covered',
            'premium', 'benefit', 'sum insured', 'conditions', 'limitations'
        ]
        
        phrase_agreement = 0
        common_phrases = 0
        for phrase in key_phrases:
            if phrase in answer1.lower() and phrase in answer2.lower():
                common_phrases += 1
        
        if common_phrases > 0:
            phrase_agreement = common_phrases / len([p for p in key_phrases if p in answer1.lower() or p in answer2.lower()])
        
        # Combine scores
        final_agreement = (number_agreement * 0.4 + word_overlap * 0.4 + phrase_agreement * 0.2)
        return final_agreement
    
    def _calculate_response_quality(self, answer: str, confidence: float, query: str) -> float:
        """Calculate overall response quality score"""
        answer_lower = answer.lower()
        
        # Base score from confidence
        score = confidence
        
        # Penalize non-informative responses
        if any(phrase in answer_lower for phrase in [
            'not provided', 'not mentioned', 'cannot find', 'insufficient information'
        ]):
            if 'document does not specify' in answer_lower:
                score *= 0.8  # Less penalty for explicit acknowledgment
            else:
                score *= 0.4
        
        # Boost for specific information
        if re.search(r'\d+\s*(?:days?|months?|years?)', answer):
            score += 0.1
        
        if re.search(r'₹[\d,]+', answer):
            score += 0.05
        
        # Boost for query relevance
        query_words = set(query.lower().split())
        answer_words = set(answer_lower.split())
        relevance = len(query_words.intersection(answer_words)) / len(query_words) if query_words else 0
        score += relevance * 0.1
        
        # Boost for comprehensive answers
        if len(answer) > 100:
            score += 0.05
        
        return min(score, 1.0)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        return {
            "document_indexed": self.document_indexed,
            "total_chunks": len(self.document_chunks),
            "models_available": {
                "gemini": bool(settings.GEMINI_API_KEY),
                "groq": bool(settings.GROQ_API_KEY)
            },
            "optimization_level": "ENHANCED_90_PERCENT_TARGET"
        }

# Global instance
enhanced_ai = EnhancedAIProcessor()

async def process_query(document: str, query: str) -> Tuple[str, float]:
    """Main function for enhanced AI processing with 90%+ accuracy target"""
    return await enhanced_ai.process_query(document, query)