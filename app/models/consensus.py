"""
Advanced Consensus Engine for Multi-Model AI Responses
Implements sophisticated voting and confidence weighting algorithms
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ConsensusResult:
    """Result of consensus analysis"""
    final_answer: str
    final_confidence: float
    consensus_method: str
    agreement_score: float
    reasoning: str
    model_contributions: Dict[str, float]

class ConsensusEngine:
    """Advanced consensus engine with multiple algorithms"""
    
    def __init__(self):
        self.similarity_threshold = 0.7
        self.confidence_threshold = 0.5
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        
        # Insurance-specific negative indicators
        self.negative_indicators = [
            'not provided', 'not mentioned', 'cannot find', 'no information',
            'insufficient information', 'not specified', 'unclear',
            'not available', 'not stated', 'does not contain'
        ]
        
        # Quality indicators for insurance responses
        self.quality_indicators = [
            'covered', 'excluded', 'benefit', 'premium', 'waiting period',
            'grace period', 'deductible', 'co-payment', 'maternity',
            'pre-existing', 'surgery', 'treatment', 'diagnosis',
            'hospital', 'medical', 'policy', 'clause', 'section'
        ]
    
    def find_consensus(
        self, 
        responses: List[Tuple[str, float, str]], 
        query: str = "",
        method: str = "auto"
    ) -> ConsensusResult:
        """
        Find consensus among multiple AI responses
        
        Args:
            responses: List of (answer, confidence, model_name) tuples
            query: Original query for context
            method: Consensus method ('auto', 'weighted', 'similarity', 'voting')
        """
        if not responses:
            return ConsensusResult(
                final_answer="No responses available",
                final_confidence=0.0,
                consensus_method="none",
                agreement_score=0.0,
                reasoning="No valid responses to process",
                model_contributions={}
            )
        
        if len(responses) == 1:
            answer, confidence, model = responses[0]
            return ConsensusResult(
                final_answer=answer,
                final_confidence=confidence,
                consensus_method="single",
                agreement_score=1.0,
                reasoning=f"Single response from {model}",
                model_contributions={model: 1.0}
            )
        
        # Filter and categorize responses
        filtered_responses = self._filter_responses(responses)
        
        if method == "auto":
            method = self._select_best_method(filtered_responses, query)
        
        # Apply selected consensus method
        if method == "weighted":
            return self._weighted_consensus(filtered_responses, query)
        elif method == "similarity":
            return self._similarity_consensus(filtered_responses, query)
        elif method == "voting":
            return self._voting_consensus(filtered_responses, query)
        else:
            return self._adaptive_consensus(filtered_responses, query)
    
    def _filter_responses(self, responses: List[Tuple[str, float, str]]) -> Dict[str, List[Tuple[str, float, str]]]:
        """Categorize responses by quality"""
        high_quality = []
        medium_quality = []
        low_quality = []
        
        for answer, confidence, model in responses:
            answer_lower = answer.lower()
            
            # Check for negative indicators
            has_negative = any(indicator in answer_lower for indicator in self.negative_indicators)
            
            # Check for quality indicators
            quality_count = sum(1 for indicator in self.quality_indicators if indicator in answer_lower)
            
            # Check for numerical information (important in insurance)
            has_numbers = bool(re.search(r'\d+', answer))
            
            # Categorize based on multiple factors
            if has_negative and confidence < 0.4:
                low_quality.append((answer, confidence, model))
            elif quality_count >= 3 or (has_numbers and confidence > 0.6):
                high_quality.append((answer, confidence, model))
            else:
                medium_quality.append((answer, confidence, model))
        
        return {
            'high': high_quality,
            'medium': medium_quality,
            'low': low_quality
        }
    
    def _select_best_method(self, filtered_responses: Dict, query: str) -> str:
        """Select the best consensus method based on response characteristics"""
        high_quality = filtered_responses['high']
        medium_quality = filtered_responses['medium']
        total_responses = len(high_quality) + len(medium_quality) + len(filtered_responses['low'])
        
        # If we have multiple high-quality responses, use similarity
        if len(high_quality) >= 2:
            return "similarity"
        
        # If we have mixed quality, use weighted approach
        if len(high_quality) >= 1 and len(medium_quality) >= 1:
            return "weighted"
        
        # If responses are sparse, use voting
        if total_responses <= 2:
            return "voting"
        
        # Default to adaptive
        return "adaptive"
    
    def _weighted_consensus(self, filtered_responses: Dict, query: str) -> ConsensusResult:
        """Weighted consensus based on confidence and quality"""
        all_responses = filtered_responses['high'] + filtered_responses['medium'] + filtered_responses['low']
        
        if not all_responses:
            return self._create_empty_result("weighted")
        
        # Calculate weights based on quality tier and confidence
        weighted_responses = []
        total_weight = 0
        
        for answer, confidence, model in all_responses:
            # Base weight from confidence
            weight = confidence
            
            # Quality tier multiplier
            if (answer, confidence, model) in filtered_responses['high']:
                weight *= 1.5
            elif (answer, confidence, model) in filtered_responses['medium']:
                weight *= 1.0
            else:
                weight *= 0.5
            
            # Query relevance boost
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            relevance = len(query_words.intersection(answer_words)) / len(query_words) if query_words else 0
            weight *= (1 + relevance * 0.3)
            
            weighted_responses.append((answer, weight, model))
            total_weight += weight
        
        # Select highest weighted response
        best_response = max(weighted_responses, key=lambda x: x[1])
        final_answer, final_weight, best_model = best_response
        
        # Calculate final confidence
        final_confidence = min(final_weight / total_weight * len(weighted_responses), 1.0)
        
        # Calculate model contributions
        model_contributions = {}
        for answer, weight, model in weighted_responses:
            model_contributions[model] = weight / total_weight
        
        return ConsensusResult(
            final_answer=final_answer,
            final_confidence=final_confidence,
            consensus_method="weighted",
            agreement_score=final_weight / total_weight,
            reasoning=f"Selected highest weighted response from {best_model} (weight: {final_weight:.3f})",
            model_contributions=model_contributions
        )
    
    def _similarity_consensus(self, filtered_responses: Dict, query: str) -> ConsensusResult:
        """Consensus based on answer similarity"""
        # Prioritize high-quality responses
        responses_to_analyze = filtered_responses['high']
        if len(responses_to_analyze) < 2:
            responses_to_analyze = filtered_responses['high'] + filtered_responses['medium']
        
        if len(responses_to_analyze) < 2:
            # Fall back to single best response
            all_responses = filtered_responses['high'] + filtered_responses['medium'] + filtered_responses['low']
            if all_responses:
                best = max(all_responses, key=lambda x: x[1])
                return self._create_single_result(best, "similarity")
            return self._create_empty_result("similarity")
        
        try:
            # Extract answers for similarity calculation
            answers = [resp[0] for resp in responses_to_analyze]
            
            # Calculate TF-IDF similarity matrix
            tfidf_matrix = self.vectorizer.fit_transform(answers)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find groups of similar answers
            similar_groups = self._find_similarity_groups(similarity_matrix, responses_to_analyze)
            
            if similar_groups:
                # Select the group with highest combined confidence
                best_group = max(similar_groups, key=lambda g: sum(resp[1] for resp in g['responses']))
                
                # Within the best group, select highest confidence response
                best_response = max(best_group['responses'], key=lambda x: x[1])
                
                # Boost confidence due to agreement
                agreement_boost = min(len(best_group['responses']) * 0.05, 0.2)
                boosted_confidence = min(best_response[1] + agreement_boost, 1.0)
                
                model_contributions = {}
                for _, _, model in best_group['responses']:
                    model_contributions[model] = 1.0 / len(best_group['responses'])
                
                return ConsensusResult(
                    final_answer=best_response[0],
                    final_confidence=boosted_confidence,
                    consensus_method="similarity",
                    agreement_score=best_group['avg_similarity'],
                    reasoning=f"Selected from group of {len(best_group['responses'])} similar responses (avg similarity: {best_group['avg_similarity']:.3f})",
                    model_contributions=model_contributions
                )
            
            # No similar groups found, fall back to highest confidence
            best_response = max(responses_to_analyze, key=lambda x: x[1])
            return self._create_single_result(best_response, "similarity")
            
        except Exception as e:
            logger.error(f"Similarity consensus failed: {e}")
            # Fall back to highest confidence
            all_responses = filtered_responses['high'] + filtered_responses['medium'] + filtered_responses['low']
            if all_responses:
                best = max(all_responses, key=lambda x: x[1])
                return self._create_single_result(best, "similarity")
            return self._create_empty_result("similarity")
    
    def _voting_consensus(self, filtered_responses: Dict, query: str) -> ConsensusResult:
        """Simple voting consensus"""
        all_responses = filtered_responses['high'] + filtered_responses['medium'] + filtered_responses['low']
        
        if not all_responses:
            return self._create_empty_result("voting")
        
        # Simple approach: return highest confidence response
        # In a real voting system, you might implement answer clustering
        best_response = max(all_responses, key=lambda x: x[1])
        
        # Check if multiple responses agree (simple word overlap)
        agreements = 0
        best_answer_words = set(best_response[0].lower().split())
        
        for answer, _, _ in all_responses:
            answer_words = set(answer.lower().split())
            overlap = len(best_answer_words.intersection(answer_words))
            if overlap > len(best_answer_words) * 0.3:  # 30% word overlap
                agreements += 1
        
        agreement_score = agreements / len(all_responses)
        
        # Boost confidence if there's agreement
        boosted_confidence = best_response[1]
        if agreement_score > 0.5:
            boosted_confidence = min(boosted_confidence + 0.1, 1.0)
        
        model_contributions = {best_response[2]: 1.0}
        
        return ConsensusResult(
            final_answer=best_response[0],
            final_confidence=boosted_confidence,
            consensus_method="voting",
            agreement_score=agreement_score,
            reasoning=f"Selected highest confidence response with {agreement_score:.1%} agreement",
            model_contributions=model_contributions
        )
    
    def _adaptive_consensus(self, filtered_responses: Dict, query: str) -> ConsensusResult:
        """Adaptive consensus combining multiple methods"""
        # Try similarity first if we have enough responses
        high_medium = filtered_responses['high'] + filtered_responses['medium']
        
        if len(high_medium) >= 2:
            similarity_result = self._similarity_consensus(filtered_responses, query)
            if similarity_result.agreement_score > 0.6:
                similarity_result.consensus_method = "adaptive(similarity)"
                return similarity_result
        
        # Fall back to weighted consensus
        weighted_result = self._weighted_consensus(filtered_responses, query)
        weighted_result.consensus_method = "adaptive(weighted)"
        return weighted_result
    
    def _find_similarity_groups(self, similarity_matrix: np.ndarray, responses: List) -> List[Dict]:
        """Find groups of similar responses"""
        n = len(responses)
        visited = [False] * n
        groups = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Start a new group
            group_responses = [responses[i]]
            group_similarities = []
            visited[i] = True
            
            # Find similar responses
            for j in range(i + 1, n):
                if not visited[j] and similarity_matrix[i][j] > self.similarity_threshold:
                    group_responses.append(responses[j])
                    group_similarities.append(similarity_matrix[i][j])
                    visited[j] = True
            
            # Only keep groups with multiple responses
            if len(group_responses) > 1:
                avg_similarity = np.mean(group_similarities) if group_similarities else 0
                groups.append({
                    'responses': group_responses,
                    'avg_similarity': avg_similarity
                })
        
        return groups
    
    def _create_single_result(self, response: Tuple[str, float, str], method: str) -> ConsensusResult:
        """Create result for single response"""
        answer, confidence, model = response
        return ConsensusResult(
            final_answer=answer,
            final_confidence=confidence,
            consensus_method=method,
            agreement_score=1.0,
            reasoning=f"Single response from {model}",
            model_contributions={model: 1.0}
        )
    
    def _create_empty_result(self, method: str) -> ConsensusResult:
        """Create empty result"""
        return ConsensusResult(
            final_answer="No valid responses available",
            final_confidence=0.0,
            consensus_method=method,
            agreement_score=0.0,
            reasoning="No responses to process",
            model_contributions={}
        )

# Legacy compatibility function for existing code
def aggregate_responses(responses: List[Tuple[str, float]]) -> Tuple[str, float]:
    """Legacy function for backward compatibility"""
    if not responses:
        return "No responses available", 0.0
    
    # Convert to format expected by ConsensusEngine
    converted_responses = [(answer, confidence, f"model_{i}") for i, (answer, confidence) in enumerate(responses)]
    
    engine = ConsensusEngine()
    result = engine.find_consensus(converted_responses)
    
    return result.final_answer, result.final_confidence