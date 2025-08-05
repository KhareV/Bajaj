"""
Enhanced Consensus Engine for Insurance AI Responses
Optimized algorithms for better accuracy and decision making
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ConsensusResult:
    """Enhanced result of consensus analysis"""
    final_answer: str
    final_confidence: float
    consensus_method: str
    agreement_score: float
    reasoning: str
    model_contributions: Dict[str, float]
    quality_metrics: Dict[str, float]

class EnhancedConsensusEngine:
    """Enhanced consensus engine optimized for insurance responses"""
    
    def __init__(self):
        self.confidence_threshold = 0.4
        self.agreement_threshold = 0.6
        
        # Insurance-specific quality indicators
        self.quality_indicators = {
            'high_value': [
                'grace period of', 'waiting period of', 'sum insured of',
                'covered up to', 'limited to', 'as per section',
                'provided that', 'subject to', 'in accordance with'
            ],
            'medium_value': [
                'covered', 'excluded', 'benefit', 'premium', 'deductible',
                'hospital', 'treatment', 'surgery', 'diagnosis', 'policy'
            ],
            'negative_indicators': [
                'not provided', 'not mentioned', 'cannot find', 'no information',
                'insufficient information', 'unclear', 'not specified',
                'not available', 'not stated', 'may be', 'might be', 'possibly'
            ],
            'uncertainty_phrases': [
                'appears to', 'seems to', 'likely', 'probably', 'perhaps',
                'could be', 'should be', 'generally', 'typically'
            ]
        }
        
        # Numerical pattern weights for insurance
        self.numerical_patterns = {
            r'\d+\s*days?': 3.0,           # Grace period days
            r'\d+\s*months?': 3.0,         # Waiting periods
            r'\d+\s*years?': 2.5,          # Long-term periods
            r'₹[\d,]+': 2.0,               # Currency amounts
            r'\d+%': 2.0,                  # Percentages
            r'\d+\s*lakhs?': 2.0,          # Large amounts
        }
    
    def find_consensus(
        self, 
        responses: List[Tuple[str, float, str]], 
        query: str = "",
        method: str = "enhanced_auto"
    ) -> ConsensusResult:
        """
        Enhanced consensus finding with insurance-specific logic
        """
        if not responses:
            return self._create_empty_result("none")
        
        if len(responses) == 1:
            return self._create_single_result(responses[0], "single")
        
        # Enhanced response analysis
        analyzed_responses = self._analyze_responses(responses, query)
        
        # Select optimal consensus method
        if method == "enhanced_auto":
            method = self._select_optimal_method(analyzed_responses, query)
        
        # Apply selected method
        if method == "quality_weighted":
            return self._quality_weighted_consensus(analyzed_responses, query)
        elif method == "agreement_based":
            return self._agreement_based_consensus(analyzed_responses, query)
        elif method == "confidence_tiered":
            return self._confidence_tiered_consensus(analyzed_responses, query)
        else:
            return self._adaptive_consensus(analyzed_responses, query)
    
    def _analyze_responses(self, responses: List[Tuple[str, float, str]], query: str) -> List[Dict]:
        """Comprehensive response analysis"""
        analyzed = []
        
        for answer, confidence, model in responses:
            analysis = {
                'answer': answer,
                'confidence': confidence,
                'model': model,
                'quality_score': self._calculate_quality_score(answer),
                'specificity_score': self._calculate_specificity_score(answer),
                'relevance_score': self._calculate_relevance_score(answer, query),
                'uncertainty_penalty': self._calculate_uncertainty_penalty(answer),
                'numerical_content': self._extract_numerical_content(answer),
                'length_factor': self._calculate_length_factor(answer)
            }
            
            # Calculate composite score
            analysis['composite_score'] = self._calculate_composite_score(analysis)
            analyzed.append(analysis)
        
        return analyzed
    
    def _calculate_quality_score(self, answer: str) -> float:
        """Calculate response quality based on insurance-specific indicators"""
        answer_lower = answer.lower()
        score = 0.5  # Base score
        
        # Check for high-value phrases
        for phrase in self.quality_indicators['high_value']:
            if phrase in answer_lower:
                score += 0.15
        
        # Check for medium-value phrases
        medium_count = sum(1 for phrase in self.quality_indicators['medium_value'] if phrase in answer_lower)
        score += min(medium_count * 0.05, 0.2)
        
        # Penalty for negative indicators
        negative_count = sum(1 for phrase in self.quality_indicators['negative_indicators'] if phrase in answer_lower)
        score -= negative_count * 0.2
        
        # Penalty for uncertainty
        uncertainty_count = sum(1 for phrase in self.quality_indicators['uncertainty_phrases'] if phrase in answer_lower)
        score -= uncertainty_count * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_specificity_score(self, answer: str) -> float:
        """Calculate how specific/detailed the answer is"""
        score = 0.3  # Base score
        
        # Boost for numerical information
        for pattern, weight in self.numerical_patterns.items():
            matches = len(re.findall(pattern, answer, re.IGNORECASE))
            score += matches * weight * 0.05
        
        # Boost for specific terms
        specific_terms = [
            'section', 'clause', 'paragraph', 'table', 'schedule',
            'annexure', 'appendix', 'terms and conditions'
        ]
        
        for term in specific_terms:
            if term in answer.lower():
                score += 0.08
        
        # Boost for definitive language
        definitive_phrases = [
            'is defined as', 'shall be', 'must be', 'is required',
            'is covered', 'is excluded', 'is limited to'
        ]
        
        for phrase in definitive_phrases:
            if phrase in answer.lower():
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_relevance_score(self, answer: str, query: str) -> float:
        """Calculate relevance to the original query"""
        if not query:
            return 0.5
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= common_words
        answer_words -= common_words
        
        if not query_words:
            return 0.5
        
        # Calculate word overlap
        overlap = len(query_words.intersection(answer_words))
        relevance = overlap / len(query_words)
        
        # Boost for exact phrase matches
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        key_phrases = ['grace period', 'waiting period', 'maternity', 'coverage', 'exclusion']
        for phrase in key_phrases:
            if phrase in query_lower and phrase in answer_lower:
                relevance += 0.2
        
        return min(1.0, relevance)
    
    def _calculate_uncertainty_penalty(self, answer: str) -> float:
        """Calculate penalty for uncertain language"""
        answer_lower = answer.lower()
        penalty = 0.0
        
        # Strong uncertainty indicators
        strong_uncertainty = ['not provided', 'not mentioned', 'cannot find', 'no information']
        for phrase in strong_uncertainty:
            if phrase in answer_lower:
                penalty += 0.3
        
        # Mild uncertainty indicators
        mild_uncertainty = ['may be', 'might be', 'possibly', 'appears to', 'seems to']
        for phrase in mild_uncertainty:
            if phrase in answer_lower:
                penalty += 0.1
        
        # Hedge words
        hedge_words = ['generally', 'typically', 'usually', 'often', 'sometimes']
        hedge_count = sum(1 for word in hedge_words if word in answer_lower)
        penalty += hedge_count * 0.05
        
        return min(0.8, penalty)  # Cap penalty
    
    def _extract_numerical_content(self, answer: str) -> Dict[str, int]:
        """Extract and categorize numerical content"""
        content = {
            'days': len(re.findall(r'\d+\s*days?', answer, re.IGNORECASE)),
            'months': len(re.findall(r'\d+\s*months?', answer, re.IGNORECASE)),
            'years': len(re.findall(r'\d+\s*years?', answer, re.IGNORECASE)),
            'amounts': len(re.findall(r'₹[\d,]+', answer)),
            'percentages': len(re.findall(r'\d+%', answer)),
            'total_numbers': len(re.findall(r'\d+', answer))
        }
        return content
    
    def _calculate_length_factor(self, answer: str) -> float:
        """Calculate length appropriateness factor"""
        length = len(answer)
        
        if length < 20:
            return 0.3  # Too short
        elif length < 50:
            return 0.6  # Short but acceptable
        elif length < 200:
            return 1.0  # Good length
        elif length < 400:
            return 0.9  # Bit long but detailed
        else:
            return 0.7  # Too verbose
    
    def _calculate_composite_score(self, analysis: Dict) -> float:
        """Calculate composite score from all factors"""
        base_confidence = analysis['confidence']
        quality = analysis['quality_score']
        specificity = analysis['specificity_score']
        relevance = analysis['relevance_score']
        uncertainty_penalty = analysis['uncertainty_penalty']
        length_factor = analysis['length_factor']
        
        # Weighted combination
        composite = (
            base_confidence * 0.35 +
            quality * 0.25 +
            specificity * 0.20 +
            relevance * 0.15 +
            length_factor * 0.05
        ) - uncertainty_penalty
        
        return max(0.0, min(1.0, composite))
    
    def _select_optimal_method(self, analyzed_responses: List[Dict], query: str) -> str:
        """Select the best consensus method based on response characteristics"""
        if len(analyzed_responses) < 2:
            return "single"
        
        # Check quality distribution
        quality_scores = [resp['quality_score'] for resp in analyzed_responses]
        confidence_scores = [resp['confidence'] for resp in analyzed_responses]
        
        high_quality_count = sum(1 for score in quality_scores if score > 0.7)
        high_confidence_count = sum(1 for score in confidence_scores if score > 0.7)
        
        # Calculate agreement between top responses
        top_responses = sorted(analyzed_responses, key=lambda x: x['composite_score'], reverse=True)[:2]
        agreement = self._calculate_response_agreement(top_responses[0]['answer'], top_responses[1]['answer'])
        
        if high_quality_count >= 2 and agreement > 0.6:
            return "agreement_based"
        elif high_confidence_count >= 1:
            return "confidence_tiered"
        else:
            return "quality_weighted"
    
    def _quality_weighted_consensus(self, analyzed_responses: List[Dict], query: str) -> ConsensusResult:
        """Consensus based on quality-weighted scoring"""
        if not analyzed_responses:
            return self._create_empty_result("quality_weighted")
        
        # Sort by composite score
        sorted_responses = sorted(analyzed_responses, key=lambda x: x['composite_score'], reverse=True)
        best_response = sorted_responses[0]
        
        # Calculate model contributions
        total_score = sum(resp['composite_score'] for resp in analyzed_responses)
        model_contributions = {}
        for resp in analyzed_responses:
            model = resp['model']
            contribution = resp['composite_score'] / total_score if total_score > 0 else 1.0 / len(analyzed_responses)
            model_contributions[model] = model_contributions.get(model, 0) + contribution
        
        # Calculate final confidence with quality boost
        base_confidence = best_response['confidence']
        quality_boost = (best_response['quality_score'] - 0.5) * 0.2  # Up to 0.1 boost
        final_confidence = min(1.0, base_confidence + quality_boost)
        
        quality_metrics = {
            'quality_score': best_response['quality_score'],
            'specificity_score': best_response['specificity_score'],
            'relevance_score': best_response['relevance_score'],
            'composite_score': best_response['composite_score']
        }
        
        return ConsensusResult(
            final_answer=best_response['answer'],
            final_confidence=final_confidence,
            consensus_method="quality_weighted",
            agreement_score=best_response['composite_score'],
            reasoning=f"Selected highest quality response (composite score: {best_response['composite_score']:.3f})",
            model_contributions=model_contributions,
            quality_metrics=quality_metrics
        )
    
    def _agreement_based_consensus(self, analyzed_responses: List[Dict], query: str) -> ConsensusResult:
        """Consensus based on response agreement"""
        if len(analyzed_responses) < 2:
            return self._quality_weighted_consensus(analyzed_responses, query)
        
        # Find pairs with high agreement
        best_agreement = 0
        best_pair = None
        
        for i in range(len(analyzed_responses)):
            for j in range(i + 1, len(analyzed_responses)):
                agreement = self._calculate_response_agreement(
                    analyzed_responses[i]['answer'], 
                    analyzed_responses[j]['answer']
                )
                
                if agreement > best_agreement:
                    best_agreement = agreement
                    best_pair = (i, j)
        
        if best_pair and best_agreement > self.agreement_threshold:
            # Select better response from agreeing pair
            resp1, resp2 = analyzed_responses[best_pair[0]], analyzed_responses[best_pair[1]]
            
            if resp1['composite_score'] >= resp2['composite_score']:
                selected_response = resp1
            else:
                selected_response = resp2
            
            # Boost confidence due to agreement
            agreement_boost = min(best_agreement * 0.15, 0.2)
            final_confidence = min(1.0, selected_response['confidence'] + agreement_boost)
            
            model_contributions = {
                resp1['model']: 0.5,
                resp2['model']: 0.5
            }
            
            quality_metrics = {
                'agreement_score': best_agreement,
                'quality_score': selected_response['quality_score'],
                'composite_score': selected_response['composite_score']
            }
            
            return ConsensusResult(
                final_answer=selected_response['answer'],
                final_confidence=final_confidence,
                consensus_method="agreement_based",
                agreement_score=best_agreement,
                reasoning=f"Selected from agreeing responses (agreement: {best_agreement:.3f})",
                model_contributions=model_contributions,
                quality_metrics=quality_metrics
            )
        
        # No good agreement found, fall back to quality-weighted
        return self._quality_weighted_consensus(analyzed_responses, query)
    
    def _confidence_tiered_consensus(self, analyzed_responses: List[Dict], query: str) -> ConsensusResult:
        """Consensus based on confidence tiers"""
        if not analyzed_responses:
            return self._create_empty_result("confidence_tiered")
        
        # Group by confidence tiers
        high_confidence = [resp for resp in analyzed_responses if resp['confidence'] > 0.7]
        medium_confidence = [resp for resp in analyzed_responses if 0.4 <= resp['confidence'] <= 0.7]
        low_confidence = [resp for resp in analyzed_responses if resp['confidence'] < 0.4]
        
        # Select from highest available tier
        if high_confidence:
            candidates = high_confidence
            tier = "high"
        elif medium_confidence:
            candidates = medium_confidence
            tier = "medium"
        else:
            candidates = low_confidence
            tier = "low"
        
        # Within tier, select by composite score
        best_candidate = max(candidates, key=lambda x: x['composite_score'])
        
        # Calculate model contributions from tier
        model_contributions = {}
        for resp in candidates:
            model = resp['model']
            weight = resp['composite_score']
            model_contributions[model] = model_contributions.get(model, 0) + weight
        
        # Normalize contributions
        total_weight = sum(model_contributions.values())
        if total_weight > 0:
            model_contributions = {k: v/total_weight for k, v in model_contributions.items()}
        
        quality_metrics = {
            'confidence_tier': tier,
            'tier_size': len(candidates),
            'quality_score': best_candidate['quality_score'],
            'composite_score': best_candidate['composite_score']
        }
        
        return ConsensusResult(
            final_answer=best_candidate['answer'],
            final_confidence=best_candidate['confidence'],
            consensus_method="confidence_tiered",
            agreement_score=best_candidate['composite_score'],
            reasoning=f"Selected from {tier} confidence tier ({len(candidates)} candidates)",
            model_contributions=model_contributions,
            quality_metrics=quality_metrics
        )
    
    def _adaptive_consensus(self, analyzed_responses: List[Dict], query: str) -> ConsensusResult:
        """Adaptive consensus combining multiple strategies"""
        # Try agreement-based first if we have good candidates
        high_quality = [resp for resp in analyzed_responses if resp['quality_score'] > 0.6]
        
        if len(high_quality) >= 2:
            agreement_result = self._agreement_based_consensus(analyzed_responses, query)
            if agreement_result.agreement_score > 0.5:
                agreement_result.consensus_method = "adaptive(agreement)"
                return agreement_result
        
        # Fall back to confidence-tiered
        tiered_result = self._confidence_tiered_consensus(analyzed_responses, query)
        tiered_result.consensus_method = "adaptive(tiered)"
        return tiered_result
    
    def _calculate_response_agreement(self, answer1: str, answer2: str) -> float:
        """Enhanced agreement calculation"""
        # Numerical agreement (most important for insurance)
        numbers1 = set(re.findall(r'\d+', answer1))
        numbers2 = set(re.findall(r'\d+', answer2))
        
        numerical_agreement = 0
        if numbers1 and numbers2:
            common_numbers = numbers1.intersection(numbers2)
            total_numbers = numbers1.union(numbers2)
            numerical_agreement = len(common_numbers) / len(total_numbers)
        elif not numbers1 and not numbers2:
            numerical_agreement = 1.0  # Both have no numbers
        
        # Key phrase agreement
        key_phrases = [
            'grace period', 'waiting period', 'covered', 'excluded', 'not covered',
            'sum insured', 'benefit', 'premium', 'deductible', 'hospital'
        ]
        
        phrase_matches = 0
        phrase_total = 0
        
        for phrase in key_phrases:
            in_answer1 = phrase in answer1.lower()
            in_answer2 = phrase in answer2.lower()
            
            if in_answer1 or in_answer2:
                phrase_total += 1
                if in_answer1 and in_answer2:
                    phrase_matches += 1
        
        phrase_agreement = phrase_matches / phrase_total if phrase_total > 0 else 0
        
        # Semantic similarity (word overlap)
        words1 = set(answer1.lower().split()) - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words2 = set(answer2.lower().split()) - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            word_overlap = 0
        
        # Combined agreement score
        agreement = (
            numerical_agreement * 0.5 +  # Numbers are crucial
            phrase_agreement * 0.3 +     # Key phrases important
            word_overlap * 0.2           # General similarity
        )
        
        return agreement
    
    def _create_single_result(self, response: Tuple[str, float, str], method: str) -> ConsensusResult:
        """Create result for single response"""
        answer, confidence, model = response
        
        # Analyze the single response
        quality_score = self._calculate_quality_score(answer)
        
        quality_metrics = {
            'quality_score': quality_score,
            'is_single_response': True
        }
        
        return ConsensusResult(
            final_answer=answer,
            final_confidence=confidence,
            consensus_method=method,
            agreement_score=1.0,
            reasoning=f"Single response from {model}",
            model_contributions={model: 1.0},
            quality_metrics=quality_metrics
        )
    
    def _create_empty_result(self, method: str) -> ConsensusResult:
        """Create empty result"""
        return ConsensusResult(
            final_answer="No valid responses available",
            final_confidence=0.0,
            consensus_method=method,
            agreement_score=0.0,
            reasoning="No responses to process",
            model_contributions={},
            quality_metrics={'error': True}
        )

# Legacy compatibility function
def aggregate_responses(responses: List[Tuple[str, float]]) -> Tuple[str, float]:
    """Enhanced legacy function for backward compatibility"""
    if not responses:
        return "No responses available", 0.0
    
    # Convert to format expected by EnhancedConsensusEngine
    converted_responses = [(answer, confidence, f"model_{i}") for i, (answer, confidence) in enumerate(responses)]
    
    engine = EnhancedConsensusEngine()
    result = engine.find_consensus(converted_responses)
    
    return result.final_answer, result.final_confidence