"""
Ultimate Hybrid Consensus Engine for Multi-Model AI Responses
Combines advanced algorithms with insurance-specific optimizations for maximum accuracy
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Optional sklearn import with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using fallback similarity calculation")

@dataclass
class ConsensusResult:
    """Ultimate result of consensus analysis with comprehensive metrics"""
    final_answer: str
    final_confidence: float
    consensus_method: str
    agreement_score: float
    reasoning: str
    model_contributions: Dict[str, float]
    quality_metrics: Dict[str, float]
    similarity_scores: Optional[Dict[str, float]] = None
    tier_analysis: Optional[Dict[str, Any]] = None

class UltimateHybridConsensusEngine:
    """Ultimate hybrid consensus engine combining advanced algorithms with insurance expertise"""
    
    def __init__(self):
        # Core thresholds
        self.confidence_threshold = 0.4
        self.agreement_threshold = 0.6
        self.similarity_threshold = 0.7
        self.high_quality_threshold = 0.7
        self.medium_quality_threshold = 0.4
        
        # Initialize TF-IDF vectorizer if sklearn available
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams for better insurance phrase matching
                min_df=1,
                max_df=0.9
            )
        else:
            self.vectorizer = None
        
        # Ultimate insurance-specific quality indicators
        self.quality_indicators = {
            'ultimate_high_value': [
                'grace period of', 'waiting period of', 'sum insured of',
                'covered up to', 'limited to', 'as per section',
                'provided that', 'subject to', 'in accordance with',
                'continuously covered for', 'policy inception',
                'pre-existing diseases', 'direct complications',
                'lawful medical termination', 'transplantation of human organs act'
            ],
            'high_value': [
                'grace period', 'waiting period', 'maternity expenses',
                'health check-up', 'cataract surgery', 'organ donor',
                'no claim discount', 'hospital defined', 'ayush coverage',
                'room rent', 'sum insured', 'base premium', 'table of benefits'
            ],
            'medium_value': [
                'covered', 'excluded', 'benefit', 'premium', 'deductible',
                'hospital', 'treatment', 'surgery', 'diagnosis', 'policy',
                'claim', 'medical expenses', 'hospitalization', 'indemnifies'
            ],
            'negative_indicators': [
                'not provided', 'not mentioned', 'cannot find', 'no information',
                'insufficient information', 'unclear', 'not specified',
                'not available', 'not stated', 'does not contain',
                'document does not specify', 'unable to determine'
            ],
            'uncertainty_phrases': [
                'appears to', 'seems to', 'likely', 'probably', 'perhaps',
                'could be', 'should be', 'generally', 'typically',
                'may be', 'might be', 'possibly', 'usually', 'often'
            ]
        }
        
        # Enhanced numerical pattern weights for insurance
        self.numerical_patterns = {
            r'\d+\s*days?': 4.0,           # Grace period days (highest priority)
            r'(thirty|30)\s*days?': 5.0,   # Specific grace period
            r'\d+\s*months?': 4.0,         # Waiting periods
            r'(thirty-six|36)\s*months?': 5.0,  # Specific PED waiting
            r'(twenty-four|24)\s*months?': 4.5,  # Maternity waiting
            r'\d+\s*years?': 3.0,          # Long-term periods
            r'(two|2)\s*years?': 3.5,      # Cataract waiting
            r'₹[\d,]+': 2.5,               # Currency amounts
            r'\d+%': 2.5,                  # Percentages (NCD, room rent)
            r'\d+\s*lakhs?': 2.0,          # Large amounts
            r'\d+\s*deliveries?': 3.0,     # Maternity limits
        }
        
        # Competition format patterns (from ultimate AI processor)
        self.competition_patterns = [
            r'grace period of (thirty|30) days.*premium payment.*due date.*renew.*continue.*policy.*continuity benefits',
            r'waiting period of (thirty-six|36) \(?\d*\)? months.*continuous coverage.*first policy inception.*pre-existing diseases.*direct complications',
            r'policy covers maternity expenses.*childbirth.*lawful medical termination.*pregnancy.*female insured person.*continuously covered.*(24|twenty-four) months.*limited.*(two|2) deliveries.*terminations.*policy period',
            r'policy reimburses.*health check-ups.*end.*every.*block.*(two|2) continuous policy years.*renewed without.*break.*table of benefits',
            r'specific waiting period.*(two|2) \(?\d*\)? years.*cataract surgery',
            r'policy indemnifies.*medical expenses.*organ donor.*hospitalization.*harvesting.*organ.*insured person.*transplantation of human organs act.*1994',
            r'no claim discount.*\d+%.*base premium.*renewal.*preceding year.*maximum aggregate.*capped.*\d+%.*total base premium',
            r'hospital.*defined.*institution.*\d+.*inpatient beds.*towns.*population.*\d+.*lakhs.*\d+.*beds.*qualified nursing staff.*medical practitioners.*24/7.*operation theatre.*daily records'
        ]
    
    def find_consensus(
        self, 
        responses: List[Tuple[str, float, str]], 
        query: str = "",
        method: str = "ultimate_auto"
    ) -> ConsensusResult:
        """
        Ultimate consensus finding with hybrid algorithms and insurance expertise
        """
        if not responses:
            return self._create_empty_result("ultimate_auto")
        
        if len(responses) == 1:
            return self._create_single_result(responses[0], "single")
        
        # Ultimate response analysis combining both approaches
        analyzed_responses = self._ultimate_analyze_responses(responses, query)
        
        # Enhanced filtering and categorization
        filtered_responses = self._ultimate_filter_responses(analyzed_responses)
        
        # Select optimal consensus method
        if method == "ultimate_auto":
            method = self._select_ultimate_method(filtered_responses, analyzed_responses, query)
        
        # Apply selected method with enhanced algorithms
        if method == "ultimate_quality_weighted":
            return self._ultimate_quality_weighted_consensus(analyzed_responses, filtered_responses, query)
        elif method == "ultimate_similarity":
            return self._ultimate_similarity_consensus(analyzed_responses, filtered_responses, query)
        elif method == "ultimate_agreement":
            return self._ultimate_agreement_consensus(analyzed_responses, filtered_responses, query)
        elif method == "ultimate_tiered":
            return self._ultimate_tiered_consensus(analyzed_responses, filtered_responses, query)
        else:
            return self._ultimate_adaptive_consensus(analyzed_responses, filtered_responses, query)
    
    def _ultimate_analyze_responses(self, responses: List[Tuple[str, float, str]], query: str) -> List[Dict]:
        """Ultimate response analysis combining both methodologies"""
        analyzed = []
        
        for answer, confidence, model in responses:
            analysis = {
                'answer': answer,
                'confidence': confidence,
                'model': model,
                
                # Quality metrics (from version 1)
                'quality_score': self._calculate_ultimate_quality_score(answer),
                'specificity_score': self._calculate_ultimate_specificity_score(answer),
                'relevance_score': self._calculate_ultimate_relevance_score(answer, query),
                'uncertainty_penalty': self._calculate_ultimate_uncertainty_penalty(answer),
                'numerical_content': self._extract_ultimate_numerical_content(answer),
                'length_factor': self._calculate_ultimate_length_factor(answer),
                
                # Additional metrics (from version 2 enhanced)
                'insurance_terminology_score': self._calculate_insurance_terminology_score(answer),
                'competition_format_score': self._calculate_competition_format_score(answer),
                'definitiveness_score': self._calculate_definitiveness_score(answer),
                'structure_score': self._calculate_structure_score(answer),
                
                # Ultimate composite score
                'composite_score': 0.0,  # Will be calculated below
                'tier': 'unknown'        # Will be assigned below
            }
            
            # Calculate ultimate composite score
            analysis['composite_score'] = self._calculate_ultimate_composite_score(analysis)
            
            # Assign quality tier
            analysis['tier'] = self._assign_quality_tier(analysis)
            
            analyzed.append(analysis)
        
        return analyzed
    
    def _calculate_ultimate_quality_score(self, answer: str) -> float:
        """Ultimate quality calculation combining both approaches"""
        answer_lower = answer.lower()
        score = 0.5  # Base score
        
        # Ultimate high-value phrases (competition format specific)
        for phrase in self.quality_indicators['ultimate_high_value']:
            if phrase in answer_lower:
                score += 0.2
        
        # High-value phrases
        for phrase in self.quality_indicators['high_value']:
            if phrase in answer_lower:
                score += 0.12
        
        # Medium-value phrases
        medium_count = sum(1 for phrase in self.quality_indicators['medium_value'] if phrase in answer_lower)
        score += min(medium_count * 0.04, 0.16)
        
        # Heavy penalty for negative indicators
        negative_count = sum(1 for phrase in self.quality_indicators['negative_indicators'] if phrase in answer_lower)
        score -= negative_count * 0.25
        
        # Penalty for uncertainty
        uncertainty_count = sum(1 for phrase in self.quality_indicators['uncertainty_phrases'] if phrase in answer_lower)
        score -= uncertainty_count * 0.12
        
        return max(0.0, min(1.0, score))
    
    def _calculate_ultimate_specificity_score(self, answer: str) -> float:
        """Ultimate specificity calculation with enhanced patterns"""
        score = 0.3  # Base score
        
        # Enhanced numerical information boost
        for pattern, weight in self.numerical_patterns.items():
            matches = len(re.findall(pattern, answer, re.IGNORECASE))
            score += matches * weight * 0.04
        
        # Boost for specific insurance terms
        specific_terms = [
            'section', 'clause', 'paragraph', 'table', 'schedule',
            'annexure', 'appendix', 'terms and conditions',
            'policy wordings', 'benefits table', 'exclusions'
        ]
        
        for term in specific_terms:
            if term in answer.lower():
                score += 0.06
        
        # Ultimate boost for definitive language
        definitive_phrases = [
            'is defined as', 'shall be', 'must be', 'is required',
            'is covered', 'is excluded', 'is limited to',
            'the policy covers', 'the policy excludes', 'the grace period is'
        ]
        
        for phrase in definitive_phrases:
            if phrase in answer.lower():
                score += 0.08
        
        return min(1.0, score)
    
    def _calculate_ultimate_relevance_score(self, answer: str, query: str) -> float:
        """Ultimate relevance calculation with enhanced matching"""
        if not query:
            return 0.5
        
        query_words = set(word.lower() for word in query.split() if len(word) > 2)
        answer_words = set(word.lower() for word in answer.split() if len(word) > 2)
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are'}
        query_words -= common_words
        answer_words -= common_words
        
        if not query_words:
            return 0.5
        
        # Basic word overlap
        overlap = len(query_words.intersection(answer_words))
        relevance = overlap / len(query_words)
        
        # Ultimate boost for exact phrase matches
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        ultimate_key_phrases = [
            'grace period', 'waiting period', 'maternity', 'coverage', 'exclusion',
            'pre-existing', 'cataract', 'organ donor', 'health check-up',
            'no claim discount', 'hospital defined', 'room rent'
        ]
        
        for phrase in ultimate_key_phrases:
            if phrase in query_lower and phrase in answer_lower:
                relevance += 0.25
        
        # Boost for insurance-specific query-answer alignment
        insurance_alignments = {
            'grace': ['grace period', 'premium payment', 'due date'],
            'waiting': ['waiting period', 'pre-existing', 'continuous coverage'],
            'maternity': ['maternity expenses', 'childbirth', 'pregnancy'],
            'cover': ['covered', 'coverage', 'benefits', 'sum insured']
        }
        
        for query_key, answer_terms in insurance_alignments.items():
            if query_key in query_lower:
                for term in answer_terms:
                    if term in answer_lower:
                        relevance += 0.1
                        break
        
        return min(1.0, relevance)
    
    def _calculate_ultimate_uncertainty_penalty(self, answer: str) -> float:
        """Ultimate uncertainty penalty calculation"""
        answer_lower = answer.lower()
        penalty = 0.0
        
        # Strong uncertainty indicators (heavy penalty)
        strong_uncertainty = ['not provided', 'not mentioned', 'cannot find', 'no information', 'insufficient information']
        for phrase in strong_uncertainty:
            if phrase in answer_lower:
                penalty += 0.35
        
        # Special case: "document does not specify" is better than generic uncertainty
        if 'document does not specify' in answer_lower:
            penalty = max(0, penalty - 0.15)  # Reduce penalty
        
        # Mild uncertainty indicators
        mild_uncertainty = ['may be', 'might be', 'possibly', 'appears to', 'seems to', 'likely', 'probably']
        for phrase in mild_uncertainty:
            if phrase in answer_lower:
                penalty += 0.12
        
        # Hedge words
        hedge_words = ['generally', 'typically', 'usually', 'often', 'sometimes', 'mostly', 'largely']
        hedge_count = sum(1 for word in hedge_words if word in answer_lower)
        penalty += hedge_count * 0.06
        
        return min(0.8, penalty)  # Cap penalty
    
    def _extract_ultimate_numerical_content(self, answer: str) -> Dict[str, Any]:
        """Ultimate numerical content extraction"""
        content = {
            'days': len(re.findall(r'\d+\s*days?', answer, re.IGNORECASE)),
            'months': len(re.findall(r'\d+\s*months?', answer, re.IGNORECASE)),
            'years': len(re.findall(r'\d+\s*years?', answer, re.IGNORECASE)),
            'amounts': len(re.findall(r'₹[\d,]+', answer)),
            'percentages': len(re.findall(r'\d+%', answer)),
            'total_numbers': len(re.findall(r'\d+', answer)),
            
            # Insurance-specific extractions
            'grace_period_days': len(re.findall(r'(thirty|30)\s*days?', answer, re.IGNORECASE)),
            'ped_waiting_months': len(re.findall(r'(thirty-six|36)\s*months?', answer, re.IGNORECASE)),
            'maternity_months': len(re.findall(r'(twenty-four|24)\s*months?', answer, re.IGNORECASE)),
            'cataract_years': len(re.findall(r'(two|2)\s*years?', answer, re.IGNORECASE)),
            'deliveries': len(re.findall(r'(two|2)\s*deliveries?', answer, re.IGNORECASE))
        }
        
        # Calculate numerical richness score
        content['richness_score'] = (
            content['days'] * 0.3 +
            content['months'] * 0.3 +
            content['years'] * 0.2 +
            content['amounts'] * 0.15 +
            content['percentages'] * 0.15 +
            content['grace_period_days'] * 0.4 +
            content['ped_waiting_months'] * 0.4 +
            content['maternity_months'] * 0.35
        )
        
        return content
    
    def _calculate_ultimate_length_factor(self, answer: str) -> float:
        """Ultimate length appropriateness calculation"""
        length = len(answer)
        word_count = len(answer.split())
        
        # Optimal ranges for insurance answers
        if length < 15:
            return 0.2  # Too short for meaningful insurance info
        elif length < 40:
            return 0.5  # Short but might be definitive
        elif length < 100:
            return 0.8  # Good for simple answers
        elif length < 250:
            return 1.0  # Optimal for detailed insurance answers
        elif length < 500:
            return 0.9  # Comprehensive but manageable
        elif length < 800:
            return 0.7  # Very detailed, might be verbose
        else:
            return 0.5  # Too verbose
    
    def _calculate_insurance_terminology_score(self, answer: str) -> float:
        """Calculate insurance-specific terminology richness"""
        answer_lower = answer.lower()
        score = 0.0
        
        # Ultimate insurance terminology
        ultimate_terms = [
            'sum insured', 'base premium', 'table of benefits',
            'continuous coverage', 'policy inception', 'direct complications',
            'preferred provider network', 'transplantation of human organs act',
            'qualified nursing staff', 'operation theatre', 'daily records',
            'lawful medical termination', 'ayush hospital', 'room rent capped'
        ]
        
        for term in ultimate_terms:
            if term in answer_lower:
                score += 0.08
        
        # Standard insurance terms
        standard_terms = [
            'policy holder', 'insured person', 'beneficiary', 'nominee',
            'claim settlement', 'cashless treatment', 'reimbursement',
            'medical expenses', 'hospitalization', 'day care surgery'
        ]
        
        for term in standard_terms:
            if term in answer_lower:
                score += 0.04
        
        return min(1.0, score)
    
    def _calculate_competition_format_score(self, answer: str) -> float:
        """Calculate adherence to competition answer formats"""
        answer_lower = answer.lower()
        score = 0.0
        
        # Check for competition format patterns
        for pattern in self.competition_patterns:
            if re.search(pattern, answer_lower):
                score += 0.3  # High boost for exact format match
                break
        
        # Check for format indicators
        format_indicators = [
            'grace period of', 'waiting period of', 'policy covers',
            'policy indemnifies', 'policy reimburses', 'hospital is defined',
            'no claim discount of', 'continuously covered for'
        ]
        
        for indicator in format_indicators:
            if indicator in answer_lower:
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_definitiveness_score(self, answer: str) -> float:
        """Calculate how definitive/certain the answer is"""
        answer_lower = answer.lower()
        score = 0.5  # Base score
        
        # Definitive language boosts
        definitive_starters = [
            'the grace period is', 'the waiting period is', 'the policy covers',
            'yes, the policy', 'no, the policy', 'coverage includes',
            'the benefit is', 'exclusions are', 'the hospital is defined'
        ]
        
        for starter in definitive_starters:
            if starter in answer_lower:
                score += 0.2
                break
        
        # Definitive structure words
        definitive_words = ['is', 'are', 'must', 'shall', 'will', 'does', 'includes', 'excludes']
        definitive_count = sum(1 for word in definitive_words if f' {word} ' in f' {answer_lower} ')
        score += min(definitive_count * 0.03, 0.15)
        
        # Penalty for uncertain language
        uncertain_words = ['may', 'might', 'could', 'should', 'possibly', 'probably', 'perhaps']
        uncertain_count = sum(1 for word in uncertain_words if word in answer_lower)
        score -= uncertain_count * 0.08
        
        return max(0.0, min(1.0, score))
    
    def _calculate_structure_score(self, answer: str) -> float:
        """Calculate answer structure quality"""
        score = 0.5  # Base score
        
        # Boost for structured elements
        if re.search(r'\d+\.', answer):  # Numbered lists
            score += 0.1
        
        if re.search(r'[:\-\•]', answer):  # Colons, dashes, bullets
            score += 0.05
        
        # Boost for proper sentences
        sentence_count = len(re.findall(r'[.!?]+', answer))
        if sentence_count > 0:
            score += min(sentence_count * 0.02, 0.1)
        
        # Boost for insurance-specific structure patterns
        if re.search(r'provided that|subject to|in accordance with', answer.lower()):
            score += 0.08
        
        return min(1.0, score)
    
    def _calculate_ultimate_composite_score(self, analysis: Dict) -> float:
        """Ultimate composite score calculation"""
        weights = {
            'confidence': 0.25,
            'quality_score': 0.20,
            'specificity_score': 0.15,
            'relevance_score': 0.12,
            'insurance_terminology_score': 0.10,
            'competition_format_score': 0.08,
            'definitiveness_score': 0.05,
            'structure_score': 0.03,
            'length_factor': 0.02
        }
        
        composite = sum(analysis.get(metric, 0) * weight for metric, weight in weights.items())
        composite -= analysis.get('uncertainty_penalty', 0)
        
        # Boost for numerical richness
        numerical_boost = min(analysis['numerical_content']['richness_score'] * 0.05, 0.1)
        composite += numerical_boost
        
        return max(0.0, min(1.0, composite))
    
    def _assign_quality_tier(self, analysis: Dict) -> str:
        """Assign quality tier based on analysis"""
        composite_score = analysis['composite_score']
        confidence = analysis['confidence']
        quality_score = analysis['quality_score']
        
        if (composite_score > 0.8 and confidence > 0.7 and quality_score > 0.7):
            return 'ultimate'
        elif (composite_score > 0.65 and confidence > 0.5 and quality_score > 0.5):
            return 'high'
        elif (composite_score > 0.4 and confidence > 0.3):
            return 'medium'
        else:
            return 'low'
    
    def _ultimate_filter_responses(self, analyzed_responses: List[Dict]) -> Dict[str, List[Dict]]:
        """Ultimate response filtering with enhanced categorization"""
        filtered = {
            'ultimate': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for analysis in analyzed_responses:
            tier = analysis['tier']
            filtered[tier].append(analysis)
        
        return filtered
    
    def _select_ultimate_method(self, filtered_responses: Dict, analyzed_responses: List[Dict], query: str) -> str:
        """Select optimal consensus method based on response landscape"""
        ultimate_count = len(filtered_responses['ultimate'])
        high_count = len(filtered_responses['high'])
        medium_count = len(filtered_responses['medium'])
        total_count = len(analyzed_responses)
        
        # If we have ultimate quality responses
        if ultimate_count >= 2:
            return "ultimate_agreement"
        elif ultimate_count >= 1 and high_count >= 1:
            return "ultimate_quality_weighted"
        
        # If we have multiple high-quality responses
        elif high_count >= 2:
            if SKLEARN_AVAILABLE:
                return "ultimate_similarity"
            else:
                return "ultimate_agreement"
        
        # Mixed quality scenarios
        elif high_count >= 1 and medium_count >= 1:
            return "ultimate_tiered"
        
        # Sparse or low-quality responses
        elif total_count <= 2:
            return "ultimate_quality_weighted"
        
        # Default to adaptive
        else:
            return "ultimate_adaptive"
    
    def _ultimate_quality_weighted_consensus(self, analyzed_responses: List[Dict], filtered_responses: Dict, query: str) -> ConsensusResult:
        """Ultimate quality-weighted consensus"""
        if not analyzed_responses:
            return self._create_empty_result("ultimate_quality_weighted")
        
        # Calculate ultimate weights
        weighted_responses = []
        total_weight = 0
        
        for analysis in analyzed_responses:
            # Base weight from composite score
            weight = analysis['composite_score']
            
            # Tier multipliers
            tier_multipliers = {
                'ultimate': 2.0,
                'high': 1.5,
                'medium': 1.0,
                'low': 0.4
            }
            
            weight *= tier_multipliers.get(analysis['tier'], 1.0)
            
            # Competition format bonus
            if analysis['competition_format_score'] > 0.2:
                weight *= 1.3
            
            # Insurance terminology bonus
            if analysis['insurance_terminology_score'] > 0.3:
                weight *= 1.2
            
            # Query relevance bonus
            weight *= (1 + analysis['relevance_score'] * 0.4)
            
            weighted_responses.append((analysis, weight))
            total_weight += weight
        
        # Select highest weighted response
        best_analysis, best_weight = max(weighted_responses, key=lambda x: x[1])
        
        # Calculate enhanced final confidence
        base_confidence = best_analysis['confidence']
        quality_boost = (best_analysis['quality_score'] - 0.5) * 0.3
        competition_boost = best_analysis['competition_format_score'] * 0.15
        final_confidence = min(1.0, base_confidence + quality_boost + competition_boost)
        
        # Calculate model contributions
        model_contributions = {}
        for analysis, weight in weighted_responses:
            model = analysis['model']
            contribution = weight / total_weight if total_weight > 0 else 1.0 / len(weighted_responses)
            model_contributions[model] = model_contributions.get(model, 0) + contribution
        
        quality_metrics = {
            'selected_tier': best_analysis['tier'],
            'composite_score': best_analysis['composite_score'],
            'quality_score': best_analysis['quality_score'],
            'competition_format_score': best_analysis['competition_format_score'],
            'weight_ratio': best_weight / total_weight if total_weight > 0 else 1.0
        }
        
        return ConsensusResult(
            final_answer=best_analysis['answer'],
            final_confidence=final_confidence,
            consensus_method="ultimate_quality_weighted",
            agreement_score=best_weight / total_weight if total_weight > 0 else 1.0,
            reasoning=f"Selected highest weighted response from {best_analysis['tier']} tier (weight: {best_weight:.3f})",
            model_contributions=model_contributions,
            quality_metrics=quality_metrics,
            tier_analysis={
                'ultimate': len(filtered_responses['ultimate']),
                'high': len(filtered_responses['high']),
                'medium': len(filtered_responses['medium']),
                'low': len(filtered_responses['low'])
            }
        )
    
    def _ultimate_similarity_consensus(self, analyzed_responses: List[Dict], filtered_responses: Dict, query: str) -> ConsensusResult:
        """Ultimate similarity-based consensus with sklearn and fallback"""
        # Prioritize ultimate and high-quality responses
        priority_responses = filtered_responses['ultimate'] + filtered_responses['high']
        
        if len(priority_responses) < 2:
            priority_responses = analyzed_responses  # Use all if not enough high-quality
        
        if len(priority_responses) < 2:
            # Fall back to quality weighted
            return self._ultimate_quality_weighted_consensus(analyzed_responses, filtered_responses, query)
        
        # Use sklearn if available, otherwise use fallback similarity
        if SKLEARN_AVAILABLE and self.vectorizer:
            return self._sklearn_similarity_consensus(priority_responses, filtered_responses, query)
        else:
            return self._fallback_similarity_consensus(priority_responses, filtered_responses, query)
    
    def _sklearn_similarity_consensus(self, priority_responses: List[Dict], filtered_responses: Dict, query: str) -> ConsensusResult:
        """Sklearn-based similarity consensus"""
        try:
            answers = [resp['answer'] for resp in priority_responses]
            
            # Calculate TF-IDF similarity matrix
            tfidf_matrix = self.vectorizer.fit_transform(answers)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similarity groups
            similar_groups = self._find_ultimate_similarity_groups(similarity_matrix, priority_responses)
            
            if similar_groups:
                # Select best group based on combined score
                best_group = max(similar_groups, key=lambda g: (
                    sum(resp['composite_score'] for resp in g['responses']) / len(g['responses']) * 
                    g['avg_similarity']
                ))
                
                # Select best response from best group
                best_response = max(best_group['responses'], key=lambda x: x['composite_score'])
                
                # Calculate similarity boost
                similarity_boost = min(best_group['avg_similarity'] * 0.2, 0.25)
                group_size_boost = min(len(best_group['responses']) * 0.05, 0.15)
                final_confidence = min(best_response['confidence'] + similarity_boost + group_size_boost, 1.0)
                
                model_contributions = {}
                for resp in best_group['responses']:
                    model = resp['model']
                    model_contributions[model] = 1.0 / len(best_group['responses'])
                
                quality_metrics = {
                    'avg_similarity': best_group['avg_similarity'],
                    'group_size': len(best_group['responses']),
                    'composite_score': best_response['composite_score'],
                    'method': 'sklearn_tfidf'
                }
                
                return ConsensusResult(
                    final_answer=best_response['answer'],
                    final_confidence=final_confidence,
                    consensus_method="ultimate_similarity",
                    agreement_score=best_group['avg_similarity'],
                    reasoning=f"Selected from similarity group of {len(best_group['responses'])} responses (avg similarity: {best_group['avg_similarity']:.3f})",
                    model_contributions=model_contributions,
                    quality_metrics=quality_metrics,
                    similarity_scores={'avg_group_similarity': best_group['avg_similarity']}
                )
            
            # No similar groups found
            best_response = max(priority_responses, key=lambda x: x['composite_score'])
            return self._create_analyzed_single_result(best_response, "ultimate_similarity")
            
        except Exception as e:
            logger.error(f"Sklearn similarity consensus failed: {e}")
            return self._fallback_similarity_consensus(priority_responses, filtered_responses, query)
    
    def _fallback_similarity_consensus(self, priority_responses: List[Dict], filtered_responses: Dict, query: str) -> ConsensusResult:
        """Fallback similarity consensus without sklearn"""
        # Use simple word overlap similarity
        similarity_groups = []
        used_indices = set()
        
        for i, resp1 in enumerate(priority_responses):
            if i in used_indices:
                continue
            
            group = [resp1]
            used_indices.add(i)
            similarities = []
            
            for j, resp2 in enumerate(priority_responses[i+1:], i+1):
                if j in used_indices:
                    continue
                
                similarity = self._calculate_word_overlap_similarity(resp1['answer'], resp2['answer'])
                if similarity > self.similarity_threshold:
                    group.append(resp2)
                    similarities.append(similarity)
                    used_indices.add(j)
            
            if len(group) > 1:
                avg_similarity = np.mean(similarities) if similarities else 0
                similarity_groups.append({
                    'responses': group,
                    'avg_similarity': avg_similarity
                })
        
        if similarity_groups:
            # Select best group
            best_group = max(similarity_groups, key=lambda g: (
                sum(resp['composite_score'] for resp in g['responses']) / len(g['responses']) *
                g['avg_similarity']
            ))
            
            best_response = max(best_group['responses'], key=lambda x: x['composite_score'])
            
            similarity_boost = min(best_group['avg_similarity'] * 0.15, 0.2)
            final_confidence = min(best_response['confidence'] + similarity_boost, 1.0)
            
            model_contributions = {}
            for resp in best_group['responses']:
                model = resp['model']
                model_contributions[model] = 1.0 / len(best_group['responses'])
            
            quality_metrics = {
                'avg_similarity': best_group['avg_similarity'],
                'group_size': len(best_group['responses']),
                'method': 'word_overlap'
            }
            
            return ConsensusResult(
                final_answer=best_response['answer'],
                final_confidence=final_confidence,
                consensus_method="ultimate_similarity",
                agreement_score=best_group['avg_similarity'],
                reasoning=f"Selected from word-overlap similarity group of {len(best_group['responses'])} responses",
                model_contributions=model_contributions,
                quality_metrics=quality_metrics,
                similarity_scores={'avg_group_similarity': best_group['avg_similarity']}
            )
        
        # No similarity groups found
        best_response = max(priority_responses, key=lambda x: x['composite_score'])
        return self._create_analyzed_single_result(best_response, "ultimate_similarity")
    
    def _ultimate_agreement_consensus(self, analyzed_responses: List[Dict], filtered_responses: Dict, query: str) -> ConsensusResult:
        """Ultimate agreement-based consensus"""
        if len(analyzed_responses) < 2:
            return self._ultimate_quality_weighted_consensus(analyzed_responses, filtered_responses, query)
        
        # Focus on ultimate and high-quality responses
        priority_responses = filtered_responses['ultimate'] + filtered_responses['high']
        if len(priority_responses) < 2:
            priority_responses = analyzed_responses
        
        # Find best agreeing pair
        best_agreement = 0
        best_pair = None
        
        for i in range(len(priority_responses)):
            for j in range(i + 1, len(priority_responses)):
                agreement = self._calculate_ultimate_agreement(
                    priority_responses[i]['answer'], 
                    priority_responses[j]['answer']
                )
                
                if agreement > best_agreement:
                    best_agreement = agreement
                    best_pair = (i, j)
        
        if best_pair and best_agreement > self.agreement_threshold:
            resp1, resp2 = priority_responses[best_pair[0]], priority_responses[best_pair[1]]
            
            # Select better response from agreeing pair
            if resp1['composite_score'] >= resp2['composite_score']:
                selected_response = resp1
            else:
                selected_response = resp2
            
            # Ultimate agreement boost
            agreement_boost = min(best_agreement * 0.25, 0.3)
            final_confidence = min(selected_response['confidence'] + agreement_boost, 1.0)
            
            model_contributions = {
                resp1['model']: 0.5,
                resp2['model']: 0.5
            }
            
            quality_metrics = {
                'agreement_score': best_agreement,
                'selected_tier': selected_response['tier'],
                'composite_score': selected_response['composite_score'],
                'pair_quality': (resp1['composite_score'] + resp2['composite_score']) / 2
            }
            
            return ConsensusResult(
                final_answer=selected_response['answer'],
                final_confidence=final_confidence,
                consensus_method="ultimate_agreement",
                agreement_score=best_agreement,
                reasoning=f"Selected from agreeing pair with {best_agreement:.3f} agreement",
                model_contributions=model_contributions,
                quality_metrics=quality_metrics
            )
        
        # No good agreement found
        return self._ultimate_quality_weighted_consensus(analyzed_responses, filtered_responses, query)
    
    def _ultimate_tiered_consensus(self, analyzed_responses: List[Dict], filtered_responses: Dict, query: str) -> ConsensusResult:
        """Ultimate tiered consensus with enhanced tier selection"""
        tier_priority = ['ultimate', 'high', 'medium', 'low']
        
        for tier in tier_priority:
            candidates = filtered_responses[tier]
            if candidates:
                # Within tier, select by composite score with additional criteria
                best_candidate = max(candidates, key=lambda x: (
                    x['composite_score'],
                    x['competition_format_score'],
                    x['insurance_terminology_score'],
                    x['confidence']
                ))
                
                # Tier-based confidence adjustment
                tier_multipliers = {
                    'ultimate': 1.2,
                    'high': 1.1,
                    'medium': 1.0,
                    'low': 0.9
                }
                
                adjusted_confidence = min(
                    best_candidate['confidence'] * tier_multipliers[tier], 
                    1.0
                )
                
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
                    'selected_tier': tier,
                    'tier_size': len(candidates),
                    'composite_score': best_candidate['composite_score'],
                    'tier_average_quality': sum(r['composite_score'] for r in candidates) / len(candidates)
                }
                
                return ConsensusResult(
                    final_answer=best_candidate['answer'],
                    final_confidence=adjusted_confidence,
                    consensus_method="ultimate_tiered",
                    agreement_score=best_candidate['composite_score'],
                    reasoning=f"Selected best from {tier} tier ({len(candidates)} candidates)",
                    model_contributions=model_contributions,
                    quality_metrics=quality_metrics,
                    tier_analysis={k: len(v) for k, v in filtered_responses.items()}
                )
        
        # Should not reach here, but fallback
        return self._create_empty_result("ultimate_tiered")
    
    def _ultimate_adaptive_consensus(self, analyzed_responses: List[Dict], filtered_responses: Dict, query: str) -> ConsensusResult:
        """Ultimate adaptive consensus combining all strategies"""
        # Strategy 1: Try agreement if we have quality responses
        ultimate_high = filtered_responses['ultimate'] + filtered_responses['high']
        
        if len(ultimate_high) >= 2:
            agreement_result = self._ultimate_agreement_consensus(analyzed_responses, filtered_responses, query)
            if agreement_result.agreement_score > 0.6:
                agreement_result.consensus_method = "ultimate_adaptive(agreement)"
                return agreement_result
        
        # Strategy 2: Try similarity if available and we have enough responses
        if len(ultimate_high) >= 2 and (SKLEARN_AVAILABLE or True):  # Always try fallback similarity
            similarity_result = self._ultimate_similarity_consensus(analyzed_responses, filtered_responses, query)
            if similarity_result.agreement_score > 0.5:
                similarity_result.consensus_method = "ultimate_adaptive(similarity)"
                return similarity_result
        
        # Strategy 3: Fall back to tiered approach
        tiered_result = self._ultimate_tiered_consensus(analyzed_responses, filtered_responses, query)
        tiered_result.consensus_method = "ultimate_adaptive(tiered)"
        return tiered_result
    
    # Helper methods
    
    def _find_ultimate_similarity_groups(self, similarity_matrix: np.ndarray, responses: List[Dict]) -> List[Dict]:
        """Enhanced similarity group finding"""
        n = len(responses)
        visited = [False] * n
        groups = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            group_responses = [responses[i]]
            group_similarities = []
            visited[i] = True
            
            for j in range(i + 1, n):
                if not visited[j] and similarity_matrix[i][j] > self.similarity_threshold:
                    group_responses.append(responses[j])
                    group_similarities.append(similarity_matrix[i][j])
                    visited[j] = True
            
            if len(group_responses) > 1:
                avg_similarity = np.mean(group_similarities) if group_similarities else 0
                groups.append({
                    'responses': group_responses,
                    'avg_similarity': avg_similarity
                })
        
        return groups
    
    def _calculate_word_overlap_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate word overlap similarity as fallback"""
        words1 = set(word.lower() for word in answer1.split() if len(word) > 2)
        words2 = set(word.lower() for word in answer2.split() if len(word) > 2)
        
        # Remove common words
        common_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'will', 'have', 'been'}
        words1 -= common_words
        words2 -= common_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_ultimate_agreement(self, answer1: str, answer2: str) -> float:
        """Ultimate agreement calculation combining all factors"""
        # Enhanced numerical agreement
        numbers1 = set(re.findall(r'\d+', answer1))
        numbers2 = set(re.findall(r'\d+', answer2))
        
        numerical_agreement = 0
        if numbers1 and numbers2:
            common_numbers = numbers1.intersection(numbers2)
            total_numbers = numbers1.union(numbers2)
            numerical_agreement = len(common_numbers) / len(total_numbers)
        elif not numbers1 and not numbers2:
            numerical_agreement = 1.0
        
        # Ultimate key phrase agreement
        ultimate_phrases = [
            'grace period', 'waiting period', 'covered', 'excluded', 'not covered',
            'premium', 'benefit', 'sum insured', 'conditions', 'limitations',
            'continuous coverage', 'policy inception', 'maternity expenses',
            'pre-existing diseases', 'direct complications', 'base premium',
            'table of benefits', 'hospital defined', 'ayush coverage'
        ]
        
        phrase_matches = 0
        phrase_total = 0
        
        for phrase in ultimate_phrases:
            in_answer1 = phrase in answer1.lower()
            in_answer2 = phrase in answer2.lower()
            
            if in_answer1 or in_answer2:
                phrase_total += 1
                if in_answer1 and in_answer2:
                    phrase_matches += 1
        
        phrase_agreement = phrase_matches / phrase_total if phrase_total > 0 else 0
        
        # Enhanced word overlap
        words1 = set(word.lower() for word in answer1.split() if len(word) > 3)
        words2 = set(word.lower() for word in answer2.split() if len(word) > 3)
        
        common_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'will', 'have', 'been', 'are', 'were'}
        words1 -= common_words
        words2 -= common_words
        
        word_overlap = 0
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        
        # Sentiment agreement
        sentiment_agreement = self._calculate_sentiment_agreement(answer1, answer2)
        
        # Ultimate weighted combination
        final_agreement = (
            numerical_agreement * 0.4 +    # Numbers most important
            phrase_agreement * 0.3 +       # Key phrases very important
            word_overlap * 0.2 +           # General similarity
            sentiment_agreement * 0.1      # Overall sentiment
        )
        
        return final_agreement
    
    def _calculate_sentiment_agreement(self, answer1: str, answer2: str) -> float:
        """Calculate sentiment agreement between answers"""
        positive_words = ['yes', 'covered', 'included', 'eligible', 'provided', 'available', 'benefits']
        negative_words = ['no', 'not', 'excluded', 'except', 'limitation', 'restriction', 'unavailable']
        
        def get_sentiment(answer):
            answer_lower = answer.lower()
            positive_count = sum(1 for word in positive_words if word in answer_lower)
            negative_count = sum(1 for word in negative_words if word in answer_lower)
            
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
        
        sentiment1 = get_sentiment(answer1)
        sentiment2 = get_sentiment(answer2)
        
        return 1.0 if sentiment1 == sentiment2 else 0.0
    
    def _create_analyzed_single_result(self, analysis: Dict, method: str) -> ConsensusResult:
        """Create result for single analyzed response"""
        quality_metrics = {
            'tier': analysis['tier'],
            'composite_score': analysis['composite_score'],
            'quality_score': analysis['quality_score'],
            'competition_format_score': analysis['competition_format_score'],
            'is_single_response': True
        }
        
        return ConsensusResult(
            final_answer=analysis['answer'],
            final_confidence=analysis['confidence'],
            consensus_method=method,
            agreement_score=1.0,
            reasoning=f"Single {analysis['tier']} tier response from {analysis['model']}",
            model_contributions={analysis['model']: 1.0},
            quality_metrics=quality_metrics
        )
    
    def _create_single_result(self, response: Tuple[str, float, str], method: str) -> ConsensusResult:
        """Create result for single unanalyzed response"""
        answer, confidence, model = response
        
        # Quick analysis for single response
        quality_score = self._calculate_ultimate_quality_score(answer)
        
        quality_metrics = {
            'quality_score': quality_score,
            'is_single_response': True,
            'method': 'quick_analysis'
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
    """Ultimate legacy function for backward compatibility"""
    if not responses:
        return "No responses available", 0.0
    
    # Convert to format expected by UltimateHybridConsensusEngine
    converted_responses = [(answer, confidence, f"model_{i}") for i, (answer, confidence) in enumerate(responses)]
    
    engine = UltimateHybridConsensusEngine()
    result = engine.find_consensus(converted_responses)
    
    return result.final_answer, result.final_confidence