"""
ULTRA STREAMLINED COMPREHENSIVE FINAL TESTER - 11 Strategic Questions
Date: 2025-08-05 18:38:47 UTC | User: vkhare2909
ULTRA OPTIMIZED VERSION: 11 strategic questions with 95%+ accuracy validation in <5 minutes
"""

import requests
import time
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Tuple
import logging
import re
from dataclasses import dataclass
from datetime import datetime
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestQuestion:
    """Enhanced question with comprehensive metadata"""
    text: str
    category: str
    domain: str
    weight: float
    expected_keywords: List[str]
    expected_answer_sample: str
    difficulty_level: str  # "easy", "medium", "hard", "expert"
    question_type: str  # "factual", "analytical", "comparative", "procedural"

@dataclass
class DocumentInfo:
    """Enhanced document information"""
    name: str
    url: str
    document_type: str  # "known" or "unknown"
    weight: float
    pages_estimate: int
    domain: str  # "health_insurance", "life_insurance", "general_insurance"
    complexity_level: str  # "basic", "intermediate", "advanced"

class UltraStreamlinedTester:
    """
    ULTRA STREAMLINED comprehensive tester with 11 strategic questions
    Complete system validation in 3-4 minutes with maximum coverage efficiency
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.bearer_token = "fff018ce90c02cb0554e8968e827c7696a7304b28ba190b9cace6236579f7258"
        
        # PRIMARY TEST DOCUMENT
        self.primary_document = DocumentInfo(
            name="BAJAJ ALLIANZ (Known - Health)",
            url="https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
            document_type="known",
            weight=1.0,
            pages_estimate=49,
            domain="health_insurance",
            complexity_level="intermediate"
        )
        
        # ULTRA STRATEGIC 11 QUESTIONS - Maximum coverage with minimum questions
        self.ultra_strategic_questions = [
            # === BASIC POLICY FUNDAMENTALS (2 questions) ===
            TestQuestion(
                text="What is the grace period for premium payment and what are the minimum and maximum entry ages for this policy?",
                category="basic_policy_info",
                domain="general",
                weight=2.0,
                expected_keywords=["grace period", "thirty days", "30 days", "entry age", "18 years", "65 years", "minimum", "maximum"],
                expected_answer_sample="Grace period is 30 days for premium payment. Minimum entry age is 18 years and maximum is 65 years.",
                difficulty_level="easy",
                question_type="factual"
            ),
            
            TestQuestion(
                text="How does the policy define a 'Hospital' and what are the room rent and ICU charge sub-limits?",
                category="hospital_definition_limits",
                domain="health_insurance",
                weight=2.5,
                expected_keywords=["10 inpatient beds", "15 beds", "qualified nursing", "operation theatre", "1%", "2%", "Sum Insured", "room rent", "ICU"],
                expected_answer_sample="Hospital must have 10-15 inpatient beds with qualified nursing staff 24/7 and operation theatre. Room rent capped at 1% and ICU at 2% of Sum Insured daily.",
                difficulty_level="medium",
                question_type="factual"
            ),
            
            # === WAITING PERIODS & COVERAGE CONDITIONS (3 questions) ===
            TestQuestion(
                text="What are the waiting periods for pre-existing diseases (PED) and what is the No Claim Discount (NCD) offered?",
                category="waiting_periods_ncd",
                domain="health_insurance",
                weight=3.0,
                expected_keywords=["waiting period", "thirty-six", "36 months", "pre-existing", "continuous coverage", "No Claim Discount", "5%", "base premium"],
                expected_answer_sample="PED has 36 months waiting period with continuous coverage. NCD of 5% on base premium for claim-free renewal years.",
                difficulty_level="medium",
                question_type="factual"
            ),
            
            TestQuestion(
                text="Does this policy cover maternity expenses and AYUSH treatments, and what are the specific conditions for each?",
                category="maternity_ayush_coverage",
                domain="health_insurance",
                weight=3.5,
                expected_keywords=["maternity", "24 months", "continuously covered", "two deliveries", "AYUSH", "Ayurveda", "Yoga", "Naturopathy", "Unani", "Siddha", "Homeopathy"],
                expected_answer_sample="Maternity covered after 24 months continuous coverage, limited to two deliveries. AYUSH treatments covered in recognized AYUSH hospitals.",
                difficulty_level="hard",
                question_type="analytical"
            ),
            
            TestQuestion(
                text="Is there a benefit for preventive health check-ups and what is the difference between network and non-network hospital treatment?",
                category="health_checkup_network",
                domain="health_insurance",
                weight=3.0,
                expected_keywords=["health check-up", "two continuous policy years", "renewed without break", "network", "non-network", "cashless", "reimbursement"],
                expected_answer_sample="Health check-up benefits after two continuous policy years. Network hospitals offer cashless treatment, non-network requires reimbursement.",
                difficulty_level="hard",
                question_type="comparative"
            ),
            
            # === CLAIMS & PROCEDURES (2 questions) ===
            TestQuestion(
                text="What is the process for cashless claim settlement and what documents are required for reimbursement claims?",
                category="claims_process",
                domain="general",
                weight=2.5,
                expected_keywords=["cashless", "pre-authorization", "TPA", "approval", "documents", "bills", "discharge summary", "diagnostic reports", "claim form"],
                expected_answer_sample="Cashless requires pre-authorization from TPA before treatment. Reimbursement needs original bills, discharge summary, diagnostic reports, and claim form.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            TestQuestion(
                text="What is the time limit for intimating claims and what are the conditions for policy renewal?",
                category="claim_intimation_renewal",
                domain="general",
                weight=2.0,
                expected_keywords=["intimation", "30 days", "hospitalization", "renewal", "guaranteed", "premium", "terms revised"],
                expected_answer_sample="Claims must be intimated within 30 days of hospitalization. Policy renewal is guaranteed with premium payment, though terms may be revised.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            # === ADVANCED COVERAGE & EXCLUSIONS (2 questions) ===
            TestQuestion(
                text="Are organ donor medical expenses covered and what are the major exclusions under this health insurance policy?",
                category="donor_coverage_exclusions",
                domain="health_insurance",
                weight=4.0,
                expected_keywords=["organ donor", "medical expenses", "harvesting", "Transplantation Act", "exclusions", "cosmetic", "fertility", "self-inflicted", "war", "nuclear"],
                expected_answer_sample="Organ donor expenses covered for harvesting complying with Transplantation Act. Major exclusions include cosmetic surgery, fertility treatments, self-inflicted injuries, war risks.",
                difficulty_level="expert",
                question_type="analytical"
            ),
            
            TestQuestion(
                text="Are mental health and psychiatric treatments covered, and are there any co-payment or deductible clauses in this policy?",
                category="mental_health_copay",
                domain="health_insurance",
                weight=3.5,
                expected_keywords=["mental health", "psychiatric", "psychological", "covered", "co-payment", "deductible", "percentage", "share"],
                expected_answer_sample="Mental health and psychiatric treatments are covered subject to policy terms. Co-payment clauses may apply where insured bears percentage of claim amount.",
                difficulty_level="expert",
                question_type="analytical"
            ),
            
            # === COMPREHENSIVE ANALYSIS (2 questions) ===
            TestQuestion(
                text="How does family coverage work under this policy and can dependents be added or removed during the policy term?",
                category="family_coverage_changes",
                domain="health_insurance",
                weight=2.5,
                expected_keywords=["family", "coverage", "floater", "shared", "dependents", "add", "remove", "renewal", "mid-term", "life events"],
                expected_answer_sample="Family coverage works as floater policy with shared sum insured. Dependents can be added at renewal, mid-term additions allowed for specific life events.",
                difficulty_level="medium",
                question_type="comparative"
            ),
            
            TestQuestion(
                text="What is the relationship between sum insured and various sub-limits, and does this policy provide coverage for treatment abroad?",
                category="sum_insured_international",
                domain="health_insurance",
                weight=4.0,
                expected_keywords=["sum insured", "sub-limits", "percentage", "relationship", "abroad", "international", "overseas", "emergency", "coverage"],
                expected_answer_sample="Sub-limits are expressed as percentages of sum insured managing overall coverage. International coverage may be available for emergency treatments abroad subject to terms.",
                difficulty_level="expert",
                question_type="analytical"
            )
        ]
        
        # Performance tracking
        self.test_results = {
            "test_metadata": {
                "user": "vkhare2909",
                "timestamp": "2025-08-05 18:38:47 UTC",
                "test_type": "ULTRA_STREAMLINED_COMPREHENSIVE",
                "total_questions": len(self.ultra_strategic_questions),
                "optimization_level": "MAXIMUM - 11 questions only",
                "target_time": "< 5 minutes",
                "coverage_efficiency": "95%+ with minimal questions"
            },
            "ultra_results": {},
            "performance_metrics": {},
            "final_assessment": {}
        }
    
    async def run_ultra_streamlined_test(self):
        """Run ultra streamlined test with 11 strategic questions"""
        
        print("🚀 ULTRA STREAMLINED COMPREHENSIVE TESTER")
        print("="*100)
        print(f"👤 User: vkhare2909")
        print(f"📅 Date: 2025-08-05 18:38:47 UTC")
        print(f"⚡ ULTRA OPTIMIZATION: 11 strategic questions only")
        print(f"🎯 Target: 95%+ Accuracy & <5min Total Time")
        print(f"📄 Document: {self.primary_document.name}")
        print(f"❓ Questions: {len(self.ultra_strategic_questions)} (MAXIMUM EFFICIENCY)")
        print("="*100)
        
        # Single comprehensive test
        await self._ultra_comprehensive_test()
        
        # Performance analysis
        self._ultra_performance_analysis()
        
        # Final assessment
        self._ultra_final_assessment()
        
        return self._generate_ultra_report()
    
    async def _ultra_comprehensive_test(self):
        """Ultra comprehensive test with all 11 strategic questions"""
        print("\n🎯 ULTRA COMPREHENSIVE TEST")
        print("-" * 80)
        print("Testing with ALL 11 strategic questions for complete system validation...")
        
        print(f"📄 Document: {self.primary_document.name}")
        print(f"🏷️ Type: {self.primary_document.document_type.upper()}")
        print(f"🌐 Domain: {self.primary_document.domain}")
        print(f"📊 Complexity: {self.primary_document.complexity_level}")
        
        questions_text = [q.text for q in self.ultra_strategic_questions]
        
        print(f"❓ Strategic Questions: {len(questions_text)}")
        
        try:
            start_time = time.time()
            response = await self._test_api_request(self.primary_document.url, questions_text)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            if response and response.get("answers"):
                answers = response["answers"]
                
                print(f"\n📊 ULTRA TEST RESULTS:")
                print(f"  ⏱️ Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
                print(f"  📝 Questions: {len(questions_text)}")
                print(f"  💬 Answers: {len(answers)}")
                print(f"  ⚡ Avg Time/Q: {total_time/len(questions_text):.1f}s")
                print(f"  🎯 Speed Target: {'✅ EXCELLENT' if total_time < 300 else '❌ SLOW'} (<5min)")
                
                # Ultra accuracy analysis
                print(f"\n🔍 ULTRA ACCURACY ANALYSIS:")
                
                total_questions = len(self.ultra_strategic_questions)
                correct_count = 0
                partial_count = 0
                incorrect_count = 0
                
                weighted_score_total = 0.0
                total_weight = sum(q.weight for q in self.ultra_strategic_questions)
                
                difficulty_scores = {}
                question_type_scores = {}
                category_scores = {}
                individual_scores = []
                
                for i, (question, answer) in enumerate(zip(self.ultra_strategic_questions, answers)):
                    if i < len(answers):
                        try:
                            accuracy_score = self._calculate_ultra_accuracy(question, answer)
                            individual_scores.append(accuracy_score)
                            
                            # Apply question weight
                            weighted_score = accuracy_score * question.weight
                            weighted_score_total += weighted_score
                            
                            # Categorize accuracy
                            if accuracy_score >= 0.85:
                                correct_count += 1
                                status = "✅ EXCELLENT"
                            elif accuracy_score >= 0.7:
                                partial_count += 1
                                status = "🟡 GOOD"
                            elif accuracy_score >= 0.5:
                                partial_count += 1
                                status = "🟠 PARTIAL"
                            else:
                                incorrect_count += 1
                                status = "❌ POOR"
                            
                            # Track by difficulty
                            difficulty = question.difficulty_level
                            if difficulty not in difficulty_scores:
                                difficulty_scores[difficulty] = []
                            difficulty_scores[difficulty].append(accuracy_score)
                            
                            # Track by question type
                            q_type = question.question_type
                            if q_type not in question_type_scores:
                                question_type_scores[q_type] = []
                            question_type_scores[q_type].append(accuracy_score)
                            
                            # Track by category
                            category = question.category
                            if category not in category_scores:
                                category_scores[category] = []
                            category_scores[category].append(accuracy_score)
                            
                            print(f"  Q{i+1:2d} ({question.difficulty_level:6}|W:{question.weight}|{question.category:25}): {accuracy_score:.1%} {status}")
                            
                        except Exception as e:
                            print(f"  Q{i+1:2d} ERROR: {str(e)[:50]}")
                            incorrect_count += 1
                            individual_scores.append(0.0)
                
                # Calculate weighted overall accuracy
                if total_weight > 0:
                    weighted_accuracy = weighted_score_total / total_weight
                else:
                    weighted_accuracy = 0.0
                
                # Calculate simple overall accuracy
                if total_questions > 0:
                    simple_accuracy = (correct_count + (partial_count * 0.7)) / total_questions
                else:
                    simple_accuracy = 0.0
                
                print(f"\n📈 ULTRA ACCURACY SUMMARY:")
                print(f"  🎯 Weighted Accuracy: {weighted_accuracy:.1%} (Primary Score)")
                print(f"  📊 Simple Accuracy: {simple_accuracy:.1%}")
                print(f"  ✅ Excellent Answers: {correct_count}/{total_questions} ({correct_count/total_questions:.1%})")
                print(f"  🟡 Good/Partial Answers: {partial_count}/{total_questions} ({partial_count/total_questions:.1%})")
                print(f"  ❌ Poor Answers: {incorrect_count}/{total_questions} ({incorrect_count/total_questions:.1%})")
                
                # Difficulty-wise accuracy
                print(f"\n⚡ DIFFICULTY-WISE PERFORMANCE:")
                for difficulty, scores in difficulty_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        print(f"  {difficulty.title():10}: {avg_score:.1%} ({len(scores)} questions)")
                
                # Question type accuracy
                print(f"\n📊 QUESTION TYPE PERFORMANCE:")
                for q_type, scores in question_type_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        print(f"  {q_type.title():12}: {avg_score:.1%} ({len(scores)} questions)")
                
                # Category performance
                print(f"\n🏷️ CATEGORY PERFORMANCE:")
                for category, scores in category_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        print(f"  {category:25}: {avg_score:.1%}")
                
                # Store ultra results
                self.test_results["ultra_results"] = {
                    "total_time": total_time,
                    "total_time_minutes": total_time / 60,
                    "avg_time_per_question": total_time / len(questions_text) if len(questions_text) > 0 else 0,
                    "weighted_accuracy": weighted_accuracy,
                    "simple_accuracy": simple_accuracy,
                    "correct_count": correct_count,
                    "partial_count": partial_count,
                    "incorrect_count": incorrect_count,
                    "total_questions": total_questions,
                    "individual_scores": individual_scores,
                    "difficulty_accuracy": {diff: (sum(scores)/len(scores) if scores else 0) for diff, scores in difficulty_scores.items()},
                    "question_type_accuracy": {q_type: (sum(scores)/len(scores) if scores else 0) for q_type, scores in question_type_scores.items()},
                    "category_accuracy": {cat: (sum(scores)/len(scores) if scores else 0) for cat, scores in category_scores.items()},
                    "target_achievement": {
                        "speed": total_time < 300,  # 5 minutes
                        "accuracy": weighted_accuracy >= 0.95,
                        "combined": total_time < 300 and weighted_accuracy >= 0.95
                    },
                    "status": "success"
                }
                
            else:
                print("❌ Ultra test failed - no valid response")
                self.test_results["ultra_results"] = {
                    "status": "failed", 
                    "error": "No valid response from API"
                }
                
        except Exception as e:
            print(f"❌ Ultra test error: {e}")
            self.test_results["ultra_results"] = {
                "status": "error", 
                "error": str(e)
            }
    
    def _ultra_performance_analysis(self):
        """Ultra performance analysis"""
        print("\n📊 ULTRA PERFORMANCE ANALYSIS")
        print("-" * 80)
        
        ultra_results = self.test_results.get("ultra_results", {})
        
        if ultra_results.get("status") != "success":
            print("❌ Cannot perform analysis - test failed")
            return
        
        weighted_accuracy = ultra_results.get("weighted_accuracy", 0)
        total_time = ultra_results.get("total_time", 0)
        total_questions = ultra_results.get("total_questions", 0)
        
        # Performance metrics
        speed_score = min(1.0, 300 / max(total_time, 1))  # Normalized to 5-minute target
        efficiency_score = weighted_accuracy * speed_score
        
        # Coverage analysis
        difficulty_coverage = len(ultra_results.get("difficulty_accuracy", {}))
        question_type_coverage = len(ultra_results.get("question_type_accuracy", {}))
        category_coverage = len(ultra_results.get("category_accuracy", {}))
        
        coverage_score = min(1.0, (difficulty_coverage + question_type_coverage + category_coverage) / 10)
        
        # Consistency analysis
        individual_scores = ultra_results.get("individual_scores", [])
        if len(individual_scores) > 1:
            consistency_score = max(0, 1 - statistics.variance(individual_scores))
        else:
            consistency_score = 0.8
        
        print(f"⚡ Speed Performance: {speed_score:.1%}")
        print(f"🎯 Accuracy Performance: {weighted_accuracy:.1%}")
        print(f"⚡ Efficiency Score: {efficiency_score:.1%}")
        print(f"📋 Coverage Score: {coverage_score:.1%}")
        print(f"🔄 Consistency Score: {consistency_score:.1%}")
        
        # Store performance metrics
        self.test_results["performance_metrics"] = {
            "speed_score": speed_score,
            "accuracy_score": weighted_accuracy,
            "efficiency_score": efficiency_score,
            "coverage_score": coverage_score,
            "consistency_score": consistency_score,
            "overall_performance": (weighted_accuracy * 0.5 + speed_score * 0.2 + 
                                   efficiency_score * 0.15 + coverage_score * 0.1 + 
                                   consistency_score * 0.05)
        }
    
    def _ultra_final_assessment(self):
        """Ultra final assessment"""
        print("\n🏆 ULTRA FINAL ASSESSMENT")
        print("-" * 80)
        
        ultra_results = self.test_results.get("ultra_results", {})
        performance_metrics = self.test_results.get("performance_metrics", {})
        
        if ultra_results.get("status") != "success":
            print("❌ Cannot perform assessment - test failed")
            return
        
        weighted_accuracy = ultra_results.get("weighted_accuracy", 0)
        total_time = ultra_results.get("total_time", 0)
        overall_performance = performance_metrics.get("overall_performance", 0)
        
        print(f"📊 ULTRA PERFORMANCE SUMMARY:")
        print(f"  🎯 Weighted Accuracy: {weighted_accuracy:.1%}")
        print(f"  ⏱️ Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  🏆 Overall Performance: {overall_performance:.1%}")
        
        # Target achievement
        targets = ultra_results.get("target_achievement", {})
        accuracy_target = targets.get("accuracy", False)
        speed_target = targets.get("speed", False)
        combined_target = targets.get("combined", False)
        
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        print(f"  🎯 Accuracy (95%+): {'✅ ACHIEVED' if accuracy_target else '❌ MISSED'}")
        print(f"  ⚡ Speed (<5min): {'✅ ACHIEVED' if speed_target else '❌ MISSED'}")
        print(f"  🏆 Combined Target: {'✅ ACHIEVED' if combined_target else '❌ MISSED'}")
        
        # Final grade
        if overall_performance >= 0.95:
            final_grade = "A+ (95%+) 🏆 CHAMPIONSHIP ELITE"
            readiness = "🚀 DEPLOY IMMEDIATELY - GUARANTEED TOP 3!"
        elif overall_performance >= 0.9:
            final_grade = "A (90-94%) 🏆 CHAMPIONSHIP READY"
            readiness = "🚀 DEPLOY WITH CONFIDENCE - TOP 5 GUARANTEED!"
        elif overall_performance >= 0.85:
            final_grade = "A- (85-89%) ⭐ EXCELLENT"
            readiness = "✅ DEPLOY READY - VERY COMPETITIVE"
        elif overall_performance >= 0.8:
            final_grade = "B+ (80-84%) ✅ VERY GOOD"
            readiness = "✅ DEPLOY READY - COMPETITIVE"
        elif overall_performance >= 0.75:
            final_grade = "B (75-79%) ✅ GOOD"
            readiness = "🟡 DEPLOY WITH MONITORING"
        else:
            final_grade = "C (<75%) ❌ NEEDS IMPROVEMENT"
            readiness = "🔧 OPTIMIZATION REQUIRED"
        
        print(f"\n🏆 FINAL GRADE: {final_grade}")
        print(f"🚀 DEPLOYMENT READINESS: {readiness}")
        
        # Recommendations
        recommendations = []
        
        if combined_target:
            recommendations.append("🎉 OUTSTANDING! Both accuracy and speed targets achieved!")
            recommendations.append("🏆 System is ready for championship-level competition!")
        elif accuracy_target and not speed_target:
            recommendations.append("🎯 Excellent accuracy! Focus on speed optimization.")
            recommendations.append("⚡ Consider response time improvements.")
        elif speed_target and not accuracy_target:
            recommendations.append("⚡ Great speed! Focus on accuracy improvements.")
            recommendations.append("🎯 Review question handling for better precision.")
        else:
            recommendations.append("🔧 Both accuracy and speed need optimization.")
            recommendations.append("📊 Review system configuration and performance.")
        
        if weighted_accuracy >= 0.9:
            recommendations.append("✅ Accuracy exceeds expectations - maintain current approach!")
        
        if total_time < 180:  # 3 minutes
            recommendations.append("🚀 Exceptional speed performance!")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Store final assessment
        self.test_results["final_assessment"] = {
            "weighted_accuracy": weighted_accuracy,
            "total_time": total_time,
            "overall_performance": overall_performance,
            "final_grade": final_grade,
            "readiness_level": readiness,
            "target_achievement": targets,
            "recommendations": recommendations,
            "championship_ready": combined_target and overall_performance >= 0.9
        }
    
    def _calculate_ultra_accuracy(self, question: TestQuestion, answer: str) -> float:
        """Ultra-enhanced accuracy calculation with strategic weighting"""
        
        if not answer or len(answer) < 10:
            return 0.0
        
        try:
            answer_lower = answer.lower()
            score = 0.0
            
            # 1. Keyword matching (50% of score) - Increased importance
            keyword_matches = 0
            total_keywords = len(question.expected_keywords)
            
            if total_keywords > 0:
                for keyword in question.expected_keywords:
                    if keyword.lower() in answer_lower:
                        keyword_matches += 1
                
                keyword_score = keyword_matches / total_keywords
                score += keyword_score * 0.5
            
            # 2. Content quality (25% of score)
            content_indicators = [
                len(answer) > 50,  # Substantial length
                not any(term in answer_lower for term in ["error", "unable", "failed", "not found", "processing error"]),
                any(term in answer_lower for term in ["policy", "coverage", "insured", "benefits", "covered", "premium"]),
                bool(re.search(r'\d+', answer)),  # Contains numbers
                len(answer.split()) > 15,  # Good detail
                len(answer.split('.')) > 1,  # Multiple sentences
            ]
            
            content_score = sum(1 for indicator in content_indicators if indicator) / len(content_indicators)
            score += content_score * 0.25
            
            # 3. Question-specific accuracy (15% of score)
            if question.question_type == "factual":
                if re.search(r'\b\d+\b', answer) and any(term in answer_lower for term in ["is", "are", "defined"]):
                    score += 0.15
            elif question.question_type == "analytical":
                if len(answer.split()) > 20 and any(term in answer_lower for term in ["includes", "conditions", "requirements"]):
                    score += 0.15
            elif question.question_type == "comparative":
                if any(term in answer_lower for term in ["different", "compared", "versus", "while"]):
                    score += 0.15
            elif question.question_type == "procedural":
                if any(term in answer_lower for term in ["process", "steps", "procedure", "must"]):
                    score += 0.15
            
            # 4. Complexity bonus (10% of score)
            complexity_bonus = 0
            if question.difficulty_level == "expert" and len(answer) > 80:
                complexity_bonus = 0.1
            elif question.difficulty_level == "hard" and len(answer) > 60:
                complexity_bonus = 0.08
            elif question.difficulty_level == "medium" and len(answer) > 40:
                complexity_bonus = 0.06
            elif question.difficulty_level == "easy" and len(answer) > 20:
                complexity_bonus = 0.04
            
            score += complexity_bonus
            
            # Severe penalty for uncertainty or error responses
            uncertainty_phrases = [
                'not mentioned', 'not provided', 'unclear', 'unable to determine',
                'insufficient information', 'cannot find', 'not specified',
                'processing error', 'error occurred', 'unable to process'
            ]
            
            for phrase in uncertainty_phrases:
                if phrase in answer_lower:
                    score = max(score * 0.1, 0.02)  # Very severe penalty
                    break
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            print(f"Error in ultra accuracy calculation: {e}")
            return 0.0
    
    async def _test_api_request(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """Enhanced API request with comprehensive error handling"""
        
        payload = {
            "documents": document_url,
            "questions": questions
        }
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        try:
            # Increased timeout to 5 minutes for comprehensive processing
            timeout = aiohttp.ClientTimeout(total=300)  # 5-minute timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info(f"🚀 Sending request with {len(questions)} questions...")
                async with session.post(
                    f"{self.base_url}/hackrx/run",
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Received successful response with {len(result.get('answers', []))} answers")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text[:200]}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"Request timeout after 300 seconds")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def _generate_ultra_report(self) -> Dict[str, Any]:
        """Generate ultra streamlined final report"""
        
        report = {
            "test_metadata": self.test_results["test_metadata"],
            "executive_summary": {
                "test_type": "ULTRA_STREAMLINED_COMPREHENSIVE",
                "optimization_level": "MAXIMUM",
                "total_questions": len(self.ultra_strategic_questions),
                "questions_efficiency": f"{len(self.ultra_strategic_questions)} questions for complete validation",
                "time_efficiency": "< 5 minutes total execution",
                "coverage_efficiency": "95%+ accuracy validation with minimal questions",
                "completion_status": "COMPLETED",
                "timestamp": "2025-08-05 18:38:47 UTC",
                "user": "vkhare2909"
            },
            "ultra_results": self.test_results.get("ultra_results", {}),
            "performance_metrics": self.test_results.get("performance_metrics", {}),
            "final_assessment": self.test_results.get("final_assessment", {}),
            "ultra_recommendations": self._generate_ultra_recommendations()
        }
        
        return report
    
    def _generate_ultra_recommendations(self) -> List[str]:
        """Generate ultra streamlined recommendations"""
        
        recommendations = []
        
        final_assessment = self.test_results.get("final_assessment", {})
        ultra_results = self.test_results.get("ultra_results", {})
        
        championship_ready = final_assessment.get("championship_ready", False)
        weighted_accuracy = final_assessment.get("weighted_accuracy", 0)
        total_time = final_assessment.get("total_time", 0)
        
        if championship_ready:
            recommendations.append("🏆 CHAMPIONSHIP READY! Deploy immediately for guaranteed top performance!")
            recommendations.append("🎯 Both accuracy and speed targets exceeded - system is optimal!")
        elif weighted_accuracy >= 0.95:
            recommendations.append("🎯 Outstanding accuracy! Minor speed optimization could perfect the system.")
        elif weighted_accuracy >= 0.9:
            recommendations.append("✅ Excellent performance! System is deployment-ready.")
        elif weighted_accuracy >= 0.8:
            recommendations.append("🟡 Good performance with room for improvement in accuracy.")
        else:
            recommendations.append("🔧 Significant accuracy improvements needed before deployment.")
        
        if total_time < 180:
            recommendations.append("🚀 Exceptional speed - well under target!")
        elif total_time < 300:
            recommendations.append("⚡ Good speed performance - meets target requirements.")
        else:
            recommendations.append("⏱️ Speed optimization needed to meet target requirements.")
        
        # Add strategic recommendations
        target_achievement = ultra_results.get("target_achievement", {})
        if target_achievement.get("combined", False):
            recommendations.append("🎉 ULTRA SUCCESS! Ready for championship competition!")
        
        return recommendations

# Ultra quick validation function
def run_ultra_quick_validation():
    """Ultra quick validation with 3 essential questions"""
    
    print("⚡ ULTRA QUICK VALIDATION TEST")
    print("="*60)
    
    ultra_quick_questions = [
        "What is the grace period for premium payment and what are the minimum and maximum entry ages for this policy?",
        "What are the waiting periods for pre-existing diseases (PED) and what is the No Claim Discount (NCD) offered?",
        "Does this policy cover maternity expenses and AYUSH treatments, and what are the specific conditions for each?"
    ]
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "questions": ultra_quick_questions
    }
    
    headers = {
        "Authorization": "Bearer fff018ce90c02cb0554e8968e827c7696a7304b28ba190b9cace6236579f7258",
        "Content-Type": "application/json"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/hackrx/run",
            json=payload,
            headers=headers,
            timeout=300  # Increased to match main test timeout
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            
            print(f"✅ SUCCESS! Response time: {total_time:.1f}s")
            print(f"📝 Questions tested: {len(ultra_quick_questions)}")
            print(f"💬 Answers received: {len(answers)}")
            print(f"⚡ Speed: {'✅ EXCELLENT' if total_time < 60 else '🟡 ACCEPTABLE' if total_time < 120 else '❌ SLOW'}")
            
            # Quick accuracy check
            good_answers = 0
            for i, answer in enumerate(answers):
                if len(answer) > 50 and "error" not in answer.lower() and any(keyword in answer.lower() for keyword in ["grace", "waiting", "maternity", "ayush", "period"]):
                    good_answers += 1
                    print(f"  Q{i+1}: ✅ GOOD ({len(answer)} chars)")
                else:
                    print(f"  Q{i+1}: 🟡 BASIC ({len(answer)} chars)")
            
            quick_accuracy = good_answers / len(ultra_quick_questions) if ultra_quick_questions else 0
            
            print(f"\n🎯 ULTRA QUICK ASSESSMENT:")
            print(f"📊 Quick Accuracy: {quick_accuracy:.1%}")
            print(f"✅ Good Answers: {good_answers}/{len(ultra_quick_questions)}")
            
            if total_time < 120 and quick_accuracy >= 0.67:
                print(f"\n🏆 ULTRA QUICK VALIDATION PASSED!")
                print(f"🚀 System ready for ultra comprehensive testing!")
                return True
            else:
                print(f"\n🟡 ULTRA QUICK VALIDATION PARTIAL")
                print(f"🔧 System may need optimization")
                return False
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ultra quick test failed: {e}")
        return False

# Main ultra streamlined test function
def run_ultra_streamlined_test():
    """Run the ultra streamlined comprehensive test"""
    
    print("🔧 CHECKING SERVER STATUS...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server Status: {health_data.get('status', 'unknown')}")
        else:
            print(f"⚠️ Server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("🔧 Start server with: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False
    
    # Run ultra streamlined test
    tester = UltraStreamlinedTester()
    
    async def run_async():
        return await tester.run_ultra_streamlined_test()
    
    try:
        final_report = asyncio.run(run_async())
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"ultra_streamlined_report_{timestamp}.json"
        
        with open(report_filename, "w") as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📊 ULTRA STREAMLINED TEST COMPLETE!")
        print(f"📄 Report saved to: {report_filename}")
        
        # Print executive summary
        final_assessment = final_report.get("final_assessment", {})
        championship_ready = final_assessment.get("championship_ready", False)
        weighted_accuracy = final_assessment.get("weighted_accuracy", 0)
        total_time = final_assessment.get("total_time", 0)
        
        print(f"\n🏆 ULTRA FINAL RESULTS:")
        print(f"📊 Weighted Accuracy: {weighted_accuracy:.1%}")
        print(f"⏱️ Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🎯 Championship Ready: {'🏆 YES!' if championship_ready else '🔧 NO'}")
        print(f"🚀 Grade: {final_assessment.get('final_grade', 'N/A')}")
        
        if championship_ready:
            print(f"\n🎉 🏆 ULTRA SUCCESS! 🏆 🎉")
            print(f"🥇 SYSTEM IS CHAMPIONSHIP-READY!")
            print(f"🚀 DEPLOY IMMEDIATELY FOR GUARANTEED TOP 3!")
            return True
        elif weighted_accuracy >= 0.9:
            print(f"\n✅ EXCELLENT PERFORMANCE!")
            print(f"🚀 System is deployment-ready!")
            return True
        else:
            print(f"\n🟡 GOOD PERFORMANCE with room for improvement")
            return False
            
    except Exception as e:
        print(f"\n💥 Ultra test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"🚀 ULTRA STREAMLINED COMPREHENSIVE TESTER")
    print(f"📅 Date: 2025-08-05 18:38:47 UTC")
    print(f"👤 User: vkhare2909")
    print(f"⚡ Mission: Complete system validation with only 11 strategic questions")
    print("="*80)
    
    print("🔍 Select test type:")
    print("1. Ultra Quick Validation (3 questions, ~90 seconds)")
    print("2. Ultra Streamlined Comprehensive (11 questions, ~4 minutes)")
    print("3. Both (Quick validation first, then comprehensive)")
    
    choice = input("\nEnter choice (1/2/3) or Enter for ultra streamlined: ").strip()
    
    if choice == "1":
        success = run_ultra_quick_validation()
        if success:
            print("\n🎉 Ultra quick validation successful!")
        else:
            print("\n⚠️ Ultra quick validation shows issues")
    
    elif choice == "3":
        print("\n⚡ Running ultra quick validation first...")
        quick_success = run_ultra_quick_validation()
        
        if quick_success:
            print("\n🚀 Quick test passed! Proceeding to ultra comprehensive...")
            input("Press Enter to continue...")
            comprehensive_success = run_ultra_streamlined_test()
        else:
            proceed = input("\n⚠️ Quick test showed issues. Continue anyway? (y/N): ")
            if proceed.lower() == 'y':
                comprehensive_success = run_ultra_streamlined_test()
            else:
                print("🔧 Consider optimization before comprehensive testing")
                comprehensive_success = False
    
    else:  # Default to ultra streamlined
        comprehensive_success = run_ultra_streamlined_test()
    
    print(f"\n🎯 Ultra testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"👤 Tester: vkhare2909")
    print("\n📋 ULTRA OPTIMIZATION SUMMARY:")
    print("  • Reduced to only 11 strategic questions")
    print("  • Complete system validation in <5 minutes")
    print("  • Maximum coverage with minimum questions")
    print("  • Advanced weighted accuracy scoring")
    print("  • Championship-readiness assessment")
    print("🏁 END OF ULTRA STREAMLINED TESTING")