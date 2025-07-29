"""
Comprehensive accuracy testing script for the Bajaj Finserv AI system
Tests against known insurance policy questions and expected answers
"""

import asyncio
import json
import time
from typing import List, Dict, Any
from pathlib import Path
import statistics

from app.models.ai_processor import AIProcessor
from app.config import settings

class AccuracyTester:
    """Comprehensive accuracy testing system"""
    
    def __init__(self):
        self.ai_processor = AIProcessor()
        self.test_results = []
        
        # Insurance domain test cases based on Bajaj Allianz policy
        self.test_cases = [
            {
                "category": "grace_period",
                "question": "What is the grace period for premium payment?",
                "expected_keywords": ["30", "days", "grace", "period"],
                "expected_concepts": ["premium payment", "due date"],
                "weight": 1.0
            },
            {
                "category": "waiting_period",
                "question": "What is the waiting period for pre-existing diseases?",
                "expected_keywords": ["36", "months", "pre-existing", "diseases"],
                "expected_concepts": ["continuous coverage", "policy inception"],
                "weight": 1.2
            },
            {
                "category": "waiting_period", 
                "question": "What is the waiting period for cataract surgery?",
                "expected_keywords": ["24", "months", "cataract", "surgery"],
                "expected_concepts": ["specific diseases", "surgical procedure"],
                "weight": 1.0
            },
            {
                "category": "maternity",
                "question": "What are the maternity benefits and waiting period?",
                "expected_keywords": ["24", "months", "maternity", "pregnancy"],
                "expected_concepts": ["childbirth", "delivery", "continuous coverage"],
                "weight": 1.1
            },
            {
                "category": "coverage",
                "question": "Does the policy cover knee surgery?",
                "expected_keywords": ["surgery", "covered", "surgical"],
                "expected_concepts": ["hospitalization", "treatment"],
                "weight": 0.9
            },
            {
                "category": "exclusions",
                "question": "What are the main exclusions under this policy?",
                "expected_keywords": ["excluded", "exclusions", "not covered"],
                "expected_concepts": ["cosmetic", "dental", "pre-existing"],
                "weight": 1.0
            },
            {
                "category": "limits",
                "question": "What is the ambulance charge limit?",
                "expected_keywords": ["ambulance", "charges", "limit"],
                "expected_concepts": ["emergency", "transportation"],
                "weight": 0.8
            },
            {
                "category": "coverage",
                "question": "Are organ transplant expenses covered?",
                "expected_keywords": ["organ", "transplant", "covered"],
                "expected_concepts": ["medical expenses", "surgery"],
                "weight": 1.0
            },
            {
                "category": "definitions",
                "question": "How is 'Hospital' defined under this policy?",
                "expected_keywords": ["hospital", "definition", "medical"],
                "expected_concepts": ["healthcare facility", "treatment"],
                "weight": 0.7
            },
            {
                "category": "claims",
                "question": "What is the claim settlement process?",
                "expected_keywords": ["claim", "settlement", "process"],
                "expected_concepts": ["documentation", "reimbursement"],
                "weight": 1.0
            }
        ]
    
    async def run_comprehensive_accuracy_test(self, document_text: str) -> Dict[str, Any]:
        """Run comprehensive accuracy testing"""
        print("🧪 Starting Comprehensive Accuracy Testing...")
        print(f"📋 Total test cases: {len(self.test_cases)}")
        
        # Initialize AI processor
        success = await self.ai_processor.initialize_document(document_text)
        if not success:
            raise ValueError("Failed to initialize document for testing")
        
        # Run all test cases
        start_time = time.time()
        results = []
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\n🔍 Running test case {i+1}/{len(self.test_cases)}: {test_case['category']}")
            result = await self._run_single_test(document_text, test_case, i+1)
            results.append(result)
            
            # Print immediate feedback
            print(f"✅ Accuracy: {result['accuracy_score']:.2f}, Confidence: {result['confidence']:.2f}")
        
        total_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(results, total_time)
        
        # Print summary
        self._print_test_summary(metrics, results)
        
        return {
            "metrics": metrics,
            "individual_results": results,
            "test_metadata": {
                "total_test_cases": len(self.test_cases),
                "total_time": total_time,
                "timestamp": time.time()
            }
        }
    
    async def _run_single_test(self, document: str, test_case: Dict, test_number: int) -> Dict[str, Any]:
        """Run a single test case"""
        start_time = time.time()
        
        try:
            # Get AI response
            answer, confidence = await self.ai_processor.process_query(
                document, test_case["question"]
            )
            
            processing_time = time.time() - start_time
            
            # Calculate accuracy
            accuracy_score = self._calculate_accuracy(answer, test_case)
            
            return {
                "test_number": test_number,
                "category": test_case["category"],
                "question": test_case["question"],
                "answer": answer,
                "confidence": confidence,
                "accuracy_score": accuracy_score,
                "processing_time": processing_time,
                "weight": test_case["weight"],
                "keyword_matches": self._count_keyword_matches(answer, test_case["expected_keywords"]),
                "concept_matches": self._count_concept_matches(answer, test_case["expected_concepts"]),
                "answer_length": len(answer),
                "status": "passed" if accuracy_score > 0.6 else "failed"
            }
            
        except Exception as e:
            return {
                "test_number": test_number,
                "category": test_case["category"],
                "question": test_case["question"],
                "answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "accuracy_score": 0.0,
                "processing_time": time.time() - start_time,
                "weight": test_case["weight"],
                "error": str(e),
                "status": "error"
            }
    
    def _calculate_accuracy(self, answer: str, test_case: Dict) -> float:
        """Calculate accuracy score for an answer"""
        if not answer or "error" in answer.lower():
            return 0.0
        
        answer_lower = answer.lower()
        
        # Keyword matching (40% weight)
        keyword_score = self._count_keyword_matches(answer_lower, test_case["expected_keywords"])
        keyword_accuracy = keyword_score / len(test_case["expected_keywords"]) if test_case["expected_keywords"] else 0
        
        # Concept matching (30% weight)
        concept_score = self._count_concept_matches(answer_lower, test_case["expected_concepts"])
        concept_accuracy = concept_score / len(test_case["expected_concepts"]) if test_case["expected_concepts"] else 0
        
        # Answer quality (30% weight)
        quality_score = self._assess_answer_quality(answer, test_case)
        
        # Weighted final score
        final_score = (
            keyword_accuracy * 0.4 + 
            concept_accuracy * 0.3 + 
            quality_score * 0.3
        )
        
        return min(final_score, 1.0)
    
    def _count_keyword_matches(self, answer: str, keywords: List[str]) -> int:
        """Count keyword matches in answer"""
        if not keywords:
            return 0
        
        answer_lower = answer.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword.lower() in answer_lower:
                matches += 1
        
        return matches
    
    def _count_concept_matches(self, answer: str, concepts: List[str]) -> int:
        """Count concept matches in answer"""
        if not concepts:
            return 0
        
        answer_lower = answer.lower()
        matches = 0
        
        for concept in concepts:
            if concept.lower() in answer_lower:
                matches += 1
        
        return matches
    
    def _assess_answer_quality(self, answer: str, test_case: Dict) -> float:
        """Assess overall answer quality"""
        if not answer:
            return 0.0
        
        quality_score = 0.0
        
        # Length check (not too short, not too long)
        if 20 <= len(answer) <= 500:
            quality_score += 0.3
        
        # Contains numbers (important for insurance)
        if any(char.isdigit() for char in answer):
            quality_score += 0.2
        
        # Contains insurance terms
        insurance_terms = ["policy", "coverage", "benefit", "premium", "exclusion", 
                          "waiting", "period", "treatment", "medical", "hospital"]
        if any(term in answer.lower() for term in insurance_terms):
            quality_score += 0.3
        
        # Not a negative response
        negative_phrases = ["not provided", "cannot find", "no information", 
                           "not mentioned", "insufficient information"]
        if not any(phrase in answer.lower() for phrase in negative_phrases):
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _calculate_comprehensive_metrics(self, results: List[Dict], total_time: float) -> Dict[str, Any]:
        """Calculate comprehensive test metrics"""
        if not results:
            return {}
        
        # Basic metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["status"] == "passed")
        failed_tests = sum(1 for r in results if r["status"] == "failed")
        error_tests = sum(1 for r in results if r["status"] == "error")
        
        # Accuracy metrics
        accuracy_scores = [r["accuracy_score"] for r in results]
        confidence_scores = [r["confidence"] for r in results]
        processing_times = [r["processing_time"] for r in results]
        
        # Weighted accuracy (considering test case importance)
        weighted_accuracy = sum(r["accuracy_score"] * r["weight"] for r in results) / sum(r["weight"] for r in results)
        
        # Category-wise metrics
        category_metrics = self._calculate_category_metrics(results)
        
        # Performance metrics
        avg_processing_time = statistics.mean(processing_times)
        max_processing_time = max(processing_times)
        min_processing_time = min(processing_times)
        
        return {
            "overall_accuracy": statistics.mean(accuracy_scores),
            "weighted_accuracy": weighted_accuracy,
            "confidence_average": statistics.mean(confidence_scores),
            "pass_rate": passed_tests / total_tests,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "category_metrics": category_metrics,
            "performance_metrics": {
                "total_time": total_time,
                "average_processing_time": avg_processing_time,
                "max_processing_time": max_processing_time,
                "min_processing_time": min_processing_time,
                "tests_under_5s": sum(1 for t in processing_times if t < 5),
                "tests_under_10s": sum(1 for t in processing_times if t < 10)
            },
            "quality_metrics": {
                "high_confidence_answers": sum(1 for r in results if r["confidence"] > 0.8),
                "medium_confidence_answers": sum(1 for r in results if 0.5 < r["confidence"] <= 0.8),
                "low_confidence_answers": sum(1 for r in results if r["confidence"] <= 0.5)
            }
        }
    
    def _calculate_category_metrics(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate metrics by category"""
        categories = {}
        
        for result in results:
            category = result["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        category_metrics = {}
        for category, category_results in categories.items():
            accuracy_scores = [r["accuracy_score"] for r in category_results]
            confidence_scores = [r["confidence"] for r in category_results]
            
            category_metrics[category] = {
                "test_count": len(category_results),
                "average_accuracy": statistics.mean(accuracy_scores),
                "average_confidence": statistics.mean(confidence_scores),
                "pass_rate": sum(1 for r in category_results if r["status"] == "passed") / len(category_results)
            }
        
        return category_metrics
    
    def _print_test_summary(self, metrics: Dict, results: List[Dict]):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE ACCURACY TEST RESULTS")
        print("="*80)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"   Weighted Accuracy: {metrics['weighted_accuracy']:.2%}")
        print(f"   Average Confidence: {metrics['confidence_average']:.2f}")
        print(f"   Pass Rate: {metrics['pass_rate']:.2%}")
        
        print(f"\n🧪 TEST STATISTICS:")
        print(f"   Total Tests: {metrics['total_tests']}")
        print(f"   Passed: {metrics['passed_tests']}")
        print(f"   Failed: {metrics['failed_tests']}")
        print(f"   Errors: {metrics['error_tests']}")
        
        print(f"\n⚡ PERFORMANCE:")
        perf = metrics['performance_metrics']
        print(f"   Total Time: {perf['total_time']:.2f}s")
        print(f"   Average per Test: {perf['average_processing_time']:.2f}s")
        print(f"   Fastest: {perf['min_processing_time']:.2f}s")
        print(f"   Slowest: {perf['max_processing_time']:.2f}s")
        print(f"   Tests under 5s: {perf['tests_under_5s']}/{metrics['total_tests']}")
        
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, cat_metrics in metrics['category_metrics'].items():
            print(f"   {category.upper()}:")
            print(f"     Accuracy: {cat_metrics['average_accuracy']:.2%}")
            print(f"     Confidence: {cat_metrics['average_confidence']:.2f}")
            print(f"     Pass Rate: {cat_metrics['pass_rate']:.2%}")
        
        print(f"\n🎖️ TOP PERFORMING TESTS:")
        sorted_results = sorted(results, key=lambda x: x['accuracy_score'], reverse=True)
        for i, result in enumerate(sorted_results[:3]):
            print(f"   #{i+1}: {result['question'][:50]}...")
            print(f"       Accuracy: {result['accuracy_score']:.2%}, Confidence: {result['confidence']:.2f}")
        
        print(f"\n⚠️ AREAS FOR IMPROVEMENT:")
        poor_results = [r for r in results if r['accuracy_score'] < 0.5]
        for result in poor_results[:3]:
            print(f"   • {result['question'][:50]}...")
            print(f"     Accuracy: {result['accuracy_score']:.2%}, Issue: Low keyword/concept match")
        
        print("\n" + "="*80)

async def main():
    """Main testing function"""
    
    # Sample Bajaj Allianz policy content for testing
    sample_document = """
    National Parivar Mediclaim Plus Policy Document
    
    GRACE PERIOD:
    A grace period of thirty (30) days is allowed for payment of any premium after its due date, 
    during which period the Policy shall continue to remain in force. If the premium is not paid 
    within the grace period, the Policy shall lapse.
    
    WAITING PERIODS:
    1. Pre-existing Diseases (PED): There shall be a waiting period of thirty-six (36) months 
       of continuous coverage from the first Policy inception for pre-existing diseases and 
       their direct complications to be covered.
    
    2. Specific Diseases: Waiting period of twenty-four (24) months shall apply for cataract, 
       benign prostatic hypertrophy, hysterectomy for menorrhagia or fibromyoma only, 
       hernia, hydrocele, congenital internal diseases, fistula in anus, piles, sinusitis and related disorders.
    
    3. Maternity Benefits: Waiting period of twenty-four (24) months of continuous coverage 
       shall apply for maternity benefits including childbirth, miscarriage, or termination 
       of pregnancy for any reason.
    
    COVERAGE BENEFITS:
    The Company will pay for reasonable and customary charges for the following:
    - Room, boarding, nursing expenses, surgeon's fees, anaesthetist's fees, medical practitioner's fees
    - Pre-hospitalization and post-hospitalization expenses
    - Day care procedures
    - Ambulance charges up to a maximum of Rs. 2,000 per policy year
    - Organ transplant (heart, liver, kidney, pancreas, bone marrow) expenses
    
    EXCLUSIONS:
    The following are not covered under this policy:
    - Cosmetic or plastic surgery unless necessitated due to an accident
    - Dental treatment unless requiring hospitalization due to accident
    - Treatment taken outside India
    - Expenses related to admission primarily for diagnostic, X-ray or laboratory examinations
    
    DEFINITIONS:
    Hospital: A hospital means any institution established for in-patient care and day care treatment 
    of illness and/or injuries and which has been registered as a hospital with the local authorities.
    
    CLAIMS PROCEDURE:
    For cashless treatment, the insured should contact the TPA at least 24 hours before 
    planned hospitalization. For reimbursement claims, submit all documents within 30 days 
    of discharge from hospital.
    """
    
    print("🚀 Starting Bajaj Finserv AI Accuracy Testing System")
    print(f"🕒 Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        tester = AccuracyTester()
        results = await tester.run_comprehensive_accuracy_test(sample_document)
        
        # Save results to file
        output_file = f"accuracy_test_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Final verdict
        overall_accuracy = results["metrics"]["overall_accuracy"]
        weighted_accuracy = results["metrics"]["weighted_accuracy"]
        
        print(f"\n🏆 FINAL VERDICT:")
        if weighted_accuracy > 0.85:
            print(f"   EXCELLENT: {weighted_accuracy:.1%} - Ready for hackathon submission! 🥇")
        elif weighted_accuracy > 0.75:
            print(f"   GOOD: {weighted_accuracy:.1%} - Strong performance, minor optimizations needed 🥈")
        elif weighted_accuracy > 0.65:
            print(f"   ACCEPTABLE: {weighted_accuracy:.1%} - Meets requirements, room for improvement 🥉")
        else:
            print(f"   NEEDS WORK: {weighted_accuracy:.1%} - Requires significant improvements ⚠️")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())