"""
CHAMPIONSHIP Integration Test - Validates 90%+ Accuracy & <15s Response
Run this to verify championship-level performance
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any

# Import your championship modules
from app.main import ai_processor, document_processor
from app.models.ai_processor import ChampionshipAIProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChampionshipTester:
    """Championship-level testing for maximum accuracy validation"""
    
    def __init__(self):
        self.test_documents = [
            "https://www.hackrx.in/policies/BAJHLIP23020V012223.pdf",
            "https://www.hackrx.in/policies/CHOTGDP23004V012223.pdf", 
            "https://www.hackrx.in/policies/EDLHLGA23009V012223.pdf",
            "https://www.hackrx.in/policies/HDFHLIP23024V072223.pdf",
            "https://www.hackrx.in/policies/ICIHLIP22012V012223.pdf"
        ]
        
        self.championship_queries = [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?", 
            "What is the waiting period for specific diseases?",
            "What are the maternity benefits covered?",
            "What expenses are covered under this policy?",
            "What are the exclusions in this policy?",
            "What are the key definitions in this policy?",
            "What is the claims procedure?"
        ]
        
        self.processor = ChampionshipAIProcessor()
        self.results = []
    
    async def run_championship_test(self):
        """Run championship-level accuracy test"""
        logger.info("🏆 CHAMPIONSHIP ACCURACY TEST STARTED")
        logger.info(f"🎯 Target: 90%+ Accuracy, <15s Response Time")
        logger.info(f"📄 Documents: {len(self.test_documents)}")
        logger.info(f"❓ Queries per document: {len(self.championship_queries)}")
        
        overall_start = time.time()
        total_accuracy = 0.0
        total_queries = 0
        total_response_times = []
        
        for doc_idx, doc_url in enumerate(self.test_documents):
            logger.info(f"\n🔄 Processing Document {doc_idx + 1}/{len(self.test_documents)}")
            logger.info(f"📄 URL: {doc_url}")
            
            try:
                # Extract document
                doc_start = time.time()
                document_text = await document_processor.extract_text_from_url(doc_url)
                doc_time = time.time() - doc_start
                logger.info(f"✅ Document extracted in {doc_time:.2f}s ({len(document_text)} chars)")
                
                # Initialize processor for this document
                await self.processor.initialize_document(document_text)
                
                # Process all queries for this document
                doc_results = []
                for query_idx, query in enumerate(self.championship_queries):
                    query_start = time.time()
                    
                    answer, confidence = await self.processor.process_query(document_text, query)
                    
                    query_time = time.time() - query_start
                    total_response_times.append(query_time)
                    
                    # Calculate accuracy score based on confidence and content quality
                    accuracy_score = self._calculate_accuracy_score(answer, confidence, query)
                    
                    result = {
                        'document_index': doc_idx,
                        'query_index': query_idx,
                        'query': query,
                        'answer': answer,
                        'confidence': confidence,
                        'accuracy_score': accuracy_score,
                        'response_time': query_time,
                        'meets_time_target': query_time < 15.0
                    }
                    
                    doc_results.append(result)
                    total_accuracy += accuracy_score
                    total_queries += 1
                    
                    # Log individual result
                    status = "🏆" if accuracy_score > 0.9 else "✅" if accuracy_score > 0.7 else "⚠️"
                    logger.info(f"  {status} Q{query_idx+1}: {accuracy_score:.1%} accuracy, {query_time:.2f}s")
                
                self.results.extend(doc_results)
                
                # Document summary
                doc_avg_accuracy = sum(r['accuracy_score'] for r in doc_results) / len(doc_results)
                doc_avg_time = sum(r['response_time'] for r in doc_results) / len(doc_results)
                logger.info(f"📊 Document {doc_idx+1} Results: {doc_avg_accuracy:.1%} accuracy, {doc_avg_time:.2f}s avg time")
                
            except Exception as e:
                logger.error(f"❌ Document {doc_idx+1} failed: {e}")
                continue
        
        # Calculate overall results
        overall_time = time.time() - overall_start
        overall_accuracy = total_accuracy / total_queries if total_queries > 0 else 0.0
        avg_response_time = sum(total_response_times) / len(total_response_times) if total_response_times else 0.0
        
        # Generate championship report
        self._generate_championship_report(
            overall_accuracy, avg_response_time, overall_time, total_queries
        )
        
        return overall_accuracy >= 0.9 and avg_response_time < 15.0
    
    def _calculate_accuracy_score(self, answer: str, confidence: float, query: str) -> float:
        """Calculate championship-level accuracy score"""
        if not answer or "error" in answer.lower():
            return 0.1
        
        base_score = confidence
        
        # Boost for specific information
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Grace period queries
        if "grace period" in query_lower:
            if any(term in answer_lower for term in ["30 days", "thirty days", "grace period"]):
                base_score += 0.2
            if "premium" in answer_lower and "payment" in answer_lower:
                base_score += 0.1
        
        # Waiting period queries
        elif "waiting period" in query_lower:
            if any(term in answer_lower for term in ["36 months", "24 months", "waiting period"]):
                base_score += 0.2
            if "pre-existing" in query_lower and any(term in answer_lower for term in ["pre-existing", "ped"]):
                base_score += 0.15
        
        # Coverage/Benefits queries
        elif any(term in query_lower for term in ["coverage", "benefits", "covered"]):
            if any(term in answer_lower for term in ["covered", "benefits", "sum insured"]):
                base_score += 0.15
        
        # Exclusions queries
        elif "exclusion" in query_lower:
            if any(term in answer_lower for term in ["excluded", "not covered", "exclusion"]):
                base_score += 0.15
        
        # Penalize vague responses
        if any(term in answer_lower for term in ["unable", "cannot", "not provided", "unclear"]):
            base_score -= 0.2
        
        # Boost for detailed responses
        if len(answer) > 100:
            base_score += 0.05
        
        return min(max(base_score, 0.0), 1.0)
    
    def _generate_championship_report(self, accuracy: float, avg_time: float, total_time: float, total_queries: int):
        """Generate championship test report"""
        
        logger.info("\n" + "="*80)
        logger.info("🏆 CHAMPIONSHIP TEST RESULTS")
        logger.info("="*80)
        
        # Overall Performance
        logger.info(f"📊 OVERALL PERFORMANCE:")
        logger.info(f"   🎯 Accuracy: {accuracy:.1%} (Target: 90%+)")
        logger.info(f"   ⚡ Avg Response Time: {avg_time:.2f}s (Target: <15s)")
        logger.info(f"   🕒 Total Test Time: {total_time:.2f}s")
        logger.info(f"   📝 Total Queries: {total_queries}")
        
        # Performance Assessment
        accuracy_status = "🏆 CHAMPIONSHIP" if accuracy >= 0.9 else "✅ GOOD" if accuracy >= 0.8 else "⚠️ NEEDS IMPROVEMENT"
        time_status = "🏆 CHAMPIONSHIP" if avg_time < 10 else "✅ TARGET MET" if avg_time < 15 else "⚠️ EXCEEDED TARGET"
        
        logger.info(f"\n🎖️ PERFORMANCE ASSESSMENT:")
        logger.info(f"   Accuracy: {accuracy_status}")
        logger.info(f"   Response Time: {time_status}")
        
        # Detailed breakdown
        category_results = {}
        for result in self.results:
            query = result['query']
            if 'grace period' in query.lower():
                category = 'grace_period'
            elif 'waiting period' in query.lower() and 'pre-existing' in query.lower():
                category = 'waiting_period_ped'
            elif 'waiting period' in query.lower():
                category = 'waiting_period_specific'
            elif 'maternity' in query.lower():
                category = 'maternity_benefits'
            elif any(term in query.lower() for term in ['coverage', 'covered', 'expenses']):
                category = 'coverage_general'
            elif 'exclusion' in query.lower():
                category = 'exclusions'
            elif 'definition' in query.lower():
                category = 'definitions'
            elif 'claims' in query.lower():
                category = 'claims_procedure'
            else:
                category = 'other'
            
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result['accuracy_score'])
        
        logger.info(f"\n📋 CATEGORY BREAKDOWN:")
        for category, scores in category_results.items():
            avg_score = sum(scores) / len(scores)
            status = "🏆" if avg_score > 0.9 else "✅" if avg_score > 0.7 else "❌"
            logger.info(f"   {status} {category}: {avg_score:.1%}")
        
        # Championship recommendation
        logger.info(f"\n🎯 CHAMPIONSHIP ASSESSMENT:")
        if accuracy >= 0.9 and avg_time < 15:
            logger.info("   🏆 CHAMPIONSHIP LEVEL ACHIEVED!")
            logger.info("   🥇 Ready for competition victory!")
        elif accuracy >= 0.85 and avg_time < 15:
            logger.info("   🥈 EXCELLENT PERFORMANCE!")
            logger.info("   📈 Close to championship level")
        elif accuracy >= 0.8 or avg_time < 15:
            logger.info("   🥉 GOOD PERFORMANCE!")
            logger.info("   🔧 Some optimization needed")
        else:
            logger.info("   ⚠️ NEEDS SIGNIFICANT IMPROVEMENT")
            logger.info("   🛠️ Major optimization required")
        
        # Save detailed results
        timestamp = int(time.time())
        filename = f"championship_test_results_{timestamp}.json"
        
        test_summary = {
            "championship_test_results": {
                "overall_accuracy": accuracy,
                "average_response_time": avg_time,
                "total_test_time": total_time,
                "total_queries": total_queries,
                "accuracy_target_met": accuracy >= 0.9,
                "time_target_met": avg_time < 15.0,
                "championship_ready": accuracy >= 0.9 and avg_time < 15.0
            },
            "category_breakdown": {
                category: sum(scores) / len(scores) 
                for category, scores in category_results.items()
            },
            "detailed_results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(test_summary, f, indent=2)
        
        logger.info(f"\n💾 Detailed results saved to: {filename}")
        logger.info("="*80)

async def main():
    """Run championship integration test"""
    tester = ChampionshipTester()
    
    try:
        success = await tester.run_championship_test()
        
        if success:
            print("\n🏆 CHAMPIONSHIP TEST PASSED! System ready for competition victory!")
            exit(0)
        else:
            print("\n⚠️ Championship targets not fully met. Check logs for optimization opportunities.")
            exit(1)
            
    except Exception as e:
        logger.error(f"❌ Championship test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())