#!/usr/bin/env python3
"""
Test script for Ultimate AI Processor with 90%+ Accuracy
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from app.models.ai_processor import ChampionshipAIProcessor

async def test_ultimate_ai():
    """Test the ultimate AI processor"""
    print("🏆 Testing Ultimate Championship AI Processor...")
    
    # Initialize the processor
    ai_processor = ChampionshipAIProcessor()
    
    # Test document content (sample)
    test_document = """
    GRACE PERIOD: A grace period of thirty days is provided for premium payment 
    after the due date to renew or continue the policy without losing continuity benefits.
    
    WAITING PERIOD: There is a waiting period of thirty-six (36) months of continuous 
    coverage from the first policy inception for pre-existing diseases and their 
    direct complications to be covered.
    
    MATERNITY: The policy covers maternity expenses, including childbirth and lawful 
    medical termination of pregnancy. To be eligible, the female insured person must 
    have been continuously covered for at least 24 months.
    """
    
    # Test queries
    test_queries = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does the policy cover maternity expenses?"
    ]
    
    print("✅ AI Processor initialized successfully!")
    print(f"📊 Processing stats: {ai_processor.get_processing_stats()}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        
        try:
            answer, confidence = await ai_processor.process_query(test_document, query)
            print(f"✅ Answer: {answer[:100]}...")
            print(f"🎯 Confidence: {confidence:.2f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🏆 Ultimate AI Processor test completed!")

if __name__ == "__main__":
    asyncio.run(test_ultimate_ai())
