#!/usr/bin/env python3
"""
Vercel Deployment Test Script
Test the deployed API endpoint on Vercel
"""

import requests
import json
import time

def test_vercel_deployment(base_url):
    """Test the Vercel deployment"""
    
    print(f"🧪 Testing Vercel deployment at: {base_url}")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Environment: {health_data.get('environment', 'unknown')}")
            print(f"   Version: {health_data.get('version', 'unknown')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Authentication test
    headers = {
        "Authorization": "Bearer fff018ce90c02cb0554e8968e827c7696a7304b28ba190b9cace6236579f7258",
        "Content-Type": "application/json"
    }
    
    test_payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "questions": ["What is the grace period for premium payment?"]
    }
    
    try:
        print("🔍 Testing hackrx/run endpoint...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/hackrx/run",
            headers=headers,
            json=test_payload,
            timeout=120
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️ Response time: {processing_time:.1f}s")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API test successful!")
            print(f"📝 Answer: {data['answers'][0][:100]}...")
            print(f"🏆 Vercel deployment working correctly!")
            return True
        else:
            print(f"❌ API test failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    # Example Vercel URLs
    print("🏆 VERCEL DEPLOYMENT TESTING")
    print("=" * 50)
    print("After deploying to Vercel, test with your URL:")
    print()
    
    # Test with your actual Vercel URL
    vercel_url = input("Enter your Vercel deployment URL (e.g., https://bajaj-ai.vercel.app): ").strip()
    
    if not vercel_url:
        print("❌ Please provide a valid Vercel URL")
        print("📝 Example: https://bajaj-ai-hackathon.vercel.app")
    else:
        success = test_vercel_deployment(vercel_url)
        if success:
            print(f"🎉 COMPETITION READY!")
            print(f"📝 Submit this URL to competition: {vercel_url}/hackrx/run")
        else:
            print("❌ Please check deployment and try again")
