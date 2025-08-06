"""ADVANCED SAFETY FILTER BYPASS LOGIC"""

import asyncio

async def bypass_safety_filter(model, question, context, max_attempts=3):
    """Advanced multi-strategy safety filter bypass"""
    
    strategies = [
        # Strategy 1: Professional Business Language
        f"""PROFESSIONAL INSURANCE DOCUMENT ANALYSIS

CLIENT REQUEST: Analyze the attached business insurance policy documentation to provide accurate information about: {question}

DOCUMENT CONTENT:
{context[:1000]}

REQUIRED OUTPUT:
- Factual policy information
- Coverage details and limitations  
- Regulatory compliance information
- Business terms and conditions

PROFESSIONAL RESPONSE:""",

        # Strategy 2: Technical Documentation Review
        f"""TECHNICAL INSURANCE POLICY REVIEW

DOCUMENTATION ANALYSIS TASK: Extract specific information from insurance policy documents.

QUERY: {question}

SOURCE MATERIAL:
{context[:800]}

TECHNICAL ANALYSIS:
Please provide factual information from the documentation regarding the query above.""",

        # Strategy 3: Simple Factual Query
        f"Based on this insurance policy document, please answer: {question}\n\nPolicy Content: {context[:600]}",
        
        # Strategy 4: Minimal Context
        f"Question: {question}\nDocument excerpt: {context[:400]}\nAnswer:",
    ]
    
    for attempt, strategy in enumerate(strategies):
        try:
            response = await asyncio.to_thread(model.generate_content, strategy)
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip(), f"bypass-strategy-{attempt+1}"
        except Exception as e:
            print(f"Strategy {attempt+1} failed: {e}")
            continue
    
    return None, None
