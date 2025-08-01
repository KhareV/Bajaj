"""
ULTIMATE COMPREHENSIVE FINAL TESTER - Multi-Domain & Multi-Difficulty
Date: 2025-08-01 17:09:28 UTC | User: vkhare2909
FINAL VERSION: Extensive testing across all domains with 95%+ accuracy validation
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

class UltimateComprehensiveTester:
    """
    ULTIMATE comprehensive tester with extensive question coverage
    Tests across multiple domains, difficulty levels, and question types
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.bearer_token = "fff018ce90c02cb0554e8968e827c7696a7304b28ba190b9cace6236579f7258"
        
        # COMPREHENSIVE DOCUMENTS with enhanced metadata
        self.test_documents = [
            DocumentInfo(
                name="BAJAJ ALLIANZ (Known - Health)",
                url="https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
                document_type="known",
                weight=0.5,
                pages_estimate=49,
                domain="health_insurance",
                complexity_level="intermediate"
            ),
            DocumentInfo(
                name="CHOLAMANDALAM (Unknown - General)",
                url="https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/CHOTGDP23004V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
                document_type="unknown",
                weight=2.0,
                pages_estimate=85,
                domain="general_insurance",
                complexity_level="advanced"
            ),
            DocumentInfo(
                name="EDELWEISS (Unknown - Health)",
                url="https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/EDLHLGA23009V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
                document_type="unknown",
                weight=2.0,
                pages_estimate=101,
                domain="health_insurance",
                complexity_level="advanced"
            ),
            DocumentInfo(
                name="HDFC ERGO (Known - Health)",
                url="https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
                document_type="known",
                weight=0.5,
                pages_estimate=67,
                domain="health_insurance",
                complexity_level="intermediate"
            ),
            DocumentInfo(
                name="ICICI LOMBARD (Unknown - Health)",
                url="https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/ICIHLIP22012V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
                document_type="unknown",
                weight=2.0,
                pages_estimate=78,
                domain="health_insurance",
                complexity_level="advanced"
            )
        ]
        
        # COMPREHENSIVE QUESTION BANK - 50+ questions across domains
        self.comprehensive_questions = [
            # === BASIC POLICY TERMS (Easy - Factual) ===
            TestQuestion(
                text="What is the grace period for premium payment under this policy?",
                category="grace_period",
                domain="general",
                weight=1.0,
                expected_keywords=["grace period", "thirty days", "30 days", "premium", "due date"],
                expected_answer_sample="A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
                difficulty_level="easy",
                question_type="factual"
            ),
            
            TestQuestion(
                text="What is the policy term duration mentioned in this document?",
                category="policy_term",
                domain="general",
                weight=1.0,
                expected_keywords=["policy term", "duration", "years", "annual", "renewable"],
                expected_answer_sample="The policy term is for one year and is renewable annually.",
                difficulty_level="easy",
                question_type="factual"
            ),
            
            TestQuestion(
                text="What is the minimum and maximum entry age for this policy?",
                category="entry_age",
                domain="general",
                weight=1.0,
                expected_keywords=["entry age", "minimum", "maximum", "years", "eligibility"],
                expected_answer_sample="The minimum entry age is 18 years and maximum entry age is 65 years.",
                difficulty_level="easy",
                question_type="factual"
            ),
            
            # === WAITING PERIODS (Medium - Factual) ===
            TestQuestion(
                text="What is the waiting period for pre-existing diseases (PED) to be covered?",
                category="waiting_period_ped",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["waiting period", "thirty-six", "36 months", "pre-existing", "continuous coverage"],
                expected_answer_sample="There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
                difficulty_level="medium",
                question_type="factual"
            ),
            
            TestQuestion(
                text="What is the waiting period for specific diseases like cataract surgery?",
                category="waiting_period_specific",
                domain="health_insurance",
                weight=1.5,
                expected_keywords=["waiting period", "two years", "2 years", "cataract surgery", "specific"],
                expected_answer_sample="The policy has a specific waiting period of two (2) years for cataract surgery.",
                difficulty_level="medium",
                question_type="factual"
            ),
            
            TestQuestion(
                text="Is there any waiting period for accidental injuries and emergency treatments?",
                category="waiting_period_emergency",
                domain="health_insurance",
                weight=1.5,
                expected_keywords=["no waiting period", "accidental", "emergency", "immediate", "coverage"],
                expected_answer_sample="There is no waiting period for accidental injuries and emergency treatments are covered immediately.",
                difficulty_level="medium",
                question_type="factual"
            ),
            
            # === MATERNITY & REPRODUCTIVE HEALTH (Hard - Analytical) ===
            TestQuestion(
                text="Does this policy cover maternity expenses, and what are the conditions?",
                category="maternity_coverage",
                domain="health_insurance",
                weight=2.5,
                expected_keywords=["maternity", "24 months", "continuously covered", "two deliveries", "childbirth"],
                expected_answer_sample="Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
                difficulty_level="hard",
                question_type="analytical"
            ),
            
            TestQuestion(
                text="What are the specific inclusions and exclusions for maternity benefits?",
                category="maternity_details",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["maternity", "inclusions", "exclusions", "normal delivery", "cesarean", "complications"],
                expected_answer_sample="Maternity benefits include normal delivery, cesarean section, and pregnancy-related complications. Exclusions may include fertility treatments and cosmetic procedures.",
                difficulty_level="hard",
                question_type="analytical"
            ),
            
            TestQuestion(
                text="Does the policy cover newborn baby expenses and for how long?",
                category="newborn_coverage",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["newborn", "baby", "coverage", "days", "automatic", "dependent"],
                expected_answer_sample="Yes, the policy covers newborn baby expenses from day one and the baby is automatically covered as a dependent.",
                difficulty_level="hard",
                question_type="analytical"
            ),
            
            # === ORGAN DONATION & TRANSPLANT (Expert - Procedural) ===
            TestQuestion(
                text="Are the medical expenses for an organ donor covered under this policy?",
                category="organ_donor",
                domain="health_insurance",
                weight=2.5,
                expected_keywords=["organ donor", "medical expenses", "harvesting", "Transplantation of Human Organs Act"],
                expected_answer_sample="Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
                difficulty_level="expert",
                question_type="procedural"
            ),
            
            TestQuestion(
                text="What are the conditions and limitations for organ transplant coverage?",
                category="organ_transplant",
                domain="health_insurance",
                weight=2.5,
                expected_keywords=["organ transplant", "conditions", "limitations", "medical necessity", "approval"],
                expected_answer_sample="Organ transplant coverage requires pre-authorization, medical necessity, and compliance with legal requirements. Coverage includes both donor and recipient expenses.",
                difficulty_level="expert",
                question_type="procedural"
            ),
            
            # === NO CLAIM DISCOUNT & RENEWALS (Medium - Factual) ===
            TestQuestion(
                text="What is the No Claim Discount (NCD) offered in this policy?",
                category="ncd",
                domain="general",
                weight=1.0,
                expected_keywords=["No Claim Discount", "5%", "base premium", "renewal", "claim-free"],
                expected_answer_sample="A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
                difficulty_level="medium",
                question_type="factual"
            ),
            
            TestQuestion(
                text="How does the No Claim Discount accumulate over multiple years?",
                category="ncd_accumulation",
                domain="general",
                weight=1.5,
                expected_keywords=["NCD", "accumulate", "years", "maximum", "percentage", "increase"],
                expected_answer_sample="The No Claim Discount accumulates annually for each claim-free year, typically increasing by 5% each year up to a maximum limit.",
                difficulty_level="medium",
                question_type="analytical"
            ),
            
            TestQuestion(
                text="What happens to the No Claim Discount if a claim is made?",
                category="ncd_claim_impact",
                domain="general",
                weight=1.5,
                expected_keywords=["claim made", "discount", "reset", "lost", "next renewal"],
                expected_answer_sample="If a claim is made during the policy year, the No Claim Discount is typically reset or reduced for the next renewal.",
                difficulty_level="medium",
                question_type="analytical"
            ),
            
            # === PREVENTIVE HEALTHCARE (Hard - Analytical) ===
            TestQuestion(
                text="Is there a benefit for preventive health check-ups?",
                category="health_checkup",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["health check-up", "two continuous policy years", "renewed without break", "Table of Benefits"],
                expected_answer_sample="Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
                difficulty_level="hard",
                question_type="analytical"
            ),
            
            TestQuestion(
                text="What specific preventive health services are covered under this policy?",
                category="preventive_services",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["preventive", "services", "vaccinations", "screenings", "annual", "checkup"],
                expected_answer_sample="The policy covers annual health checkups, vaccinations, cancer screenings, and other preventive diagnostic services as specified in the policy schedule.",
                difficulty_level="hard",
                question_type="analytical"
            ),
            
            # === HOSPITAL & PROVIDER NETWORK (Medium - Factual) ===
            TestQuestion(
                text="How does the policy define a 'Hospital'?",
                category="hospital_definition",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["10 inpatient beds", "15 beds", "qualified nursing staff", "operation theatre", "daily records"],
                expected_answer_sample="A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
                difficulty_level="medium",
                question_type="factual"
            ),
            
            TestQuestion(
                text="What is the difference between network and non-network hospitals?",
                category="hospital_network",
                domain="health_insurance",
                weight=1.5,
                expected_keywords=["network", "non-network", "cashless", "reimbursement", "settlement"],
                expected_answer_sample="Network hospitals offer cashless treatment with direct settlement, while non-network hospitals require reimbursement process where you pay first and claim later.",
                difficulty_level="medium",
                question_type="comparative"
            ),
            
            TestQuestion(
                text="Are there any restrictions on choice of hospitals or doctors?",
                category="hospital_choice",
                domain="health_insurance",
                weight=1.5,
                expected_keywords=["choice", "restrictions", "empanelled", "approved", "list"],
                expected_answer_sample="Treatment can be taken at any hospital meeting the policy definition, though cashless facility is available only at network hospitals.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            # === ALTERNATIVE TREATMENTS (Medium - Factual) ===
            TestQuestion(
                text="What is the extent of coverage for AYUSH treatments?",
                category="ayush_treatment",
                domain="health_insurance",
                weight=1.5,
                expected_keywords=["AYUSH", "Ayurveda", "Yoga", "Naturopathy", "Unani", "Siddha", "Homeopathy", "Sum Insured limit"],
                expected_answer_sample="The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
                difficulty_level="medium",
                question_type="factual"
            ),
            
            TestQuestion(
                text="Are there specific conditions for AYUSH treatment coverage?",
                category="ayush_conditions",
                domain="health_insurance",
                weight=1.5,
                expected_keywords=["AYUSH", "conditions", "qualified", "practitioner", "hospital", "inpatient"],
                expected_answer_sample="AYUSH treatment coverage requires treatment by qualified practitioners in recognized AYUSH hospitals and is typically limited to inpatient treatments.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            # === ROOM RENT & SUB-LIMITS (Hard - Analytical) ===
            TestQuestion(
                text="Are there any sub-limits on room rent and ICU charges for Plan A?",
                category="room_rent_limits",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["Plan A", "1%", "2%", "Sum Insured", "room rent", "ICU charges", "Preferred Provider Network"],
                expected_answer_sample="Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN).",
                difficulty_level="hard",
                question_type="analytical"
            ),
            
            TestQuestion(
                text="How do room rent restrictions affect other medical expenses?",
                category="room_rent_impact",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["room rent", "proportionate", "deduction", "medical expenses", "ratio"],
                expected_answer_sample="If room rent exceeds the policy limit, all other medical expenses may be reduced proportionately based on the ratio of eligible to actual room rent.",
                difficulty_level="hard",
                question_type="analytical"
            ),
            
            # === CLAIMS PROCESS (Medium - Procedural) ===
            TestQuestion(
                text="What is the process for cashless claim settlement?",
                category="cashless_claims",
                domain="general",
                weight=1.5,
                expected_keywords=["cashless", "pre-authorization", "TPA", "approval", "discharge"],
                expected_answer_sample="For cashless claims, obtain pre-authorization from TPA, get approval before treatment, and settle directly with the hospital at discharge.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            TestQuestion(
                text="What documents are required for reimbursement claims?",
                category="reimbursement_documents",
                domain="general",
                weight=1.5,
                expected_keywords=["documents", "bills", "discharge summary", "diagnostic reports", "claim form"],
                expected_answer_sample="Required documents include original bills, discharge summary, diagnostic reports, claim form, and policy documents.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            TestQuestion(
                text="What is the time limit for intimating claims to the insurance company?",
                category="claim_intimation",
                domain="general",
                weight=1.5,
                expected_keywords=["intimation", "time limit", "days", "notice", "delay"],
                expected_answer_sample="Claims must be intimated to the insurance company within 30 days of hospitalization or as soon as reasonably possible.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            # === EXCLUSIONS & LIMITATIONS (Expert - Analytical) ===
            TestQuestion(
                text="What are the major exclusions under this health insurance policy?",
                category="major_exclusions",
                domain="health_insurance",
                weight=2.5,
                expected_keywords=["exclusions", "cosmetic", "fertility", "self-inflicted", "war", "nuclear"],
                expected_answer_sample="Major exclusions include cosmetic surgery, fertility treatments, self-inflicted injuries, war risks, nuclear hazards, and experimental treatments.",
                difficulty_level="expert",
                question_type="analytical"
            ),
            
            TestQuestion(
                text="Are mental health and psychiatric treatments covered?",
                category="mental_health",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["mental health", "psychiatric", "psychological", "covered", "excluded"],
                expected_answer_sample="Mental health and psychiatric treatments are covered subject to policy terms and may have specific sub-limits or conditions.",
                difficulty_level="expert",
                question_type="analytical"
            ),
            
            TestQuestion(
                text="Does the policy cover treatment for congenital diseases?",
                category="congenital_diseases",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["congenital", "birth defects", "genetic", "covered", "excluded"],
                expected_answer_sample="Coverage for congenital diseases varies by policy and may be excluded or covered with waiting periods and sub-limits.",
                difficulty_level="expert",
                question_type="analytical"
            ),
            
            # === PREMIUM & PAYMENT (Easy - Factual) ===
            TestQuestion(
                text="What are the available modes of premium payment?",
                category="premium_payment",
                domain="general",
                weight=1.0,
                expected_keywords=["premium", "payment", "annual", "monthly", "quarterly", "online"],
                expected_answer_sample="Premium can be paid annually, and may also offer monthly or quarterly payment options through various channels including online payment.",
                difficulty_level="easy",
                question_type="factual"
            ),
            
            TestQuestion(
                text="Is there any discount for online premium payment?",
                category="online_discount",
                domain="general",
                weight=1.0,
                expected_keywords=["online", "discount", "premium", "digital", "payment"],
                expected_answer_sample="Some policies offer discounts for online premium payment as part of digital initiatives.",
                difficulty_level="easy",
                question_type="factual"
            ),
            
            # === POLICY BENEFITS & FEATURES (Medium - Comparative) ===
            TestQuestion(
                text="What are the key benefits and features of this insurance policy?",
                category="key_benefits",
                domain="general",
                weight=1.5,
                expected_keywords=["benefits", "features", "coverage", "services", "advantages"],
                expected_answer_sample="Key benefits include comprehensive health coverage, cashless treatment, preventive care, maternity benefits, and AYUSH treatment coverage.",
                difficulty_level="medium",
                question_type="comparative"
            ),
            
            TestQuestion(
                text="Does the policy offer any value-added services?",
                category="value_added_services",
                domain="general",
                weight=1.5,
                expected_keywords=["value-added", "services", "teleconsultation", "second opinion", "wellness"],
                expected_answer_sample="The policy may offer value-added services like teleconsultation, second medical opinion, health check-ups, and wellness programs.",
                difficulty_level="medium",
                question_type="comparative"
            ),
            
            # === INTERNATIONAL COVERAGE (Expert - Procedural) ===
            TestQuestion(
                text="Does this policy provide coverage for treatment abroad?",
                category="international_coverage",
                domain="health_insurance",
                weight=2.5,
                expected_keywords=["abroad", "international", "overseas", "emergency", "coverage"],
                expected_answer_sample="International coverage may be available for emergency treatments abroad, subject to specific terms and sub-limits.",
                difficulty_level="expert",
                question_type="procedural"
            ),
            
            # === CO-PAYMENT & DEDUCTIBLES (Hard - Analytical) ===
            TestQuestion(
                text="Are there any co-payment or deductible clauses in this policy?",
                category="copayment_deductible",
                domain="health_insurance",
                weight=2.0,
                expected_keywords=["co-payment", "deductible", "percentage", "amount", "share"],
                expected_answer_sample="The policy may have co-payment clauses where the insured bears a percentage of the claim amount, particularly for certain age groups or treatments.",
                difficulty_level="hard",
                question_type="analytical"
            ),
            
            # === FAMILY COVERAGE (Medium - Comparative) ===
            TestQuestion(
                text="How does family coverage work under this policy?",
                category="family_coverage",
                domain="health_insurance",
                weight=1.5,
                expected_keywords=["family", "coverage", "floater", "individual", "members"],
                expected_answer_sample="Family coverage typically works as a floater policy where the sum insured is shared among all family members, or individual limits may apply.",
                difficulty_level="medium",
                question_type="comparative"
            ),
            
            TestQuestion(
                text="Can dependents be added or removed during the policy term?",
                category="dependent_changes",
                domain="health_insurance",
                weight=1.5,
                expected_keywords=["dependents", "add", "remove", "policy term", "mid-term"],
                expected_answer_sample="Dependents can typically be added at renewal, while mid-term additions may be allowed for specific life events like marriage or childbirth.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            # === RENEWAL & PORTABILITY (Medium - Procedural) ===
            TestQuestion(
                text="What are the conditions for policy renewal?",
                category="renewal_conditions",
                domain="general",
                weight=1.5,
                expected_keywords=["renewal", "conditions", "automatic", "premium", "terms"],
                expected_answer_sample="Policy renewal is typically guaranteed with payment of renewal premium, though terms and conditions may be revised.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            TestQuestion(
                text="Is policy portability allowed under this insurance?",
                category="portability",
                domain="general",
                weight=1.5,
                expected_keywords=["portability", "transfer", "insurer", "waiting period", "credit"],
                expected_answer_sample="Policy portability is allowed subject to IRDAI guidelines, with credit for waiting periods served and continuous coverage.",
                difficulty_level="medium",
                question_type="procedural"
            ),
            
            # === COMPLEX ANALYTICAL QUESTIONS (Expert) ===
            TestQuestion(
                text="How does this policy handle pre-existing conditions differently from new illnesses?",
                category="ped_vs_new",
                domain="health_insurance",
                weight=3.0,
                expected_keywords=["pre-existing", "new illness", "waiting period", "immediate", "coverage"],
                expected_answer_sample="Pre-existing conditions have a waiting period of 36 months while new illnesses are covered immediately after the initial waiting period, demonstrating the policy's risk-based approach.",
                difficulty_level="expert",
                question_type="comparative"
            ),
            
            TestQuestion(
                text="What is the relationship between sum insured and various sub-limits in the policy?",
                category="sum_insured_sublimits",
                domain="health_insurance",
                weight=3.0,
                expected_keywords=["sum insured", "sub-limits", "percentage", "relationship", "overall"],
                expected_answer_sample="Sub-limits are typically expressed as percentages of the sum insured and may apply to specific treatments, ensuring the overall sum insured is managed across different coverage areas.",
                difficulty_level="expert",
                question_type="analytical"
            )
        ]
        
        # Performance tracking
        self.test_results = {
            "test_metadata": {
                "user": "vkhare2909",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "test_type": "ULTIMATE_COMPREHENSIVE_MULTI_DOMAIN",
                "total_questions": len(self.comprehensive_questions),
                "total_documents": len(self.test_documents),
                "domains_covered": list(set([q.domain for q in self.comprehensive_questions])),
                "difficulty_levels": list(set([q.difficulty_level for q in self.comprehensive_questions])),
                "question_types": list(set([q.question_type for q in self.comprehensive_questions]))
            },
            "domain_analysis": {},
            "difficulty_analysis": {},
            "question_type_analysis": {},
            "document_performance": {},
            "comprehensive_metrics": {},
            "final_assessment": {}
        }
    
    async def run_ultimate_comprehensive_test(self):
        """Run ultimate comprehensive test across all domains and difficulties"""
        
        print("🏆 ULTIMATE COMPREHENSIVE MULTI-DOMAIN FINAL TESTER")
        print("="*100)
        print(f"👤 User: vkhare2909")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"🎯 Target: 90%+ Accuracy & <15s Response")
        print(f"📋 Total Documents: {len(self.test_documents)}")
        print(f"❓ Total Questions: {len(self.comprehensive_questions)}")
        print(f"🌐 Domains: {', '.join(self.test_results['test_metadata']['domains_covered'])}")
        print(f"⚡ Difficulty Levels: {', '.join(self.test_results['test_metadata']['difficulty_levels'])}")
        print(f"📊 Question Types: {', '.join(self.test_results['test_metadata']['question_types'])}")
        print("="*100)
        
        # Phase 1: Comprehensive Primary Test
        await self._phase1_comprehensive_primary_test()
        
        # Phase 2: Domain-wise Analysis
        await self._phase2_domain_wise_analysis()
        
        # Phase 3: Difficulty Level Analysis
        await self._phase3_difficulty_analysis()
        
        # Phase 4: Question Type Analysis
        await self._phase4_question_type_analysis()
        
        # Phase 5: Document-wise Performance
        await self._phase5_document_performance()
        
        # Phase 6: Final Comprehensive Assessment
        self._phase6_final_comprehensive_assessment()
        
        return self._generate_comprehensive_report()
    
    async def _phase1_comprehensive_primary_test(self):
        """Phase 1: Comprehensive primary test with all questions"""
        print("\n🎯 PHASE 1: COMPREHENSIVE PRIMARY TEST")
        print("-" * 80)
        print("Testing with BAJAJ ALLIANZ document using ALL comprehensive questions...")
        
        baseline_doc = self.test_documents[0]  # BAJAJ ALLIANZ
        
        print(f"📄 Testing with: {baseline_doc.name}")
        print(f"🏷️ Document Type: {baseline_doc.document_type.upper()} (Weight: {baseline_doc.weight}x)")
        print(f"🌐 Domain: {baseline_doc.domain}")
        print(f"📊 Complexity: {baseline_doc.complexity_level}")
        
        questions_text = [q.text for q in self.comprehensive_questions]
        
        print(f"❓ Total Questions: {len(questions_text)}")
        if len(questions_text) > 20:
            print(f"📦 Will be processed in batches of 20 questions")
        
        try:
            start_time = time.time()
            response = await self._test_api_request(baseline_doc.url, questions_text)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            if response and response.get("answers"):
                answers = response["answers"]
                
                print(f"\n📊 COMPREHENSIVE PRIMARY TEST RESULTS:")
                print(f"  ⏱️ Total Time: {total_time:.1f}s")
                print(f"  📝 Questions: {len(questions_text)}")
                print(f"  💬 Answers: {len(answers)}")
                print(f"  ⚡ Avg Time/Q: {total_time/len(questions_text):.1f}s")
                print(f"  🎯 Speed Target: {'✅ MET' if total_time < 15 else '❌ MISSED'} (<15s)")
                
                # Comprehensive accuracy analysis
                print(f"\n🔍 COMPREHENSIVE ACCURACY ANALYSIS:")
                
                total_questions = len(self.comprehensive_questions)
                correct_count = 0
                partial_count = 0
                incorrect_count = 0
                
                domain_scores = {}
                difficulty_scores = {}
                question_type_scores = {}
                category_scores = {}
                individual_scores = []
                
                for i, (question, answer) in enumerate(zip(self.comprehensive_questions, answers)):
                    if i < len(answers):
                        try:
                            accuracy_score = self._calculate_comprehensive_accuracy(question, answer)
                            individual_scores.append(accuracy_score)
                            
                            # Categorize accuracy
                            if accuracy_score >= 0.8:
                                correct_count += 1
                                status = "✅ CORRECT"
                            elif accuracy_score >= 0.5:
                                partial_count += 1
                                status = "🟡 PARTIAL"
                            else:
                                incorrect_count += 1
                                status = "❌ INCORRECT"
                            
                            # Track by domain
                            domain = question.domain
                            if domain not in domain_scores:
                                domain_scores[domain] = []
                            domain_scores[domain].append(accuracy_score)
                            
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
                            
                            print(f"  Q{i+1:2d} ({question.difficulty_level:6}|{question.domain:15}|{question.category:20}): {accuracy_score:.1%} {status}")
                            
                        except Exception as e:
                            print(f"  Q{i+1:2d} ERROR: {str(e)[:50]}")
                            incorrect_count += 1
                            individual_scores.append(0.0)
                
                # Calculate overall accuracy
                if total_questions > 0:
                    overall_accuracy = (correct_count + (partial_count * 0.5)) / total_questions
                else:
                    overall_accuracy = 0.0
                
                print(f"\n📈 COMPREHENSIVE ACCURACY SUMMARY:")
                print(f"  🎯 Overall Accuracy: {overall_accuracy:.1%}")
                print(f"  ✅ Correct Answers: {correct_count}/{total_questions} ({correct_count/total_questions:.1%})")
                print(f"  🟡 Partial Answers: {partial_count}/{total_questions} ({partial_count/total_questions:.1%})")
                print(f"  ❌ Incorrect Answers: {incorrect_count}/{total_questions} ({incorrect_count/total_questions:.1%})")
                
                # Domain-wise accuracy
                print(f"\n🌐 DOMAIN-WISE ACCURACY:")
                for domain, scores in domain_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        print(f"  {domain:20}: {avg_score:.1%} ({len(scores)} questions)")
                
                # Difficulty-wise accuracy
                print(f"\n⚡ DIFFICULTY-WISE ACCURACY:")
                for difficulty, scores in difficulty_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        print(f"  {difficulty.title():10}: {avg_score:.1%} ({len(scores)} questions)")
                
                # Question type accuracy
                print(f"\n📊 QUESTION TYPE ACCURACY:")
                for q_type, scores in question_type_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        print(f"  {q_type.title():12}: {avg_score:.1%} ({len(scores)} questions)")
                
                # Top and bottom performing categories
                category_avgs = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items() if scores}
                if category_avgs:
                    sorted_categories = sorted(category_avgs.items(), key=lambda x: x[1], reverse=True)
                    
                    print(f"\n🏆 TOP 5 PERFORMING CATEGORIES:")
                    for cat, score in sorted_categories[:5]:
                        print(f"  {cat:25}: {score:.1%}")
                    
                    print(f"\n⚠️ BOTTOM 5 PERFORMING CATEGORIES:")
                    for cat, score in sorted_categories[-5:]:
                        print(f"  {cat:25}: {score:.1%}")
                
                # Store primary results
                self.test_results["primary_comprehensive_results"] = {
                    "total_time": total_time,
                    "avg_time_per_question": total_time / len(questions_text) if len(questions_text) > 0 else 0,
                    "overall_accuracy": overall_accuracy,
                    "correct_count": correct_count,
                    "partial_count": partial_count,
                    "incorrect_count": incorrect_count,
                    "total_questions": total_questions,
                    "individual_scores": individual_scores,
                    "domain_accuracy": {domain: (sum(scores)/len(scores) if scores else 0) for domain, scores in domain_scores.items()},
                    "difficulty_accuracy": {diff: (sum(scores)/len(scores) if scores else 0) for diff, scores in difficulty_scores.items()},
                    "question_type_accuracy": {q_type: (sum(scores)/len(scores) if scores else 0) for q_type, scores in question_type_scores.items()},
                    "category_accuracy": {cat: (sum(scores)/len(scores) if scores else 0) for cat, scores in category_scores.items()},
                    "target_met": {
                        "speed": total_time < 15,
                        "accuracy": overall_accuracy >= 0.9
                    },
                    "status": "success"
                }
                
            else:
                print("❌ Comprehensive primary test failed - no valid response")
                self.test_results["primary_comprehensive_results"] = {
                    "status": "failed", 
                    "error": "No valid response from API"
                }
                
        except Exception as e:
            print(f"❌ Comprehensive primary test error: {e}")
            self.test_results["primary_comprehensive_results"] = {
                "status": "error", 
                "error": str(e)
            }
    
    async def _phase2_domain_wise_analysis(self):
        """Phase 2: Domain-wise analysis with subset of questions"""
        print("\n🌐 PHASE 2: DOMAIN-WISE ANALYSIS")
        print("-" * 80)
        
        # Group questions by domain
        domain_questions = {}
        for question in self.comprehensive_questions:
            domain = question.domain
            if domain not in domain_questions:
                domain_questions[domain] = []
            domain_questions[domain].append(question)
        
        domain_results = {}
        
        for domain, questions in domain_questions.items():
            print(f"\n📋 Testing Domain: {domain.upper()}")
            print(f"  📊 Questions in domain: {len(questions)}")
            
            # Test with first 5 questions from each domain
            test_questions = questions[:5]
            questions_text = [q.text for q in test_questions]
            
            # Test with different documents for variety
            test_doc = self.test_documents[0]  # Use primary document
            
            try:
                start_time = time.time()
                response = await self._test_api_request(test_doc.url, questions_text)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                if response and response.get("answers"):
                    answers = response["answers"]
                    
                    # Calculate domain-specific accuracy
                    domain_correct = 0
                    domain_partial = 0
                    domain_total = len(test_questions)
                    domain_scores = []
                    
                    for i, (question, answer) in enumerate(zip(test_questions, answers)):
                        if i < len(answers):
                            try:
                                accuracy_score = self._calculate_comprehensive_accuracy(question, answer)
                                domain_scores.append(accuracy_score)
                                
                                if accuracy_score >= 0.8:
                                    domain_correct += 1
                                elif accuracy_score >= 0.5:
                                    domain_partial += 1
                            except Exception as e:
                                print(f"    Q{i+1} accuracy calculation error: {e}")
                                domain_scores.append(0.0)
                    
                    if domain_total > 0:
                        domain_accuracy = (domain_correct + (domain_partial * 0.5)) / domain_total
                    else:
                        domain_accuracy = 0.0
                    
                    print(f"  ⚡ Response Time: {response_time:.1f}s")
                    print(f"  🎯 Domain Accuracy: {domain_accuracy:.1%}")
                    print(f"  ✅ Correct: {domain_correct}/{domain_total}")
                    print(f"  🟡 Partial: {domain_partial}/{domain_total}")
                    
                    domain_results[domain] = {
                        "accuracy": domain_accuracy,
                        "response_time": response_time,
                        "correct_count": domain_correct,
                        "partial_count": domain_partial,
                        "total_count": domain_total,
                        "individual_scores": domain_scores,
                        "questions_tested": len(test_questions),
                        "status": "success"
                    }
                    
                else:
                    print(f"  ❌ Failed - No valid response")
                    domain_results[domain] = {
                        "status": "failed",
                        "error": "No valid response from API"
                    }
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:50]}...")
                domain_results[domain] = {
                    "status": "error", 
                    "error": str(e)[:100]
                }
        
        self.test_results["domain_analysis"] = domain_results
    
    async def _phase3_difficulty_analysis(self):
        """Phase 3: Difficulty level analysis"""
        print("\n⚡ PHASE 3: DIFFICULTY LEVEL ANALYSIS")
        print("-" * 80)
        
        # Group questions by difficulty
        difficulty_questions = {}
        for question in self.comprehensive_questions:
            difficulty = question.difficulty_level
            if difficulty not in difficulty_questions:
                difficulty_questions[difficulty] = []
            difficulty_questions[difficulty].append(question)
        
        difficulty_results = {}
        
        for difficulty, questions in difficulty_questions.items():
            print(f"\n📊 Testing Difficulty: {difficulty.upper()}")
            print(f"  📊 Questions at this level: {len(questions)}")
            
            # Test with subset of questions from each difficulty
            test_questions = questions[:8]  # Test more for comprehensive analysis
            questions_text = [q.text for q in test_questions]
            
            test_doc = self.test_documents[0]
            
            try:
                start_time = time.time()
                response = await self._test_api_request(test_doc.url, questions_text)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                if response and response.get("answers"):
                    answers = response["answers"]
                    
                    # Calculate difficulty-specific metrics
                    diff_correct = 0
                    diff_partial = 0
                    diff_total = len(test_questions)
                    diff_scores = []
                    
                    for i, (question, answer) in enumerate(zip(test_questions, answers)):
                        if i < len(answers):
                            try:
                                accuracy_score = self._calculate_comprehensive_accuracy(question, answer)
                                diff_scores.append(accuracy_score)
                                
                                if accuracy_score >= 0.8:
                                    diff_correct += 1
                                elif accuracy_score >= 0.5:
                                    diff_partial += 1
                            except Exception as e:
                                diff_scores.append(0.0)
                    
                    if diff_total > 0:
                        diff_accuracy = (diff_correct + (diff_partial * 0.5)) / diff_total
                        avg_score = sum(diff_scores) / len(diff_scores) if diff_scores else 0
                        score_variance = statistics.variance(diff_scores) if len(diff_scores) > 1 else 0
                    else:
                        diff_accuracy = 0.0
                        avg_score = 0.0
                        score_variance = 0.0
                    
                    print(f"  ⚡ Response Time: {response_time:.1f}s")
                    print(f"  🎯 Difficulty Accuracy: {diff_accuracy:.1%}")
                    print(f"  📊 Average Score: {avg_score:.1%}")
                    print(f"  📈 Score Variance: {score_variance:.3f}")
                    print(f"  ✅ Correct: {diff_correct}/{diff_total}")
                    print(f"  🟡 Partial: {diff_partial}/{diff_total}")
                    
                    difficulty_results[difficulty] = {
                        "accuracy": diff_accuracy,
                        "average_score": avg_score,
                        "score_variance": score_variance,
                        "response_time": response_time,
                        "correct_count": diff_correct,
                        "partial_count": diff_partial,
                        "total_count": diff_total,
                        "individual_scores": diff_scores,
                        "questions_tested": len(test_questions),
                        "status": "success"
                    }
                    
                else:
                    difficulty_results[difficulty] = {
                        "status": "failed",
                        "error": "No valid response from API"
                    }
                    
            except Exception as e:
                difficulty_results[difficulty] = {
                    "status": "error", 
                    "error": str(e)[:100]
                }
        
        self.test_results["difficulty_analysis"] = difficulty_results
    
    async def _phase4_question_type_analysis(self):
        """Phase 4: Question type analysis"""
        print("\n📊 PHASE 4: QUESTION TYPE ANALYSIS")
        print("-" * 80)
        
        # Group questions by type
        type_questions = {}
        for question in self.comprehensive_questions:
            q_type = question.question_type
            if q_type not in type_questions:
                type_questions[q_type] = []
            type_questions[q_type].append(question)
        
        type_results = {}
        
        for q_type, questions in type_questions.items():
            print(f"\n📝 Testing Question Type: {q_type.upper()}")
            print(f"  📊 Questions of this type: {len(questions)}")
            
            # Test with subset of questions from each type
            test_questions = questions[:6]
            questions_text = [q.text for q in test_questions]
            
            test_doc = self.test_documents[0]
            
            try:
                start_time = time.time()
                response = await self._test_api_request(test_doc.url, questions_text)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                if response and response.get("answers"):
                    answers = response["answers"]
                    
                    # Calculate type-specific metrics
                    type_correct = 0
                    type_partial = 0
                    type_total = len(test_questions)
                    type_scores = []
                    
                    for i, (question, answer) in enumerate(zip(test_questions, answers)):
                        if i < len(answers):
                            try:
                                accuracy_score = self._calculate_comprehensive_accuracy(question, answer)
                                type_scores.append(accuracy_score)
                                
                                if accuracy_score >= 0.8:
                                    type_correct += 1
                                elif accuracy_score >= 0.5:
                                    type_partial += 1
                            except Exception as e:
                                type_scores.append(0.0)
                    
                    if type_total > 0:
                        type_accuracy = (type_correct + (type_partial * 0.5)) / type_total
                    else:
                        type_accuracy = 0.0
                    
                    print(f"  ⚡ Response Time: {response_time:.1f}s")
                    print(f"  🎯 Type Accuracy: {type_accuracy:.1%}")
                    print(f"  ✅ Correct: {type_correct}/{type_total}")
                    print(f"  🟡 Partial: {type_partial}/{type_total}")
                    
                    type_results[q_type] = {
                        "accuracy": type_accuracy,
                        "response_time": response_time,
                        "correct_count": type_correct,
                        "partial_count": type_partial,
                        "total_count": type_total,
                        "individual_scores": type_scores,
                        "questions_tested": len(test_questions),
                        "status": "success"
                    }
                    
                else:
                    type_results[q_type] = {
                        "status": "failed",
                        "error": "No valid response from API"
                    }
                    
            except Exception as e:
                type_results[q_type] = {
                    "status": "error", 
                    "error": str(e)[:100]
                }
        
        self.test_results["question_type_analysis"] = type_results
    
    async def _phase5_document_performance(self):
        """Phase 5: Document-wise performance analysis"""
        print("\n📄 PHASE 5: DOCUMENT PERFORMANCE ANALYSIS")
        print("-" * 80)
        
        # Test subset of key questions with each document
        key_questions = self.comprehensive_questions[:10]  # First 10 comprehensive questions
        questions_text = [q.text for q in key_questions]
        
        document_results = {}
        
        for doc in self.test_documents:
            print(f"\n📋 Testing: {doc.name}")
            print(f"  🏷️ Type: {doc.document_type.upper()} (Weight: {doc.weight}x)")
            print(f"  🌐 Domain: {doc.domain}")
            print(f"  📊 Complexity: {doc.complexity_level}")
            
            try:
                start_time = time.time()
                response = await self._test_api_request(doc.url, questions_text)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                if response and response.get("answers"):
                    answers = response["answers"]
                    
                    # Calculate document-specific accuracy
                    doc_correct = 0
                    doc_partial = 0
                    doc_total = len(key_questions)
                    doc_scores = []
                    
                    for i, (question, answer) in enumerate(zip(key_questions, answers)):
                        if i < len(answers):
                            try:
                                accuracy_score = self._calculate_comprehensive_accuracy(question, answer)
                                doc_scores.append(accuracy_score)
                                
                                if accuracy_score >= 0.8:
                                    doc_correct += 1
                                elif accuracy_score >= 0.5:
                                    doc_partial += 1
                            except Exception as e:
                                doc_scores.append(0.0)
                    
                    if doc_total > 0:
                        doc_accuracy = (doc_correct + (doc_partial * 0.5)) / doc_total
                        weighted_accuracy = doc_accuracy * doc.weight
                    else:
                        doc_accuracy = 0.0
                        weighted_accuracy = 0.0
                    
                    print(f"  ⚡ Response Time: {response_time:.1f}s")
                    print(f"  🎯 Document Accuracy: {doc_accuracy:.1%}")
                    print(f"  ⭐ Weighted Accuracy: {weighted_accuracy:.1%}")
                    print(f"  ✅ Correct: {doc_correct}/{doc_total}")
                    print(f"  🟡 Partial: {doc_partial}/{doc_total}")
                    
                    document_results[doc.name] = {
                        "accuracy": doc_accuracy,
                        "weighted_accuracy": weighted_accuracy,
                        "response_time": response_time,
                        "correct_count": doc_correct,
                        "partial_count": doc_partial,
                        "total_count": doc_total,
                        "individual_scores": doc_scores,
                        "document_type": doc.document_type,
                        "domain": doc.domain,
                        "complexity_level": doc.complexity_level,
                        "weight": doc.weight,
                        "status": "success"
                    }
                    
                else:
                    print(f"  ❌ Failed - No valid response")
                    document_results[doc.name] = {
                        "status": "failed",
                        "error": "No valid response from API"
                    }
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:50]}...")
                document_results[doc.name] = {
                    "status": "error", 
                    "error": str(e)[:100]
                }
        
        self.test_results["document_performance"] = document_results
    
    def _phase6_final_comprehensive_assessment(self):
        """Phase 6: Final comprehensive assessment"""
        print("\n🏆 PHASE 6: FINAL COMPREHENSIVE ASSESSMENT")
        print("-" * 80)
        
        # Extract metrics from all phases
        primary_results = self.test_results.get("primary_comprehensive_results", {})
        domain_results = self.test_results.get("domain_analysis", {})
        difficulty_results = self.test_results.get("difficulty_analysis", {})
        type_results = self.test_results.get("question_type_analysis", {})
        document_results = self.test_results.get("document_performance", {})
        
        # Calculate comprehensive metrics
        final_metrics = {
            "overall_accuracy": 0.0,
            "speed_performance": 0.0,
            "domain_performance": {},
            "difficulty_performance": {},
            "type_performance": {},
            "document_performance": {},
            "comprehensive_score": 0.0
        }
        
        if primary_results.get("overall_accuracy") is not None:
            final_metrics["overall_accuracy"] = primary_results["overall_accuracy"]
            final_metrics["speed_performance"] = primary_results.get("avg_time_per_question", 0)
        
        # Domain performance
        for domain, results in domain_results.items():
            if results.get("accuracy") is not None:
                final_metrics["domain_performance"][domain] = results["accuracy"]
        
        # Difficulty performance
        for difficulty, results in difficulty_results.items():
            if results.get("accuracy") is not None:
                final_metrics["difficulty_performance"][difficulty] = results["accuracy"]
        
        # Question type performance
        for q_type, results in type_results.items():
            if results.get("accuracy") is not None:
                final_metrics["type_performance"][q_type] = results["accuracy"]
        
        # Document performance
        for doc_name, results in document_results.items():
            if results.get("accuracy") is not None:
                final_metrics["document_performance"][doc_name] = results["accuracy"]
        
        # Calculate comprehensive score
        accuracy_weight = 0.6
        speed_weight = 0.2
        consistency_weight = 0.2
        
        accuracy_score = final_metrics["overall_accuracy"]
        speed_score = min(1.0, 15 / max(final_metrics["speed_performance"], 1))  # Normalized speed score
        
        # Consistency score based on variance across domains/difficulties
        domain_accuracies = list(final_metrics["domain_performance"].values())
        difficulty_accuracies = list(final_metrics["difficulty_performance"].values())
        
        if domain_accuracies and difficulty_accuracies:
            domain_variance = statistics.variance(domain_accuracies) if len(domain_accuracies) > 1 else 0
            difficulty_variance = statistics.variance(difficulty_accuracies) if len(difficulty_accuracies) > 1 else 0
            consistency_score = max(0, 1 - (domain_variance + difficulty_variance) / 2)
        else:
            consistency_score = 0.5
        
        comprehensive_score = (accuracy_score * accuracy_weight + 
                             speed_score * speed_weight + 
                             consistency_score * consistency_weight)
        
        final_metrics["comprehensive_score"] = comprehensive_score
        
        # Print detailed assessment
        print(f"📊 COMPREHENSIVE PERFORMANCE METRICS:")
        print(f"  🎯 Overall Accuracy: {final_metrics['overall_accuracy']:.1%}")
        print(f"  ⚡ Speed Performance: {final_metrics['speed_performance']:.1f}s per question")
        print(f"  🏆 Comprehensive Score: {comprehensive_score:.1%}")
        
        if final_metrics["domain_performance"]:
            print(f"\n🌐 DOMAIN PERFORMANCE:")
            for domain, accuracy in final_metrics["domain_performance"].items():
                print(f"  {domain:20}: {accuracy:.1%}")
        
        if final_metrics["difficulty_performance"]:
            print(f"\n⚡ DIFFICULTY PERFORMANCE:")
            for difficulty, accuracy in final_metrics["difficulty_performance"].items():
                print(f"  {difficulty.title():10}: {accuracy:.1%}")
        
        if final_metrics["type_performance"]:
            print(f"\n📊 QUESTION TYPE PERFORMANCE:")
            for q_type, accuracy in final_metrics["type_performance"].items():
                print(f"  {q_type.title():12}: {accuracy:.1%}")
        
        # Final grade assignment
        if comprehensive_score >= 0.95:
            final_grade = "A+ (95%+) 🏆 CHAMPIONSHIP ELITE"
        elif comprehensive_score >= 0.9:
            final_grade = "A (90-94%) 🏆 CHAMPIONSHIP READY"
        elif comprehensive_score >= 0.85:
            final_grade = "A- (85-89%) ⭐ EXCELLENT"
        elif comprehensive_score >= 0.8:
            final_grade = "B+ (80-84%) ✅ VERY GOOD"
        elif comprehensive_score >= 0.75:
            final_grade = "B (75-79%) ✅ GOOD"
        elif comprehensive_score >= 0.7:
            final_grade = "B- (70-74%) 🟡 ACCEPTABLE"
        else:
            final_grade = "C (<70%) ❌ NEEDS IMPROVEMENT"
        
        print(f"\n🎯 FINAL COMPREHENSIVE ASSESSMENT:")
        print(f"  📊 Comprehensive Score: {comprehensive_score:.1%}")
        print(f"  🏆 Grade: {final_grade}")
        print(f"  🎯 Hackathon Target (90%): {'✅ ACHIEVED' if comprehensive_score >= 0.9 else '❌ MISSED'}")
        
        # Readiness assessment
        readiness_criteria = {
            "accuracy_target": comprehensive_score >= 0.9,
            "speed_target": final_metrics["speed_performance"] < 15,
            "domain_coverage": len(final_metrics["domain_performance"]) >= 2,
            "difficulty_coverage": len(final_metrics["difficulty_performance"]) >= 3,
            "consistency": consistency_score >= 0.7
        }
        
        criteria_met = sum(1 for passed in readiness_criteria.values() if passed)
        total_criteria = len(readiness_criteria)
        
        print(f"\n📋 HACKATHON READINESS CRITERIA:")
        for criterion, passed in readiness_criteria.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion.replace('_', ' ').title()}")
        
        print(f"\n🚀 READINESS ASSESSMENT:")
        print(f"  📊 Criteria Met: {criteria_met}/{total_criteria}")
        print(f"  📈 Readiness Score: {(criteria_met/total_criteria)*100:.0f}%")
        
        if criteria_met >= 4 and comprehensive_score >= 0.9:
            readiness_level = "🏆 CHAMPIONSHIP READY"
            recommendation = "DEPLOY IMMEDIATELY! System exceeds all targets and ready for TOP 3 position!"
        elif criteria_met >= 3 and comprehensive_score >= 0.8:
            readiness_level = "✅ HACKATHON READY"
            recommendation = "Deploy with confidence! System meets most requirements and ready for competition!"
        elif criteria_met >= 2:
            readiness_level = "🟡 NEEDS MINOR TUNING"
            recommendation = "Minor optimizations needed before deployment."
        else:
            readiness_level = "🔧 NEEDS MAJOR IMPROVEMENT"
            recommendation = "Significant improvements required before deployment."
        
        print(f"  🎯 Readiness Level: {readiness_level}")
        print(f"  💡 Recommendation: {recommendation}")
        
        self.test_results["comprehensive_metrics"] = final_metrics
        self.test_results["final_assessment"] = {
            "comprehensive_score": comprehensive_score,
            "final_grade": final_grade,
            "readiness_level": readiness_level,
            "recommendation": recommendation,
            "criteria_met": criteria_met,
            "total_criteria": total_criteria,
            "readiness_criteria": readiness_criteria,
            "target_achievement": {
                "accuracy": comprehensive_score >= 0.9,
                "speed": final_metrics["speed_performance"] < 15,
                "overall": comprehensive_score >= 0.9 and final_metrics["speed_performance"] < 15
            }
        }
    
    def _calculate_comprehensive_accuracy(self, question: TestQuestion, answer: str) -> float:
        """Enhanced accuracy calculation for comprehensive testing"""
        
        if not answer or len(answer) < 10:
            return 0.0
        
        try:
            answer_lower = answer.lower()
            score = 0.0
            
            # 1. Keyword matching (40% of score) - Enhanced for comprehensive testing
            keyword_matches = 0
            total_keywords = len(question.expected_keywords)
            
            if total_keywords > 0:
                for keyword in question.expected_keywords:
                    if keyword.lower() in answer_lower:
                        keyword_matches += 1
                
                keyword_score = keyword_matches / total_keywords
                score += keyword_score * 0.4
            
            # 2. Content relevance (25% of score) - Enhanced
            content_indicators = [
                len(answer) > 30,  # Adequate length
                not any(term in answer_lower for term in ["error", "unable", "failed", "not found", "processing error"]),
                any(term in answer_lower for term in ["policy", "coverage", "insured", "benefits", "covered", "premium"]),
                bool(re.search(r'\d+', answer)),  # Contains numbers
                len(answer.split()) > 8,  # Sufficient detail
            ]
            
            content_score = sum(1 for indicator in content_indicators if indicator) / len(content_indicators)
            score += content_score * 0.25
            
            # 3. Domain-specific accuracy (20% of score) - New for comprehensive testing
            domain_specific_terms = {
                "health_insurance": ["medical", "treatment", "hospital", "doctor", "diagnosis", "surgery"],
                "general_insurance": ["premium", "policy", "coverage", "claim", "deductible"],
                "general": ["insurance", "benefit", "terms", "conditions"]
            }
            
            domain_terms = domain_specific_terms.get(question.domain, domain_specific_terms["general"])
            domain_matches = sum(1 for term in domain_terms if term in answer_lower)
            domain_score = min(domain_matches / len(domain_terms), 1.0)
            score += domain_score * 0.2
            
            # 4. Question type specific scoring (10% of score) - New
            type_specific_bonus = 0
            if question.question_type == "factual":
                # Factual questions should have specific numbers and clear statements
                if re.search(r'\b\d+\b', answer) and any(term in answer_lower for term in ["is", "are", "defined", "specified"]):
                    type_specific_bonus = 0.1
            elif question.question_type == "analytical":
                # Analytical questions should have comprehensive explanations
                if len(answer.split()) > 15 and any(term in answer_lower for term in ["includes", "conditions", "requirements"]):
                    type_specific_bonus = 0.1
            elif question.question_type == "comparative":
                # Comparative questions should show differences or similarities
                if any(term in answer_lower for term in ["different", "compared", "versus", "while", "whereas"]):
                    type_specific_bonus = 0.1
            elif question.question_type == "procedural":
                # Procedural questions should have step-by-step or process information
                if any(term in answer_lower for term in ["process", "steps", "procedure", "must", "required"]):
                    type_specific_bonus = 0.1
            
            score += type_specific_bonus
            
            # 5. Difficulty level adjustment (5% of score) - New
            difficulty_bonus = 0
            if question.difficulty_level == "expert" and len(answer) > 100:
                difficulty_bonus = 0.05  # Bonus for detailed expert answers
            elif question.difficulty_level == "easy" and len(answer) > 20:
                difficulty_bonus = 0.05  # Bonus for adequate easy answers
            
            score += difficulty_bonus
            
            # Penalty for uncertainty or error responses (severe)
            uncertainty_phrases = [
                'not mentioned', 'not provided', 'unclear', 'unable to determine',
                'insufficient information', 'cannot find', 'not specified',
                'appears to', 'seems to', 'might be', 'could be', 'possibly',
                'processing error', 'error occurred', 'unable to process'
            ]
            
            for phrase in uncertainty_phrases:
                if phrase in answer_lower:
                    score = max(score * 0.2, 0.05)  # Severe penalty
                    break
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            print(f"Error in comprehensive accuracy calculation: {e}")
            return 0.0
    
    async def _test_api_request_batched(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """Enhanced API request with automatic batching for 20-question limit"""
        
        # Split questions into batches of 20
        batch_size = 20
        all_answers = []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        
        print(f"📦 Batching {len(questions)} questions into {total_batches} batches of max {batch_size}")
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"  🔄 Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)")
            
            payload = {
                "documents": document_url,
                "questions": batch_questions
            }
            
            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json"
            }
            
            try:
                timeout = aiohttp.ClientTimeout(total=60)  # 1-minute timeout per batch
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.base_url}/hackrx/run",
                        json=payload,
                        headers=headers
                    ) as response:
                        
                        if response.status == 200:
                            batch_response = await response.json()
                            if batch_response and batch_response.get("answers"):
                                all_answers.extend(batch_response["answers"])
                                print(f"    ✅ Batch {batch_num} completed ({len(batch_response['answers'])} answers)")
                            else:
                                print(f"    ❌ Batch {batch_num} failed - no answers")
                                return None
                        else:
                            error_text = await response.text()
                            logger.error(f"Batch {batch_num} API error {response.status}: {error_text[:200]}")
                            return None
                            
            except Exception as e:
                logger.error(f"Batch {batch_num} request failed: {e}")
                return None
            
            # Small delay between batches to be respectful
            if batch_num < total_batches:
                await asyncio.sleep(1)
        
        print(f"✅ All batches completed. Total answers: {len(all_answers)}")
        return {"answers": all_answers}

    async def _test_api_request(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """Enhanced API request with comprehensive error handling"""
        
        # If more than 20 questions, use batched approach
        if len(questions) > 20:
            return await self._test_api_request_batched(document_url, questions)
        
        payload = {
            "documents": document_url,
            "questions": questions
        }
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=300)  # 5-minute timeout for comprehensive testing
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/hackrx/run",
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text[:200]}")
                        return None
                        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        report = {
            "test_metadata": self.test_results["test_metadata"],
            "executive_summary": {
                "test_type": "ULTIMATE_COMPREHENSIVE_MULTI_DOMAIN",
                "total_questions": len(self.comprehensive_questions),
                "total_documents": len(self.test_documents),
                "domains_tested": len(set([q.domain for q in self.comprehensive_questions])),
                "difficulty_levels_tested": len(set([q.difficulty_level for q in self.comprehensive_questions])),
                "question_types_tested": len(set([q.question_type for q in self.comprehensive_questions])),
                "completion_status": "COMPLETED",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "user": "vkhare2909"
            },
            "comprehensive_analysis": {
                "primary_test": self.test_results.get("primary_comprehensive_results", {}),
                "domain_analysis": self.test_results.get("domain_analysis", {}),
                "difficulty_analysis": self.test_results.get("difficulty_analysis", {}),
                "question_type_analysis": self.test_results.get("question_type_analysis", {}),
                "document_performance": self.test_results.get("document_performance", {}),
            },
            "comprehensive_metrics": self.test_results.get("comprehensive_metrics", {}),
            "final_assessment": self.test_results.get("final_assessment", {}),
            "recommendations": self._generate_comprehensive_recommendations()
        }
        
        return report
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Generate comprehensive recommendations based on analysis"""
        
        recommendations = []
        
        final_assessment = self.test_results.get("final_assessment", {})
        comprehensive_metrics = self.test_results.get("comprehensive_metrics", {})
        
        comprehensive_score = final_assessment.get("comprehensive_score", 0.0)
        
        # Overall performance recommendations
        if comprehensive_score >= 0.95:
            recommendations.append("🏆 OUTSTANDING! System exceeds championship standards across all domains!")
            recommendations.append("🚀 Deploy immediately for guaranteed TOP 3 position!")
        elif comprehensive_score >= 0.9:
            recommendations.append("🏆 EXCELLENT performance! System meets championship requirements!")
            recommendations.append("✅ Ready for hackathon deployment with high confidence!")
        elif comprehensive_score >= 0.8:
            recommendations.append("⭐ Good performance with room for minor improvements.")
            recommendations.append("🔧 Consider fine-tuning weak categories before deployment.")
        else:
            recommendations.append("🔧 Significant improvements needed for competitive performance.")
            recommendations.append("📊 Focus on accuracy optimization and response time reduction.")
        
        # Domain-specific recommendations
        domain_performance = comprehensive_metrics.get("domain_performance", {})
        if domain_performance:
            weak_domains = [domain for domain, accuracy in domain_performance.items() if accuracy < 0.8]
            strong_domains = [domain for domain, accuracy in domain_performance.items() if accuracy >= 0.9]
            
            if weak_domains:
                recommendations.append(f"🎯 Improve performance in: {', '.join(weak_domains)}")
            if strong_domains:
                recommendations.append(f"✅ Excellent performance in: {', '.join(strong_domains)}")
        
        # Difficulty-specific recommendations
        difficulty_performance = comprehensive_metrics.get("difficulty_performance", {})
        if difficulty_performance:
            if difficulty_performance.get("expert", 0) < 0.7:
                recommendations.append("🔬 Enhance handling of expert-level questions with more detailed responses.")
            if difficulty_performance.get("easy", 0) < 0.9:
                recommendations.append("⚡ Optimize basic factual question processing for higher accuracy.")
        
        # Speed recommendations
        speed_performance = comprehensive_metrics.get("speed_performance", 0)
        if speed_performance > 15:
            recommendations.append("⚡ Optimize response time to meet <15s target requirement.")
        elif speed_performance < 10:
            recommendations.append("🚀 Excellent speed performance! Well under target requirements.")
        
        # System readiness
        target_achievement = final_assessment.get("target_achievement", {})
        if target_achievement.get("overall", False):
            recommendations.append("🎉 BOTH speed and accuracy targets achieved! System is championship-ready!")
        
        return recommendations

# Synchronous runner function for comprehensive testing
def run_ultimate_comprehensive_test():
    """Run the ultimate comprehensive test"""
    
    print("🔧 CHECKING SERVER STATUS...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server Status: {health_data.get('status', 'unknown')}")
            print(f"📦 Version: {health_data.get('version', 'unknown')}")
        else:
            print(f"⚠️ Server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("🔧 Start server with: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False
    
    # Run ultimate comprehensive test
    tester = UltimateComprehensiveTester()
    
    async def run_async():
        return await tester.run_ultimate_comprehensive_test()
    
    try:
        final_report = asyncio.run(run_async())
        
        # Save detailed comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"ultimate_comprehensive_report_{timestamp}.json"
        
        with open(report_filename, "w") as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📊 ULTIMATE COMPREHENSIVE TEST COMPLETE!")
        print(f"📄 Detailed report saved to: {report_filename}")
        
        # Print executive summary
        final_assessment = final_report.get("final_assessment", {})
        comprehensive_score = final_assessment.get("comprehensive_score", 0)
        readiness_level = final_assessment.get("readiness_level", "UNKNOWN")
        
        print(f"\n🏆 FINAL COMPREHENSIVE RESULTS:")
        print(f"📊 Comprehensive Score: {comprehensive_score:.1%}")
        print(f"🎯 Readiness Level: {readiness_level}")
        print(f"🚀 Recommendation: {final_assessment.get('recommendation', 'No recommendation available')}")
        
        # Target achievement summary
        target_achievement = final_assessment.get("target_achievement", {})
        if target_achievement.get("overall", False):
            print(f"\n🎉 🏆 BOTH TARGETS ACHIEVED! 🏆 🎉")
            print(f"✅ Accuracy Target: {target_achievement.get('accuracy', False)}")
            print(f"⚡ Speed Target: {target_achievement.get('speed', False)}")
            print(f"🥇 SYSTEM IS CHAMPIONSHIP-READY FOR TOP 3 POSITION!")
            return True
        else:
            accuracy_met = target_achievement.get('accuracy', False)
            speed_met = target_achievement.get('speed', False)
            print(f"\n📈 TARGET ACHIEVEMENT STATUS:")
            print(f"🎯 Accuracy (90%+): {'✅ ACHIEVED' if accuracy_met else '❌ NOT MET'}")
            print(f"⚡ Speed (<15s): {'✅ ACHIEVED' if speed_met else '❌ NOT MET'}")
            
            if accuracy_met or speed_met:
                print(f"🟡 PARTIAL SUCCESS - Continue with deployment and monitor performance")
                return True
            else:
                print(f"🔧 NEEDS IMPROVEMENT - Review recommendations before deployment")
                return False
            
    except Exception as e:
        print(f"\n💥 Comprehensive test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Quick test function for faster validation
def run_quick_validation_test():
    """Run a quick validation test with essential questions"""
    
    print("⚡ QUICK VALIDATION TEST")
    print("="*60)
    
    quick_questions = [
        "What is the grace period for premium payment under this policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?"
    ]
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
        "questions": quick_questions
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
            timeout=60
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            
            print(f"✅ SUCCESS! Response time: {total_time:.1f}s")
            print(f"📝 Questions tested: {len(quick_questions)}")
            print(f"💬 Answers received: {len(answers)}")
            print(f"⚡ Speed: {'✅ FAST' if total_time < 15 else '❌ SLOW'} (Target: <15s)")
            
            print(f"\n📋 QUICK RESULTS:")
            for i, (question, answer) in enumerate(zip(quick_questions, answers)):
                print(f"\nQ{i+1}: {question}")
                print(f"A{i+1}: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            
            # Quick accuracy assessment
            good_answers = sum(1 for answer in answers if len(answer) > 30 and "error" not in answer.lower())
            quick_accuracy = good_answers / len(quick_questions) if quick_questions else 0
            
            print(f"\n🎯 QUICK ASSESSMENT:")
            print(f"📊 Quick Accuracy Estimate: {quick_accuracy:.1%}")
            print(f"✅ Good Answers: {good_answers}/{len(quick_questions)}")
            
            if total_time < 15 and quick_accuracy >= 0.8:
                print(f"\n🏆 QUICK VALIDATION PASSED!")
                print(f"🚀 System appears ready for comprehensive testing!")
                return True
            else:
                print(f"\n🟡 QUICK VALIDATION PARTIAL")
                print(f"🔧 Consider optimization before full testing")
                return False
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print(f"🚀 ULTIMATE COMPREHENSIVE FINAL TESTER")
    print(f"📅 Current Date: 2025-08-01 17:13:05 UTC")
    print(f"👤 User: vkhare2909")
    print(f"🎯 Mission: Achieve 90%+ accuracy & <15s response for TOP 3 position")
    print("="*80)
    
    # Option to run quick test first
    print("🔍 Select test type:")
    print("1. Quick Validation Test (5 questions, ~1 minute)")
    print("2. Ultimate Comprehensive Test (50+ questions, ~15 minutes)")
    print("3. Both (Quick first, then comprehensive)")
    
    choice = input("\nEnter your choice (1/2/3) or press Enter for comprehensive: ").strip()
    
    if choice == "1":
        success = run_quick_validation_test()
        if success:
            print("\n🎉 Quick validation successful!")
        else:
            print("\n⚠️ Quick validation shows issues - consider optimization")
    
    elif choice == "3":
        print("\n⚡ Running quick validation first...")
        quick_success = run_quick_validation_test()
        
        if quick_success:
            print("\n🚀 Quick test passed! Proceeding to comprehensive test...")
            input("Press Enter to continue with comprehensive test...")
            comprehensive_success = run_ultimate_comprehensive_test()
        else:
            proceed = input("\n⚠️ Quick test showed issues. Continue with comprehensive test? (y/N): ")
            if proceed.lower() == 'y':
                comprehensive_success = run_ultimate_comprehensive_test()
            else:
                print("🔧 Consider system optimization before comprehensive testing")
                comprehensive_success = False
    
    else:  # Default to comprehensive test
        comprehensive_success = run_ultimate_comprehensive_test()
    
    # Final status
    if 'comprehensive_success' in locals() and comprehensive_success:
        print("\n🥇 ULTIMATE COMPREHENSIVE TEST SUCCESSFUL!")
        print("🏆 SYSTEM IS READY FOR CHAMPIONSHIP COMPETITION!")
        print("🚀 DEPLOY IMMEDIATELY FOR TOP 3 POSITION!")
    elif 'success' in locals() and success:
        print("\n✅ Quick test successful - system shows promise!")
    else:
        print("\n📊 Testing completed - review results and recommendations")
    
    print(f"\n🎯 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"👤 Tester: vkhare2909")
    print("🏁 END OF ULTIMATE COMPREHENSIVE TESTING")