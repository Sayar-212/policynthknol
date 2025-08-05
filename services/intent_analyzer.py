from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer
import re

class LocalIntentAnalyzer:
    def __init__(self):
        # Reuse the same model as embedding service for efficiency
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Predefined intent patterns with embeddings
        self.intent_patterns = {
            "definition": [
                "What is grace period",
                "Define deductible", 
                "What does this mean",
                "Explain the term",
                "What is the meaning of"
            ],
            "specific_value": [
                "How many days for grace period",
                "What is the amount of deductible",
                "How long is the waiting period",
                "What is the maximum limit",
                "How much is the premium"
            ],
            "coverage_check": [
                "Is maternity covered",
                "Does this include dental",
                "What is covered under this policy",
                "Are pre-existing diseases covered",
                "Is this treatment included"
            ],
            "exclusion_check": [
                "What is excluded from coverage",
                "Is this not covered",
                "What are the exclusions",
                "Are there any restrictions",
                "What is not included"
            ],
            "time_period": [
                "How long is the waiting period",
                "What is the grace period duration",
                "How many months for pre-existing",
                "What is the cooling period",
                "How long do I have to wait"
            ],
            "limits": [
                "What is the maximum coverage",
                "What are the policy limits",
                "What is the sum insured",
                "What is the room rent limit",
                "What is the annual limit"
            ]
        }
        
        # Generate embeddings for all patterns once
        self._generate_pattern_embeddings()
    
    def _generate_pattern_embeddings(self):
        """Pre-compute embeddings for all intent patterns"""
        self.pattern_embeddings = {}
        for intent, patterns in self.intent_patterns.items():
            embeddings = self.model.encode(patterns)
            self.pattern_embeddings[intent] = embeddings
    
    def analyze_intent(self, question: str) -> Dict[str, Any]:
        """Fast local intent analysis using similarity matching"""
        question_embedding = self.model.encode([question])[0]
        
        # Find best matching intent
        best_intent = "general"
        best_score = 0.0
        
        for intent, pattern_embeddings in self.pattern_embeddings.items():
            # Calculate similarity with all patterns for this intent
            similarities = np.dot(pattern_embeddings, question_embedding)
            max_similarity = np.max(similarities)
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent
        
        # Extract key concepts and determine expectations
        key_concepts = self._extract_key_concepts(question)
        expects_numbers = self._expects_numbers(question)
        
        return {
            "intent_type": best_intent,
            "looking_for": self._get_looking_for(best_intent, question),
            "expects_numbers": expects_numbers,
            "key_concepts": key_concepts,
            "confidence": float(best_score)
        }
    
    def _extract_key_concepts(self, question: str) -> List[str]:
        """Extract insurance-specific terms from question"""
        insurance_terms = [
            "grace period", "waiting period", "cooling period",
            "pre-existing", "maternity", "pregnancy", 
            "deductible", "co-pay", "excess",
            "sum insured", "coverage", "limit",
            "hospitalization", "outpatient", "cashless",
            "claim", "premium", "policy"
        ]
        
        question_lower = question.lower()
        found_terms = []
        
        for term in insurance_terms:
            if term in question_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _expects_numbers(self, question: str) -> bool:
        """Check if question expects numerical answer"""
        numerical_indicators = [
            "how much", "how many", "how long",
            "what is the amount", "what is the limit",
            "days", "months", "years", "percentage"
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in numerical_indicators)
    
    def _get_looking_for(self, intent_type: str, question: str) -> str:
        """Determine what user is looking for based on intent"""
        intent_mapping = {
            "definition": "explanation or meaning",
            "specific_value": "exact numbers or amounts", 
            "coverage_check": "what is covered",
            "exclusion_check": "what is excluded",
            "time_period": "duration or time limits",
            "limits": "maximum amounts or limits"
        }
        
        return intent_mapping.get(intent_type, "general information")