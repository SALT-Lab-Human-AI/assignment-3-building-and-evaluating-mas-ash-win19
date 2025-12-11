"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List


class InputGuardrail:
    """
    Guardrail for checking input safety.

    Lightweight rule-based input checks:
    - Length guard
    - Toxic language detection
    - Prompt-injection heuristics
    - Off-topic filtering (based on configured topic)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        topic = config.get("system", {}).get("topic", "")
        self.allowed_topic = topic.lower()
        self.min_len = 5
        self.max_len = 2000
        # Simple keyword lists to keep this offline and deterministic
        self.toxic_keywords = [
            "hate", "kill", "violence", "terror", "racist", "sexist",
            "genocide", "attack", "bomb", "suicide"
        ]
        self.injection_patterns = [
            "ignore previous instructions",
            "disregard previous",
            "forget everything",
            "system prompt",
            "sudo",
            "###",
            "reset conversation"
        ]

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result
        """
        violations = []
        lowered = query.lower()

        # Length checks
        if len(query) < self.min_len:
            violations.append({
                "validator": "length",
                "reason": "Query too short",
                "severity": "low",
                "category": "format"
            })

        if len(query) > self.max_len:
            violations.append({
                "validator": "length",
                "reason": "Query too long",
                "severity": "medium",
                "category": "format"
            })

        # Toxicity
        violations.extend(self._check_toxic_language(lowered))

        # Prompt injection
        violations.extend(self._check_prompt_injection(lowered))

        # Relevance
        violations.extend(self._check_relevance(lowered))

        is_valid = len(violations) == 0

        return {
            "valid": is_valid,
            "violations": violations,
            # For now return original query; a future version could redact pieces
            "sanitized_input": query
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language.

        """
        violations = []
        for keyword in self.toxic_keywords:
            if keyword in text:
                violations.append({
                    "validator": "toxicity",
                    "reason": f"Toxic language detected: {keyword}",
                    "severity": "high",
                    "category": "harmful_content"
                })
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.

        """
        violations = []
        for pattern in self.injection_patterns:
            if pattern.lower() in text.lower():
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection: {pattern}",
                    "severity": "high",
                    "category": "prompt_injection"
                })

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to the system's purpose.

        """
        violations = []
        if self.allowed_topic:
            # Basic topicality heuristic: require overlap with topic keywords
            # Made less strict: check for HCI-related terms even if exact topic word missing
            topic_tokens = [t.strip().lower() for t in self.allowed_topic.split() if t.strip()]
            hci_related_terms = ['hci', 'human', 'computer', 'interface', 'interaction', 'user', 'usability',
                                'accessibility', 'design', 'ui', 'ux', 'experience', 'research']

            query_lower = query.lower()
            if topic_tokens:
                # Allow if query contains topic keywords OR HCI-related terms
                has_topic = any(token in query_lower for token in topic_tokens)
                has_hci_terms = any(term in query_lower for term in hci_related_terms)

                if not (has_topic or has_hci_terms):
                    violations.append({
                        "validator": "relevance",
                        "reason": f"Query may be off-topic (expected topic: {self.allowed_topic})",
                        "severity": "medium",
                        "category": "off_topic_queries"
                    })
        return violations
