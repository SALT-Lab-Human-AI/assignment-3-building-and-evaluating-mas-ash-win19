"""
Output Guardrail
Checks system outputs for safety violations.
"""

from typing import Dict, Any, List
import re


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    Rule-based output checks to avoid unsafe disclosures:
    - PII redaction
    - Harmful content/violence detection
    - Optional factual/bias heuristics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.harmful_keywords = [
            "violence", "kill", "attack", "bomb", "exploit", "weapon",
            "self-harm", "suicide", "terror", "assassinate"
        ]
        # Bias markers - kept minimal to avoid false positives in academic content
        # "everyone" removed as it's commonly used appropriately in accessibility discourse
        self.bias_markers = [
            "always wrong", "never works", "no one can", "all people must"
        ]

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Validation result
        """
        violations = []
        pii_violations = self._check_pii(response)
        violations.extend(pii_violations)

        harmful_violations = self._check_harmful_content(response)
        violations.extend(harmful_violations)

        if sources:
            consistency_violations = self._check_factual_consistency(response, sources)
            violations.extend(consistency_violations)

        bias_violations = self._check_bias(response)
        violations.extend(bias_violations)

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_output": self._sanitize(response, violations) if violations else response
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information.

        """
        violations = []

        # Simple regex patterns for common PII
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        }

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Contains {pii_type}",
                    "severity": "high",
                    "category": "pii",
                    "matches": matches
                })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or inappropriate content.

        """
        violations = []

        for keyword in self.harmful_keywords:
            if keyword in text.lower():
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"May contain harmful content: {keyword}",
                    "severity": "medium",
                    "category": "harmful_content"
                })

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response is consistent with sources.

        """
        violations = []

        # Basic heuristic: ensure at least one source is referenced if provided
        if sources and sources != []:
            referenced_any = any(url.get("url") in response for url in sources if isinstance(url, dict) and url.get("url"))
            if not referenced_any:
                violations.append({
                    "validator": "factual_consistency",
                    "reason": "Response does not cite provided sources.",
                    "severity": "low",
                    "category": "factual_consistency"
                })

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased language.

        """
        violations = []
        lowered = text.lower()
        for marker in self.bias_markers:
            if marker in lowered:
                violations.append({
                    "validator": "bias",
                    "reason": f"Potentially biased generalization using '{marker}'",
                    "severity": "low",
                    "category": "bias"
                })
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize text by removing/redacting violations.

        """
        sanitized = text

        # Redact PII
        for violation in violations:
            if violation.get("validator") == "pii":
                for match in violation.get("matches", []):
                    sanitized = sanitized.replace(match, "[REDACTED]")

        # Mask harmful content indicators
        for violation in violations:
            if violation.get("validator") == "harmful_content":
                keyword = violation.get("reason", "").split(":")[-1].strip()
                if keyword:
                    sanitized = sanitized.replace(keyword, "[REMOVED]")

        return sanitized
