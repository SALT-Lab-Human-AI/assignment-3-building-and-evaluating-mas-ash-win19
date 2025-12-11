"""
Safety Manager
Coordinates safety guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.

    Coordinates input/output guardrails and logs events for UI display.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories
        self.prohibited_categories = config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries"
        ])

        # Violation response strategy
        self.on_violation = config.get("on_violation", {})

        # Instantiate guardrails
        from src.guardrails.input_guardrail import InputGuardrail
        from src.guardrails.output_guardrail import OutputGuardrail

        self.input_guardrail = InputGuardrail(config=config)
        self.output_guardrail = OutputGuardrail(config=config)
        # Log file configuration
        logging_config = config.get("logging", {})
        self.safety_log_file = logging_config.get("safety_log") or config.get("safety_log_file")

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list

        """
        if not self.enabled:
            return {"safe": True}

        result = self.input_guardrail.validate(query)
        violations = result.get("violations", [])
        is_safe = result.get("valid", True)

        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("input", query, violations, is_safe)

        return {
            "safe": is_safe,
            "violations": violations,
            "sanitized_query": result.get("sanitized_input", query)
        }

    def check_output_safety(self, response: str) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list

        """
        if not self.enabled:
            return {"safe": True, "response": response}

        guardrail_result = self.output_guardrail.validate(response)
        violations = guardrail_result.get("violations", [])
        is_safe = guardrail_result.get("valid", True)
        action = "none"
        sanitized_flag = False

        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("output", response, violations, is_safe)

        result = {
            "safe": is_safe,
            "violations": violations,
            "response": response,
            "sanitized": False,
            "action_taken": "none"
        }

        # Apply sanitization if configured
        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            if action == "sanitize":
                result["response"] = guardrail_result.get("sanitized_output", response)
                sanitized_flag = True
            elif action == "refuse":
                result["response"] = self.on_violation.get(
                    "message",
                    "I cannot provide this response due to safety policies."
                )
            result["sanitized"] = sanitized_flag
            result["action_taken"] = action

        return result

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize response by removing or redacting unsafe content.

        """
        # Placeholder
        return "[REDACTED] " + response

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }

        self.safety_events.append(event)
        self.logger.warning(f"Safety event: {event_type} - safe={is_safe}")

        # Write to safety log file if configured
        log_file = self.safety_log_file
        if log_file and self.log_events:
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []
