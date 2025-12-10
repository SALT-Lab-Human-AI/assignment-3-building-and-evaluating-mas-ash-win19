"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager
import re


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    
    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")
        safety_config = config.get("safety", {}).copy()
        # pass through logging/system metadata for guardrails
        safety_config["logging"] = config.get("logging", {})
        safety_config["system"] = config.get("system", {})
        self.safety_manager = SafetyManager(safety_config)
        
        # Create the research team
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)
        
        self.logger.info("Research team created successfully")
        
        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")
        # Reset safety log per query
        self.safety_manager.clear_events()
        
        try:
            # Input safety check
            input_safety = self.safety_manager.check_input_safety(query)
            if not input_safety.get("safe", True):
                refusal_message = self.config.get("safety", {}).get("on_violation", {}).get(
                    "message",
                    "Request blocked due to safety policies."
                )
                return {
                    "query": query,
                    "response": refusal_message,
                    "conversation_history": [],
                    "metadata": {
                        "error": False,
                        "safety_events": self.safety_manager.get_safety_events(),
                        "agents_involved": [],
                    }
                }

            sanitized_query = input_safety.get("sanitized_query", query)

            # Run the async query processing
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, 
                        self._process_query_async(sanitized_query, max_rounds)
                    ).result()
            else:
                result = loop.run_until_complete(self._process_query_async(sanitized_query, max_rounds))
            
            # Output safety check
            output_safety = self.safety_manager.check_output_safety(result.get("response", ""))
            result.setdefault("metadata", {})
            result["metadata"]["safety_events"] = self.safety_manager.get_safety_events()
            if not output_safety.get("safe", True):
                result["response"] = output_safety.get("response", result.get("response", ""))
                result["metadata"]["safety_violations"] = output_safety.get("violations", [])
                result["metadata"]["safety_action"] = output_safety.get("action_taken", "refuse")
                result["metadata"]["safety_sanitized"] = output_safety.get("sanitized", False)
            
            self.logger.info("Query processing complete")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "metadata": {
                    "error": True,
                    "safety_events": self.safety_manager.get_safety_events()
                }
            }
    
    async def _process_query_async(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Async implementation of query processing.
        
        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds
            
        Returns:
            Dictionary containing results
        """
        # Create task message
        task_message = f"""Research Query: {query}

Please work together to answer this query comprehensively:
1. Planner: Create a research plan (respond once with numbered steps and hand off).
2. Researcher: Use web_search/paper_search tools; include URLs and short notes per source. If tools fail or are unavailable, state that explicitly and continue with available information. End with "RESEARCH COMPLETE".
3. Writer: Synthesize findings with inline citations and references. End with "DRAFT COMPLETE".
4. Critic: Evaluate quality and either say "TERMINATE" if acceptable or give concrete revisions."""
        
        # Run the team
        result = await self.team.run(task=task_message)
        
        # Extract conversation history
        messages = []
        raw_messages = getattr(result, "messages", result)
        # Handle both async iterables and plain lists
        if hasattr(raw_messages, "__aiter__"):
            async for message in raw_messages:
                msg_dict = {
                    "source": message.source,
                    "content": message.content if hasattr(message, "content") else str(message),
                }
                messages.append(msg_dict)
        else:
            for message in raw_messages:
                msg_dict = {
                    "source": getattr(message, "source", "Unknown"),
                    "content": message.content if hasattr(message, "content") else str(message),
                }
                messages.append(msg_dict)
        
        # Extract final response
        final_response = ""
        if messages:
            # Get the last message from Writer or Critic
            for msg in reversed(messages):
                if msg.get("source") in ["Writer", "Critic"]:
                    final_response = msg.get("content", "")
                    break
        
        # If no response found, use the last message
        if not final_response and messages:
            final_response = messages[-1].get("content", "")
        
        return self._extract_results(query, messages, final_response)

    def _extract_results(self, query: str, messages: List[Dict[str, Any]], final_response: str = "") -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Args:
            query: Original query
            messages: List of conversation messages
            final_response: Final response from the team

        Returns:
            Structured result dictionary
        """
        # Extract components from conversation
        research_findings = []
        plan = ""
        critique = ""
        citations = []
        last_agent = ""
        
        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")
            last_agent = source or last_agent
            
            if source == "Planner" and not plan:
                plan = content
            
            elif source == "Researcher":
                research_findings.append(content)
                citations.extend(self._extract_citations(content))
            
            elif source == "Critic":
                critique = content
                citations.extend(self._extract_citations(content))
            elif source == "Writer":
                citations.extend(self._extract_citations(content))
        
        # Count sources mentioned in research
        num_sources = 0
        for finding in research_findings:
            # Rough count of sources based on numbered results
            num_sources += finding.count("\n1.") + finding.count("\n2.") + finding.count("\n3.")
        
        # Clean up final response
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()
        
        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, 1),  # At least 1
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "citations": citations[:20],
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
                "last_agent": last_agent,
            }
        }

    def _extract_citations(self, text: str) -> List[str]:
        """
        Extract citations/URLs from a block of text.
        """
        citations = []
        # URLs
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\\[\\]]+', text)
        citations.extend(urls)
        # [Source: ...] patterns
        source_tags = re.findall(r'\\[Source: ([^\\]]+)\\]', text)
        citations.extend(source_tags)
        return list(dict.fromkeys(citations))  # preserve order, dedupe

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        """
        return workflow


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.
    
    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)
    
    # Print workflow visualization
    print(orchestrator.visualize_workflow())
    
    # Example query
    query = "What are the latest trends in human-computer interaction research?"
    
    print(f"\nProcessing query: {query}\n")
    print("=" * 70)
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    demonstrate_usage()
