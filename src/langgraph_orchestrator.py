"""
LangGraph-Based Orchestrator

This orchestrator uses LangGraph to explicitly control the multi-agent workflow.
Unlike AutoGen's autonomous approach, LangGraph gives us precise control over
the execution flow.

Workflow:
1. Planner: Breaks down query into research steps
2. Researcher: Gathers evidence using tools
3. Writer: Synthesizes findings with citations
4. Critic: Evaluates and decides whether to approve or revise
"""

import os
import logging
from typing import Dict, Any, List, TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.tools.web_search import web_search
from src.tools.paper_search import paper_search
from src.guardrails.safety_manager import SafetyManager


# Define the state that flows through the graph
class ResearchState(TypedDict):
    query: str
    plan: str
    research_findings: str
    draft: str
    critique: str
    final_response: str
    conversation_history: List[Dict[str, str]]
    sources: List[str]
    revision_count: int
    approved: bool


class LangGraphOrchestrator:
    """
    LangGraph-based multi-agent orchestrator.

    Provides explicit control over the research workflow with
    deterministic agent transitions.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the LangGraph orchestrator."""
        self.config = config
        self.logger = logging.getLogger("langgraph_orchestrator")

        # Initialize safety manager
        safety_config = config.get("safety", {}).copy()
        safety_config["logging"] = config.get("logging", {})
        safety_config["system"] = config.get("system", {})
        self.safety_manager = SafetyManager(safety_config)

        # Initialize LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        model_config = config.get("models", {}).get("default", {})
        self.llm = ChatGroq(
            api_key=api_key,
            model=model_config.get("name", "llama-3.3-70b-versatile"),
            temperature=model_config.get("temperature", 0.7),
        )

        # Build the workflow graph
        self.graph = self._build_graph()

        self.logger.info("LangGraph orchestrator initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ResearchState)

        # Add nodes for each agent
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("critic", self._critic_node)

        # Define the flow
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "writer")
        workflow.add_edge("writer", "critic")

        # Critic decides: approve (END) or revise (back to writer)
        workflow.add_conditional_edges(
            "critic",
            self._should_continue,
            {
                "revise": "writer",
                "approve": END
            }
        )

        return workflow.compile()

    def _planner_node(self, state: ResearchState) -> ResearchState:
        """Planner agent: Creates research plan."""
        self.logger.info("Planner: Creating research plan...")

        system_prompt = """You are a Research Planner specializing in HCI topics.
Break down the query into specific, actionable research steps.

Provide a numbered list of 3-5 research steps. Be specific about:
- What information to find
- What sources to prioritize (academic papers vs web articles)
- Key concepts to explore

Keep it concise and actionable."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {state['query']}\n\nCreate a research plan:")
        ]

        response = self.llm.invoke(messages)
        plan = response.content

        state["plan"] = plan
        state["conversation_history"].append({
            "agent": "Planner",
            "content": plan
        })

        self.logger.info("Planner: Plan created")
        return state

    def _researcher_node(self, state: ResearchState) -> ResearchState:
        """Researcher agent: Gathers evidence using tools."""
        self.logger.info("Researcher: Gathering evidence...")

        system_prompt = """You are a Research Assistant. Use the available tools to gather information.

You have access to:
- web_search(query): Search the web
- paper_search(query): Search academic papers

For each source, provide:
- Title and URL
- Key findings relevant to the query
- Brief summary

Gather 5-8 high-quality sources."""

        # Extract search terms from plan
        query_text = state['query']

        # Perform web search
        findings = ["**Web Search Results:**\n"]
        try:
            web_results = web_search(query_text, max_results=3)
            findings.append(web_results)
        except Exception as e:
            findings.append(f"Web search unavailable: {str(e)}\n")

        # Perform paper search
        findings.append("\n**Academic Paper Results:**\n")
        try:
            paper_results = paper_search(query_text, max_results=3)
            findings.append(paper_results)
        except Exception as e:
            findings.append(f"Paper search unavailable: {str(e)}\n")

        research_findings = "\n".join(findings)

        state["research_findings"] = research_findings
        state["conversation_history"].append({
            "agent": "Researcher",
            "content": research_findings
        })

        # Extract sources
        import re
        urls = re.findall(r'URL: (https?://[^\s]+)', research_findings)
        state["sources"].extend(urls)

        self.logger.info(f"Researcher: Gathered {len(urls)} sources")
        return state

    def _writer_node(self, state: ResearchState) -> ResearchState:
        """Writer agent: Synthesizes findings."""
        self.logger.info("Writer: Synthesizing response...")

        system_prompt = """You are a Research Writer. Synthesize the research findings into a clear, well-organized response.

Structure:
1. **Introduction**: Brief overview answering the query
2. **Main Content**: Organized findings with inline citations
3. **References**: List all sources with URLs

Citation format: [Source Title](URL) or (Author, Year)

Write in clear, accessible language. Be comprehensive but concise."""

        # Include critique if this is a revision
        critique_context = ""
        if state.get("critique"):
            critique_context = f"\n\nPREVIOUS CRITIQUE TO ADDRESS:\n{state['critique']}\n"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Query: {state['query']}

Research Plan:
{state['plan']}

Research Findings:
{state['research_findings']}
{critique_context}

Write a comprehensive response with citations:""")
        ]

        response = self.llm.invoke(messages)
        draft = response.content

        state["draft"] = draft
        state["conversation_history"].append({
            "agent": "Writer",
            "content": draft
        })

        self.logger.info("Writer: Draft completed")
        return state

    def _critic_node(self, state: ResearchState) -> ResearchState:
        """Critic agent: Evaluates quality."""
        self.logger.info("Critic: Evaluating response...")

        system_prompt = """You are a Research Critic. Evaluate the response on:

1. **Relevance**: Does it answer the query?
2. **Evidence**: Are sources cited properly?
3. **Completeness**: Are all aspects covered?
4. **Clarity**: Is it well-organized?

Provide:
- Overall assessment (APPROVE or NEEDS REVISION)
- Specific feedback if revisions needed
- What's done well

Keep feedback constructive and specific."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Query: {state['query']}

Draft Response:
{state['draft']}

Sources Available: {len(state.get('sources', []))}

Evaluate this response:""")
        ]

        response = self.llm.invoke(messages)
        critique = response.content

        state["critique"] = critique
        state["conversation_history"].append({
            "agent": "Critic",
            "content": critique
        })

        # Check if approved
        if "APPROVE" in critique.upper() and "NEEDS REVISION" not in critique.upper():
            state["approved"] = True
            state["final_response"] = state["draft"]
            self.logger.info("Critic: Response APPROVED")
        else:
            state["revision_count"] = state.get("revision_count", 0) + 1
            self.logger.info(f"Critic: Revision requested (count: {state['revision_count']})")

        return state

    def _should_continue(self, state: ResearchState) -> str:
        """Decide whether to continue revising or finish."""
        if state.get("approved", False):
            return "approve"

        # Limit revisions to avoid infinite loops
        if state.get("revision_count", 0) >= 2:
            self.logger.warning("Max revisions reached, approving anyway")
            state["approved"] = True
            state["final_response"] = state["draft"]
            return "approve"

        return "revise"

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a research query through the LangGraph workflow.

        Args:
            query: Research question

        Returns:
            Dictionary with response, metadata, conversation history
        """
        self.logger.info(f"Processing query: {query}")
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

            # Initialize state
            initial_state: ResearchState = {
                "query": sanitized_query,
                "plan": "",
                "research_findings": "",
                "draft": "",
                "critique": "",
                "final_response": "",
                "conversation_history": [],
                "sources": [],
                "revision_count": 0,
                "approved": False
            }

            # Run the graph
            final_state = self.graph.invoke(initial_state)

            # Extract result
            response = final_state.get("final_response", final_state.get("draft", ""))

            # Output safety check
            output_safety = self.safety_manager.check_output_safety(response)
            if not output_safety.get("safe", True):
                response = output_safety.get("response", response)

            # Build result
            result = {
                "query": query,
                "response": response,
                "conversation_history": final_state.get("conversation_history", []),
                "metadata": {
                    "num_messages": len(final_state.get("conversation_history", [])),
                    "num_sources": len(final_state.get("sources", [])),
                    "agents_involved": list(set([
                        msg["agent"] for msg in final_state.get("conversation_history", [])
                    ])),
                    "plan": final_state.get("plan", ""),
                    "research_findings": [final_state.get("research_findings", "")],
                    "critique": final_state.get("critique", ""),
                    "citations": final_state.get("sources", [])[:20],
                    "revision_count": final_state.get("revision_count", 0),
                    "approved": final_state.get("approved", False),
                    "safety_events": self.safety_manager.get_safety_events(),
                    "last_agent": "Critic"
                }
            }

            if not output_safety.get("safe", True):
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

    def get_agent_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all agents."""
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """Generate a text visualization of the workflow."""
        workflow = """
LangGraph Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Calls web_search() tool
   - Calls paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds inline citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer (max 2 revisions)
"""
        return workflow


if __name__ == "__main__":
    # Quick test
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    orchestrator = LangGraphOrchestrator(config)

    print("Testing LangGraph orchestrator...\n")
    result = orchestrator.process_query("What are key principles of accessible UI design?")

    print("\n=== RESULT ===")
    print(f"Response: {result['response'][:300]}...")
    print(f"\nMetadata:")
    print(f"  Agents: {result['metadata']['agents_involved']}")
    print(f"  Messages: {result['metadata']['num_messages']}")
    print(f"  Sources: {result['metadata']['num_sources']}")
    print(f"  Approved: {result['metadata']['approved']}")
