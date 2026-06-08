"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    retrieved_context: str = field(default="")
    """Context retrieved from local knowledge base or web search."""

    search_decision: str = field(default="")
    """Result of the relevance check: 'YES' or 'NO'."""

    local_search_performed: bool = field(default=False)
    """Whether the workflow executed a local vector store retrieval step."""

    data_source: str = field(default="")
    """Which source is used for generation: 'direct', 'local', 'web', or 'none'."""

    intent: str = field(default="")
    """User intent: 'explain' (teaching/explaining) or 'solve' (problem solving)."""

    solve_mode: str = field(default="")
    """Solve mode: 'direct' for simple problems or 'retrieval' for complex problems."""

    solution_steps: str = field(default="")
    """Generated solution steps for problem solving."""

    rewritten_content: str = field(default="")
    """Rewritten content for teaching."""
