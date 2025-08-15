from langgraph.graph import StateGraph, END
from agentai.nodes import node_load, node_supervisor, node_inspect
from agentai.modules.common import AgentState
from typing import Literal

def should_continue(state: AgentState) -> Literal["inspect", "end"]:
    """Determines the next step based on the supervisor's decision."""
    if state.get("next", "").lower() == "inspect":
        return "inspect"
    else:
        return "end"

def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("loaddata", node_load)
    workflow.add_node("supervisor", node_supervisor)
    workflow.add_node("inspect", node_inspect)

    workflow.set_entry_point("loaddata")

    # Define as transições
    workflow.add_edge("loaddata", "supervisor")
    workflow.add_edge("inspect", "supervisor")
    
    # Adiciona a transição condicional do supervisor
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "inspect": "inspect",
            "end": END,
        },
    )

    return workflow.compile()