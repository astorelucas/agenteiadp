
from langgraph.graph import StateGraph, END
from getpass import getpass
from sklearn.experimental import enable_iterative_imputer
from agentai.nodes import (
    node_load,
    node_inspect
)
import operator
from agentai.modules.common import AgentState

# Build Workflow
def build_workflow() -> StateGraph:
    
    workflow = StateGraph(AgentState)

    workflow.add_node("loaddata", node_load)
    workflow.add_node("inspect", node_inspect)

    workflow.add_edge("loaddata", "inspect")

    workflow.set_entry_point("loaddata")
    workflow.set_finish_point("inspect")

    app = workflow.compile()

    from IPython.display import Image, display
    print(app.get_graph().draw_ascii())

    return app