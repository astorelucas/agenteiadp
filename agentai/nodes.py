import pandas as pd
from typing import Any, Dict
import numpy as np

from agentai.agents import (
    create_inspection_agent
)

from agentai.modules.common import AgentState

from langchain_core.messages import HumanMessage

class AgentNode:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def load_dataset(state: AgentState) -> AgentState:
    path = state.get("path", "")
    messages = state.get("messages", [])

    try:
        df = pd.read_csv(path)
        messages.append(f"Loaded dataset from {path} with {len(df)} rows.")
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")

    return {**state, "df": df, "messages": messages, "next": "inspect"}


def inspect_node(state: AgentState) -> AgentState:
    df = state["df"]

    if df.empty:
        return {**state, "output": " No dataset available for inspection."}

    # Create and run the inspection agent
    assert isinstance(df, pd.DataFrame), "Expected df to be a pandas DataFrame"
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected df to be a pandas DataFrame")
    agent = create_inspection_agent(df)
    response = agent.invoke(state["msg"])
    inspection_report = response.get("output", "") or str(response)

    # Ensure output is stored
    return {
        **state,
        "output": inspection_report,
        "logs": state.get("messages", []) + [
            "Inspection completed.",
            f"Report:\n{inspection_report}"
        ]
    }


node_load = AgentNode("loaddata", load_dataset)
node_inspect = AgentNode("inspect", inspect_node)
