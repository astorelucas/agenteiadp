# agentai/nodes.py

import pandas as pd
from typing import Any, Dict
import numpy as np
import json
import re

from agentai.agents import (
    create_inspection_agent,
    create_supervisor_agent
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
    logs = state.get("logs", [])

    try:
        df = pd.read_csv(path)
        logs.append(f"Loaded dataset from {path} with {len(df)} rows.")
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")

    return {**state, "df": df, "logs": logs, "next": "supervisor"}

def inspect_node(state: AgentState) -> AgentState:
    df = state["df"]
    msg = state["msg"]
    logs = state.get("logs", [])

    if df.empty:
        return {**state, "output": " No dataset available for inspection."}

    agent = create_inspection_agent(df)
    response = agent.invoke({"input": msg})
    inspection_report = response.get("output", "") or str(response)

    logs.extend(["Inspection completed.", f"Report:\n{inspection_report}"])
    
    return {**state, "output": inspection_report, "logs": logs}

def supervisor_node(state: AgentState) -> AgentState:
    if state.get("df") is None:
        return {**state, "output": "No dataset loaded.", "next": "END"}

    supervisor_agent = create_supervisor_agent()
    
    input_message = (
        f"Current task: {state.get('msg')}\n"
        f"Previous step output:\n{state.get('output')}"
    )

    response = supervisor_agent.invoke({"messages": [HumanMessage(content=input_message)]})
    
    logs = state.get("logs", [])
    raw_output = str(response.get("messages", [])[-1].content)
    
    json_str_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    
    if not json_str_match:
        logs.append(f"Supervisor failed to produce any JSON. Raw output: {raw_output}")
        return {**state, "next": "END", "logs": logs}
    
    json_str = json_str_match.group(0)
    try:
        # Tenta carregar o JSON
        plan = json.loads(json_str)
    except json.JSONDecodeError:
        # Se falhar, registra o erro e encerra o fluxo
        logs.append(f"Supervisor produced invalid JSON. Raw output: {json_str}")
        return {**state, "next": "END", "logs": logs}

    next_step = plan.get("next", "END")
    msg_out = plan.get("msg", state.get("msg"))
    output = plan.get("output", "")

    logs.append(f"Supervisor decision: {output}")

    return {
        **state,
        "output": output,
        "next": next_step,
        "msg": msg_out,
        "logs": logs
    }

node_load = AgentNode("loaddata", load_dataset)
node_supervisor = AgentNode("supervisor", supervisor_node)
node_inspect = AgentNode("inspect", inspect_node)