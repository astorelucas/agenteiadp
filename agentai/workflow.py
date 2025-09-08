import pandas as pd
import re
import json
from typing import Literal
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agentai.agents import create_pandas_agent, create_supervisor_agent, create_imputator_agent
from agentai.modules.common import AgentState
from agentai.tools import ImputationStrategyFactory
from agentai.nodes import (
    FeatureEngineeringNode,
    PandasNode,
    ImputatorNode,
    SupervisorNode,
)


class WorkflowExecutor:
    def __init__(self, csv_path: str):
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Falha ao carregar o dataset: {e}")
        
        self.factory = ImputationStrategyFactory()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        supervisor_node = SupervisorNode(self)
        inspect_node = PandasNode(self)
        feature_engineer_node = FeatureEngineeringNode(self)
        imputator_node = ImputatorNode(self)

        # register nodes using their execute methods
        workflow.add_node("supervisor", supervisor_node.execute)
        workflow.add_node("inspect", inspect_node.execute)
        workflow.add_node("feature_engineer", feature_engineer_node.execute)
        workflow.add_node("imputator", imputator_node.execute)

        workflow.set_entry_point("supervisor")

        workflow.add_edge("inspect", "supervisor")
        workflow.add_edge("feature_engineer", "supervisor") 
        workflow.add_edge("imputator", "supervisor")


        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "inspect": "inspect",
                "imputator": "imputator",
                "feature_engineer": "feature_engineer", 
                "end": END,
            },
        )

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _should_continue(self, state: AgentState) -> Literal["inspect","imputator","feature_engineer","end"]:
        next_decision = state.get("next", "").lower()
        if  next_decision in ["inspect", "imputator", "feature_engineer"]:
            return next_decision
        else:
            return "end"

    def invoke(self, initial_message: str, thread_id: str):
        """Executa o grafo e imprime apenas o resultado final."""
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"msg": initial_message, "logs": [], "main_goal": initial_message}
     
        final_state = self.graph.invoke(initial_state, config=config)
        #, recursion_limit=15
        
        print("\n--- RESULTADO FINAL DO GRAFO ---")
        for key, value in final_state.items():
            if key in ['subagents_report', 'next']:
                continue
            print(f"  {key}: {value}")
        return final_state

    # same as 'invoke' but better for debug (should be removed when we start using langsmith correctly)
    def stream(self, initial_message: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"msg": initial_message, "logs": [], "main_goal": initial_message}
        
        for event in self.graph.stream(initial_state, config=config):
            #, recursion_limit=15
            for key, value in event.items():
                print(f"--- Evento do Nó: {key} ---")
                print(value)
                print("\n")

