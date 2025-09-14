import pandas as pd
import re
import json
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agentai.agents import create_pandas_agent, create_supervisor_agent, create_imputator_agent, create_summarizer_agent, create_plotter_agent
from agentai.modules.common import AgentState
from agentai.tools import ImputationStrategyFactory
from agentai.nodes import (
    FeatureEngineeringNode,
    PandasNode,
    ImputatorNode,
    SupervisorNode,
    PlotterNode
)


class WorkflowExecutor:
    def __init__(self, csv_path: str, plot_images_path: str, llm):
        try:
            self.df = pd.read_csv(csv_path)
            self.images_path = plot_images_path
            self.llm = llm
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
        plotter_node = PlotterNode(self)

        # register nodes using their execute methods

        workflow.add_node("supervisor", supervisor_node.execute)
        workflow.add_node("inspect", inspect_node.execute)
        workflow.add_node("feature_engineer", feature_engineer_node.execute)
        workflow.add_node("imputator", imputator_node.execute)
        workflow.add_node("plot", plotter_node.execute)
        workflow.add_node("summarizer", self._summarizer_node)
        
        workflow.set_entry_point("plot")

        workflow.add_edge("plot", "supervisor")
        workflow.add_edge("inspect", "supervisor")
        workflow.add_edge("feature_engineer", "supervisor") 
        workflow.add_edge("imputator", "supervisor")
        workflow.add_edge("summarizer", END)
        
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "inspect": "inspect",
                "plot": "plot",
                "imputator": "imputator",
                "feature_engineer": "feature_engineer", 
                "end": "summarizer",
            },
        )

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _should_continue(self, state: AgentState) -> Literal["inspect","imputator","feature_engineer","end"]:
        next_decision = state.get("next", "").lower()
        if  next_decision in ["inspect", "imputator", "plot", "feature_engineer"]:
            return next_decision
        else:
            return "end"
        

    def _summarizer_node(self, state:AgentState) -> dict:
        summarizer_agent = create_summarizer_agent(self.llm)
        
        logs = state.get('logs', [])
        logs_to_summarize = "\n".join(logs)
        prompt = f"summarize the following logs:\n{logs_to_summarize}"

        summary_text = ""
        try:
            response = summarizer_agent.invoke({"messages": [HumanMessage(content=prompt)]})
            summary_text = str(response.get("messages", [])[-1].content)
            logs.append("Finished summarizing.")
        except Exception as e:
            summary_text = f"ERRO: Falha ao invocar o agente de resumo: {e}"
            logs.append("An error occurred whilst summarizing the logs")

        return {"logs": logs, "summary": summary_text}
        
    def invoke(self, initial_message: str, thread_id: str):
        """Executa o grafo e imprime apenas o resultado final."""
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"msg": initial_message, "logs": [], "main_goal": initial_message, "is_before_dp": True}
     
        final_state = self.graph.invoke(initial_state, config=config)
        #, recursion_limit=15
        
        print("\n--- RESULTADO FINAL DO GRAFO ---")
        for key, value in final_state.items():
            if key in ['subagents_report', 'next', 'summary']:
                continue
            print(f"  {key}: {value}")

        summary = final_state.get("summary", "ERRO: Nenhum resumo foi gerado.")

        print(f"\n\nRESUMO:\n {summary}")

        return final_state