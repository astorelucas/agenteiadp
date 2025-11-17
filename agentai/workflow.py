import pandas as pd
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agentai.modules.common import AgentState
from agentai.tools import ImputationStrategyFactory
from agentai.nodes import (
    FeatureEngineeringNode,
    PandasNode,
    ImputatorNode,
    SupervisorNode,
    FeedbackNode,
    AutoMLNode,
    SummarizerNode
)

class WorkflowExecutor:
    def __init__(self, llm, csv_path: str, plot_images_path: str):
        try:
            self.df = pd.read_csv(csv_path)
            self.images_path = plot_images_path
            self.llm = llm
            self.factory = ImputationStrategyFactory()
            self.graph = self._build_graph()
        except Exception as e:
            raise ValueError(f"Falha ao carregar o dataset: {e}")
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        supervisor_node = SupervisorNode(self)
        inspect_node = PandasNode(self)
        feature_engineer_node = FeatureEngineeringNode(self)
        imputator_node = ImputatorNode(self)
        feedback_node = FeedbackNode(self)
        automl_node = AutoMLNode(self)
        summarizer_node = SummarizerNode(self)
        
        # register nodes using their execute methods
        workflow.add_node("supervisor", supervisor_node.execute)
        workflow.add_node("inspect", inspect_node.execute)
        workflow.add_node("feature_engineer", feature_engineer_node.execute)
        workflow.add_node("imputator", imputator_node.execute)

        workflow.add_node("summarizer", summarizer_node.execute)
        workflow.add_node("feedback", feedback_node.execute)
        workflow.add_node("automl", automl_node.execute)
        
        workflow.set_entry_point("supervisor")

        workflow.add_edge("inspect", "supervisor")
        workflow.add_edge("feature_engineer", "supervisor") 
        workflow.add_edge("imputator", "supervisor")
        workflow.add_edge("automl", "supervisor")
        workflow.add_edge("feedback", "summarizer")
        workflow.add_edge("summarizer", END)
        
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "inspect": "inspect",
                "imputator": "imputator",
                "feature_engineer": "feature_engineer", 
                "automl": "automl",
                "end": "feedback",
            },
        )

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _should_continue(self, state: AgentState) -> Literal["inspect","imputator","feature_engineer","automl", "end"]:
        next_decision = state.get("next", "").lower()

        if  next_decision in ["inspect", "imputator", "feature_engineer","automl"]:
            return next_decision
        else:
            return "end"
        
    def invoke(self, initial_message: str, thread_id: str):
        """Executa o grafo e imprime apenas o resultado final."""
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {"msg": initial_message, "logs": [], "main_goal": initial_message}

        final_state = {}

        print("\n--- INICIANDO EXECUÇÃO DO GRAFO ---")

        for chunk in self.graph.stream(initial_state, config=config, recursion_limit=35):
            
            for node_name, state in chunk.items():
                print(f"\n--- [ Nó Executado: {node_name} ] ---")

                if report := state.get("subagents_report"):
                    print("Relatório:")
                    print(report)
                    if node_name == "supervisor":
                        print("-" * 25)

                if node_name == "supervisor":
                    print(f"Próximo passo planejado: '{state.get('next')}'")
                    print(f"Instrução para o próximo agente: \"{state.get('msg')}\"")

                elif node_name == "summarizer":
                    print("Execução concluída. Gerando resumo...")

                if state:
                    final_state = state

        print("\n\n--- FIM DA EXECUÇÃO DO GRAFO ---")
        
        summary = final_state.get("summary", "ERRO: Nenhum resumo foi gerado.")
        print(f"\n\nRESUMO:\n{summary}")

        return final_state