import pandas as pd
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
    RetrieverNode,
    PlotterNode,
    FeedbackNode
)


class WorkflowExecutor:
    def __init__(self, llm, plot_images_path: str, csv_path: str = None):
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
        retriever_node = RetrieverNode()
        plotter_node = PlotterNode(self)
        feedback_node = FeedbackNode(self)
        
        # register nodes using their execute methods
        workflow.add_node("supervisor", supervisor_node.execute)
        workflow.add_node("inspect", inspect_node.execute)
        workflow.add_node("feature_engineer", feature_engineer_node.execute)
        workflow.add_node("imputator", imputator_node.execute)
        workflow.add_node("retriever", retriever_node.execute)
        workflow.add_node("plot", plotter_node.execute)
        workflow.add_node("summarizer", self._summarizer_node)
        workflow.add_node("feedback", feedback_node.execute)
        
        workflow.set_entry_point("plot")

        workflow.add_edge("plot", "supervisor")
        workflow.add_edge("inspect", "supervisor")
        workflow.add_edge("feature_engineer", "supervisor") 
        workflow.add_edge("imputator", "supervisor")
        workflow.add_edge("retriever", "supervisor")
        workflow.add_edge("feedback", "summarizer")
        workflow.add_edge("summarizer", END)

        
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "inspect": "inspect",
                "plot": "plot",
                "imputator": "imputator",
                "feature_engineer": "feature_engineer", 
                "retriever": "retriever",
                "end": "feedback",
            },
        )

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _should_continue(self, state: AgentState) -> Literal["inspect","imputator","feature_engineer", "retriever", "end"]:
        next_decision = state.get("next", "").lower()

        if  next_decision in ["inspect", "imputator", "feature_engineer", "retriever", "plot"]:
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
            logs.append("\n[Summarizer Node] Finished summarizing.")
        except Exception as e:
            summary_text = f"ERRO: Falha ao invocar o agente de resumo: {e}"
            logs.append("\n[Summarizer Node] An error occurred whilst summarizing the logs")

        return {"logs": logs, "summary": summary_text}

    def invoke(self, initial_message: str, thread_id: str):
        """Executa o grafo e imprime apenas o resultado final."""
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {"msg": initial_message, "logs": [], "main_goal": initial_message, "is_before_dp": True}

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