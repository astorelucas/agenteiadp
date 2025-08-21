import pandas as pd
import re
import json
from typing import Literal
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agentai.agents import create_pandas_agent, create_supervisor_agent
from agentai.modules.common import AgentState


class WorkflowExecutor:
    def __init__(self, csv_path: str):
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Falha ao carregar o dataset: {e}")
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("inspect", self._pandas_node)
        
        workflow.set_entry_point("supervisor")

        workflow.add_edge("inspect", "supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "inspect": "inspect",
                "end": END,
            },
        )

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _should_continue(self, state: AgentState) -> Literal["inspect", "end"]:
        if state.get("next", "").lower() == "inspect":
            return "inspect"
        else:
            return "end"

    def _pandas_node(self, state: AgentState) -> dict:
        msg = state.get("msg")
        logs = state.get("logs", [])
        max_retries = 2
        
        agent = create_pandas_agent(self.df)
        current_input = msg
        inspection_report = ""

        for attempt in range(max_retries + 1):
            try:
                response = agent.invoke({"input": current_input})
                inspection_report = response.get("output", "") or str(response)
                logs.append(f"Inspection agent successfully executed instruction: '{msg}'")
                break
            except Exception as e:
                logs.append(f"Attempt {attempt + 1}/{max_retries + 1} failed for instruction '{msg}'. Error: {e}")
                if attempt == max_retries:
                    inspection_report = f"Agent failed after {max_retries + 1} attempts. Final Error: {e}"
                    break
                
                current_input = f"Your previous attempt failed with this error: {e}. Please correct your code and try again to accomplish the original task: {msg}"
        
        return {"subagents_report": inspection_report, "logs": logs}

    def _supervisor_node(self, state: AgentState) -> dict:
        supervisor_agent = create_supervisor_agent()
        
        previous_report = state.get("subagents_report")
        
        main_goal = state.get("main_goal", state.get('msg'))
        
        # Contexto do supervisor agora em inglês
        input_message = (
            f"Main Goal: {main_goal}\n\n"
            f"The dataset has {len(self.df)} rows and {len(self.df.columns)} columns.\n"
            f"Current Task: {state.get('msg')}\n"
            f"Logs from previous steps:\n{state.get('logs')}\n"
        )
        if previous_report:
            input_message += f"Report from the previous step:\n{previous_report}"

        response = supervisor_agent.invoke({"messages": [HumanMessage(content=input_message)]})
        
        logs = state.get("logs", [])
        raw_output = str(response.get("messages", [])[-1].content)
        json_str_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        
        if not json_str_match:
            # Logs de erro em inglês
            logs.append(f"Supervisor failed to produce JSON. Output: {raw_output}")
            return {"next": "END", "logs": logs}
        
        try:
            plan = json.loads(json_str_match.group(0))
        except json.JSONDecodeError:
            # Logs de erro em inglês
            logs.append(f"Supervisor produced invalid JSON. Output: {json_str_match.group(0)}")
            return {"next": "END", "logs": logs}
        
        next_step = plan.get("next", "END")
        msg_out = plan.get("msg", state.get("msg"))
        output = plan.get("output", "")
        # Log de decisão em inglês
        logs.append(f"Supervisor decision: {output}")
        
        return {
            "next": next_step, 
            "msg": msg_out, 
            "logs": logs, 
            "subagents_report": None,
            "main_goal": main_goal 
        }

    def invoke(self, initial_message: str, thread_id: str):
        """Executa o grafo e imprime apenas o resultado final."""
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"msg": initial_message, "logs": [], "main_goal": initial_message}
     
        final_state = self.graph.invoke(initial_state, config=config, recursion_limit=15)
        
        print("\n--- RESULTADO FINAL DO GRAFO ---")
        for key, value in final_state.items():
            if key in ['subagents_report', 'next']:
                continue
            print(f"  {key}: {value}")


    def stream(self, initial_message: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"msg": initial_message, "logs": [], "main_goal": initial_message}
        
        for event in self.graph.stream(initial_state, config=config, recursion_limit=15):
            for key, value in event.items():
                print(f"--- Evento do Nó: {key} ---")
                print(value)
                print("\n")