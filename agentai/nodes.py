import json
import re
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any


from langchain_core.messages import HumanMessage
from agentai.modules.common import AgentState
from agentai.rag import RAG
from agentai.agents import (
    create_pandas_agent,
    create_supervisor_agent,
    create_imputator_agent,
    create_plotter_agent,
    create_feedback_agent,
    create_feature_engineering_agent,
    create_automl_agent
)

class Node:
    def __init__(self, name: str, executor=None):
        self.name = name
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self, state: AgentState) -> dict:
        try:
            return self.execute(state)
        except Exception as e:
            logs = state.get("logs", [])
            logs.append(f"[Node '{self.name}]' error: {e}")
            return {"subagents_report": f"Error in node '{self.name}': {e}", "logs": logs}

    def __call__(self, state: AgentState) -> dict:
        return self.run(state)


class AgentThoughtCollector(BaseCallbackHandler):
    """
        Callback para coletar os pensamentos internos de uma a├º├úo de agente.
        Utilidade: o pandas agente, por exemplo, n├úo coloca seus pensamentos na sua reposta final. Essa classe resolve isso
    """
    def __init__(self):
        self.thoughts = []

    def on_agent_action(self, action: Dict[str, Any], **kwargs: Any) -> Any:
        if hasattr(action, 'log'):
            self.thoughts.append(action.log)


class FeatureEngineeringNode(Node):
    def __init__(self, executor):
        super().__init__("feature_engineer")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        logs = state.get("logs", [])
        msg = state.get("msg", "")
        df = getattr(self.executor, "df", None)

        if df is None:
            error_report = "[FeatureEngineeringNode] No DataFrame available on executor."
            logs.append(error_report)
            return {"subagents_report": error_report, "logs": logs}

        try:
            agent = create_feature_engineering_agent(df, self.executor.llm)
            response = agent.invoke({"input": msg})
            fe_report = response.get("output", "") or str(response)

            # Pega o df atualizado
            self.executor.df = df  

            logs.append(f"[FeatureEngineeringNode] Executed instruction: '{msg}' -> {fe_report}")
            return {"subagents_report": fe_report, "logs": logs}

        except Exception as e:
            error_report = f"[FeatureEngineeringNode] Error: {e}"
            logs.append(error_report)
            return {"subagents_report": error_report, "logs": logs}


class PandasNode(Node):
    """Run a pandas-capable inspection agent against the executor.df"""
    def __init__(self, executor):
        super().__init__("inspect")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        msg = state.get("msg", "")
        logs = state.get("logs", [])
        max_retries = 2

        agent = create_pandas_agent(self.executor.df, self.executor.llm)
        current_input = msg
        report = "\n[Pandas Node] "

        thought_collector = AgentThoughtCollector()


        for attempt in range(max_retries + 1):
            try:
                response = agent.invoke(
                    {"input": current_input},
                    config={"callbacks": [thought_collector]}
                )
                report += response.get("output", "") or str(response)
                break
            except Exception as e:
                logs.append(f"Attempt {attempt + 1}/{max_retries + 1} failed for instruction '{msg}'. Error: {e}")
                if attempt == max_retries:
                    report += f"Agent failed after {max_retries + 1} attempts. Final Error: {e}"
                    break

                current_input = f"Your previous attempt failed with this error: {e}. Please correct your code and try again to accomplish the original task: {msg}"
            
        full_thought_process = "\n".join(thought_collector.thoughts)
        
        complete_report = (
            f"{full_thought_process}\n"
            f"FINAL REPORT:{report}"
        )

        logs.append(f"[Pandas Node]: {complete_report}")

        return {"subagents_report": complete_report, "logs": logs}


class ImputatorNode(Node):
    def __init__(self, executor):
        super().__init__("imputator")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        context = state.get("msg", "")
        logs = state.get("logs", [])
        
        imputator_agent = create_imputator_agent(self.executor.llm)
        response = imputator_agent.invoke({"messages": [HumanMessage(content=context)]})
        raw_output = str(response.get("messages", [])[-1].content)
        json_str_match = re.search(r'\{.*\}', raw_output, re.DOTALL)

        report = f"\n[Imputator Node] "

        if not json_str_match:
            report += f"Error: Imputator agent failed to produce valid JSON. Output: {raw_output}"
            logs.append(f"\n {report}")
            return {"subagents_report": report, "logs": logs}

        try:
            decision = json.loads(json_str_match.group(0))
            method = decision.get("method")
            params = decision.get("params", {})

            report += f"Imputator agent decided on method '{method}' with params {params}."

            strategy = self.executor.factory.create_strategy(name=method, **params)
            imputed_df = strategy.execute(self.executor.df)
            self.executor.df = imputed_df
            report += f"Imputation using '{method}' strategy completed successfully."

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            report += f"JSON error processing imputator agent decision: {e}. Raw output: {raw_output}"
            

        logs.append(report)
        return {"subagents_report": report, "logs": logs}


class SupervisorNode(Node):
    def __init__(self, executor):
        super().__init__("supervisor")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        supervisor_agent = create_supervisor_agent(self.executor.llm)

        previous_report = state.get("subagents_report")

        main_goal = state.get("main_goal", state.get('msg'))

        input_message = (
            f"Main Goal: {main_goal}\n\n"
            f"The dataset has {len(self.executor.df)} rows and {len(self.executor.df.columns)} columns.\n"
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
            logs.append(f"\n[Supervisor Node] Supervisor failed to produce JSON. Output: {raw_output}")
            return {"next": "END", "logs": logs}

        try:
            plan = json.loads(json_str_match.group(0))
        except json.JSONDecodeError:
            logs.append(f"\n[Supervisor Node] Supervisor produced invalid JSON. Output: {json_str_match.group(0)}")
            return {"next": "END", "logs": logs}

        next_step = plan.get("next", "END")
        msg_out = plan.get("msg", state.get("msg"))
        output = plan.get("output", "")
        is_before_dp = plan.get("is_before_dp").lower() == "true"
        logs.append(f"\n[Supervisor Node] Decision made: {output}")
        
        return_state = {
            "next": next_step,
            "msg": msg_out,
            "logs": logs,
            "subagents_report": None,
            "main_goal": main_goal,
            "is_before_dp": is_before_dp
        }

        if next_step == "automl":
            test_size = plan.get("test_size")
            target = plan.get("target")
            return_state = {**return_state, "test_size": test_size, "target": target}

        return return_state
    
class RetrieverNode(Node):
    def __init__(self):
        super().__init__("retriever")
        self.rag = RAG()

    def execute(self, state: AgentState) -> dict:
        logs = state.get("logs", [])
        msg = state.get("msg", "")

        try:
            report = self.rag.retrieve(msg)
            logs.append("\n[Retriever Node]: " + report)
            return {"subagents_report": report, "logs": logs}

        except Exception as e:
            error_report = f"[Retriever Node]: Error: {e}"
            logs.append(error_report)
            return {"subagents_report": error_report, "logs": logs}
    
class PlotterNode(Node):
    def __init__(self, executor):
        super().__init__("plot")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        msg = state.get("msg", "").lower()
        logs = state.get("logs", [])
        is_before_dp = state.get('is_before_dp')

        input_message = (
            f"Create plots to help analyze the dataset based on the following instruction: '{msg}'.\n"
            f"If the instruction is not clear, create simple plots like scatter, time series, heatmap and histogram.\n"
        )
        
        report = f"\n[Plotter Node] "

        try:
            agent = create_plotter_agent(self.executor.df, self.executor.images_path, self.executor.llm, is_before_dp=is_before_dp)
            response = agent.invoke({"input": input_message})
            report += response.get("output", "") or str(response)
        except Exception as e:
            report += f"Time series agent failed to execute instruction. Error: {e}"
        
        logs.append(report)
        return {"subagents_report": report, "logs": logs}



class FeedbackNode(Node):
    def __init__(self, executor):
        super().__init__("feedback")
        self.executor = executor
        self.agent = create_feedback_agent(self.executor.llm)
        self.rag = RAG()

    def execute(self, state: AgentState) -> dict:
        logs = state.get("logs", [])
        summary = state.get("summary", "")

        input_message = (
            f"Execution Logs:\n{logs}\n\n"
            f"Summary:\n{summary}\n\n"
            "Decide if there is knowledge worth storing."
        )

        try:
            response = self.agent.invoke({"messages": [HumanMessage(content=input_message)]})
            raw_output = str(response.get("messages", [])[-1].content)
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)

            report = f"\n[Feedback Node] "

            if not json_match:
                report += f"No valid JSON produced by the agent. Raw Output: {raw_output}"
                logs.append(report)
                return {"logs": logs, "feedback": None, "summary": summary, "subagents_report": report}

            decision = json.loads(json_match.group(0))
            if decision.get("store"):
                insight = decision.get("insight", "").strip()
                if insight:
                    self.rag.store(insight)
                    report += f"Stored new insight: {insight}"
                    logs.append(report)
                    return {"logs": logs, "feedback": insight, "summary": summary, "subagents_report": report}
            
            report += "No relevant insight to store. Note: This is not an error."
            logs.append(report)
            return {"logs": logs, "feedback": None, "summary": summary, "subagents_report": report}

        except Exception as e:
            report += f"Error during execution: {e}"
            logs.append(report)
            return {"logs": logs, "feedback": None, "summary": summary, "subagents_report": report}
    
class AutoMLNode(Node):
    def __init__(self, executor):
        super().__init__("automl")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        logs = state.get("logs", [])
        msg = state.get("msg", "")

        # Extract parameters from the state
        test_size = float(state.get("test_size"))
        target = state.get("target")

        # Validate inputs
        if not isinstance(test_size, (float)) or not (0 < test_size < 1):
            error_message = "Invalid test_size. Must be a float between 0 and 1."
            logs.append(f"[AutoML Node] {error_message}")
            return {"subagents_report": error_message, "logs": logs}

        if target not in self.executor.df.columns:
            error_message = f"Target column '{target}' not found in the dataset."
            logs.append(f"[AutoML Node] {error_message}")
            return {"subagents_report": error_message, "logs": logs}
        

        automl_agent = create_automl_agent(self.executor.df, self.executor.llm, target, test_size)
    
        try:
            # Invoke the AutoML agent
            input_message = (
                f"Based on the following instruction: '{msg}', select the best time series forecasting model and its hyperparameters using PyCaret.\n"
            )

            response = automl_agent.invoke({"input": input_message})

            # Parse the response
            raw_output = response.get("output", "") or str(response)
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)

            if not json_match:
                report = f"Error: AutoML agent failed to produce valid JSON. Output: {raw_output}"
                logs.append(report)
                return {"subagents_report": report, "logs": logs}

            decision = json.loads(json_match.group(0))
            model = decision.get("model")
            params = decision.get("params", {})

            report = f"AutoML selected model: {model} with parameters: {params}"
            logs.append(report)

        except Exception as e:
            report = f"[AutoML Node] Error during execution: {e}"
            logs.append(report)

        return {"subagents_report": report, "logs": logs}