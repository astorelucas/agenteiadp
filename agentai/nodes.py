import json
import re

from langchain_core.messages import HumanMessage
from agentai.modules.common import AgentState
from agentai.rag import RAG
from agentai.agents import (
    create_pandas_agent,
    create_supervisor_agent,
    create_imputator_agent,
    create_plotter_agent,
    create_feedback_agent,
    create_summarizer_agent,
    create_automl_agent,
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


class FeatureEngineeringNode(Node):
    def __init__(self, executor):
        super().__init__("feature_engineer")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        logs = state.get("logs", [])
        msg = state.get("msg", "")
        df = getattr(self.executor, "df", None)

        if df is None:
            error_report = "\n[FeatureEngineeringNode] no DataFrame available on executor."
            logs.append(error_report)
            return {"subagents_report": error_report, "logs": logs}

        try:
            # Rolling average for 'temperature'
            if "rolling average" in msg and "temperature" in msg:
                logs.append("\n[FeatureEngineeringNode] Executing: Create rolling average for temperature.")
                new_col = 'temperature_rolling_avg_3h'
                df[new_col] = df['temperature'].rolling(window=3, min_periods=1).mean().fillna(method="bfill")
                report = f"Successfully created column: {new_col}"

            # Rolling standard deviation for 'temperature'
            elif "standard deviation" in msg and "temperature" in msg:
                logs.append("\n[FeatureEngineeringNode] Executing: Create rolling standard deviation for temperature.")
                new_col = 'temperature_rolling_std_3h'
                df[new_col] = df['temperature'].rolling(window=3, min_periods=1).std().fillna(0)
                report = f"Successfully created column: {new_col}"

            else:
                report = "ERROR: No specific feature engineering task found in the instruction."

            # Persist changes back to the executor
            self.executor.df = df
            logs.append(report)
            return {"subagents_report": report, "logs": logs}

        except Exception as e:
            error_report = f"Error in feature engineering node: {e}"
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


class ImputatorNode(Node):
    def __init__(self, executor):
        super().__init__("imputator")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        context = state.get("msg", "")
        logs = state.get("logs", [])
        logs.append("\n[Imputator Node] Executing imputation node.")

        imputator_agent = create_imputator_agent(self.executor.llm)
        response = imputator_agent.invoke({"messages": [HumanMessage(content=context)]})

        raw_output = str(response.get("messages", [])[-1].content)
        json_str_match = re.search(r'\{.*\}', raw_output, re.DOTALL)

        if not json_str_match:
            report = f"Error: Imputator agent failed to produce valid JSON. Output: {raw_output}"
            logs.append(report)
            return {"subagents_report": report, "logs": logs}

        try:
            decision = json.loads(json_str_match.group(0))
            method = decision.get("method")
            params = decision.get("params", {})

            logs.append(f"Imputator agent decided on method '{method}' with params {params}.")

            strategy = self.executor.factory.create_strategy(name=method, **params)
            imputed_df = strategy.execute(self.executor.df)
            self.executor.df = imputed_df
            report = f"Imputation using '{method}' strategy completed successfully."
            logs.append(report)

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            report = f"Error processing imputator agent decision: {e}. Raw output: {raw_output}"
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
            logs.append(f"Supervisor failed to produce JSON. Output: {raw_output}")
            return {"next": "END", "logs": logs}

        try:
            plan = json.loads(json_str_match.group(0))
        except json.JSONDecodeError:
            logs.append(f"Supervisor produced invalid JSON. Output: {json_str_match.group(0)}")
            return {"next": "END", "logs": logs}

        next_step = plan.get("next", "END")
        msg_out = plan.get("msg", state.get("msg"))
        output = plan.get("output", "")
        is_before_dp = plan.get("is_before_dp")
        logs.append(f"Supervisor decision: {output}")

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
        
        try:
            agent = create_plotter_agent(self.executor.df, self.executor.images_path, self.executor.llm, is_before_dp=is_before_dp)
            response = agent.invoke({"input": input_message})
            plotter_report = response.get("output", "") or str(response)
            logs.append(f"[Plotter Node]: Time series agent successfully executed instruction: '{msg}'")
        except Exception as e:
            plotter_report = f"[Plotter Node]: Time series agent failed to execute instruction '{msg}'. Error: {e}"
            logs.append(plotter_report)
        
        return {"subagents_report": plotter_report, "logs": logs}



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

            import json, re
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if not json_match:
                report = "[Feedback Node] No valid JSON produced by the agent."
                logs.append(report)
                return {"logs": logs, "feedback": None, "summary": summary, "subagents_report": report}

            decision = json.loads(json_match.group(0))
            if decision.get("store"):
                insight = decision.get("insight", "").strip()
                if insight:
                    self.rag.store([insight])
                    report = f"[Feedback Node] Stored new insight: {insight}"
                    logs.append(report)
                    return {"logs": logs, "feedback": insight, "summary": summary, "subagents_report": report}
            
            report = "[Feedback Node] No relevant insight to store."
            logs.append(report)
            return {"logs": logs, "feedback": None, "summary": summary, "subagents_report": report}

        except Exception as e:
            report = f"[Feedback Node] Error during execution: {e}"
            logs.append(report)
            return {"logs": logs, "feedback": None, "summary": summary, "subagents_report": report}

class SummarizerNode(Node):
    def __init__(self, executor):
        super().__init__("summarizer")
        self.executor = executor

    def execute(self, state:AgentState) -> dict:
        summarizer_agent = create_summarizer_agent(self.executor.llm)

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
    
class AutoMLNode(Node):
    def __init__(self, executor):
        super().__init__("automl")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        automl_agent = create_automl_agent(self.executor.df, self.executor.llm)
        logs = state.get("logs", [])
        msg = state.get("msg", "")

        # Extract parameters from the state
        test_size = int(state.get("test_size"))
        target = state.get("target")

        # Validate inputs
        if not isinstance(test_size, (int)) or test_size <= 0:
            error_message = "Invalid test_size. Must be a positive integer."
            logs.append(f"[AutoML Node] {error_message}")
            return {"subagents_report": error_message, "logs": logs}

        if target not in self.executor.df.columns:
            error_message = f"Target column '{target}' not found in the dataset."
            logs.append(f"[AutoML Node] {error_message}")
            return {"subagents_report": error_message, "logs": logs}

        try:
            # Invoke the AutoML agent
            input_message = (
                f"test_size: {test_size}\n"
                f"target: {target}\n"
                f"{msg}"
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