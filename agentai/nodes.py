from typing import Any, Dict
import json
import re

from langchain_core.messages import HumanMessage
from agentai.modules.common import AgentState
from agentai.agents import (
    create_pandas_agent,
    create_supervisor_agent,
    create_imputator_agent
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
            logs.append(f"Node '{self.name}' error: {e}")
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
            error_report = "FeatureEngineeringNode: no DataFrame available on executor."
            logs.append(error_report)
            return {"subagents_report": error_report, "logs": logs}

        try:
            # Rolling average for 'temperature'
            if "rolling average" in msg and "temperature" in msg:
                logs.append("Executing: Create rolling average for temperature.")
                new_col = 'temperature_rolling_avg_3h'
                df[new_col] = df['temperature'].rolling(window=3, min_periods=1).mean().fillna(method="bfill")
                report = f"Successfully created column: {new_col}"

            # Rolling standard deviation for 'temperature'
            elif "standard deviation" in msg and "temperature" in msg:
                logs.append("Executing: Create rolling standard deviation for temperature.")
                new_col = 'temperature_rolling_std_3h'
                df[new_col] = df['temperature'].rolling(window=3, min_periods=1).std().fillna(0)
                report = f"Successfully created column: {new_col}"

            else:
                report = "No specific feature engineering task found in the instruction."

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

        agent = create_pandas_agent(self.executor.df)
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
        logs.append("Executing imputation node.")

        imputator_agent = create_imputator_agent()
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
        supervisor_agent = create_supervisor_agent()

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
        logs.append(f"Supervisor decision: {output}")

        return {
            "next": next_step,
            "msg": msg_out,
            "logs": logs,
            "subagents_report": None,
            "main_goal": main_goal,
        }


class RetrieverNode(Node):
    def __init__(self, executor):
        super().__init__("retriever")
        self.executor = executor

    def execute(self, state: AgentState) -> dict:
        logs = state.get("logs", [])
        msg = state.get("msg", "")
        df = getattr(self.executor, "df", None)

        if df is None:
            error_report = "RetrieverNode: no DataFrame available."
            logs.append(error_report)
            return {"subagents_report": error_report, "logs": logs}

        try:
            # MUST CREATE RAG HERE.
            report = ...
            log_report = "[Retriever Node]" + report

            # Persist changes back to the executor
            self.executor.df = df
            logs.append(report)
            return {"subagents_report": report, "logs": logs}

        except Exception as e:
            error_report = f"Error in feature engineering node: {e}"
            logs.append(error_report)
            return {"subagents_report": error_report, "logs": logs}
