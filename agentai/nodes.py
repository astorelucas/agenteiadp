import json
import re
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any
import pandas as pd
import numpy as np

from langchain_core.messages import HumanMessage
from agentai.modules.common import AgentState
from agentai.rag import RAG
from agentai.agents import (
    create_pandas_agent,
    create_supervisor_agent,
    create_imputator_agent,
    create_feedback_agent,
    create_feature_engineering_agent,
    create_automl_agent,
    create_summarizer_agent
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
        self.agent = create_feature_engineering_agent(self.executor.df, self.executor.llm)

    def check_data_leak(self, features, target):
        num_rows = len(self.executor.df)

        if num_rows>500:
            n = random.randint(0, num_rows-500)
            sample =  self.executor.df.iloc[n:n+500].copy()
        else:
            sample = self.executor.df.copy()

        didLeak = False
        leaks = []

        for feature in features:
            
            correlations = []
            lags = []

            for i in range(-3,4, 1):
                valid = sample[[feature, target]].copy()
                valid[target] = valid[target].shift(i)
                valid.dropna(inplace=True)
                corr = valid[feature].corr( valid[target] )
                if not np.isnan(corr):
                    correlations.append(corr)
                    lags.append(i)

            if not correlations:
                continue

            idx = np.argmax( np.abs(correlations) )
            lag = lags[idx]
            value = correlations[idx]

            # pegar leak futuro
            if lag < 0 and abs(value) > 0.3: 
                didLeak = True
                leaks.append(feature)
                print("VAZAMENTO FUTURO\n")
            
            # pegar leak presente
            else:
                if 0 in lags:
                    corr_0 = correlations[lags.index(0)]
                    if abs(corr_0) > 0.9:  
                        didLeak = True
                        leaks.append(feature)
                        print("VAZAMENTO PRESENTE\n")


        return (didLeak, leaks)



    def execute(self, state: AgentState) -> dict:
        logs = state.get("logs", [])
        msg = state.get("msg", "")
        
        thought_collector = AgentThoughtCollector()

        if self.executor.df is None:
            error_report = "[FeatureEngineeringNode] No DataFrame available on executor."
            logs.append(error_report)
            return {"subagents_report": error_report, "logs": logs}

        try:
            count = 0
            max_attempts = 3
            base_msg = msg

            while count < max_attempts:
                count+=1

                original_cols = set(self.executor.df.columns)

                response = self.agent.invoke({"input": msg}, config={"callbacks": [thought_collector]})
                report = response.get("output", "") or str(response)
                
                new_features = list( set(self.executor.df.columns) - original_cols)

                if not new_features:
                    break

                target = state.get("target")

                didLeak, leaks = self.check_data_leak(new_features, target=target)


                if didLeak:
                    print(f"\n[LeakDetection] Vazamento detectado nas features: {leaks}\n")
                    self.executor.df.drop(columns=leaks, inplace=True, errors="ignore")
                    self.agent = create_feature_engineering_agent(self.executor.df, self.executor.llm)
                    msg = f"{base_msg}\nAvoid creating features that use future information. Previous features {leaks} caused data leakage."
                    continue
                print(f"\n[LeakDetection] Não houve vazamento de dados\n")
                break
            

            if count == max_attempts:
                logs.append("[LeakDetection] Maximum number of attempts reached in Feature Engineering, some features were removed. Agent thoughts are shown below:")

            full_thought_process = thought_collector.thoughts
            
            complete_report = (
                f"{full_thought_process}\n"
                f"FINAL REPORT:{report}"
            )

            logs.append(f"[FeatureEngineeringNode] {complete_report}")
            return {"subagents_report": complete_report, "logs": logs}

        except Exception as e:
            error_report = f"[FeatureEngineeringNode] Error: {e}"
            logs.append(error_report)
            return {"subagents_report": error_report, "logs": logs}


class PandasNode(Node):
    """Run a pandas-capable inspection agent against the executor.df"""
    def __init__(self, executor):
        super().__init__("inspect")
        self.executor = executor
        self.agent = create_pandas_agent(self.executor.df, self.executor.llm)

    def execute(self, state: AgentState) -> dict:
        msg = state.get("msg", "")
        logs = state.get("logs", [])
        max_retries = 2
        current_input = msg
        report = "\n[Pandas Node] "

        thought_collector = AgentThoughtCollector()


        for attempt in range(max_retries + 1):
            try:
                response = self.agent.invoke(
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
        self.imputator_agent = create_imputator_agent(self.executor.llm)

    def execute(self, state: AgentState) -> dict:
        context = state.get("msg", "")
        logs = state.get("logs", [])
    
        response = self.imputator_agent.invoke({"messages": [HumanMessage(content=context)]})
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
        self.supervisor_agent = create_supervisor_agent(self.executor.llm)

    def execute(self, state: AgentState) -> dict:
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

        response = self.supervisor_agent.invoke({"messages": [HumanMessage(content=input_message)]})

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
        logs.append(f"\n[Supervisor Node] Decision made: {output}")
        
        return_state = {
            "next": next_step,
            "msg": msg_out,
            "logs": logs,
            "subagents_report": None,
            "main_goal": main_goal
        }

        if next_step == "automl":
            test_size = plan.get("test_size")
            target = plan.get("target")
            prediction_length = plan.get("prediction_length")
            eval_metric = plan.get("eval_metric")
            return_state = {**return_state, "test_size": test_size, "target": target, "prediction_length": prediction_length, "eval_metric": eval_metric}

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
    

class FeedbackNode(Node):
    def __init__(self, executor):
        super().__init__("feedback")
        self.executor = executor
        self.agent = create_feedback_agent(executor.llm)
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
        self.automl_agent = None

    def execute(self, state: AgentState) -> dict:
        logs = state.get("logs", [])
        msg = state.get("msg", "")

        # Robust Extraction for test_size
        test_size_raw = state.get("test_size")
        try:
            test_size = float(test_size_raw)
        except (TypeError, ValueError):
            error_message = f"Invalid or missing test_size: {test_size_raw!r}. It must be a float between 0 and 1."
            logs.append(f"[AutoML Node] {error_message}")
            return {"subagents_report": error_message, "logs": logs}

        # Robust Extraction for target
        target = state.get("target")
        if not isinstance(target, str) or not target or target not in self.executor.df.columns:
            error_message = f"Invalid or missing target column: {target!r}. It must be a non-empty string present in the dataset columns."
            logs.append(f"[AutoML Node] {error_message}. RETURN TO SUPERVISOR OR USE RETRIEVER TO DECIDE.")
            return {"subagents_report": error_message, "logs": logs}

        # Range validation (already robust)
        if not (0 < test_size < 1):
            error_message = "Invalid test_size. Must be a float between 0 and 1."
            logs.append(f"[AutoML Node] {error_message}")
            return {"subagents_report": error_message, "logs": logs}

        prediction_length = state.get("prediction_length")
        if prediction_length is None:
            logs.append(f"[AutoML Node] Forecasting horizon (prediction_length) is not specified, RETURN TO SUPERVISOR OR USE RETRIEVER TO DECIDE.")
            return {"subagents_report": f"Forecasting horizon (prediction_length) is not specified, please select one before proceeding.", "logs": logs}
        else:
            logs.append(f"[AutoML Node] Using default forecasting horizon (prediction_length) of {prediction_length}.")

        eval_metric = state.get("eval_metric")  # default evaluation metric
        if eval_metric is None:
            logs.append(f"[AutoML Node] Evaluation metric (eval_metric) is not specified, RETURN TO SUPERVISOR OR USE RETRIEVER TO DECIDE.")
            return {"subagents_report": f"Evaluation metric (eval_metric) is not specified, please select one before proceeding.", "logs": logs}
        else:
            logs.append(f"[AutoML Node] Using evaluation metric: {eval_metric}.")

        df = self.executor.df

        # Normalize column names (remove stray spaces) to help timestamp detection
        try:
            df.columns = df.columns.str.strip()
        except Exception:
            pass

        # Resolve ambiguity of timestamp:
        # - If the DataFrame has a DatetimeIndex (even unnamed), bring it back as a column so the autogluon tool can detect it.
        # - Also handle the case where the index has a name like 'date' or 'time' but that name is not present in columns.
        try:
            is_dt_index = isinstance(df.index, pd.DatetimeIndex)
        except Exception:
            is_dt_index = False

        if is_dt_index and (df.index.name is None or df.index.name not in df.columns):
            df = df.reset_index()
        elif df.index.name and df.index.name not in df.columns and any(sub in str(df.index.name).lower() for sub in ("date", "time", "stamp")):
            df = df.reset_index()

        # Remove duplicated columns silently
        df = df.loc[:, ~df.columns.duplicated()]

        self.executor.df = df

        # Create and persist the automl agent so it can be reused
        if self.automl_agent is None:
            print(f"-------DF columns: {self.executor.df.columns}--")
            self.automl_agent = create_automl_agent(self.executor.df, self.executor.llm, target, test_size, prediction_length, eval_metric)

        automl_agent = self.automl_agent

        try:
            # Invoke the AutoML agent
            input_message = (
                f"Based on the following instruction: '{msg}', select the best time series forecasting model and its hyperparameters using automl tools.\n"
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

            # If the tool itself returned an error field, report it clearly and return the full tool result
            if isinstance(decision, dict) and decision.get("error"):
                error_msg = f"[AutoML Node] Tool reported error: {decision.get('error')}"
                # include any logs provided by the tool
                tool_logs = decision.get("logs") or []
                if tool_logs:
                    logs.extend(tool_logs)
                logs.append(error_msg)
                return {"subagents_report": error_msg, "automl_result": decision, "logs": logs}

            # Build a concise report of the most important keys, but preserve all prediction data
            model = decision.get("best_model") or decision.get("model")
            real = decision.get("real")
            forecast = decision.get("forecast")
            logs_agent = decision.get("logs") or []

            report_lines = []
            if model:
                report_lines.append(f"Best Model: {model}")
            if real is not None and forecast is not None:
                report_lines.append(f"Forecast on {len(forecast)} points completed.")
            if logs_agent:
                report_lines.append(f"Logs: {logs_agent[-2:]}")  # Show last two log lines
            report = " | ".join(report_lines) if report_lines else str(decision)

            logs.append(report)
            # Return the full original result dict for downstream usage
            return {"subagents_report": report, "automl_result": decision, "logs": logs}

        except Exception as e:
            report = f"[AutoML Node] Error during execution: {e}"
            logs.append(report)
            return {"subagents_report": report, "logs": logs}

class SummarizerNode(Node):
    def __init__(self, executor):
        super().__init__("summarizer")
        self.executor = executor
        self.summarizer_agent = create_summarizer_agent(self.executor.llm)

    def execute(self, state:AgentState) -> dict:
        msg = state.get("msg", "")
        logs = state.get('logs', [])
        logs_to_summarize = "\n".join(logs)
        prompt = f"summarize the following logs and the main goal:\n{logs_to_summarize}\n\nMain Goal: {msg}"

        try:
            output_path = "agentai/preprocessed_dataset.csv"
            self.executor.df.to_csv(output_path, index=False)
            logs.append(f"Preprocessed dataset saved to {output_path}")
        except Exception as e:
                logs.append(f"Error saving preprocessed dataset: {e}")

        summary_text = ""
        try:
            response = self.summarizer_agent.invoke({"messages": [HumanMessage(content=prompt)]})
            summary_text = str(response.get("messages", [])[-1].content)
            logs.append("\n[Summarizer Node] Finished summarizing.")
        except Exception as e:
            summary_text = f"ERRO: Falha ao invocar o agente de resumo: {e}"
            logs.append("\n[Summarizer Node] An error occurred whilst summarizing the logs")

        return {"logs": logs, "summary": summary_text}