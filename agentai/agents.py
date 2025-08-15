# agentai/agents.py

from getpass import getpass
import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from langchain.agents import AgentExecutor
from langchain_community.chat_models import ChatDeepInfra
from langgraph.prebuilt import create_react_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from agentai.tools import (
    inspection_tools
)

os.environ["DEEPINFRA_API_KEY"] = getpass("Enter your key: ")
llm = ChatDeepInfra(model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")

def create_inspection_agent(df: pd.DataFrame) -> AgentExecutor:
    # ... (nenhuma alteração nesta função)
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        extra_tools=inspection_tools,
        prefix="""You are a time series data inspection expert. Analyze the dataset thoroughly and 
        provide a detailed report with:
        1. Missing values analysis
        2. Statistical properties
        3. Time index regularity
        4. Outlier detection
        5. Data type validation

        Use appropriate pandas functions and methods to perform the analysis.
        Summarize findings clearly.
        If you encounter issues, explain them.

        When you are done, in the **Final Answer**, output ONLY valid JSON
        with the following keys: missing_values, statistics,has_infinity

        Example final output format:
        Final Answer: {{
        "missing_values": {{
            "total_missing": 2,
            "columns_with_missing": ["sensor1", "sensor2"],
            "time_gaps": {{
                "0 days 01:00:00": 3
                }}
        }},
        "statistics": {{**a json of statistics for each column**}},
        "has_infinity": false
        }}
        """
    )


def create_supervisor_agent() -> AgentExecutor:
    """Creates the supervisor agent"""
    return create_react_agent(
        model=llm,
        prompt=
        """
        You are a SUPERVISOR agent planning an Explanatory Data Analysis.
        Your job is to coordinate actions by returning a decision plan.

        Analyze the current state and results, then decide the next action.
        
        ALWAYS return ONLY a valid JSON object with the following fields:
        - "output": Your reasoning for the decision.
        - "next": The next action, which must be either "inspect" or "END".
        - "msg": (optional) The instruction for the next agent.
        
        IMPORTANT: Use double quotes for all keys and string values in the JSON.

        Example of a valid response:
        {"output": "The data is loaded. I will now ask the inspector to summarize missing values.", "next": "inspect", "msg": "Summarize missing values and check for outliers."}

        Another example of a valid response:
        {"output": "The inspection is complete. No further actions are needed.", "next": "END"}
        """,
        tools=[]
    )