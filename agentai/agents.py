
from getpass import getpass
import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_models import ChatDeepInfra
from langchain_experimental.agents import create_pandas_dataframe_agent
from agentai.tools import (
    inspection_tools
)

# -----Chat model Instantiation ----
os.environ["DEEPINFRA_API_KEY"] = getpass("Enter your  key: ")
llm = ChatDeepInfra(model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")

# Create Pandas DataFrame Agent for Inspection
def create_inspection_agent(df: pd.DataFrame) -> AgentExecutor:
    """Create specialized inspection agent for time series data"""
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="zero-shot-react-description",
        # max_iterations=10,
        allow_dangerous_code=True,
        # early_stopping_method="generate",
        handle_parsing_errors=True,
        extra_tools=inspection_tools,  # Add any additional tools if needed
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