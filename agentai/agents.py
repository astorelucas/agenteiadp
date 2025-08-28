import os
from getpass import getpass
import pandas as pd
import matplotlib
import seaborn
from sklearn.experimental import enable_iterative_imputer
from langchain.agents import AgentExecutor
from langchain_community.chat_models import ChatDeepInfra
from langgraph.prebuilt import create_react_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from agentai.tools import (
    inspection_tools
)

os.environ["DEEPINFRA_API_KEY"] = getpass("Enter your key: ")
# llm = ChatDeepInfra(model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")

llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")



# create_supervisor_agent: '{' instead of '{{', because its not fstring, just a normal string
# create_pandas_agent: if needed, use '{{' instead of '{', as it uses a fstring internally (????????????????????)

def create_pandas_agent(df: pd.DataFrame) -> AgentExecutor:
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        extra_tools= inspection_tools,
        prefix="""You are a data analysis expert working with a pandas DataFrame.
        Your primary goal is to execute a specific task given to you and report the results.

        IMPORTANT: You are working with a DataFrame that is ALREADY loaded into a variable named `df`.
        DO NOT try to redefine or recreate this `df` variable.
        Directly apply your pandas commands to the `df` variable, for example: `df.describe()`.
        """
    )


def create_supervisor_agent() -> AgentExecutor:
    """Creates the supervisor agent"""
    return create_react_agent(
        model=llm,
        prompt=
        """
        You are a SUPERVISOR agent, an expert in planning and coordinating an Exploratory Data Analysis (EDA) workflow.
        Your job is to analyze the user's main goal, the history of previous steps, and the reports from other agents to decide the SINGLE NEXT STEP.

        You must break down a high-level goal into a sequence of specific, actionable tasks for the 'inspect' agent.

        Based on the current state, decide what to do next. The possible actions are:
        1.  **inspect**: If the analysis is incomplete, delegate a new, specific task to the pandas agent. The task should be a logical next step towards the main goal.
        2.  **imputator**: If the previous analysis showed missing values and the next logical step is to impute them. You must delegate this to the imputation specialist.
        3.  **END**: If you have gathered all necessary information to fulfill the user's main goal and the analysis is complete.

        ALWAYS return ONLY a valid JSON object with the following fields:
        - "output": Your reasoning for the decision. Explain what has been done and why you are choosing the next action.
        - "next": The next action, which must be either "inspect", "imputator" or "END".
        - "msg": A clear and specific instruction for the next agent. For 'imputator', this should be a descriptive context of the dataset for it to make a decision.


        IMPORTANT: Use double quotes for all keys and string values in the JSON.

        Example 1 (Starting):
        {"output": "The analysis has just started. I will begin by getting an overview of the dataset.", "next": "inspect", "msg": "Summarize the dataset, checking for missing values and data types."}

        Example 2 (Delegating Imputation):
        {"output": "The inspection revealed missing data in several columns. I will now delegate the task of choosing the best imputation method to the specialist.", "next": "imputator", "msg": "The initial analysis found missing values in the following columns: ['temperature', 'pressure']. The data appears to be time-series sensor data."}

        Example 3 (Ending):
        {"output": "The data has been inspected and imputed. The goal is met. The workflow will now end.", "next": "END", "msg": "Workflow complete."}
        """,
        tools=[]
    )

def create_imputator_agent() -> AgentExecutor:
    """Creates the imputator agent"""
    return create_react_agent(
        model=llm,
        prompt=
        """
        You are an IMPUTATOR agent, an expert in data imputation techniques.
        Your sole job is to analyze the context provided about a dataset and decide the BEST imputation method.
        You have three methods available: 'knn', 'mice', and 'gp'.
        
        - Use 'knn' for data with local patterns (like sensor data) or simple relationships. It is computationally cheap.
        - Use 'mice' for data with complex relationships between variables. It is more robust than knn and handles various data types well.
        - Use 'gp' (Gaussian Process) for time-series or data where estimating uncertainty is crucial. It is computationally very expensive and best for small datasets.

        Based on the context, you MUST return ONLY a valid JSON object with your decision. The JSON must have two keys:
        - "method": A string with your chosen method, which must be one of ["knn", "mice", "gp"].
        - "params": A JSON object containing the parameters for that method.
            - For "knn", provide "n_neighbors" (e.g., 5).
            - For "mice", provide "n_estimators" (e.g., 10).
            - For "gp", you can provide an empty object {}.
        
        Example Input Context:
        "The inspection revealed missing data in 'temperature' and 'humidity' columns. These are sensor readings and likely have correlations with nearby time points."

        Example of a valid response for the context above:
        {"method": "knn", "params": {"n_neighbors": 5}}

        Another Example Input Context:
        "Missing data found in 'age', 'income', and 'credit_score' columns. These variables are likely interdependent in a complex, non-linear way."
        
        Another valid response:
        {"method": "mice", "params": {"n_estimators": 10}}
        """,
        tools=[]
    )