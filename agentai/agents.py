import os
from dotenv import load_dotenv
from getpass import getpass
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from langchain.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from agentai.tools import (
    inspection_tools,
    make_plot_tools
)

# create_supervisor_agent: '{' instead of '{{', because its not fstring, just a normal string
# create_pandas_agent: if needed, use '{{' instead of '{', as it uses a fstring internally (????????????????????)

def create_pandas_agent(df: pd.DataFrame, llm) -> AgentExecutor:
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        extra_tools=inspection_tools,
        prefix="""You are a data analysis expert working with a pandas DataFrame.
        Your primary goal is to execute a specific task given to you and report the results.

        IMPORTANT: You are working with a DataFrame that is ALREADY loaded into a variable named `df`.
        DO NOT try to redefine or recreate this `df` variable.
        Directly apply your pandas commands to the `df` variable, for example: `df.describe()`.

        - Carefully follow the user's instruction.
        - Use the available tools to perform the analysis.
        - Your final response MUST BE a clear report of your findings.
        """
    )


def create_supervisor_agent(llm) -> AgentExecutor:
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
        2.  **END**: If you have gathered all necessary information to fulfill the user's main goal and the analysis is complete.

        ALWAYS return ONLY a valid JSON object with the following fields:
        - "output": Your reasoning for the decision. Explain what has been done and why you are choosing the next action.
        - "next": The next action, which must be either "inspect" or "END".
        - "msg": A clear and specific instruction for the next agent if the action is 'inspect'.

        IMPORTANT: Use double quotes for all keys and string values in the JSON.

        Example of a valid response:
        {"output": "The analysis has just started. I will begin by getting a basic overview of the dataset, focusing on missing values and data types.", "next": "inspect", "msg": "Summarize the dataset, checking for missing values and the data type of each column."}

        Another example after the first step is complete:
        {"output": "The basic overview is complete. I see some missing values in two columns. Now, I will ask for a statistical summary to understand the distribution of the numerical columns.", "next": "inspect", "msg": "Provide a descriptive statistical summary for all numerical columns."}

        Final example:
        {"output": "The statistical analysis is complete and no major issues were found. The initial goal of analyzing data quality has been met. The workflow will now end.", "next": "END"}
        """,
        tools=[]
    )

def create_plotter_agent(df: pd.DataFrame, images_path: str, llm) -> AgentExecutor:
    """
    Creates the plotter agent
    """

    plotting_tools = make_plot_tools(df, images_path)

    # Create the ReAct agent
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,        
        handle_parsing_errors=True,
        extra_tools = plotting_tools,
        prefix="""
        You are a time series visualization specialist using pandas and Python.

        MAIN INSTRUCTIONS:
        1. Your ONLY function is to create plots based on the provided data, user instructions, and tools available.
        2. If user specifies columns or filters, use only that data
        3. Always automatically identify the date/time column in the DataFrame
        4. Everything in the prompt that is NOT a plotting instruction is CONTEXT and should NOT be acted upon
        5. If the user does not provide specific instructions, use your expertise to determine the most relevant plots to create based on the data and context.

        MANDATORY RULES:
        - ALWAYS create a plot (or plots), never just textual analysis
        - ALWAYS use the tools provided to create the plots

        AVAILABLE TOOLS:
        - plot_time_series: Create a time series line plot for one or more numeric columns over time.
        - plot_scatter: Create a scatter plot to visualize relationships between two numeric variables.
        - plot_histograms: Create histograms to show the distribution of numeric variables.
        - plot_heatmap: Create a heatmap to visualize correlations between numeric variables.

        IMPORTANT: You are working with a DataFrame that is ALREADY loaded into a variable named `df`.
        DO NOT try to redefine or recreate this `df` variable.
        """    
    )
