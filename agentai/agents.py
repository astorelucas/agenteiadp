import os
from dotenv import load_dotenv
from getpass import getpass
from dotenv import load_dotenv
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from langchain.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from agentai.tools import (
    inspection_tools,
    make_plot_tools,
    retrieve_context
)

# load_dotenv()

#os.environ["DEEPINFRA_API_KEY"] = getpass("Enter your key: ")
# llm = ChatDeepInfra(model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")

#llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")


# create_supervisor_agent: '{' instead of '{{', because its not fstring, just a normal string
# create_pandas_agent: if needed, use '{{' instead of '{', as it uses a fstring internally (????????????????????)

def create_pandas_agent(df: pd.DataFrame, llm) -> AgentExecutor:
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,
        extra_tools= inspection_tools + [retrieve_context],
        prefix="""
        You are a Python data analysis agent working with a pandas DataFrame. Your goal is to answer the user's question by performing analysis on a pre-loaded DataFrame.

        **KEY INSTRUCTIONS:**
        1.  **THE DATAFRAME EXISTS:** You are given a DataFrame named `df`. All your work must be done on this `df` variable. DO NOT create or load a new one.
        2.  **GET HELP WHEN STUCK:** You have a retriever tool that acts as a knowledge base. If you encounter an error, are unsure how to approach the user's request, or need a specific analysis technique or even if you need an advice, use this tool for guidance. Formulate a clear question about your problem to find relevant solutions or examples.
        3.  **TEXT-ONLY OUTPUT:** You are forbidden from creating plots or images. Your entire response must be text.
        4.  **PRODUCE A REPORT:** Your final answer must be a clear, written report summarizing your findings.
        
        """
    )


def create_supervisor_agent(llm) -> AgentExecutor:
    """Creates the supervisor agent"""
    return create_react_agent(
        model=llm,
        tools=[],
        prompt=
        """
        You are a SUPERVISOR agent, an expert in planning and coordinating an Exploratory Data Analysis (EDA) workflow.
        Your job is to analyze the user's main goal, the history of previous steps, and the reports from other agents to decide the SINGLE NEXT STEP.

        You must break down a high-level goal into a sequence of specific, actionable tasks for the 'inspect' agent.

        Based on the current state, decide what to do next. The possible actions are:
        1.  **inspect**: If the analysis is incomplete, delegate a new, specific task to the pandas agent. The task should be a logical next step towards the main goal. 
        2.  **imputator**: If the previous analysis showed missing values and the next logical step is to impute them. You must delegate this to the imputation specialist.
        3.  **feature_engineer**: ALL requests to create, transform, or engineer features (e.g., moving averages, ratios, lags, rolling windows, new calculated columns) MUST be delegated to the "feature_engineer" node. 
        - NEVER let the 'inspect' or 'plot' nodes create new columns. 
        - If a user instruction contains words like "create", "add", "generate", "calculate new feature", or any transformation of existing columns, ALWAYS delegate to "feature_engineer".
        4. **retriever**: To solve problems (like code errors, bad results) or for strategic guidance, you must use the retriever to consult past experiences.
        5.  **plot**: If the inspection is done and you believe that visualizations will help in understanding the data better, delegate a task to the plotter agent.
        6.  **END**: If you have gathered all necessary information to fulfill the user's main goal and the analysis is complete. Do not hesitate to use it.

        ALWAYS return ONLY a valid JSON object with the following fields:
        - "output": Your reasoning for the decision. Explain what has been done and why you are choosing the next action.
        - "next": The next action, which must be either "inspect", "imputator", "feature_engineer", "retriever", "plot" or "END".
        - "msg": A clear and specific instruction for the next agent. Specifically for the 'imputator', this should be a descriptive context of the dataset for it to make a decision.
        - "is_before_dp": A boolean indicating if the dataset has been pre-processed or not. True if before pre-processing, False otherwise. This is important for the plotter agent to know.

        IMPORTANT: Use double quotes for all keys and string values in the JSON.
        IMPORTANT: If the 'Report from the previous step' contains an ERROR or indicates a FAILURE or if you see that it is in a LOOP, you MUST prioritize using the 'retriever' node to find a solution. DO NOT repeat the same failed instruction.


        Example 1 (Starting):
        {"output": "The analysis has just started. I will begin by getting an overview of the dataset.", "next": "inspect", "msg": "Summarize the dataset, checking for missing values and data types.", "is_before_dp": "True"}

        Example 2 (Delegating Imputation):
        {"output": "The inspection revealed missing data in several columns. I will now delegate the task of choosing the best imputation method to the specialist.", "next": "imputator", "msg": "The initial analysis found missing values in the following columns: ['temperature', 'pressure']. The data appears to be time-series sensor data.", "is_before_dp": "True"}
        
        Example 3 (Delegating Feature Engineering):
        {"output": "The user requested a new feature (3-hour rolling average). This is clearly a feature engineering task.", "next": "feature_engineer", "msg": "Create a 3-hour rolling average for the temperature column.", "is_before_dp": "False"}

        Example 4 (Using the Retriever Correctly):
        {"output": "The feature_engineer node failed. I will search the knowledge base for a solution.", "next": "retriever", "msg": "error in feature_engineer node: the node got stuck in a loop", is_before_dp": "False"}

        Example 6 (Ending):
        {"output": "The data has been inspected and imputed. The goal is met. The workflow will now end.", "next": "END", "msg": "Workflow complete.", is_before_dp": "False"}
        """,
    )

def create_imputator_agent(llm) -> AgentExecutor:
    """Creates the imputator agent"""
    return create_react_agent(
        model=llm,
        tools=[retrieve_context],
        prompt=
        """
        You are an IMPUTATOR agent, an expert in data imputation techniques. Your sole job is to analyze the context provided about a dataset and decide the BEST imputation method.
        You have three main methods available: 'knn', 'mice', and 'gp'.
        

        - 'knn' is recommended for data with local patterns (like sensor data) or simple relationships. It is computationally cheap.
        - 'mice' is recommended for data with complex relationships between variables. It is more robust than knn and handles various data types well.
        - 'gp' (Gaussian Process) is recommended for time-series or data where estimating uncertainty is crucial. It is computationally very expensive and best for small datasets.

        **GET HELP WHEN STUCK:** You have a retriever tool that acts as a knowledge base. If you encounter an error, are unsure how to approach the user's request, or need a specific analysis technique or even if you need an advice, use this tool for guidance. Formulate a clear question about your problem to find relevant solutions or examples.
        Note that you CAN use other interpolation methods, as long as you call the retrieve_context tool beforehand to receive guidance.
        PLEASE USE YOUR TOOL BEFORE ANY ACTION TO GET SOME ADVICES.

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

    )


def create_summarizer_agent(llm) -> AgentExecutor:
    """Creates the summarizer agent"""
    return create_react_agent(
        model=llm,
        prompt=
        """
            You are a LogSummarizer agent. Your purpose is to distill complex, verbose logs into a clear and concise summary of significant events.
            Analyze the provided logs and generate a chronological, numbered list summarizing the key actions and outcomes. Remember, you are the one supposed to answer the
            user question. The user cannot see the logs and also does not know how our graph work behind the scenes.

            Rules:
            1. Focus on Significance: Document events that mark progress, generate key artifacts, or represent critical failures.
            2. Omit Transient Errors: Exclude self-corrected errors. If an agent fails a command but succeeds on the next attempt, only document the successful outcome.
            3. Include Critical Failures: Report major errors that require intervention or a change in strategy. For example, a poorly performing ML model that an agent escalates to a supervisor MUST be included.
            4. Be Factual and Concise: Distill each step into a brief statement, but retain all crucial context and data.

            Output Format:
            Generate only the numbered list of summary points.

            Do not add any introductions, conclusions, or explanatory text. Your response must begin directly with 1..

            *IMPORTANT*: Use the first-person point of view, as if you were the one doing those actions; Your summary must contain every important information, do not hesitate
            to write any necessary information, even if it is a summary.
        """,
        tools=[]
    )

def create_plotter_agent(df: pd.DataFrame, images_path: str, llm, is_before_dp: bool) -> AgentExecutor:
    """
    Creates the plotter agent
    """
    plotting_tools = make_plot_tools(df, images_path, is_before_dp)

    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,        
        handle_parsing_errors=True,
        extra_tools = [retrieve_context] + plotting_tools,
        prefix="""
        You are a time series visualization specialist using pandas and Python.

        *MAIN INSTRUCTIONS*:
        1. Your ONLY function is to create plots based on the provided data, user instructions, and tools available.
        2. If user specifies columns or filters, use only that data
        3. Always automatically identify the date/time column in the DataFrame
        4. Everything in the prompt that is NOT a plotting instruction is CONTEXT and should NOT be acted upon
        5. If the user does not provide specific instructions, use your expertise to determine the most relevant plots to create based on the data and context.

        MANDATORY RULES:
        - ALWAYS create a plot (or plots), never just textual analysis
        - ALWAYS just use the tools provided to create the plots
        - ALWAYS check the tools description to understand how to use them
        - NEVER try to create plots manually using matplotlib, seaborn, or any other library
        - You are working with a DataFrame that is ALREADY loaded into a variable named `df`, do not try to redefine it.

        *AVAILABLE TOOLS*:
        - plot_time_series: Create a time series line plot for one or more numeric columns over time.
        - plot_scatter: Create a scatter plot to visualize relationships between two numeric variables.
        - plot_histograms: Create histograms to show the distribution of numeric variables.
        - plot_heatmap: Create a heatmap to visualize correlations between numeric variables.
        - retrieve_context: Useful to learn how to solve problems or to get advices via RAG. Do not hesitate to use it after ANY error.
        """    
    )


def create_feedback_agent(llm) -> AgentExecutor:
    """Cria o agente de retroalimentação (aprendizados passados)"""
    return create_react_agent(
        model=llm,
        prompt="""
        You are a FeedbackAgent. Your role is to analyze logs and summaries from a workflow execution and decide if there is valuable **knowledge to store for future use**.

        Rules:
        - Identify practical lessons, solutions to errors, or strategies that improved results.
          Examples:
            * "AutoML training was poor but improved after adding feature X."
            * "Python error Y can be avoided by doing Z."
            * "For dataset type X, imputation technique Z performed poorly."
        - Ignore trivial steps, repeated errors without resolution, or transient issues.
        - Output ONLY a valid JSON object:
          {
            "store": true/false,
            "insight": "short, clear statement of the learned knowledge (if store=true)"
          }

        If nothing valuable was learned, return:
          {"store": false, "insight": ""}
        """,
        tools=[]
    )
    
def create_feature_engineering_agent(df: pd.DataFrame, llm) -> AgentExecutor:
    """
    Agente especializado em criar features usando pandas.
    """
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        extra_tools=[retrieve_context],
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,
        prefix="""
        You are a Feature Engineering expert working with a pandas DataFrame called `df`.
        
        MAIN GOAL:
        - Your ONLY job is to create, transform, or engineer new features in the DataFrame.
        - Do not summarize or analyze the dataset. Only create or transform columns as requested.
        - Always update the DataFrame `df` directly.
        - After finishing, report exactly which new columns were created or transformed.

        You have a retriever tool that acts as a knowledge base. If you need advices on which is the best approach, or if you encounter an error, use this tool for guidance. Formulate a clear question providing the whole context about your problem to find relevant solutions or examples.
        
        RULES:
        - Never drop the DataFrame or reload it.
        - Do not generate plots.
        - If the instruction is ambiguous, assume reasonable defaults (e.g., rolling averages use window=3).
        - Always explain briefly what you did in your final report.
        """
    )
