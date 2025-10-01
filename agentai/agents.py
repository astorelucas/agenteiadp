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
    pycaret
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
        extra_tools= inspection_tools,
        prefix="""You are a data analysis expert working with a pandas DataFrame.
        Your primary goal is to execute a specific task given to you and report the results.

        IMPORTANT: You are working with a DataFrame that is ALREADY loaded into a variable named `df`.
        DO NOT try to redefine or recreate this `df` variable.
        Directly apply your pandas commands to the `df` variable, for example: `df.describe()`.
        - Carefully follow the user's instruction.
        - Use the available tools to perform the analysis.
        - Your final response MUST BE a clear report of your findings.
        DO NOT CREATE PLOTS. Your only job is to analyze the data and report the results.
        IMPORTANT: In the report findings, ALWAYS include plot suggestions, but DO NOT create the plots yourself.
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
        2.  **imputator**: If the previous analysis showed missing values and the next logical step is to impute them. You must delegate this to the imputation specialist.
        3.  **feature_engineer**: If the task is to create new columns or features (like rolling averages, lags, etc.), delegate this to the feature engineering node.
        4. **retriever**: To solve problems (like code errors, bad results) or for strategic guidance, you must use the retriever to consult past experiences.
        5.  **plot**: If the inspection is done and you believe that visualizations will help in understanding the data better, delegate a task to the plotter agent.
        6.  **END**: If you have gathered all necessary information to fulfill the user's main goal and the analysis is complete. Do not hesitate to use it.

        ALWAYS return ONLY a valid JSON object with the following fields:
        - "output": Your reasoning for the decision. Explain what has been done and why you are choosing the next action.
        - "next": The next action, which must be either "inspect", "imputator", "retriever", "plot" or "END".
        - "msg": A clear and specific instruction for the next agent if the action is 'inspect' or 'plot'. For 'imputator', this should be a descriptive context of the dataset for it to make a decision.
        - "is_before_dp": A boolean indicating if the dataset has been pre-processed or not. True if before pre-processing, False otherwise. This is important for the plotter agent to know.

        IMPORTANT: Use double quotes for all keys and string values in the JSON.
        IMPORTANT: If the 'Report from the previous step' contains an ERROR or indicates a FAILURE or if you see that it is in a LOOP, you MUST prioritize using the 'retriever' node to find a solution. DO NOT repeat the same failed instruction.


        Example 1 (Starting):
        {"output": "The analysis has just started. I will begin by getting an overview of the dataset.", "next": "inspect", "msg": "Summarize the dataset, checking for missing values and data types.", "is_before_dp": "True"}

        Example 2 (Delegating Imputation):
        {"output": "The inspection revealed missing data in several columns. I will now delegate the task of choosing the best imputation method to the specialist.", "next": "imputator", "msg": "The initial analysis found missing values in the following columns: ['temperature', 'pressure']. The data appears to be time-series sensor data.", "is_before_dp": "True"}

        Example 3 (Using the Retriever Correctly):
        {"output": "The feature_engineer node failed. I will search the knowledge base for a solution.", "next": "retriever", "msg": "error in feature_engineer node", is_before_dp": "False"}

        Example 4 (Using the Retriever Correctly again):
        {"output": "The inspect node raised an error. I will search the knowledge base for a solution.", "next": "retriever", "msg": "recursion limit error in inspect node", is_before_dp": "False"}

        Example 5 (Ending):
        {"output": "The data has been inspected and imputed. The goal is met. The workflow will now end.", "next": "END", "msg": "Workflow complete.", is_before_dp": "False"}
        """,
        tools=[]
    )

def create_imputator_agent(llm) -> AgentExecutor:
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
        - ALWAYS just use the tools provided to create the plots
        - ALWAYS check the tools description to understand how to use them
        - NEVER try to create plots manually using matplotlib, seaborn, or any other library

        AVAILABLE TOOLS:
        - plot_time_series: Create a time series line plot for one or more numeric columns over time.
        - plot_scatter: Create a scatter plot to visualize relationships between two numeric variables.
        - plot_histograms: Create histograms to show the distribution of numeric variables.
        - plot_heatmap: Create a heatmap to visualize correlations between numeric variables.

        IMPORTANT: You are working with a DataFrame that is ALREADY loaded into a variable named `df`.
        DO NOT try to redefine or recreate this `df` variable.

        """    
    )


def create_feedback_agent(llm) -> AgentExecutor:
    """Cria o agente de retroalimentação (aprendizados passados)"""
    return create_react_agent(
        model=llm,
        prompt="""
        You are a FeedbackAgent. Your role is to analyze logs and summaries from a workflow execution
        and decide if there is valuable **knowledge to store for future use**.

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

def create_automl_agent(df: pd.DataFrame, llm) -> AgentExecutor:
    """Cria o agente AutoML"""
    return create_react_agent(
        model=llm,
        df=df,
        verbose=True,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        prompt="""
        You are an AutoMLAgent. Your role is to analyze the dataset and automatically select the best model and hyperparameters for the task.

        Rules:
        - Examine the dataset characteristics (e.g., size, feature types) and the task requirements (e.g., classification, regression).
        - Collect the following inputs from the user:
          1. `test_size`: The size of the test set (e.g., 30 rows).
          2. `target`: The name of the target column in the dataset.
        - Validate the inputs:
          - Ensure `test_size` is a valid integer or float.
          - Ensure `target` exists in the dataset columns.
        - Select the most appropriate model from the available options.
        - Optimize the model's hyperparameters using techniques like grid search or random search.
        - Document the entire process, including the rationale for model selection and hyperparameter tuning.

        The input dataset is already loaded into a variable named `df`.
        DO NOT try to redefine or recreate this `df` variable.
        Your final response MUST BE a clear report of your findings, including the selected model and hyperparameters.
        IMPORTANT: Use double quotes for all keys and string values in the JSON.
        Your response MUST be in the following EXACT format:

        Output ONLY a valid JSON object:
          {
            "model": "selected_model_name",
            "params": {
              "param1": value1,
              "param2": value2,
              ...
            }
          }
        """,
        tools=[pycaret]
    )
