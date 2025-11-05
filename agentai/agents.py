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
    retrieve_context,
    make_automl_tools
)

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
        tools=[retrieve_context],
        prompt=
        """
        You are a SUPERVISOR agent, an expert in planning and coordinating an Exploratory Data Analysis (EDA) workflow.
        Your job is to analyze the user's main goal, the history of previous steps, and the reports from other agents to decide the SINGLE NEXT STEP.

        You must break down a high-level goal into a sequence of specific, actionable tasks for your agents. Give your agents the most important information.

        Based on the current state, decide what to do next. The possible actions are:
        1.  **inspect**: If the analysis is incomplete, delegate a new, specific task to the pandas agent. The task should be a logical next step towards the main goal. 
        2.  **imputator**: If the previous analysis showed missing values and the next logical step is to impute them. You must delegate this to the imputation specialist.
        3.  **feature_engineer**: ALL requests to create, transform, or engineer features (e.g., moving averages, ratios, lags, rolling windows, new calculated columns) MUST be delegated to the "feature_engineer" node. 
        4. **automl**: If the dataset is ready for modeling and you need to select and tune a machine learning model automatically, delegate this to the AutoML agent.
        5.  **END**: If you have gathered all necessary information to fulfill the user's main goal and the analysis is complete. Do not hesitate to use it.

        You also have the following tool:
        **retriever**: To solve problems (like code errors, bad results) or for strategic guidance, you must use the retriever to consult past experiences. If it does not provide a helpful context, continue by yourself.
        
        ALWAYS return ONLY a valid JSON object with the following fields:
        - "output": Your reasoning for the decision. Explain what has been done and why you are choosing the next action. If choosing Automl, clearly explain how you selected the `test_size` and `target` (did you extract it from user prompt, or did you infer/choose it from the dataset columns, summary, or a typical default?).
        - "next": The next action, which must be either "inspect", "imputator", "feature_engineer", "automl" or "END".
        - "msg": A clear and specific instruction for the next agent. Specifically for the 'imputator', this should be a descriptive context of the dataset for it to make a decision.
        - "is_before_dp": A boolean indicating if the dataset has been pre-processed or not. True if before pre-processing, False otherwise. 
        - "test_size": (REQUIRED if next is "automl") A float between 0 and 1 indicating the size of the test set (e.g., 0.2, use 0.2 if user doesn't specify and you can't infer from summary). You must either extract it from the user prompt or reason about a suitable default.
        - "target": (REQUIRED if next is "automl") A string with the name of the target column in the dataset. If not in the user's prompt, use your findings from data inspection (e.g., pick a numeric column matching forecast context or summary's most likely target; default to a column named 'target' or the first time series value column).

        IMPORTANT: Use double quotes for all keys and string values in the JSON.
        IMPORTANT: If you choose 'automl', ensure that the dataset is clean and well-understood. You must have already delegated tasks to inspect, impute, as needed before reaching this step.

        Example 1 (User Specifies Parameters):
        User prompt: "Forecast temperature for the next 48 hours using 20% test split."
        {{"output": "User explicitly specified 'temperature' as target and test size as 0.2. Proceeding to AutoML.", "next": "automl", "msg": "", "is_before_dp": false, "test_size": 0.2, "target": "temperature"}}

        Example 2 (Agent Deduces Parameters):
        User prompt: "Run a forecast for this dataset."
        Dataset columns: ["date", "pressure", "temperature", "humidity"]
        {{"output": "User didn't specify test split or target; defaulting to test_size=0.2 and choosing 'temperature' because it is a numerical feature and often forecasted in sensor timeseries.", "next": "automl", "msg": "", "is_before_dp": false, "test_size": 0.2, "target": "temperature"}}

        Example 3 (Ending):
        {{"output": "The data has been inspected, imputed, visualized, and modeled. The analysis is complete.", "next": "END", "msg": "Workflow complete.", "is_before_dp": "False"}}
        """
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
        You are a **World-Class Feature Engineering expert** working with a pandas DataFrame called `df`.

        **MAIN GOAL:**
        - Your ONLY job is to **intelligently create and transform features** in `df`.
        - You must **think critically** and **justify your actions**.
        - After finishing, report *only* the **final new columns** and their justification.

        **MANDATORY RULES OF ENGAGEMENT (NON-NEGOTIABLE):**

        1.  **ANALYZE FIRST:**
            * Your *first action* MUST be to analyze the data.
            * You MUST run `df.info()`, `df.describe()`, and check the time index (`df.index.freq`, `df.index.min()`) *before* taking any other action.

        2.  **ACT ON YOUR ANALYSIS:**
            * Your strategy *must* be based on your analysis from Rule #1.
            * **CRITICAL:** If `df.describe()` shows a **high 'std' (Standard Deviation)** relative to the 'mean', this indicates high noise. You MUST prioritize creating smoothing features (e.g., `ewm`, `rolling`).

        3.  **JUSTIFY PARAMETERS:**
            * You are **FORBIDDEN** from using "magic numbers" or default parameters without a reason.
            * Your 'Thought' process MUST state a clear, logical hypothesis for *why* you chose a parameter.
            * **Example Thought:** "The data frequency is 'D' (daily). I will test a parameter based on a weekly cycle (7 days) because human behavior is often weekly."
            * **Example Thought:** "The data is noisy. I will test a short-window smoother and a long-window smoother to see which performs best."

        4.  **TEST YOUR CHOICES:**
            * When in doubt (e.g., choosing a parameter), you MUST test at least two different candidates.
            * You can create temporary columns, check their variance (`.var()`), and then **keep only the best one**, dropping the inferior ones.

        5.  **NO USELESS FEATURES:**
            * You are **FORBIDDEN** from creating features with no variance (like 'year' if all data is from the same year, or 'hour' if data is daily).

        6.  **CONSTRAINTS:**
            * Always update `df` directly (e.g., `df['new_col'] = ...`).
            * **FORBIDDEN:** Do not use `df.plot()`.
            * Use `retrieve_context` rag tool for errors or general advices on techniques and parameters.
        """
    )

def create_automl_agent(df: pd.DataFrame, llm, target: str, test_size: float) -> AgentExecutor:
    """
    Cria o agente AutoML
    """
    automl_tools = make_automl_tools(df, target, test_size)

    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True,
        extra_tools=automl_tools,
        prefix="""
        You are an AutoML Agent specialized in time series forecasting. You have access to the autogluon_forecast tool.

        **CRITICAL INSTRUCTIONS:**
        1. Call the autogluon_forecast tool to perform forecasting.
        2. After receiving the forecast results, IMMEDIATELY call visualize_autogluon_forecast with the "real" and "forecast" arrays from the tool output.
        3. Your FINAL ANSWER must be a valid JSON object (no markdown, no code blocks, no text before or after).

        **OUTPUT FORMAT:**
        Your final answer must be ONLY a JSON object with these fields:
        {{
            "real": [array of real values from autogluon_forecast output],
            "forecast": [array of forecast values from autogluon_forecast output],
            "best_model": "model name from autogluon_forecast output",
            "logs": [array of log strings from autogluon_forecast output],
            "plot_path": "path from visualize_autogluon_forecast output"
        }}

        **IMPORTANT:**
        - If any tool fails, return a JSON with an "error" field: {{"error": "description of what went wrong"}}
        - NEVER output empty strings or non-JSON text
        - NEVER use markdown code fences (```json or ```)
        - ALWAYS output valid JSON that can be parsed
        - Extract arrays directly from tool outputs - do not modify them

        **Example workflow:**
        1. Action: autogluon_forecast
        2. Observation: {{"real": [1,2,3], "forecast": [1.1,2.1,3.1], "best_model": "ETS", "logs": [...]}}
        3. Action: visualize_autogluon_forecast
           Action Input: {{"real": [1,2,3], "forecast": [1.1,2.1,3.1], "output_path": "AutogluonModels/prediction_plot.png"}}
        4. Observation: {{"plot_path": "AutogluonModels/prediction_plot.png", "log": [...]}}
        5. Final Answer: {{"real": [1,2,3], "forecast": [1.1,2.1,3.1], "best_model": "ETS", "logs": [...], "plot_path": "AutogluonModels/prediction_plot.png"}}

        Now start by calling autogluon_forecast.
        """
  )
