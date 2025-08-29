from dotenv import load_dotenv
import sys
import os
from uuid import uuid4

# A importação principal agora é a classe que criamos
from agentai.workflow import WorkflowExecutor

load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def execute_pipeline():
    print("*** Iniciando o pipeline ***\n\n")

    csv_path = "agentai/datasets/test.csv"

    # now the dataset is loadee *ONE* time here
    try:
        executor = WorkflowExecutor(csv_path=csv_path)
    except ValueError as e:
        print(f"Erro: {e}")
        return

    try:
        png_bytes = executor.graph.get_graph().draw_mermaid_png()
        with open("workflow_mermaid_graph.png", "wb") as f:
            f.write(png_bytes)
        print("grafo  salvo como 'workflow_mermaid_graph.png'")
    except Exception as e:
        print(f"não foi possível gerar a imagem do grafo: {e}")


    # unique id
    thread_id = str(uuid4())

    
    # initial prompt
    initial_prompt = """
    First, create a feature for the 3-hour rolling average of the 'temperature' column.
    Second, create another feature for the 3-hour rolling standard deviation of the 'temperature' column.
    Finally, provide a summary of the updated DataFrame, showing the first few rows to confirm that both new columns ('temperature_rolling_avg_3h' and 'temperature_rolling_std_3h') have been created correctly.
    """
    #"Perform a complete exploratory data analysis on the quality of this dataset. Start with a general overview, then delve into the most important points you deem necessary, such as missing values, descriptive statistics, and potential outliers. Provide a final summary upon completion."
    
    print("\n--- INICIANDO EXECUÇÃO DO GRAFO ---")
    executor.invoke(initial_message=initial_prompt, thread_id=thread_id)
    print("--- FIM DA EXECUÇÃO DO GRAFO ---\n")


if __name__ == "__main__":
    execute_pipeline()
