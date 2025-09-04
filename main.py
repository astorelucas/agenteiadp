from dotenv import load_dotenv
import sys
import os
from uuid import uuid4

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
        print("Grafo salvo como 'workflow_mermaid_graph.png'")
    except Exception as e:
        print(f"Não foi possível gerar a imagem do grafo: {e}")

    # unique ID
    thread_id = str(uuid4())

    initial_prompt = "Perform a complete exploratory data analysis on the quality of this dataset. Start with a general overview, then delve into the most important points you deem necessary, such as missing values, descriptive statistics, and potential outliers. Also create features."

    print("\n--- INICIANDO EXECUÇÃO DO GRAFO ---")
    executor.invoke(initial_message=initial_prompt, thread_id=thread_id)
    print("--- FIM DA EXECUÇÃO DO GRAFO ---\n")
    # print("\n--- DataFrame Final Após a Execução ---")
    # print(executor.df)

if __name__ == "__main__":
    execute_pipeline()
