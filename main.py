from langchain_community.chat_models import ChatDeepInfra
from dotenv import load_dotenv
import sys
import os
from uuid import uuid4

# A importação principal agora é a classe que criamos
from agentai.workflow import WorkflowExecutor

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def load_env_variables():
    load_dotenv()
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY não encontrada.")
    os.environ["DEEPINFRA_API_KEY"] = api_key

def execute_pipeline():
    load_env_variables()
    llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct")
    
    print("*** Iniciando o pipeline ***\n\n")

    csv_path = "agentai/datasets/test.csv"
    plot_images_path = "agentai/images/plots"

    # now the dataset is loadee *ONE* time here
    try:
        executor = WorkflowExecutor(csv_path=csv_path, plot_images_path=plot_images_path, llm=llm)
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
    initial_prompt = "Perform a complete exploratory data analysis on the quality of this dataset. Start with a general overview, then delve into the most important points you deem necessary, such as missing values, descriptive statistics, and potential outliers. Provide a final summary upon completion."
    
    print("\n--- INICIANDO EXECUÇÃO DO GRAFO ---")
    executor.invoke(initial_message=initial_prompt, thread_id=thread_id)
    print("--- FIM DA EXECUÇÃO DO GRAFO ---\n")


if __name__ == "__main__":
    execute_pipeline()
