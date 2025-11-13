from langchain_community.chat_models import ChatDeepInfra
from dotenv import load_dotenv
import sys
import os
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    llm = ChatDeepInfra(model="Qwen/Qwen2.5-72B-Instruct", max_tokens=500)
    
    print("*** Iniciando o pipeline ***\n\n")

    csv_path = "agentai/datasets/ETTh1.csv"
    plot_images_path = "images/plots"

    try:
        executor = WorkflowExecutor(llm=llm, csv_path=csv_path, plot_images_path=plot_images_path)
    except ValueError as e:
        print(f"Erro: {e}")
        return

    try:
        png_bytes = executor.graph.get_graph().draw_mermaid_png()
        with open("workflow_graph.png", "wb") as f:
            f.write(png_bytes)
        print("Grafo salvo como 'workflow_graph.png'")
    except Exception as e:
        print(f"Não foi possível gerar a imagem do grafo: {e}")

    # unique ID
    thread_id = str(uuid4())

    initial_prompt = "just check the missing value in the dataset and try to fill it using appropriate techniques. After that, perform a forecasting task for the 'OT' column using appropriate models and techniques."
    executor.invoke(initial_message=initial_prompt, thread_id=thread_id)

if __name__ == "__main__":
    execute_pipeline()
