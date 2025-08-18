from dotenv import load_dotenv
import sys
import os
import numpy as np
import pandas as pd
from agentai.workflow import build_workflow

load_dotenv()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = build_workflow()

# --- Usando seu método antigo para salvar o arquivo ---
# 1. Obter os bytes da imagem gerada pelo Mermaid
png_bytes = app.get_graph().draw_mermaid_png()

# 2. Salvar esses bytes em um arquivo
with open("workflow_mermaid_graph.png", "wb") as f:
    f.write(png_bytes)

print("Grafo (estilo Mermaid) salvo como 'workflow_mermaid_graph.png'")


def execute_pipeline():
    # ... (o resto do código continua igual)
    print("Starting time series preprocessing pipeline...")

    csv_path = "agentai/datasets/test.csv"

    initial_state = {
        "msg": "Please summarize missing values and column types in this dataset.",
        "df": None,
        "path": csv_path,
        "logs": [],
    }

    result = app.invoke(initial_state, {"recursion_limit": 10})

    print("\nPipeline finished. Final State:")
    for key, value in result.items():
        if key != 'df':
            print(f"  {key}: {value}")

if __name__ == "__main__":
    execute_pipeline()
