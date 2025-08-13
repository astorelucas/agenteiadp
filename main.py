from dotenv import load_dotenv

import sys
import os
import numpy as np
import pandas as pd

from agentai.workflow import build_workflow

load_dotenv()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = build_workflow()

def execute_pipeline():
    print("Starting time series preprocessing pipeline...")

    csv_path = "agentai/datasets/test.csv"

    initial_state = {
        "msg": "Please summarize missing values and column types in this dataset.",
        "df": pd.DataFrame(),
        "path": csv_path,
        "logs": [],
        "next": ""
    }

    result = app.invoke(initial_state)

    print("Pipeline finished. Logs:")
    for m in result["logs"]:
        print(m)

if __name__ == "__main__":
    execute_pipeline()