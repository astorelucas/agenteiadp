import operator
from typing import TypedDict, Annotated, Sequence, Dict, List

import numpy as np
import pandas as pd
import json
    
class AgentState(TypedDict):
    msg: str
    df: pd.DataFrame
    path: str          # CSV path or URL
    logs: list[str]
    next: str
