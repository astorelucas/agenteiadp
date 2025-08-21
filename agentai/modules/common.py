from typing import TypedDict, Optional
    
# improvements: 
# 1. removed the whole dataset from here    
# 2. removed csv_path from here
# 3. separated ephemeral from persistent states (using Optional)

class AgentState(TypedDict):
    # persistent states
    logs: list[str]
    main_goal: str

    # ephemeral states
    msg: Optional[str]
    subagents_report: Optional[str]
    next: Optional[str]