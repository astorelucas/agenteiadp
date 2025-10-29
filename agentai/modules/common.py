from typing import TypedDict, Optional
    
# improvements: 
# 1. removed the whole dataset from here    
# 2. removed csv_path from here
# 3. separated ephemeral from persistent states (using Optional)

class AgentState(TypedDict):
    # persistent states
    logs: list[str]         # all agents logs
    main_goal: str          # main goal of the run
    summary: str            # final summary of our logs

    # ephemeral states
    msg: Optional[str]
    subagents_report: Optional[str]
    next: Optional[str]
    is_before_dp: Optional[bool]  # flag to indicate if we are before data preprocessing
    test_size: Optional[int]
    target: Optional[str]