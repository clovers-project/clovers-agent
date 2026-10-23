WORKSPACE = "workspace"
KEY_SUBAGENTS = f"{WORKSPACE}.subagent_wakeups"
SUBAGENT_PROMPT = """
你是子代理
1. 只完成用户描述的任务
2. 强烈建议不要拉起新的子代理
"""
