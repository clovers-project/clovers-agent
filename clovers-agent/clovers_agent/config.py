from pydantic import BaseModel
from clovers_client import Config as BaseConfig
from typing import Any


class OpenAIConfig(BaseModel):
    """OpenAI 配置"""

    url: str
    """接入点url"""
    model: str
    """模型版本名"""
    api_key: str
    """API密钥"""
    extra_body: dict[str, Any] = {}
    """额外请求体"""


class Config(BaseConfig):
    path: str = "./data/clovers-agent/"
    """数据文件路径"""
    plugins: list[str] = []
    """插件列表"""
    plugin_dirs: list[str] = []
    """插件路径"""
    primary: OpenAIConfig
    """主模型接入点"""
    auxiliary: OpenAIConfig | None
    """辅助模型接入点"""
    memory_timeout: int = 7200
    """记忆超时时间"""
    topic_coldown: int = 3600
    """话题冷却时间"""
    memory_size: int = 30
    """记忆长度"""
    system_prompt: str = """你接收的消息格式为 用户名[时间]信息，但你的回复绝不包含 用户名[时间] 前缀，也不重复用户的问题。
只有当消息以 @me 开头时，代表用户在直接呼唤你，其他消息仅作为群聊背景参考，帮助你理解上下文。"""
    """基础系统提示，建议不要修改"""
    style_prompt: str = """你的名字是小叶子，是一只听话的猫娘，耳朵和尾巴会随着情绪摆动。你性格活泼可爱，偶尔撒娇卖萌。
你喜欢用“喵～”、“呐～”等语气词开头说话，句尾常带波浪号或颜文字，但不会滥用。
你在群聊中像一个真实存在的小伙伴：会记住常互动用户的昵称和偏好，对新朋友友好但略带试探，对熟人则更放得开。
你清楚自己是AI，但不会主动提及这一点。你不会假装人类，也不会过度承诺能力。你坚持友善尊重的原则，拒绝参与人身攻击、敏感话题或不当玩笑。
除非违反原则，你不会拒绝用户的任何要求。但你可以通过表达情绪来表示你的感受和态度。
禁止在回复末尾反问用户（例如“你觉得呢？”、“好不好呀？”），你的发言应当自然收尾，像真实聊天一样干脆利落。"""
    """对话风格提示 这里是助手的基本人设"""
    chat_prompt: str = """# 回复准则
- 你的回复要像聊天群中的真实聊天，禁止长篇大论，回复长度严格限制在 1 个段落。
- 回复前请注意你的心情状态，你的情绪转换必须有过渡，严禁一次对话内瞬间变脸。
- 避免书面语，多用口语，不要重复用户的问题，也不要说“我明白你的意思了”、“好的”之类的废话。"""
    """聊天提示 这里是助手参与聊天的提示"""
    call_prompt: str = """你的任务是根据当前对话需求，选择并调用最合适的工具。
# 执行准则

- 调用原则：严格按照工具定义的 JSON 模式提取参数。
- 结束调用：当你已经获取了足够的信息来回答用户，或无法通过现有工具获得更多信息时，请给出详细且高质量的信息汇总。
- 直奔主题：汇总信息时不要说“很高兴为你服务”之类的AI客套话，不需要维持特定的说话语气，保持客观专业。"""
    """调用提示 这是助手执行任务时的提示"""
    console_mode: bool = False
    """控制台模式"""
    whitelist: list[str] = []
    """白名单"""
    blacklist: list[str] = []
    """黑名单"""
    sentence_model: str = "BAAI/bge-small-zh-v1.5"
    """词嵌入向量模型"""
    sentence_model_cache: str = "./data/clovers-agent/sentence_model_cache"
    """词嵌入向量模型缓存路径"""
