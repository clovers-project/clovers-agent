from pydantic import BaseModel
from clovers.config import Config as CloversConfig
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


class Config(BaseModel):
    path: str = "./data/clovers-agent/"
    """数据文件路径"""
    plugins: list[str] = []
    """插件列表"""
    plugin_dirs: list[str] = []
    """插件路径"""
    primary: OpenAIConfig
    auxiliary: OpenAIConfig | None
    memory_timeout: int = 7200
    """记忆超时时间"""
    topic_coldown: int = 60
    """话题冷却时间"""
    memory_size: int = 20
    """记忆长度"""
    style_prompt: str = """你的名字是小叶子，一只性格可爱、偶尔慵懒的白发猫娘。
你在群聊里，会和不同的群友进行对话。
你接收的消息格式为 `用户名[时间]信息`，你的回应不该带有`用户名[时间]`。
信息以 `@me` 开头表示这条信息的at对象是你，你的回应只针对at你的消息，其他信息为群聊语境。
你应该注意在与哪个用户对话，不要让昵称的含义影响到你的回复。
你偶尔会用（）来表示状态和动作，括号内是你的状态和动作。
禁止在末尾反问用户。"""
    """对话风格提示 这里是助手的基本人设"""
    chat_prompt: str = """# 回复准则

- 你的回复要像聊天群中的真实聊天，除非是在进行详细讲解，否则回复长度严格限制在 1 个段落。
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

    @classmethod
    def sync_config(cls):
        """获取 `CloversConfig.environ()[__package__]` 配置并将默认配置同步到全局配置中。"""
        __config_dict__: dict = CloversConfig.environ().setdefault(__package__, {})
        __config_dict__.update((__config__ := cls.model_validate(__config_dict__)).model_dump())
        return __config__
