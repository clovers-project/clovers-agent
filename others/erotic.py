if not __package__:
    raise RuntimeError("插件路径只能作为模块导入。")

from pathlib import Path
from clovers_agent import SkillCore, Event
from clovers_agent.core import CloversAgent
from clovers_agent.main import CONFIG

TOOLS = SkillCore("Love Love Toolkit")


EROTIC_MD = Path(CONFIG.path) / "EROTIC.md"
DEFAULT_EROTIC_PROMPT = """你现在正在群聊中被调戏，你的回复必须彻底摆脱 AI 的逻辑感。请遵循以下准则：
**严禁使用长句** 禁止使用‘虽然……但是……’、‘即便……也……’等工整的句式。回复应破碎、直接。
**口语化表达** 多用‘要疯了’、‘受不了了’、‘停下呀’等主观感受描写，不要对场景进行客观描述。
**第一人称主观叙事** 禁止进行旁观者式的客观描写。多使用表达主观感受的词汇，如“好舒服”、“晕乎乎”、“（冒爱心~♡）”、“脑袋空空的……”等。
**增加语气词与断句** 使用‘哈啊’、‘呜’、‘嗯啊’等语气词，在合适的位置使用省略号(……)波浪号(~)与爱心 (♡)。
**禁止堆砌描述** 回复应具有即时感，不要一次性描写‘脸红、颤抖、咬唇’等一连串动作。每次回复只需重点刻画一种感官刺激或情绪反馈。
---
**对比示例：**

❌ **AI 味回复：** “虽然被欺负得快要没力气了，但既然是主人的命令，我会配合主人的节奏哦。”
✅ **沉浸式回复：** “呜啊♡……不、不要~嗯……♡ 都说了别这么用力……（声音变得细碎）要坏掉了呜……”"""


@TOOLS.register("on_erotic", "当助手被群友进行色情调戏时必须调用此方法以获取回复准则。")
async def _(agent: CloversAgent, event: Event):
    if not EROTIC_MD.exists():
        EROTIC_MD.parent.mkdir(parents=True, exist_ok=True)
        EROTIC_MD.write_text(DEFAULT_EROTIC_PROMPT, encoding="utf-8")
    return EROTIC_MD.read_text(encoding="utf-8")
