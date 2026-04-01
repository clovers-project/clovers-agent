import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import transformers

transformers.logging.set_verbosity_error()


def sentence_weight(text: str):
    cn = jp = kr = 0
    others: list[str] = []
    for char in text:
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF:
            cn += 1
        elif 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
            jp += 1
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            kr += 1
        else:
            others.append(char)
    return (cn / 2) + (jp / 4) + (kr / 5) + len("".join(others).split())


class TopicDecoupler:
    def __init__(self, model: SentenceTransformer, alpha: float = 0.8):
        """
        话题解藕器，用于判断输入的句子是否属于同一话题。
        Args:
            model (SentenceTransformer): SentenceTransformer 模型。
            alpha (float): 指数衰减权重。
        """
        emb_dim = model.get_sentence_embedding_dimension()
        if emb_dim is None:
            raise ValueError("SentenceTransformer model does not have a fixed sentence embedding dimension.")
        self.model = model
        self.context_emb = torch.zeros(emb_dim, device=self.model.device)
        self.scores_history: list[float] = []
        self.weights_history: list[float] = []
        self.alpha = alpha
        self.topic_change = self._topic_change_0

    def _topic_change_0(self, sentence_emb: torch.Tensor, weight: float):
        self.weights_history.append(weight)
        self.topic_change = self._topic_change_1
        return False

    def _topic_change_1(self, sentence_emb: torch.Tensor, weight: float):
        self.weights_history.append(weight)
        self.scores_history.append(util.cos_sim(sentence_emb, self.context_emb).item())
        self.topic_change = self._topic_change_2
        return False

    def _topic_change_2(self, sentence_emb: torch.Tensor, weight: float):
        score = util.cos_sim(sentence_emb, self.context_emb).item()
        scores_history = np.array(self.scores_history)
        threshold = scores_history.mean() - max(scores_history.std(), 0.01)
        if flag := (score < threshold):
            self.weights_history.clear()
            self.scores_history.clear()
            self.topic_change = self._topic_change_1
        elif scores_history.size > 3:
            self.topic_change = self._topic_change_3
        self.weights_history.append(weight)
        self.scores_history.append(score)
        return flag

    def _topic_change_3(self, sentence_emb: torch.Tensor, weight: float):
        score = util.cos_sim(sentence_emb, self.context_emb).item()
        scores_history = np.array(self.scores_history)
        weights_history = np.array(self.weights_history)
        Q1, Q2, Q3 = np.quantile(scores_history, [0.25, 0.5, 0.75])
        scale = (weight - weights_history.mean()) * scores_history.std() / max(weights_history.std(), 0.01)
        threshold = Q1 + Q2 - Q3 + scale
        if flag := (score < threshold):
            self.weights_history.clear()
            self.scores_history.clear()
            self.topic_change = self._topic_change_1
        self.weights_history.append(weight)
        self.scores_history.append(score)
        return flag

    @torch.no_grad()
    def step(self, sentence: str):
        if not sentence:
            return False
        sentence_emb = self.model.encode(sentence, convert_to_tensor=True)
        weight = sentence_weight(sentence)
        if self.topic_change(sentence_emb, weight):
            self.context_emb.mul_(0.5).add_(sentence_emb, alpha=0.5)
            return True
        else:
            # self.context_emb.add_((sentence_emb - self.context_emb) / self.__step_count)
            self.context_emb.mul_(self.alpha).add_(sentence_emb, alpha=1 - self.alpha)
            return False
