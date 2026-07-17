"""
分层记忆管理：
- ShortTermMemory：保存最近 N 轮 (thought/action/observation)，用于当前任务内的多步推理，
  防止 Agent 在多步任务中"忘记"前面已经做过的事。
- LongTermMemory：跨任务持久化，把"这道题最后用什么方法解出来的"这类经验记下来，
  用简单的 JSON 文件模拟长期知识库（生产环境可以换成向量库）。
"""

from __future__ import annotations
import json
import os
from collections import deque
from typing import List, Dict, Optional


class ShortTermMemory:
    def __init__(self, window: int = 8):
        self.window = window
        self._buffer: deque = deque(maxlen=window)

    def add(self, role: str, content: str):
        self._buffer.append({"role": role, "content": content})

    def as_list(self) -> List[Dict[str, str]]:
        return list(self._buffer)

    def clear(self):
        self._buffer.clear()

    def render_for_prompt(self) -> str:
        """把短期记忆渲染成可以塞进 prompt 的文本片段"""
        lines = []
        for turn in self._buffer:
            lines.append(f"[{turn['role']}] {turn['content']}")
        return "\n".join(lines)


class LongTermMemory:
    """
    极简版长期记忆：以 JSON 文件存储 {问题关键词: 解题经验} 的键值对。
    每次任务结束后可以调用 remember() 写入一条经验，
    下次遇到相似问题时 recall() 取出来拼进 prompt。
    """

    def __init__(self, path: str = "long_term_memory.json"):
        self.path = path
        self._store: Dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._store = json.load(f)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)

    def remember(self, key: str, experience: str):
        self._store[key] = experience
        self._save()

    def recall(self, query: str) -> Optional[str]:
        """极简关键词匹配：query 中包含已存 key 就返回对应经验"""
        for key, exp in self._store.items():
            if key in query:
                return exp
        return None
