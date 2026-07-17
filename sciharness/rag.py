"""
最小可用的 RAG 检索器：TF-IDF + 余弦相似度。
没有用向量数据库/embedding API，是刻意的——先用最简单、可解释、可离线跑的方案
把 pipeline 跑通，跑通之后再换成 embedding 检索是很直接的替换（接口不用变）。

Context Compression 的思路体现在 search() 里的 max_chars 截断：
只把最相关的片段喂给 Agent，而不是整篇文档，控制 token 消耗。
"""

from __future__ import annotations
import glob
import os
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRetriever:
    def __init__(self, kb_dir: str, top_k: int = 3, max_chars: int = 500):
        self.kb_dir = kb_dir
        self.top_k = top_k
        self.max_chars = max_chars
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self._chunks: List[Tuple[str, str]] = []  # (doc_id, chunk_text)
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._build_index()

    def _chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        # 按段落切，段落太长再按字符数硬切，保证每个 chunk 不会太大
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for p in paras:
            if len(p) <= chunk_size:
                chunks.append(p)
            else:
                for i in range(0, len(p), chunk_size):
                    chunks.append(p[i:i + chunk_size])
        return chunks

    def _build_index(self):
        for filepath in sorted(glob.glob(os.path.join(self.kb_dir, "*.txt"))):
            doc_id = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in self._chunk_text(text):
                self._chunks.append((doc_id, chunk))

        if not self._chunks:
            return

        corpus = [c[1] for c in self._chunks]
        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str) -> List[Tuple[str, str, float]]:
        """返回 [(doc_id, chunk_text, score), ...]，按相似度降序"""
        if self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(
            zip(self._chunks, sims), key=lambda x: x[1], reverse=True
        )[: self.top_k]
        results = []
        for (doc_id, chunk), score in ranked:
            if score <= 0:
                continue
            results.append((doc_id, chunk[: self.max_chars], float(score)))
        return results
