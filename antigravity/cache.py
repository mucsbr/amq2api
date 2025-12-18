"""
思考签名缓存模块
用于多轮对话中恢复思考块的签名，避免 "invalid signature" 错误
"""

import hashlib
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class SignatureCache:
    """
    思考签名 LRU 缓存

    用于缓存 Antigravity API 返回的思考块签名，
    在多轮对话中恢复历史思考块的签名。
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        初始化签名缓存

        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间（秒）
        """
        self.cache: OrderedDict[str, Tuple[str, datetime]] = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)

    def _hash_key(self, session_id: str, text: str) -> str:
        """
        生成缓存键

        Args:
            session_id: 会话 ID
            text: 思考文本内容

        Returns:
            16 字符的哈希键
        """
        content = f"{session_id}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def set(self, session_id: str, text: str, signature: str) -> None:
        """
        缓存签名

        Args:
            session_id: 会话 ID
            text: 思考文本内容
            signature: 签名字符串
        """
        if not signature or len(signature) < 50:
            # 无效签名不缓存
            return

        key = self._hash_key(session_id, text)
        self.cache[key] = (signature, datetime.now())
        self.cache.move_to_end(key)

        # 清理超出大小的条目
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        logger.debug(f"缓存签名: session={session_id[:8]}..., key={key}")

    def get(self, session_id: str, text: str) -> Optional[str]:
        """
        获取缓存的签名

        Args:
            session_id: 会话 ID
            text: 思考文本内容

        Returns:
            签名字符串，如果不存在或已过期则返回 None
        """
        key = self._hash_key(session_id, text)
        entry = self.cache.get(key)

        if not entry:
            return None

        signature, timestamp = entry

        # 检查是否过期
        if datetime.now() - timestamp > self.ttl:
            del self.cache[key]
            logger.debug(f"签名已过期: key={key}")
            return None

        # 更新 LRU 顺序
        self.cache.move_to_end(key)
        logger.debug(f"命中签名缓存: session={session_id[:8]}..., key={key}")
        return signature

    def clear(self, session_id: Optional[str] = None) -> int:
        """
        清理缓存

        Args:
            session_id: 如果指定，只清理该会话的缓存；否则清理所有

        Returns:
            清理的条目数
        """
        if session_id is None:
            count = len(self.cache)
            self.cache.clear()
            return count

        # 清理特定会话的缓存（需要遍历所有键）
        # 注意：这是一个 O(n) 操作，但通常不会频繁调用
        keys_to_remove = []
        for key in self.cache:
            # 由于我们的 key 是 hash，无法直接判断 session_id
            # 这里简化处理，不支持按 session 清理
            pass

        return 0

    def cleanup_expired(self) -> int:
        """
        清理所有过期条目

        Returns:
            清理的条目数
        """
        now = datetime.now()
        keys_to_remove = []

        for key, (_, timestamp) in self.cache.items():
            if now - timestamp > self.ttl:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

        if keys_to_remove:
            logger.debug(f"清理过期签名: {len(keys_to_remove)} 条")

        return len(keys_to_remove)

    def stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息

        Returns:
            包含 size 和 max_size 的字典
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
        }


# 全局签名缓存实例
_signature_cache = SignatureCache()


def cache_signature(session_id: str, text: str, signature: str) -> None:
    """
    缓存思考签名（全局函数）

    Args:
        session_id: 会话 ID
        text: 思考文本内容
        signature: 签名字符串
    """
    _signature_cache.set(session_id, text, signature)


def get_cached_signature(session_id: str, text: str) -> Optional[str]:
    """
    获取缓存的思考签名（全局函数）

    Args:
        session_id: 会话 ID
        text: 思考文本内容

    Returns:
        签名字符串，如果不存在则返回 None
    """
    return _signature_cache.get(session_id, text)


def clear_signature_cache(session_id: Optional[str] = None) -> int:
    """
    清理签名缓存（全局函数）

    Args:
        session_id: 如果指定，只清理该会话的缓存

    Returns:
        清理的条目数
    """
    return _signature_cache.clear(session_id)


def get_cache_stats() -> Dict[str, int]:
    """
    获取缓存统计信息（全局函数）

    Returns:
        缓存统计字典
    """
    return _signature_cache.stats()
