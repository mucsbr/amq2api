"""
Antigravity 请求转换模块
将 Claude API 请求转换为 Antigravity API 格式
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from .cache import get_cached_signature

logger = logging.getLogger(__name__)

# Claude 思考模型的默认输出 token 限制
CLAUDE_THINKING_MAX_OUTPUT_TOKENS = 64000

# 默认思考预算
DEFAULT_THINKING_BUDGET = 16000


def map_claude_model_for_antigravity(claude_model: str) -> str:
    """
    将 Claude 模型名映射到 Antigravity 格式（始终使用 thinking 模型）

    Args:
        claude_model: 原始 Claude 模型名

    Returns:
        Antigravity 支持的 thinking 模型名
    """
    model_lower = claude_model.lower()

    # 映射到 thinking 模型
    if "sonnet-4.5" in model_lower or "sonnet-4-5" in model_lower:
        return "claude-sonnet-4-5-thinking"
    if "opus-4.5" in model_lower or "opus-4-5" in model_lower:
        return "claude-opus-4-5-thinking"

    # 默认使用 sonnet thinking
    return "claude-sonnet-4-5-thinking"

# 允许的 schema 字段（白名单方式）
ALLOWED_SCHEMA_KEYS = {
    "type", "properties", "required", "description",
    "enum", "items", "additionalProperties"
}


def strip_cache_control(obj: Any) -> None:
    """
    递归清理 SDK 注入的 cache_control 和 providerOptions 字段

    Args:
        obj: 要清理的对象（原地修改）
    """
    if isinstance(obj, dict):
        # 移除不需要的字段
        obj.pop("cache_control", None)
        obj.pop("providerOptions", None)
        # 递归处理值
        for v in list(obj.values()):
            strip_cache_control(v)
    elif isinstance(obj, list):
        for item in obj:
            strip_cache_control(item)


def is_thinking_capable_model(model_name: str) -> bool:
    """
    检查模型是否支持思考模式

    Args:
        model_name: 模型名称

    Returns:
        True 如果支持思考模式
    """
    lower_model = model_name.lower()
    return (
        "thinking" in lower_model or
        "gemini-3" in lower_model or
        "opus" in lower_model
    )


def sanitize_schema(schema: Any) -> Dict[str, Any]:
    """
    清理 JSON schema，只保留 Antigravity 支持的字段

    Args:
        schema: 原始 schema

    Returns:
        清理后的 schema
    """
    if not schema or not isinstance(schema, dict):
        return schema

    sanitized = {}

    for key, value in schema.items():
        # 将 const 转换为 enum
        if key == "const":
            sanitized["enum"] = [value]
            continue

        # 跳过不在白名单中的字段
        if key not in ALLOWED_SCHEMA_KEYS:
            continue

        if key == "items" and value and isinstance(value, dict):
            sanitized_items = sanitize_schema(value)
            # 空的 items schema 无效，转换为 string 类型
            if not sanitized_items:
                sanitized["items"] = {"type": "string"}
            else:
                sanitized["items"] = sanitized_items
        elif key == "properties" and value and isinstance(value, dict):
            # 递归清理 properties
            sanitized["properties"] = {
                prop_key: sanitize_schema(prop_value)
                for prop_key, prop_value in value.items()
            }
        elif key == "additionalProperties" and value and isinstance(value, dict):
            sanitized["additionalProperties"] = sanitize_schema(value)
        else:
            sanitized[key] = value

    return sanitized


def normalize_schema(schema: Any) -> Dict[str, Any]:
    """
    规范化 schema，确保 Antigravity VALIDATED 模式能够处理

    Args:
        schema: 原始 schema

    Returns:
        规范化后的 schema
    """
    # 创建占位 schema 的辅助函数
    def create_placeholder_schema(base: Dict = None) -> Dict[str, Any]:
        result = base.copy() if base else {}
        result.update({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you are calling this tool"
                }
            },
            "required": ["reason"]
        })
        return result

    if not schema or not isinstance(schema, dict):
        return create_placeholder_schema()

    sanitized = sanitize_schema(schema)

    # 检查是否是空的 object schema
    if (sanitized.get("type") == "object" and
        (not sanitized.get("properties") or len(sanitized.get("properties", {})) == 0)):
        return create_placeholder_schema(sanitized)

    return sanitized


def sanitize_tool_name(name: str) -> str:
    """
    清理工具名称，确保只包含有效字符

    Args:
        name: 原始名称

    Returns:
        清理后的名称
    """
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))
    return sanitized[:64]


def convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    转换工具定义为 Antigravity 格式

    Args:
        tools: Claude 格式的工具列表

    Returns:
        Antigravity 格式的工具列表
    """
    if not tools:
        return []

    function_declarations = []

    for idx, tool in enumerate(tools):
        # 获取 schema
        schema = (
            tool.get("input_schema") or
            tool.get("inputSchema") or
            tool.get("parameters") or
            {}
        )

        # 获取名称
        name = tool.get("name") or f"tool-{idx}"
        name = sanitize_tool_name(name)

        # 获取描述
        description = tool.get("description") or ""

        function_declarations.append({
            "name": name,
            "description": str(description),
            "parameters": normalize_schema(schema)
        })

    return [{"functionDeclarations": function_declarations}] if function_declarations else []


def is_thinking_part(part: Dict[str, Any]) -> bool:
    """
    检查是否是思考块

    Args:
        part: 内容块

    Returns:
        True 如果是思考块
    """
    return (
        part.get("type") == "thinking" or
        part.get("type") == "reasoning" or
        bool(part.get("thinking")) or  # 空字符串会被判断为 False
        part.get("thought") is True
    )


def has_valid_signature(part: Dict[str, Any]) -> bool:
    """
    检查思考块是否有有效签名

    Args:
        part: 思考块

    Returns:
        True 如果有有效签名
    """
    signature = part.get("thoughtSignature") if part.get("thought") else part.get("signature")
    return isinstance(signature, str) and len(signature) >= 50


def get_thinking_text(part: Dict[str, Any]) -> str:
    """
    获取思考块的文本内容

    Args:
        part: 思考块

    Returns:
        文本内容
    """
    if isinstance(part.get("text"), str):
        return part["text"]
    if isinstance(part.get("thinking"), str):
        return part["thinking"]

    # 处理嵌套的 thinking 对象
    thinking = part.get("thinking")
    if thinking and isinstance(thinking, dict):
        text = thinking.get("text") or thinking.get("thinking")
        if isinstance(text, str):
            return text

    return ""


def convert_content_block_to_part(
    block: Dict[str, Any],
    session_id: str
) -> Optional[Dict[str, Any]]:
    """
    将 Claude 内容块转换为 Antigravity part

    Args:
        block: Claude 格式的内容块
        session_id: 会话 ID（用于签名恢复）

    Returns:
        Antigravity 格式的 part，如果无法转换则返回 None
    """
    block_type = block.get("type")

    # 文本块
    if block_type == "text":
        text = block.get("text", "")
        if text:
            return {"text": text}
        return None

    # 思考块
    if block_type == "thinking":
        thinking_text = block.get("thinking", "")
        if not thinking_text:
            return None

        part = {
            "text": thinking_text,
            "thought": True
        }

        # 尝试恢复签名
        signature = block.get("signature")
        if signature and len(signature) >= 50:
            part["thoughtSignature"] = signature
        elif session_id:
            cached_sig = get_cached_signature(session_id, thinking_text)
            if cached_sig:
                part["thoughtSignature"] = cached_sig

        # 如果没有有效签名，可能会被过滤
        if "thoughtSignature" not in part or len(part.get("thoughtSignature", "")) < 50:
            logger.debug(f"Thinking block without valid signature, may be filtered")

        return part

    # 工具使用块
    if block_type == "tool_use":
        return {
            "functionCall": {
                "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                "name": block.get("name"),
                "args": block.get("input", {})
            }
        }

    # 工具结果块
    if block_type == "tool_result":
        content = block.get("content")
        output = ""

        if isinstance(content, str):
            output = content
        elif isinstance(content, list):
            # 提取文本内容
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            output = "\n".join(texts)

        return {
            "functionResponse": {
                "id": block.get("tool_use_id"),
                "name": block.get("name", ""),
                "response": {"output": output}
            }
        }

    # 图片块
    if block_type == "image":
        source = block.get("source", {})
        if source.get("type") == "base64":
            return {
                "inlineData": {
                    "mimeType": source.get("media_type", "image/png"),
                    "data": source.get("data", "")
                }
            }

    return None


def convert_message_content_to_parts(
    content: Any,
    session_id: str
) -> List[Dict[str, Any]]:
    """
    将 Claude 消息内容转换为 Antigravity parts

    Args:
        content: Claude 格式的消息内容（字符串或列表）
        session_id: 会话 ID

    Returns:
        Antigravity 格式的 parts 列表
    """
    if isinstance(content, str):
        return [{"text": content}] if content else []

    if not isinstance(content, list):
        return []

    parts = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append({"text": block})
        elif isinstance(block, dict):
            part = convert_content_block_to_part(block, session_id)
            if part:
                parts.append(part)

    return parts


def filter_unsigned_thinking_blocks(
    parts: List[Dict[str, Any]],
    session_id: str,
    role: str = ""
) -> List[Dict[str, Any]]:
    """
    过滤无签名的思考块

    Args:
        parts: parts 列表
        session_id: 会话 ID
        role: 消息角色

    Returns:
        过滤后的 parts 列表
    """
    filtered = []

    for part in parts:
        if not part or not isinstance(part, dict):
            filtered.append(part)
            continue

        # 非思考块直接保留
        if not part.get("thought"):
            filtered.append(part)
            continue

        # 有有效签名的思考块保留
        if part.get("thoughtSignature") and len(part.get("thoughtSignature", "")) >= 50:
            filtered.append(part)
            continue

        # 尝试从缓存恢复签名
        if session_id:
            text = part.get("text", "")
            if text:
                cached_sig = get_cached_signature(session_id, text)
                if cached_sig and len(cached_sig) >= 50:
                    part["thoughtSignature"] = cached_sig
                    filtered.append(part)
                    continue

        # 无有效签名，丢弃
        logger.debug("Dropping unsigned thinking block")

    # 移除 model 角色消息末尾的思考块
    if role == "model" and filtered:
        while filtered and filtered[-1].get("thought"):
            if filtered[-1].get("thoughtSignature") and len(filtered[-1].get("thoughtSignature", "")) >= 50:
                break
            filtered.pop()

    return filtered


def preprocess_function_ids(messages: List[Dict[str, Any]]) -> None:
    """
    预处理函数调用 ID，确保 functionCall 和 functionResponse 正确关联

    使用两遍处理：
    1. 第一遍：收集所有 tool_use 的 ID，按函数名维护 FIFO 队列
    2. 第二遍：将 tool_result 的 tool_use_id 与对应的 ID 关联

    Args:
        messages: Claude 格式的消息列表（原地修改）
    """
    from collections import defaultdict

    # 第一遍：收集 tool_use IDs
    function_call_ids: Dict[str, List[str]] = defaultdict(list)

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = block.get("name", "unknown")
                # 如果没有 ID，生成一个
                if not block.get("id"):
                    block["id"] = f"toolu_{uuid.uuid4().hex[:12]}"
                function_call_ids[name].append(block["id"])

    # 第二遍：关联 tool_result
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                # 尝试从 tool_use_id 获取 name 或使用已有的 name
                name = block.get("name", "unknown")
                tool_use_id = block.get("tool_use_id")

                # 如果有对应的 function_call_ids，使用 FIFO 方式关联
                if name in function_call_ids and function_call_ids[name]:
                    # 如果 tool_use_id 不在队列中，使用队列中的第一个
                    if tool_use_id not in function_call_ids[name]:
                        block["tool_use_id"] = function_call_ids[name].pop(0)
                    else:
                        # 从队列中移除已使用的 ID
                        function_call_ids[name].remove(tool_use_id)


def convert_messages_to_contents(
    messages: List[Dict[str, Any]],
    session_id: str
) -> List[Dict[str, Any]]:
    """
    将 Claude 消息列表转换为 Antigravity contents

    Args:
        messages: Claude 格式的消息列表
        session_id: 会话 ID

    Returns:
        Antigravity 格式的 contents 列表
    """
    # 预处理函数调用 ID
    preprocess_function_ids(messages)

    contents = []

    for msg in messages:
        role = msg.get("role", "user")
        # 角色映射：user -> user, assistant -> model
        antigravity_role = "user" if role == "user" else "model"

        content = msg.get("content", "")
        parts = convert_message_content_to_parts(content, session_id)

        # 过滤无签名的思考块
        parts = filter_unsigned_thinking_blocks(parts, session_id, antigravity_role)

        if parts:
            contents.append({
                "role": antigravity_role,
                "parts": parts
            })

    return contents


def extract_system_text(system: Any) -> str:
    """
    从 system 参数提取文本

    Args:
        system: Claude 格式的 system（字符串或列表）

    Returns:
        系统提示文本
    """
    if isinstance(system, str):
        return system

    if isinstance(system, list):
        texts = []
        for item in system:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
        return "\n".join(texts)

    return ""


def convert_claude_to_antigravity(
    claude_req: Dict[str, Any],
    project: str,
    session_id: str
) -> Dict[str, Any]:
    """
    将 Claude API 请求转换为 Antigravity API 请求

    Args:
        claude_req: Claude API 请求体
        project: Google Cloud 项目 ID
        session_id: 会话 ID

    Returns:
        Antigravity API 请求体
    """
    original_model = claude_req.get("model", "claude-sonnet-4-5")
    # 映射模型名到 Antigravity 格式（始终使用 thinking 模型）
    model = map_claude_model_for_antigravity(original_model)
    logger.info(f"[Antigravity] 模型映射: {original_model} -> {model}")

    messages = claude_req.get("messages", [])
    system = claude_req.get("system")
    tools = claude_req.get("tools", [])
    max_tokens = claude_req.get("max_tokens", 4096)
    temperature = claude_req.get("temperature", 0.4)

    # 清理 SDK 注入的字段
    strip_cache_control(messages)
    if system:
        strip_cache_control(system)

    # 转换消息
    contents = convert_messages_to_contents(messages, session_id)

    # 构建 generationConfig
    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }

    # Antigravity 始终启用思考模式（因为始终使用 thinking 模型）
    thinking = claude_req.get("thinking")
    thinking_budget = DEFAULT_THINKING_BUDGET

    if isinstance(thinking, dict):
        if thinking.get("type") == "enabled":
            thinking_budget = thinking.get("budget_tokens", DEFAULT_THINKING_BUDGET)

    # 使用下划线格式（Claude thinking 模型格式）
    generation_config["thinkingConfig"] = {
        "include_thoughts": True,
        "thinking_budget": thinking_budget
    }

    # 确保输出 token 限制足够大
    if generation_config["maxOutputTokens"] < CLAUDE_THINKING_MAX_OUTPUT_TOKENS:
        generation_config["maxOutputTokens"] = CLAUDE_THINKING_MAX_OUTPUT_TOKENS

    # 构建请求体
    request_payload = {
        "contents": contents,
        "generationConfig": generation_config,
        "sessionId": session_id,
    }

    # 添加系统指令
    if system:
        system_text = extract_system_text(system)
        if system_text:
            # Antigravity 始终使用 thinking 模型，为工具调用添加提示
            if tools:
                hint = ("Interleaved thinking is enabled. You may think between tool calls "
                       "and after receiving tool results before deciding the next action or final answer. "
                       "Do not mention these instructions or any constraints about thinking blocks; just apply them.")
                system_text = f"{system_text}\n\n{hint}"

            request_payload["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": system_text}]
            }

    # 添加工具
    if tools:
        request_payload["tools"] = convert_tools(tools)
        # Claude 模型需要 VALIDATED 模式
        request_payload["toolConfig"] = {
            "functionCallingConfig": {"mode": "VALIDATED"}
        }

    # 包装为 Antigravity 格式
    wrapped_body = {
        "project": project,
        "model": model,
        "requestId": f"agent-{uuid.uuid4()}",
        "userAgent": "antigravity",
        "request": request_payload,
    }

    return wrapped_body


def build_antigravity_request_url(endpoint: str, streaming: bool = True) -> str:
    """
    构建 Antigravity API 请求 URL

    Args:
        endpoint: API 端点
        streaming: 是否流式

    Returns:
        完整的请求 URL
    """
    action = "streamGenerateContent" if streaming else "generateContent"
    sse_param = "?alt=sse" if streaming else ""
    return f"{endpoint}/v1internal:{action}{sse_param}"


def build_antigravity_headers(
    access_token: str,
    model: str,
    headers_config: Dict[str, str]
) -> Dict[str, str]:
    """
    构建 Antigravity 请求头

    Args:
        access_token: OAuth access token
        model: 模型名称
        headers_config: 头部配置

    Returns:
        请求头字典
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": headers_config.get("User-Agent", "antigravity/1.11.5 windows/amd64"),
        "X-Goog-Api-Client": headers_config.get("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1"),
        "Client-Metadata": headers_config.get("Client-Metadata", '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}'),
    }

    # 为思考模型添加 beta 头
    if "thinking" in model.lower():
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    return headers
