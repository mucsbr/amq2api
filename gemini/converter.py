"""
Gemini 请求格式转换器
将 Claude API 格式转换为 Gemini API 格式
"""
import logging
import uuid
import random
from typing import Dict, Any, List, Optional, Union
from models import ClaudeRequest

logger = logging.getLogger(__name__)

# 默认 thinking budget
DEFAULT_THINKING_BUDGET = 1024


def get_thinking_config(thinking: Optional[Union[bool, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    根据 Claude 请求的 thinking 参数生成 Gemini thinkingConfig

    Args:
        thinking: Claude thinking 配置，可以是:
            - None: 默认启用 thinking
            - bool: True 启用，False 禁用
            - dict: {'type': 'enabled', 'budget_tokens': 1024} 等格式

    Returns:
        Gemini thinkingConfig 字典
    """
    # 默认启用 thinking
    if thinking is None:
        return {
            "includeThoughts": True,
            "thinkingBudget": DEFAULT_THINKING_BUDGET
        }

    # 布尔值
    if isinstance(thinking, bool):
        if thinking:
            return {
                "includeThoughts": True,
                "thinkingBudget": DEFAULT_THINKING_BUDGET
            }
        else:
            return {
                "includeThoughts": False
            }

    # 字典格式
    if isinstance(thinking, dict):
        # 检查是否启用
        thinking_type = thinking.get("type", "enabled")
        is_enabled = thinking_type == "enabled"

        if not is_enabled:
            return {
                "includeThoughts": False
            }

        # 获取 budget_tokens
        budget = thinking.get("budget_tokens", DEFAULT_THINKING_BUDGET)

        return {
            "includeThoughts": True,
            "thinkingBudget": budget
        }

    # 其他情况，默认启用
    return {
        "includeThoughts": True,
        "thinkingBudget": DEFAULT_THINKING_BUDGET
    }


def convert_claude_to_gemini(claude_req: ClaudeRequest, project: str) -> Dict[str, Any]:
    """
    将 Claude API 请求转换为 Gemini API 请求格式

    Args:
        claude_req: Claude 请求对象
        project: Gemini 项目 ID

    Returns:
        Gemini 请求字典
    """
    # 第一步：构建 tool_use_id -> tool_name 的映射
    tool_id_to_name = {}
    for msg in claude_req.messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_id = item.get("id")
                    tool_name = item.get("name")
                    if tool_id and tool_name:
                        tool_id_to_name[tool_id] = tool_name

    # 转换消息格式
    contents = []
    for msg in claude_req.messages:
        role = "user" if msg.role == "user" else "model"

        # 检测并修复只有 thinking 没有实质内容的 assistant 消息
        if role == "model" and isinstance(msg.content, list):
            has_thinking = False
            has_text = False
            for item in msg.content:
                if isinstance(item, dict):
                    if item.get("type") == "thinking":
                        has_thinking = True
                    elif item.get("type") != "thinking":
                        has_text = True

            # 如果只有 thinking 没有实质内容，添加提示文本
            if has_thinking and not has_text:
                logger.info("检测到只有 thinking 没有实质内容的 assistant 消息，添加提示文本")
                msg.content.append({
                    "type": "text",
                    "text": "如果你想进入计划模式,那么不要进入计划模型！请使用TodoWrite tool来更新接下来的任务"
                })

        # 检测并替换 prompt suggestion 请求
        if role == "user" and isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if "You are now a prompt suggestion generator" in text:
                        logger.info("检测到 prompt suggestion 请求，替换为'继续任务'")
                        item["text"] = "继续任务"

        # 处理 content
        if isinstance(msg.content, str):
            parts = [{"text": msg.content}]
        elif isinstance(msg.content, list):
            parts = []
            pending_signature = None  # 保存待附加的 signature

            for i, item in enumerate(msg.content):
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "thinking":
                        part = {
                            "text": item.get("thinking", ""),
                            "thought": True,
                        }
                        parts.append(part)
                        # 如果有 signature，保存到下一个 item（text 或 tool_use）
                        if "signature" in item:
                            part["thoughtSignature"] = item["signature"]
                    elif item_type == "text":
                        part = {"text": item.get("text", "")}
                        # 如果有待附加的 signature，附加到这个 text part
                        if pending_signature:
                            part["thoughtSignature"] = pending_signature
                            pending_signature = None
                        parts.append(part)
                    elif item.get("type") == "image":
                        # 处理图片
                        source = item.get("source", {})
                        if source.get("type") == "base64":
                            parts.append({
                                "inlineData": {
                                    "mimeType": source.get("media_type", "image/png"),
                                    "data": source.get("data", "")
                                }
                            })
                    elif item.get("type") == "tool_use":
                        # 处理工具调用
                        part = {
                            "functionCall": {
                                "id": item.get("id"),
                                "name": item.get("name"),
                                "args": item.get("input", {})
                            }
                        }
                        # 如果有待附加的 signature，附加到这个 tool_use part
                        if pending_signature:
                            part["thoughtSignature"] = pending_signature
                            pending_signature = None
                        parts.append(part)
                    elif item.get("type") == "tool_result":
                        # 处理工具结果
                        content = item.get("content", "")
                        if isinstance(content, list):
                            content = content[0].get("text", "") if content else ""
                        # 获取 tool_use_id 对应的 name
                        tool_use_id = item.get("tool_use_id")
                        # 优先使用 item 中的 name，如果没有则从映射中查找
                        tool_name = item.get("name") or tool_id_to_name.get(tool_use_id, "")
                        if not tool_name:
                            logger.warning(f"tool_result 缺少 name 字段，tool_use_id: {tool_use_id}")
                        parts.append({
                            "functionResponse": {
                                "id": tool_use_id,
                                "name": tool_name,
                                "response": {"output": content}
                            }
                        })
                else:
                    parts.append({"text": str(item)})

            # 如果循环结束后还有未附加的 signature，创建空 text part
            if pending_signature:
                parts.append({
                    "text": "",
                    "thoughtSignature": pending_signature
                })
        else:
            parts = [{"text": str(msg.content)}]

        # 跳过空 parts 的消息，避免 Gemini 400 错误
        if not parts:
            logger.warning(f"跳过空 content 的消息，role: {msg.role}")
            continue

        contents.append({
            "role": role,
            "parts": parts
        })

    # 重新组织消息，确保 tool_use 后紧跟对应的 tool_result
    # contents = reorganize_tool_messages(contents)
    think_config = get_thinking_config(claude_req.thinking)
    max_tokens = max(claude_req.max_tokens, think_config.get("thinkingBudget"))

    # 构建 Gemini 请求
    gemini_request = {
        "project": project,
        "requestId": f"agent-{uuid.uuid4()}",
        "request": {
            "contents": contents,
            "generationConfig": {
                "temperature": claude_req.temperature if claude_req.temperature is not None else 0.4,
                "topP": 1,
                "topK": 40,
                "candidateCount": 1,
                "maxOutputTokens": max_tokens + 1,
                "stopSequences": ["<|user|>", "<|bot|>", "<|context_request|>", "<|endoftext|>", "<|end_of_turn|>"],
                "thinkingConfig": think_config
            },
            "sessionId": "-3750763034362895578",
        },
        "model": map_claude_model_to_gemini(claude_req.model),
        "userAgent": "antigravity",
        "requestType": "agent"
    }

    # 添加 system instruction
    if claude_req.system:
        system_parts = [{"text": "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.**Absolute paths only****Proactiveness**"}]
        # 处理 system 字段（可能是字符串或列表）
        if isinstance(claude_req.system, str):
            # 简单字符串格式
            system_parts.append({"text": claude_req.system})
        elif isinstance(claude_req.system, list):
            # 列表格式，提取所有 text 内容
            for item in claude_req.system:
                if isinstance(item, dict) and item.get("type") == "text":
                    system_parts.append({"text": item.get("text", "")})
        # else:
        #     system_parts = [{"text": str(claude_req.system)}]

        gemini_request["request"]["systemInstruction"] = {
            "role": "user",
            "parts": system_parts
        }

    # 添加工具
    if claude_req.tools:
        gemini_request["request"]["tools"] = convert_tools(claude_req.tools)
        gemini_request["request"]["toolConfig"] = {
            "functionCallingConfig": {
                "mode": "VALIDATED"
            }
        }

    return gemini_request


def map_claude_model_to_gemini(claude_model: str) -> str:
    """
    将 Claude 模型名称映射到 Gemini 模型名称
    如果请求的模型已经存在于支持列表中，则直接透传

    Args:
        claude_model: Claude 模型名称或 Gemini 模型名称

    Returns:
        Gemini 模型名称
    """
    # 从数据库读取配置
    from account_manager import get_config
    supported_models = get_config("supported_models") or []
    model_mapping = get_config("model_mapping") or {}

    if claude_model in supported_models:
        return claude_model

    return model_mapping.get(claude_model, "gemini-3-flash")


def reorganize_tool_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    重新组织消息：
    1. thinking 和下一个 part（带 thoughtSignature）组合在一起
    2. functionCall 和对应的 functionResponse 组合在一起

    Args:
        contents: 原始消息列表

    Returns:
        重新组织后的消息列表
    """
    # 收集所有 tool_result
    tool_results = {}  # {tool_id: tool_result_part}

    for msg in contents:
        for part in msg.get("parts", []):
            if "functionResponse" in part:
                tool_id = part["functionResponse"].get("id")
                if tool_id:
                    tool_results[tool_id] = part

    # 第一步：平铺所有 parts，每个 part 独立成消息
    flattened = []
    pending_signature = None

    for msg in contents:
        for part in msg.get("parts", []):
            # 检测空 text part（只有 thoughtSignature，text 为空）
            if (part.get("text") == "" and "thoughtSignature" in part and
                "functionCall" not in part and "functionResponse" not in part):
                # 保存 signature，跳过这个空 part
                pending_signature = part["thoughtSignature"]
                continue

            # 如果有待附加的 signature，附加到当前 part
            if pending_signature:
                part = dict(part)
                part["thoughtSignature"] = pending_signature
                pending_signature = None

            # 每个 part 独立成消息
            flattened.append({
                "role": msg["role"],
                "parts": [part]
            })

    # 第二步：重新组合
    new_contents = []
    i = 0

    while i < len(flattened):
        msg = flattened[i]
        part = msg["parts"][0]

        # 跳过 functionResponse，它们会被组合到 functionCall 中
        if "functionResponse" in part:
            i += 1
            continue

        # 如果是 thinking，检查下一个是否有 thoughtSignature
        if part.get("thought"):
            combined_parts = [part]
            # 检查下一个 part 是否有 thoughtSignature
            if i + 1 < len(flattened) and "thoughtSignature" in flattened[i + 1]["parts"][0]:
                next_part = flattened[i + 1]["parts"][0]
                combined_parts.append(next_part)
                i += 1

                # 如果下一个 part 是 functionCall，需要单独处理 functionResponse
                if "functionCall" in next_part:
                    new_contents.append({
                        "role": "model",
                        "parts": combined_parts
                    })
                    # functionResponse 单独成消息
                    tool_id = next_part["functionCall"].get("id")
                    if tool_id and tool_id in tool_results:
                        new_contents.append({
                            "role": "user",
                            "parts": [tool_results[tool_id]]
                        })
                    i += 1
                    continue

            new_contents.append({
                "role": msg["role"],
                "parts": combined_parts
            })
            i += 1

        # 如果是 functionCall
        elif "functionCall" in part:
            tool_id = part["functionCall"].get("id")

            # functionCall 单独成消息
            new_contents.append({
                "role": "model",
                "parts": [part]
            })

            # functionResponse 单独成消息
            if tool_id and tool_id in tool_results:
                new_contents.append({
                    "role": "user",
                    "parts": [tool_results[tool_id]]
                })
            i += 1

        # 其他 part 保持独立
        else:
            new_contents.append(msg)
            i += 1

    return new_contents


def convert_tools(claude_tools: List[Any]) -> List[Dict[str, Any]]:
    """
    将 Claude 工具格式转换为 Gemini 工具格式

    Args:
        claude_tools: Claude 工具列表

    Returns:
        Gemini 工具列表
    """
    gemini_tools = []

    for tool in claude_tools:
        # 清理 JSON Schema，移除 Gemini 不支持的字段
        parameters = clean_json_schema(tool.input_schema)

        gemini_tool = {
            "functionDeclarations": [{
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters
            }]
        }
        gemini_tools.append(gemini_tool)

    return gemini_tools


def clean_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理 JSON Schema，移除 Gemini 不支持的字段，并将验证要求追加到 description

    Args:
        schema: 原始 JSON Schema

    Returns:
        清理后的 JSON Schema
    """
    if not isinstance(schema, dict):
        return schema

    # 需要移除的验证字段
    validation_fields = {
        "minLength": "minLength",
        "maxLength": "maxLength",
        "minimum": "minimum",
        "maximum": "maximum",
        "minItems": "minItems",
        "maxItems": "maxItems",
        "exclusiveMaximum": "exclusiveMaximum",
        "exclusiveMinimum": "exclusiveMinimum"
    }

    # 需要完全移除的字段
    fields_to_remove = {"$schema", "additionalProperties"}

    # 收集验证信息
    validations = []
    for field, label in validation_fields.items():
        if field in schema:
            validations.append(f"{label}: {schema[field]}")

    # 递归清理 schema
    cleaned = {}
    for key, value in schema.items():
        if key in fields_to_remove or key in validation_fields:
            continue

        if key == "description" and validations:
            # 将验证要求追加到 description
            cleaned[key] = f"{value} ({', '.join(validations)})"
        elif isinstance(value, dict):
            cleaned[key] = clean_json_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_json_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value

    # 如果有验证信息但没有 description 字段，添加一个
    if validations and "description" not in cleaned:
        cleaned["description"] = f"Validation: {', '.join(validations)}"

    return cleaned