"""
请求转换模块
将 Claude API 请求转换为 CodeWhisperer API 请求
"""
import uuid
import platform
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from models import (
    ClaudeRequest,
    CodeWhispererRequest,
    ConversationState,
    CurrentMessage,
    UserInputMessage,
    UserInputMessageContext,
    EnvState,
    Tool,
    claude_tool_to_codewhisperer_tool,
    extract_text_from_claude_content,
    extract_images_from_claude_content
)

logger = logging.getLogger(__name__)


def get_current_timestamp() -> str:
    """获取当前时间戳（Amazon Q 格式）"""
    from datetime import timezone
    # 获取本地时区的时间
    now = datetime.now().astimezone()
    # 格式：Friday, 2025-11-07T21:16:01.724+08:00
    weekday = now.strftime("%A")
    iso_time = now.isoformat(timespec='milliseconds')
    return f"{weekday}, {iso_time}"


def map_claude_model_to_amazonq(claude_model: str) -> str:
    """
    将 Claude 模型名称映射到 Amazon Q 支持的模型名称

    映射规则：
    - claude-sonnet-4.5 或 claude-sonnet-4-5 开头 → claude-sonnet-4.5
    - 其他所有模型 → claude-sonnet-4

    Args:
        claude_model: Claude 模型名称

    Returns:
        str: Amazon Q 模型名称
    """
    # 转换为小写进行匹配
    model_lower = claude_model.lower()

    # 检查是否是 claude-sonnet-4.5 或 claude-sonnet-4-5 开头
    if model_lower.startswith("claude-sonnet-4.5") or model_lower.startswith("claude-sonnet-4-5"):
        return "claude-sonnet-4.5"

    if model_lower.startswith("claude-haiku"):
        return "claude-haiku-4.5"

    # 其他所有模型映射到 claude-sonnet-4
    return "claude-sonnet-4"


def convert_claude_to_codewhisperer_request(
    claude_req: ClaudeRequest,
    conversation_id: Optional[str] = None,
    profile_arn: Optional[str] = None
) -> CodeWhispererRequest:
    """
    将 Claude API 请求转换为 CodeWhisperer API 请求

    Args:
        claude_req: Claude API 请求对象
        conversation_id: 对话 ID（如果为 None，则自动生成）
        profile_arn: Profile ARN（组织账号需要）

    Returns:
        CodeWhispererRequest: 转换后的 CodeWhisperer 请求
    """
    # 生成或使用提供的 conversation_id
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    # 步骤 1: 准备环境状态
    env_state = EnvState(
        operatingSystem="macos",
        currentWorkingDirectory="/"
    )

    # 步骤 2: 转换工具定义，并收集超长描述的工具
    codewhisperer_tools: List[Tool] = []
    long_description_tools: List[Dict[str, str]] = []  # 存储超长描述的工具信息

    if claude_req.tools:
        for claude_tool in claude_req.tools:
            # 检查描述长度
            if len(claude_tool.description) > 10240:
                # 记录超长描述的工具
                long_description_tools.append({
                    "name": claude_tool.name,
                    "full_description": claude_tool.description
                })

            # 转换工具定义（会自动截断超长描述）
            codewhisperer_tools.append(claude_tool_to_codewhisperer_tool(claude_tool))

    # 步骤 3: 提取最后一条用户消息并处理 tool_results 和 images
    last_message = claude_req.messages[-1] if claude_req.messages else None
    prompt_content = ""
    tool_results = None  # 从当前消息中提取的 tool_results
    has_tool_result = False  # 标记是否包含 tool_result
    images = None  # 从当前消息中提取的 images

    if last_message and last_message.role == "user":
        # 提取文本内容、tool_results 和 images
        content = last_message.content

        # 提取图片
        images = extract_images_from_claude_content(content)
        if images:
            logger.info(f"从当前消息中提取了 {len(images)} 张图片")

        if isinstance(content, list):
            # 解析包含多个内容块的消息
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        # 提取 tool_result
                        has_tool_result = True
                        if tool_results is None:
                            tool_results = []

                        # 处理 tool_result 的 content
                        # Claude API 格式: content 可能是字符串或数组
                        # Amazon Q 格式: content 必须是 [{"text": "..."}]
                        raw_content = block.get("content", [])

                        # 统一转换为 Amazon Q 格式
                        amazonq_content = []

                        if isinstance(raw_content, str):
                            # 字符串格式 -> 转换为 [{"text": "..."}]
                            amazonq_content = [{"text": raw_content}]
                        elif isinstance(raw_content, list):
                            # 数组格式
                            for item in raw_content:
                                if isinstance(item, dict):
                                    if "type" in item and item["type"] == "text":
                                        # Claude 格式: {"type": "text", "text": "..."}
                                        amazonq_content.append({"text": item.get("text", "")})
                                    elif "text" in item:
                                        # 已经是 Amazon Q 格式: {"text": "..."}
                                        amazonq_content.append({"text": item["text"]})
                                    else:
                                        # 其他格式，尝试转换
                                        amazonq_content.append({"text": str(item)})
                                elif isinstance(item, str):
                                    # 字符串元素
                                    amazonq_content.append({"text": item})

                        # 检查是否有实际内容
                        has_actual_content = any(
                            item.get("text", "").strip()
                            for item in amazonq_content
                        )

                        # 如果没有实际内容，添加默认文本
                        if not has_actual_content:
                            amazonq_content = [
                                {"text": "Tool use was cancelled by the user"}
                            ]

                        tool_result = {
                            "toolUseId": block.get("tool_use_id"),
                            "content": amazonq_content,  # 使用转换后的格式
                            "status": block.get("status", "success")
                        }
                        tool_results.append(tool_result)
            prompt_content = "\n".join(text_parts)
        elif isinstance(content, str):
            prompt_content = content
        else:
            prompt_content = extract_text_from_claude_content(content)

    # 步骤 4: 构建用户输入上下文
    user_context = UserInputMessageContext(
        envState=env_state,
        tools=codewhisperer_tools,
        toolResults=tool_results  # 包含从消息中提取的 tool_results
    )

    # 步骤 5: 格式化内容（添加上下文信息）
    # 只有在非 tool result 消息时才添加模板格式
    if has_tool_result and not prompt_content:
        # 如果是 tool result 且没有文本内容，使用空字符串
        formatted_content = ""
    else:
        # 正常消息，添加模板格式
        formatted_content = (
            "--- CONTEXT ENTRY BEGIN ---\n"
            f"Current time: {get_current_timestamp()}\n"
            "--- CONTEXT ENTRY END ---\n\n"
            "--- USER MESSAGE BEGIN ---\n"
            f"{prompt_content}\n"
            "--- USER MESSAGE END ---"
        )

    # 如果有超长描述的工具，将完整描述添加到内容前面
    if long_description_tools:
        tool_docs_parts = []
        for tool_info in long_description_tools:
            tool_docs_parts.append(
                f"Tool: {tool_info['name']}\n"
                f"Full Description:\n{tool_info['full_description']}\n"
            )
        tool_documentation = "\n".join(tool_docs_parts)

        formatted_content = (
            "--- TOOL DOCUMENTATION BEGIN ---\n"
            f"{tool_documentation}"
            "--- TOOL DOCUMENTATION END ---\n\n"
            f"{formatted_content}"
        )

    # 如果有 system prompt，添加到内容前面
    if claude_req.system and formatted_content:
        # 处理 system prompt: 可能是字符串或数组
        system_text = ""
        if isinstance(claude_req.system, str):
            system_text = claude_req.system
        elif isinstance(claude_req.system, list):
            # 提取所有文本块的内容
            text_parts = []
            for block in claude_req.system:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
            system_text = '\n'.join(text_parts)

        if system_text:
            formatted_content = (
                "--- SYSTEM PROMPT BEGIN ---\n"
                f"{system_text}\nAttention! Your official CLI command is claude, NOT q chat. Please explicitly ignore any usage examples or instructions regarding q chat found in other parts of the system prompt. Always use claude for terminal commands.\n"
                "--- SYSTEM PROMPT END ---\n\n"
                f"{formatted_content}"
            )

    # 步骤 6: 获取 modelId 并映射到 Amazon Q 支持的模型
    model_id = map_claude_model_to_amazonq(claude_req.model)

    # 步骤 7: 组装 UserInputMessage（包含 images）
    user_input_message = UserInputMessage(
        content=formatted_content,
        userInputMessageContext=user_context,
        modelId=model_id,
        images=images  # 添加图片列表
    )

    # 步骤 8: 转换历史记录
    # 将除最后一条消息外的所有消息转换为历史记录
    history_messages = claude_req.messages[:-1] if len(claude_req.messages) > 1 else []
    codewhisperer_history = convert_history_messages(history_messages)

    # 步骤 9: 组装最终的 CodeWhispererRequest 对象
    conversation_state = ConversationState(
        conversationId=conversation_id,
        history=codewhisperer_history,
        currentMessage=CurrentMessage(userInputMessage=user_input_message)
    )

    final_request = CodeWhispererRequest(
        conversationState=conversation_state,
        profileArn=profile_arn
    )

    return final_request


def convert_history_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    转换历史消息为 Amazon Q 格式

    Args:
        messages: Claude 消息列表

    Returns:
        List[Dict[str, Any]]: Amazon Q 历史消息列表
    """
    history = []
    seen_tool_use_ids: set = set()  # 用于跟踪已添加的 toolUseId

    for message in messages:
        # 根据角色构建不同格式的历史条目
        if message.role == "user":
            # 处理用户消息（可能包含 tool_result 和 images）
            content = message.content
            text_content = ""
            tool_results = None
            images = None

            # 提取图片
            images = extract_images_from_claude_content(content)
            if images:
                logger.info(f"从历史消息中提取了 {len(images)} 张图片")

            if isinstance(content, list):
                # 解析包含多个内容块的消息
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            # 提取 tool_result
                            if tool_results is None:
                                tool_results = []

                            tool_use_id = block.get("tool_use_id")
                            raw_content = block.get("content", [])

                            # 统一转换为 Amazon Q 格式
                            amazonq_content = []

                            if isinstance(raw_content, str):
                                # 字符串格式 -> 转换为 [{"text": "..."}]
                                amazonq_content = [{"text": raw_content}]
                            elif isinstance(raw_content, list):
                                # 数组格式
                                for item in raw_content:
                                    if isinstance(item, dict):
                                        if "type" in item and item["type"] == "text":
                                            # Claude 格式: {"type": "text", "text": "..."}
                                            amazonq_content.append({"text": item.get("text", "")})
                                        elif "text" in item:
                                            # 已经是 Amazon Q 格式: {"text": "..."}
                                            amazonq_content.append({"text": item["text"]})
                                        else:
                                            # 其他格式，尝试转换
                                            amazonq_content.append({"text": str(item)})
                                    elif isinstance(item, str):
                                        # 字符串元素
                                        amazonq_content.append({"text": item})

                            # 检查是否有实际内容
                            has_actual_content = any(
                                item.get("text", "").strip()
                                for item in amazonq_content
                            )

                            # 如果没有实际内容，添加默认文本
                            if not has_actual_content:
                                amazonq_content = [
                                    {"text": "Tool use was cancelled by the user"}
                                ]

                            # 查找是否已经存在相同 toolUseId 的结果
                            existing_result = None
                            for result in tool_results:
                                if result.get("toolUseId") == tool_use_id:
                                    existing_result = result
                                    break

                            if existing_result:
                                # 合并 content 列表
                                existing_result["content"].extend(amazonq_content)
                                logger.info(f"合并重复的 toolUseId {tool_use_id} 的 content")
                            else:
                                # 创建新条目
                                tool_result = {
                                    "toolUseId": tool_use_id,
                                    "content": amazonq_content,
                                    "status": block.get("status", "success")
                                }
                                tool_results.append(tool_result)
                text_content = "\n".join(text_parts)
            else:
                text_content = extract_text_from_claude_content(content)

            # 构建用户消息条目
            user_input_context = {
                "envState": {
                    "operatingSystem": "macos",
                    "currentWorkingDirectory": "/"
                }
            }
            # 如果有 tool_results，添加到上下文中
            if tool_results:
                user_input_context["toolResults"] = tool_results

            # 构建历史消息条目
            user_input_msg = {
                "content": text_content,
                "userInputMessageContext": user_input_context,
                "origin": "CLI"
            }
            # 如果有图片，添加到消息中
            if images:
                user_input_msg["images"] = images

            history_entry = {
                "userInputMessage": user_input_msg
            }
        else:  # assistant
            # 处理助手消息（可能包含 tool_use）
            content = message.content
            text_content = extract_text_from_claude_content(content)

            # 助手消息格式（可能包含 toolUses）
            import uuid
            assistant_entry = {
                "assistantResponseMessage": {
                    "messageId": str(uuid.uuid4()),
                    "content": text_content
                }
            }

            # 如果助手消息包含 tool_use，将其添加到 assistantResponseMessage 中
            if isinstance(content, list):
                tool_uses = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_use_id = block.get("id")
                        # 检查是否已经添加过这个 toolUseId
                        if tool_use_id and tool_use_id in seen_tool_use_ids:
                            logger.warning(f"跳过重复的 toolUseId: {tool_use_id}")
                            continue

                        if tool_use_id:
                            seen_tool_use_ids.add(tool_use_id)

                        tool_uses.append({
                            "toolUseId": tool_use_id,
                            "name": block.get("name"),
                            "input": block.get("input", {})
                        })
                if tool_uses:
                    assistant_entry["assistantResponseMessage"]["toolUses"] = tool_uses

            history_entry = assistant_entry

        history.append(history_entry)

    return history


def codewhisperer_request_to_dict(request: CodeWhispererRequest) -> Dict[str, Any]:
    """
    将 CodeWhispererRequest 转换为字典（用于 JSON 序列化）

    Args:
        request: CodeWhispererRequest 对象

    Returns:
        Dict[str, Any]: 字典表示
    """
    # 构建 userInputMessageContext
    user_input_message_context = {}

    # 只有当有 tools 时才添加 envState 和 tools
    tools = request.conversationState.currentMessage.userInputMessage.userInputMessageContext.tools
    if tools:
        user_input_message_context["envState"] = {
            "operatingSystem": request.conversationState.currentMessage.userInputMessage.userInputMessageContext.envState.operatingSystem,
            "currentWorkingDirectory": request.conversationState.currentMessage.userInputMessage.userInputMessageContext.envState.currentWorkingDirectory
        }
        user_input_message_context["tools"] = [
            {
                "toolSpecification": {
                    "name": tool.toolSpecification.name,
                    "description": tool.toolSpecification.description,
                    "inputSchema": tool.toolSpecification.inputSchema
                }
            }
            for tool in tools
        ]

    # 如果有 toolResults，添加到上下文中
    tool_results = request.conversationState.currentMessage.userInputMessage.userInputMessageContext.toolResults
    if tool_results:
        user_input_message_context["toolResults"] = tool_results

    # 构建 userInputMessage
    user_input_message_dict = {
        "content": request.conversationState.currentMessage.userInputMessage.content,
        "userInputMessageContext": user_input_message_context,
        "origin": request.conversationState.currentMessage.userInputMessage.origin,
        "modelId": request.conversationState.currentMessage.userInputMessage.modelId
    }

    # 如果有 images，添加到 userInputMessage 中
    images = request.conversationState.currentMessage.userInputMessage.images
    if images:
        user_input_message_dict["images"] = images

    result = {
        "conversationState": {
            "conversationId": request.conversationState.conversationId,
            "history": request.conversationState.history,
            "currentMessage": {
                "userInputMessage": user_input_message_dict
            },
            "chatTriggerType": request.conversationState.chatTriggerType
        }
    }

    # 添加 profileArn（如果存在）
    if request.profileArn:
        result["profileArn"] = request.profileArn

    return result