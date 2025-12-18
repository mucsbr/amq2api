"""
Antigravity 流式响应处理器
将 Antigravity SSE 响应转换为 Claude SSE 格式
"""
import json
import logging
from typing import AsyncIterator

from .cache import cache_signature

logger = logging.getLogger(__name__)


async def handle_antigravity_stream(
    response_stream: AsyncIterator[bytes],
    model: str,
    session_id: str
) -> AsyncIterator[str]:
    """
    处理 Antigravity SSE 流式响应，转换为 Claude SSE 格式

    Args:
        response_stream: Antigravity 响应流
        model: 模型名称
        session_id: 会话 ID（用于缓存签名）

    Yields:
        Claude 格式的 SSE 事件
    """
    # 跟踪内容块和 token 统计
    content_blocks = []
    current_index = -1
    input_tokens = 0
    output_tokens = 0
    content_block_started = False
    content_block_stop_sent = False
    message_id = "msg_antigravity"
    message_start_sent = False
    finish_reason = "end_turn"  # 默认值，会从响应中更新

    # 思考文本累积（用于签名缓存）
    thought_buffer = {}  # {candidate_index: accumulated_text}

    # 处理流式响应
    buffer = ""
    byte_buffer = b""

    chunk_count = 0
    async for chunk in response_stream:
        chunk_count += 1
        if not chunk:
            logger.debug(f"[Antigravity Chunk {chunk_count}] 收到空 chunk")
            continue

        logger.debug(f"[Antigravity Chunk {chunk_count}] 收到 {len(chunk)} 字节")

        try:
            # 累积字节
            byte_buffer += chunk

            # 尝试解码
            try:
                text = byte_buffer.decode('utf-8')
                byte_buffer = b""
                logger.debug(f"[Antigravity Chunk {chunk_count}] 解码成功: {text[:200]}")
            except UnicodeDecodeError as e:
                logger.debug(f"[Antigravity Chunk {chunk_count}] 解码失败: {e}")
                if len(byte_buffer) > 4:
                    text = byte_buffer[:-4].decode('utf-8', errors='ignore')
                    byte_buffer = byte_buffer[-4:]
                else:
                    continue

            buffer += text

            # 处理完整的 SSE 事件（使用 \r\n\r\n 分隔）
            while '\r\n\r\n' in buffer:
                event_text, buffer = buffer.split('\r\n\r\n', 1)
                logger.debug(f"[Antigravity 事件] event_text: {event_text[:300]}")

                if event_text.startswith('data: '):
                    data_str = event_text[6:]
                    if data_str.strip() == '[DONE]':
                        logger.info("[Antigravity] 收到 [DONE] 标记")
                        continue

                    try:
                        data = json.loads(data_str)
                        response_data = data.get('response', data)
                        logger.info(f"[Antigravity 响应] {json.dumps(response_data, ensure_ascii=False)[:500]}")

                        # 提取 responseId
                        if 'responseId' in response_data:
                            message_id = response_data['responseId']

                        # 确保 message_start 在第一个内容之前发送
                        def ensure_message_start():
                            nonlocal message_start_sent
                            if not message_start_sent:
                                message_start_sent = True
                                return format_sse_event("message_start", {
                                    "type": "message_start",
                                    "message": {
                                        "id": message_id,
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [],
                                        "model": model,
                                        "stop_reason": None,
                                        "stop_sequence": None,
                                        "usage": {"input_tokens": 0, "output_tokens": 0}
                                    }
                                })
                            return None

                        # 提取 usageMetadata
                        if 'usageMetadata' in response_data:
                            usage_meta = response_data['usageMetadata']
                            input_tokens = usage_meta.get('promptTokenCount', 0)
                            output_tokens = usage_meta.get('candidatesTokenCount', 0)
                            logger.info(f"[Antigravity Token] input={input_tokens}, output={output_tokens}")

                        # 处理 candidates
                        if 'candidates' in response_data:
                            for cand_idx, candidate in enumerate(response_data['candidates']):
                                # 提取 finishReason 并转换为 Claude 格式
                                raw_finish = candidate.get('finishReason', '')
                                if raw_finish:
                                    finish_map = {
                                        'STOP': 'end_turn',
                                        'MAX_TOKENS': 'max_tokens',
                                        'SAFETY': 'content_filter',
                                        'RECITATION': 'content_filter',
                                        'TOOL_USE': 'tool_use',
                                        'FINISH_REASON_UNSPECIFIED': 'end_turn',
                                    }
                                    finish_reason = finish_map.get(raw_finish, 'end_turn')

                                content = candidate.get('content', {})
                                parts = content.get('parts', [])

                                for part in parts:
                                    # 处理 thinking 内容
                                    if part.get('thought'):
                                        thinking_text = part.get('text', '')
                                        if thinking_text:
                                            # 累积思考文本（用于签名缓存）
                                            if cand_idx not in thought_buffer:
                                                thought_buffer[cand_idx] = ""
                                            thought_buffer[cand_idx] += thinking_text

                                            # 开启 thinking 块
                                            if not content_block_started:
                                                # 确保 message_start 已发送
                                                msg_start_event = ensure_message_start()
                                                if msg_start_event:
                                                    yield msg_start_event

                                                current_index += 1
                                                content_blocks.append({'type': 'thinking'})
                                                yield format_sse_event("content_block_start", {
                                                    "type": "content_block_start",
                                                    "index": current_index,
                                                    "content_block": {"type": "thinking", "thinking": ""}
                                                })
                                                content_block_started = True
                                                content_block_stop_sent = False

                                            # 发送 thinking delta
                                            yield format_sse_event("content_block_delta", {
                                                "type": "content_block_delta",
                                                "index": current_index,
                                                "delta": {"type": "thinking_delta", "thinking": thinking_text}
                                            })

                                        # 处理签名
                                        if 'thoughtSignature' in part:
                                            signature = part['thoughtSignature']

                                            # 缓存签名
                                            full_text = thought_buffer.get(cand_idx, "")
                                            if full_text and session_id and signature:
                                                cache_signature(session_id, full_text, signature)
                                                logger.debug(f"[Antigravity] 缓存签名: {full_text[:50]}...")

                                            if content_block_started and not content_block_stop_sent:
                                                # 发送 signature_delta
                                                yield format_sse_event("content_block_delta", {
                                                    "type": "content_block_delta",
                                                    "index": current_index,
                                                    "delta": {"type": "signature_delta", "signature": signature}
                                                })
                                                # 发送 content_block_stop
                                                yield format_sse_event("content_block_stop", {
                                                    "type": "content_block_stop",
                                                    "index": current_index
                                                })
                                                content_block_stop_sent = True
                                                content_block_started = False
                                                # 重置思考缓冲
                                                thought_buffer[cand_idx] = ""

                                    # 处理文本内容
                                    elif 'text' in part and part['text']:
                                        text_content = part['text']

                                        # 如果之前是 thinking 块，需要先关闭
                                        if content_block_started and current_index >= 0 and content_blocks[current_index]['type'] == 'thinking':
                                            if not content_block_stop_sent:
                                                yield format_sse_event("content_block_stop", {
                                                    "type": "content_block_stop",
                                                    "index": current_index
                                                })
                                                content_block_stop_sent = True
                                            content_block_started = False

                                        # 开启新的文本块
                                        if not content_block_started or (current_index >= 0 and content_blocks[current_index]['type'] != 'text'):
                                            # 确保 message_start 已发送
                                            msg_start_event = ensure_message_start()
                                            if msg_start_event:
                                                yield msg_start_event

                                            current_index += 1
                                            content_blocks.append({'type': 'text'})
                                            yield format_sse_event("content_block_start", {
                                                "type": "content_block_start",
                                                "index": current_index,
                                                "content_block": {"type": "text", "text": ""}
                                            })
                                            content_block_started = True
                                            content_block_stop_sent = False

                                        yield format_sse_event("content_block_delta", {
                                            "type": "content_block_delta",
                                            "index": current_index,
                                            "delta": {"type": "text_delta", "text": text_content}
                                        })

                                    # 处理工具调用
                                    elif 'functionCall' in part:
                                        func_call = part['functionCall']

                                        # 如果有正在进行的块，先关闭
                                        if content_block_started and not content_block_stop_sent:
                                            yield format_sse_event("content_block_stop", {
                                                "type": "content_block_stop",
                                                "index": current_index
                                            })
                                            content_block_stop_sent = True
                                            content_block_started = False

                                        # 确保 message_start 已发送
                                        msg_start_event = ensure_message_start()
                                        if msg_start_event:
                                            yield msg_start_event

                                        current_index += 1
                                        content_blocks.append({'type': 'tool_use'})

                                        tool_id = func_call.get('id', f"toolu_{current_index}")
                                        tool_name = func_call.get('name', 'unknown_tool')
                                        tool_args = func_call.get('args', {})

                                        yield format_sse_event("content_block_start", {
                                            "type": "content_block_start",
                                            "index": current_index,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": tool_id,
                                                "name": tool_name,
                                                "input": {}
                                            }
                                        })

                                        yield format_sse_event("content_block_delta", {
                                            "type": "content_block_delta",
                                            "index": current_index,
                                            "delta": {
                                                "type": "input_json_delta",
                                                "partial_json": json.dumps(tool_args)
                                            }
                                        })

                                        yield format_sse_event("content_block_stop", {
                                            "type": "content_block_stop",
                                            "index": current_index
                                        })

                    except json.JSONDecodeError as e:
                        logger.warning(f"[Antigravity JSON错误] 解析失败: {e}, data: {data_str[:200]}")
                        continue

        except Exception as e:
            logger.error(f"[Antigravity 异常] 处理流式响应时出错: {e}", exc_info=True)
            continue

    logger.info(f"[Antigravity 流结束] 共处理 {chunk_count} 个 chunk")

    # 处理 buffer 中剩余的数据
    if buffer.strip():
        if buffer.startswith('data: '):
            data_str = buffer[6:]
            if data_str.strip() and data_str.strip() != '[DONE]':
                try:
                    data = json.loads(data_str)
                    response_data = data.get('response', data)

                    if 'candidates' in response_data:
                        for candidate in response_data['candidates']:
                            content = candidate.get('content', {})
                            parts = content.get('parts', [])

                            for part in parts:
                                if 'text' in part and part['text'] and not part.get('thought'):
                                    if current_index == -1 or content_blocks[current_index]['type'] != 'text':
                                        current_index += 1
                                        content_blocks.append({'type': 'text'})
                                        yield format_sse_event("content_block_start", {
                                            "type": "content_block_start",
                                            "index": current_index,
                                            "content_block": {"type": "text", "text": ""}
                                        })
                                        content_block_started = True

                                    yield format_sse_event("content_block_delta", {
                                        "type": "content_block_delta",
                                        "index": current_index,
                                        "delta": {"type": "text_delta", "text": part['text']}
                                    })
                except json.JSONDecodeError:
                    pass

    # 关闭最后一个内容块
    if current_index >= 0 and content_block_started and not content_block_stop_sent:
        logger.info(f"[Antigravity 结束] 关闭最后一个内容块 index={current_index}")
        yield format_sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": current_index
        })

    # 确保 message_start 已发送（即使没有内容）
    if not message_start_sent:
        yield format_sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        })

    # 发送 message_delta 事件
    logger.info(f"[Antigravity 结束] message_delta: input={input_tokens}, output={output_tokens}, stop_reason={finish_reason}")
    yield format_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": finish_reason, "stop_sequence": None},
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
    })

    # 发送 message_stop 事件
    logger.info("[Antigravity 结束] message_stop")
    yield format_sse_event("message_stop", {
        "type": "message_stop"
    })


def format_sse_event(event_type: str, data: dict) -> str:
    """
    格式化 SSE 事件

    Args:
        event_type: 事件类型
        data: 事件数据

    Returns:
        格式化的 SSE 事件字符串
    """
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
