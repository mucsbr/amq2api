"""
主服务模块
FastAPI 服务器，提供 Claude API 兼容的接口
"""
import logging
import httpx
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from config import read_global_config, get_config_sync
from auth import get_auth_headers_with_retry, refresh_account_token, NoAccountAvailableError, TokenRefreshError
from account_manager import (
    list_enabled_accounts, list_all_accounts, get_account,
    create_account, update_account, delete_account
)
from models import ClaudeRequest
from converter import convert_claude_to_codewhisperer_request, codewhisperer_request_to_dict
from stream_handler_new import handle_amazonq_stream
from message_processor import process_claude_history_for_amazonq, log_history_summary
from pydantic import BaseModel
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化配置
    logger.info("正在初始化配置...")
    try:
        await read_global_config()
        logger.info("配置初始化成功")
    except Exception as e:
        logger.error(f"配置初始化失败: {e}")
        raise

    yield

    # 关闭时清理资源
    logger.info("正在关闭服务...")


# 创建 FastAPI 应用
app = FastAPI(
    title="Amazon Q to Claude API Proxy",
    description="将 Claude API 请求转换为 Amazon Q/CodeWhisperer 请求的代理服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic 模型
class AccountCreate(BaseModel):
    label: Optional[str] = None
    clientId: str
    clientSecret: str
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = True


class AccountUpdate(BaseModel):
    label: Optional[str] = None
    clientId: Optional[str] = None
    clientSecret: Optional[str] = None
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


@app.get("/")
async def root():
    """健康检查端点"""
    return {
        "status": "ok",
        "service": "Amazon Q to Claude API Proxy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """轻量级健康检查端点 - 仅检查服务状态和账号配置"""
    try:
        all_accounts = list_all_accounts()
        enabled_accounts = [acc for acc in all_accounts if acc.get('enabled')]

        if not enabled_accounts:
            return {
                "status": "unhealthy",
                "reason": "no_enabled_accounts",
                "enabled_accounts": 0,
                "total_accounts": len(all_accounts)
            }

        return {
            "status": "healthy",
            "enabled_accounts": len(enabled_accounts),
            "total_accounts": len(all_accounts)
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "reason": "system_error",
            "error": str(e)
        }


@app.post("/v1/messages")
async def create_message(request: Request):
    """
    Claude API 兼容的消息创建端点
    接收 Claude 格式的请求，转换为 CodeWhisperer 格式并返回流式响应
    """
    try:
        # 解析请求体
        request_data = await request.json()

        # 标准 Claude API 格式 - 转换为 conversationState
        logger.info(f"收到标准 Claude API 请求: {request_data.get('model', 'unknown')}")

        # 转换为 ClaudeRequest 对象
        claude_req = parse_claude_request(request_data)

        # 获取配置
        config = await read_global_config()

        # 转换为 CodeWhisperer 请求
        codewhisperer_req = convert_claude_to_codewhisperer_request(
            claude_req,
            conversation_id=None,  # 自动生成
            profile_arn=config.profile_arn
        )

        # 转换为字典
        codewhisperer_dict = codewhisperer_request_to_dict(codewhisperer_req)
        model = claude_req.model

        # 处理历史记录：合并连续的 userInputMessage
        conversation_state = codewhisperer_dict.get("conversationState", {})
        history = conversation_state.get("history", [])

        if history:
            # 记录原始历史记录
            logger.info("=" * 80)
            logger.info("原始历史记录:")
            log_history_summary(history, prefix="[原始] ")

            # 合并连续的用户消息
            processed_history = process_claude_history_for_amazonq(history)

            # 记录处理后的历史记录
            logger.info("=" * 80)
            logger.info("处理后的历史记录:")
            log_history_summary(processed_history, prefix="[处理后] ")

            # 更新请求体
            conversation_state["history"] = processed_history
            codewhisperer_dict["conversationState"] = conversation_state

        # 处理 currentMessage 中的重复 toolResults（标准 Claude API 格式）
        conversation_state = codewhisperer_dict.get("conversationState", {})
        current_message = conversation_state.get("currentMessage", {})
        user_input_message = current_message.get("userInputMessage", {})
        user_input_message_context = user_input_message.get("userInputMessageContext", {})

        # 合并 currentMessage 中重复的 toolResults
        tool_results = user_input_message_context.get("toolResults", [])
        if tool_results:
            merged_tool_results = []
            seen_tool_use_ids = set()

            for result in tool_results:
                tool_use_id = result.get("toolUseId")
                if tool_use_id in seen_tool_use_ids:
                    # 找到已存在的条目，合并 content
                    for existing in merged_tool_results:
                        if existing.get("toolUseId") == tool_use_id:
                            existing["content"].extend(result.get("content", []))
                            logger.info(f"[CURRENT MESSAGE - CLAUDE API] 合并重复的 toolUseId {tool_use_id} 的 content")
                            break
                else:
                    # 新条目
                    seen_tool_use_ids.add(tool_use_id)
                    merged_tool_results.append(result)

            user_input_message_context["toolResults"] = merged_tool_results
            user_input_message["userInputMessageContext"] = user_input_message_context
            current_message["userInputMessage"] = user_input_message
            conversation_state["currentMessage"] = current_message
            codewhisperer_dict["conversationState"] = conversation_state

        final_request = codewhisperer_dict

        # 调试：打印请求体
        import json
        logger.info(f"转换后的请求体: {json.dumps(final_request, indent=2, ensure_ascii=False)}")

        # 获取账号和认证头（支持多账号随机选择和单账号回退）
        # 检查是否指定了特定账号（用于测试）
        specified_account_id = request.headers.get("X-Account-ID")

        try:
            if specified_account_id:
                # 使用指定的账号
                account = get_account(specified_account_id)
                if not account:
                    raise HTTPException(status_code=404, detail=f"账号不存在: {specified_account_id}")
                if not account.get('enabled'):
                    raise HTTPException(status_code=403, detail=f"账号已禁用: {specified_account_id}")

                # 获取该账号的认证头
                from auth import get_auth_headers_for_account
                base_auth_headers = await get_auth_headers_for_account(account)
                logger.info(f"使用指定账号 - 账号: {account.get('id')} (label: {account.get('label', 'N/A')})")
            else:
                # 随机选择账号
                account, base_auth_headers = await get_auth_headers_with_retry()
                if account:
                    logger.info(f"使用多账号模式 - 账号: {account.get('id')} (label: {account.get('label', 'N/A')})")
                else:
                    logger.info("使用单账号模式（.env 配置）")
        except NoAccountAvailableError as e:
            logger.error(f"无可用账号: {e}")
            raise HTTPException(status_code=503, detail="没有可用的账号，请在管理页面添加账号或配置 .env 文件")
        except TokenRefreshError as e:
            logger.error(f"Token 刷新失败: {e}")
            raise HTTPException(status_code=502, detail="Token 刷新失败")

        # 构建 Amazon Q 特定的请求头（完整版本）
        import uuid
        auth_headers = {
            **base_auth_headers,
            "Content-Type": "application/x-amz-json-1.0",
            "X-Amz-Target": "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
            "User-Agent": "aws-sdk-rust/1.3.9 ua/2.1 api/codewhispererstreaming/0.1.11582 os/macos lang/rust/1.87.0 md/appVersion-1.19.3 app/AmazonQ-For-CLI",
            "X-Amz-User-Agent": "aws-sdk-rust/1.3.9 ua/2.1 api/codewhispererstreaming/0.1.11582 os/macos lang/rust/1.87.0 m/F app/AmazonQ-For-CLI",
            "X-Amzn-Codewhisperer-Optout": "true",
            "Amz-Sdk-Request": "attempt=1; max=3",
            "Amz-Sdk-Invocation-Id": str(uuid.uuid4()),
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br"
        }

        # 发送请求到 Amazon Q
        logger.info("正在发送请求到 Amazon Q...")

        # API URL
        api_url = "https://q.us-east-1.amazonaws.com/"

        # 创建字节流响应（支持 401/403 重试）
        async def byte_stream():
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    async with client.stream(
                        "POST",
                        api_url,
                        json=final_request,
                        headers=auth_headers
                    ) as response:
                        # 检查响应状态
                        if response.status_code in (401, 403):
                            # 401/403 错误：刷新 token 并重试
                            logger.warning(f"收到 {response.status_code} 错误，尝试刷新 token 并重试")
                            error_text = await response.aread()
                            error_str = error_text.decode() if isinstance(error_text, bytes) else str(error_text)
                            logger.error(f"原始错误: {error_str}")

                            # 检测账号是否被封
                            if "TEMPORARILY_SUSPENDED" in error_str and account:
                                logger.error(f"账号 {account['id']} 已被封禁，自动禁用")
                                from datetime import datetime
                                suspend_info = {
                                    "suspended": True,
                                    "suspended_at": datetime.now().isoformat(),
                                    "suspend_reason": "TEMPORARILY_SUSPENDED"
                                }
                                current_other = account.get('other') or {}
                                current_other.update(suspend_info)
                                update_account(account['id'], enabled=False, other=current_other)
                                raise HTTPException(status_code=403, detail=f"账号已被封禁: {error_str}")

                            try:
                                # 刷新 token（支持多账号和单账号模式）
                                if account:
                                    # 多账号模式：刷新当前账号
                                    refreshed_account = await refresh_account_token(account)
                                    new_access_token = refreshed_account.get("accessToken")
                                else:
                                    # 单账号模式：刷新 .env 配置的 token
                                    from auth import refresh_legacy_token
                                    await refresh_legacy_token()
                                    from config import read_global_config
                                    config = await read_global_config()
                                    new_access_token = config.access_token

                                if not new_access_token:
                                    raise HTTPException(status_code=502, detail="Token 刷新后仍无法获取 accessToken")

                                # 更新认证头
                                auth_headers["Authorization"] = f"Bearer {new_access_token}"
                                logger.info(f"Token 刷新成功，使用新 token 重试请求")

                                # 使用新 token 重试
                                async with client.stream(
                                    "POST",
                                    api_url,
                                    json=final_request,
                                    headers=auth_headers
                                ) as retry_response:
                                    if retry_response.status_code != 200:
                                        retry_error = await retry_response.aread()
                                        retry_error_str = retry_error.decode() if isinstance(retry_error, bytes) else str(retry_error)
                                        logger.error(f"重试后仍失败: {retry_response.status_code} {retry_error_str}")

                                        # 重试后仍然失败，检测是否被封
                                        if retry_response.status_code == 403 and "TEMPORARILY_SUSPENDED" in retry_error_str and account:
                                            logger.error(f"账号 {account['id']} 已被封禁，自动禁用")
                                            from datetime import datetime
                                            suspend_info = {
                                                "suspended": True,
                                                "suspended_at": datetime.now().isoformat(),
                                                "suspend_reason": "TEMPORARILY_SUSPENDED"
                                            }
                                            current_other = account.get('other') or {}
                                            current_other.update(suspend_info)
                                            update_account(account['id'], enabled=False, other=current_other)

                                        raise HTTPException(
                                            status_code=retry_response.status_code,
                                            detail=f"重试后仍失败: {retry_error_str}"
                                        )

                                    # 重试成功，返回数据流
                                    async for chunk in retry_response.aiter_bytes():
                                        if chunk:
                                            yield chunk
                                    return

                            except TokenRefreshError as e:
                                logger.error(f"Token 刷新失败: {e}")
                                raise HTTPException(status_code=502, detail=f"Token 刷新失败: {str(e)}")

                        elif response.status_code != 200:
                            error_text = await response.aread()
                            logger.error(f"上游 API 错误: {response.status_code} {error_text}")
                            raise HTTPException(
                                status_code=response.status_code,
                                detail=f"上游 API 错误: {error_text.decode()}"
                            )

                        # 正常响应，处理 Event Stream（字节流）
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk

                except httpx.RequestError as e:
                    logger.error(f"请求错误: {e}")
                    raise HTTPException(status_code=502, detail=f"上游服务错误: {str(e)}")

        # 返回流式响应
        async def claude_stream():
            async for event in handle_amazonq_stream(byte_stream(), model=model, request_data=request_data):
                yield event

        return StreamingResponse(
            claude_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


# 账号管理 API 端点
@app.get("/v2/accounts")
async def list_accounts():
    """列出所有账号"""
    accounts = list_all_accounts()
    return JSONResponse(content=accounts)


@app.get("/v2/accounts/{account_id}")
async def get_account_detail(account_id: str):
    """获取账号详情"""
    account = get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="账号不存在")
    return JSONResponse(content=account)


@app.post("/v2/accounts")
async def create_account_endpoint(body: AccountCreate):
    """创建新账号"""
    try:
        account = create_account(
            label=body.label,
            client_id=body.clientId,
            client_secret=body.clientSecret,
            refresh_token=body.refreshToken,
            access_token=body.accessToken,
            other=body.other,
            enabled=body.enabled if body.enabled is not None else True
        )
        return JSONResponse(content=account)
    except Exception as e:
        logger.error(f"创建账号失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建账号失败: {str(e)}")


@app.patch("/v2/accounts/{account_id}")
async def update_account_endpoint(account_id: str, body: AccountUpdate):
    """更新账号信息"""
    try:
        account = update_account(
            account_id=account_id,
            label=body.label,
            client_id=body.clientId,
            client_secret=body.clientSecret,
            refresh_token=body.refreshToken,
            access_token=body.accessToken,
            other=body.other,
            enabled=body.enabled
        )
        if not account:
            raise HTTPException(status_code=404, detail="账号不存在")
        return JSONResponse(content=account)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新账号失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新账号失败: {str(e)}")


@app.delete("/v2/accounts/{account_id}")
async def delete_account_endpoint(account_id: str):
    """删除账号"""
    success = delete_account(account_id)
    if not success:
        raise HTTPException(status_code=404, detail="账号不存在")
    return JSONResponse(content={"deleted": account_id})


@app.post("/v2/accounts/{account_id}/refresh")
async def manual_refresh_endpoint(account_id: str):
    """手动刷新账号 token"""
    try:
        account = get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail="账号不存在")

        refreshed_account = await refresh_account_token(account)
        return JSONResponse(content=refreshed_account)
    except TokenRefreshError as e:
        logger.error(f"刷新 token 失败: {e}")
        raise HTTPException(status_code=502, detail=f"刷新 token 失败: {str(e)}")
    except Exception as e:
        logger.error(f"刷新 token 失败: {e}")
        raise HTTPException(status_code=500, detail=f"刷新 token 失败: {str(e)}")


# 管理页面
@app.get("/admin", response_class=FileResponse)
async def admin_page():
    """管理页面"""
    from pathlib import Path
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail="管理页面不存在")
    return FileResponse(str(frontend_path))


def parse_claude_request(data: dict) -> ClaudeRequest:
    """
    解析 Claude API 请求数据

    Args:
        data: 请求数据字典

    Returns:
        ClaudeRequest: Claude 请求对象
    """
    from models import ClaudeMessage, ClaudeTool

    # 解析消息
    messages = []
    for msg in data.get("messages", []):
        # 安全地获取 role 和 content，提供默认值
        role = msg.get("role", "user")
        content = msg.get("content", "")
        messages.append(ClaudeMessage(
            role=role,
            content=content
        ))

    # 解析工具
    tools = None
    if "tools" in data:
        tools = []
        for tool in data["tools"]:
            # 安全地获取工具字段，提供默认值
            name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = tool.get("input_schema", {})

            # 只有当 name 不为空时才添加工具
            if name:
                tools.append(ClaudeTool(
                    name=name,
                    description=description,
                    input_schema=input_schema
                ))

    return ClaudeRequest(
        model=data.get("model", "claude-sonnet-4.5"),
        messages=messages,
        max_tokens=data.get("max_tokens", 4096),
        temperature=data.get("temperature"),
        tools=tools,
        stream=data.get("stream", True),
        system=data.get("system")
    )


if __name__ == "__main__":
    import uvicorn

    # 读取配置
    try:
        import asyncio
        config = asyncio.run(read_global_config())
        port = config.port
    except Exception as e:
        logger.error(f"无法读取配置: {e}")
        port = 8080

    logger.info(f"正在启动服务，监听端口 {port}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
