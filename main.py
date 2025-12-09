"""
主服务模块
FastAPI 服务器，提供 Claude API 兼容的接口
"""
import logging
import httpx
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from config import read_global_config, get_config_sync
from auth import get_auth_headers_with_retry, refresh_account_token, NoAccountAvailableError, TokenRefreshError
from account_manager import (
    list_enabled_accounts, list_all_accounts, get_account,
    create_account, update_account, delete_account, get_random_account,
    get_random_channel_by_model
)
from models import ClaudeRequest
from converter import convert_claude_to_codewhisperer_request, codewhisperer_request_to_dict
from stream_handler_new import handle_amazonq_stream
from message_processor import process_claude_history_for_amazonq, log_history_summary
from pydantic import BaseModel
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Gemini 模块导入
from gemini.auth import GeminiTokenManager
from gemini.converter import convert_claude_to_gemini
from gemini.handler import handle_gemini_stream

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


# 管理员鉴权依赖
async def verify_admin_key(x_admin_key: Optional[str] = Header(None)):
    """验证管理员密钥"""
    import os
    admin_key = os.getenv("ADMIN_KEY")

    # 如果没有设置 ADMIN_KEY，则不需要验证
    if not admin_key:
        return True

    # 如果设置了 ADMIN_KEY，则必须验证
    if not x_admin_key or x_admin_key != admin_key:
        raise HTTPException(
            status_code=403,
            detail="访问被拒绝：需要有效的管理员密钥。请在请求头中添加 X-Admin-Key"
        )
    return True


# API Key 鉴权依赖
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """验证 API Key（Anthropic API 格式）"""
    import os
    api_key = os.getenv("API_KEY")

    # 如果没有设置 API_KEY，则不需要验证
    if not api_key:
        return True

    # 如果设置了 API_KEY，则必须验证
    if not x_api_key or x_api_key != api_key:
        raise HTTPException(
            status_code=401,
            detail="未授权：需要有效的 API Key。请在请求头中添加 x-api-key"
        )
    return True


# Pydantic 模型
class AccountCreate(BaseModel):
    label: Optional[str] = None
    clientId: str
    clientSecret: str
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = True
    type: str = "amazonq"  # amazonq 或 gemini


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
async def create_message(request: Request, _: bool = Depends(verify_api_key)):
    """
    Claude API 兼容的消息创建端点（智能路由）
    根据模型和账号数量自动选择渠道（Amazon Q 或 Gemini）
    """
    try:
        # 解析请求体
        request_data = await request.json()
        model = request_data.get('model', 'claude-sonnet-4.5')

        # 智能路由：根据模型选择渠道
        specified_account_id = request.headers.get("X-Account-ID")

        if specified_account_id:
            # 指定了账号，检查账号类型并路由到对应渠道
            account = get_account(specified_account_id)
            if not account:
                raise HTTPException(status_code=404, detail=f"账号不存在: {specified_account_id}")
            if not account.get('enabled'):
                raise HTTPException(status_code=403, detail=f"账号已禁用: {specified_account_id}")

            account_type = account.get('type', 'amazonq')
            if account_type == 'gemini':
                logger.info(f"指定账号为 Gemini 类型，转发到 Gemini 渠道")
                return await create_gemini_message(request)
        else:
            # 没有指定账号时，根据模型智能选择渠道
            channel = get_random_channel_by_model(model)

            if not channel:
                raise HTTPException(status_code=503, detail="没有可用的账号")

            logger.info(f"智能路由选择渠道: {channel}")

            # 如果选择了 Gemini 渠道，转发到 /v1/gemini/messages
            if channel == 'gemini':
                return await create_gemini_message(request)

        # 继续使用 Amazon Q 渠道的原有逻辑

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
            # logger.info("=" * 80)
            # logger.info("原始历史记录:")
            # log_history_summary(history, prefix="[原始] ")

            # 合并连续的用户消息
            processed_history = process_claude_history_for_amazonq(history)

            # 记录处理后的历史记录
            # logger.info("=" * 80)
            # logger.info("处理后的历史记录:")
            # log_history_summary(processed_history, prefix="[处理后] ")

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

        # 获取账号和认证头（支持多账号随机选择和单账号回退）
        # 检查是否指定了特定账号（用于测试）
        specified_account_id = request.headers.get("X-Account-ID")

        # 用于重试的变量
        account = None
        base_auth_headers = None

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

                                # 如果不是指定账号，抛出 TokenRefreshError 让外层重试
                                if not specified_account_id:
                                    raise TokenRefreshError(f"账号已被封禁: {error_str}")
                                else:
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
                            error_str = error_text.decode() if isinstance(error_text, bytes) else str(error_text)
                            logger.error(f"上游 API 错误: {response.status_code} {error_str}")

                            # 检测月度配额用完错误
                            if "ThrottlingException" in error_str and "MONTHLY_REQUEST_COUNT" in error_str:
                                logger.error(f"账号 {account.get('id') if account else 'legacy'} 月度配额已用完")
                                if account:
                                    # 多账号模式：禁用该账号
                                    from datetime import datetime
                                    quota_info = {
                                        "monthly_quota_exhausted": True,
                                        "exhausted_at": datetime.now().isoformat()
                                    }
                                    current_other = account.get('other') or {}
                                    current_other.update(quota_info)
                                    update_account(account['id'], enabled=False, other=current_other)
                                    raise HTTPException(
                                        status_code=429,
                                        detail="账号月度配额已用完，已自动禁用该账号。请等待下月重置或添加新账号。"
                                    )
                                else:
                                    # 单账号模式
                                    raise HTTPException(
                                        status_code=429,
                                        detail="Amazon Q 月度配额已用完，请等待下月重置。"
                                    )

                            raise HTTPException(
                                status_code=response.status_code,
                                detail=f"上游 API 错误: {error_str}"
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


@app.post("/v1/gemini/messages")
async def create_gemini_message(request: Request, _: bool = Depends(verify_api_key)):
    """
    Gemini API 端点
    接收 Claude 格式的请求，转换为 Gemini 格式并返回流式响应
    """
    try:
        # 解析请求体
        request_data = await request.json()

        # 转换为 ClaudeRequest 对象
        claude_req = parse_claude_request(request_data)

        # 检查是否指定了特定账号（用于测试）
        specified_account_id = request.headers.get("X-Account-ID")

        if specified_account_id:
            # 使用指定的账号
            account = get_account(specified_account_id)
            if not account:
                raise HTTPException(status_code=404, detail=f"账号不存在: {specified_account_id}")
            if not account.get('enabled'):
                raise HTTPException(status_code=403, detail=f"账号已禁用: {specified_account_id}")
            if account.get('type') != 'gemini':
                raise HTTPException(status_code=400, detail=f"账号类型不是 Gemini: {specified_account_id}")
            logger.info(f"使用指定 Gemini 账号: {account['label']} (ID: {account['id']})")
        else:
            # 随机选择 Gemini 账号（根据模型配额过滤）
            account = get_random_account(account_type="gemini", model=claude_req.model)
            if not account:
                raise HTTPException(status_code=503, detail=f"没有可用的 Gemini 账号支持模型 {claude_req.model}")
            logger.info(f"使用随机 Gemini 账号: {account['label']} (ID: {account['id']}) - 模型: {claude_req.model}")

        # 检查并使用数据库中的 access token
        other = account.get("other") or {}
        if isinstance(other, str):
            import json
            try:
                other = json.loads(other)
            except json.JSONDecodeError:
                other = {}

        access_token = account.get("accessToken")
        token_expires_at = None

        # 从 other 字段读取过期时间
        if access_token:
            if other.get("token_expires_at"):
                try:
                    from datetime import datetime, timedelta
                    token_expires_at = datetime.fromisoformat(other["token_expires_at"])
                    if datetime.now() >= token_expires_at - timedelta(minutes=5):
                        logger.info(f"Gemini access token 即将过期，需要刷新")
                        access_token = None
                        token_expires_at = None
                except Exception as e:
                    logger.warning(f"解析 Gemini token 过期时间失败: {e}")
                    access_token = None
                    token_expires_at = None
            else:
                # 如果有 access_token 但没有过期时间,清空 token 强制刷新一次
                logger.info(f"Gemini access token 缺少过期时间,强制刷新")
                access_token = None
                token_expires_at = None

        # 初始化 Token 管理器
        token_manager = GeminiTokenManager(
            client_id=account["clientId"],
            client_secret=account["clientSecret"],
            refresh_token=account["refreshToken"],
            api_endpoint=other.get("api_endpoint", "https://daily-cloudcode-pa.sandbox.googleapis.com"),
            access_token=access_token,
            token_expires_at=token_expires_at
        )

        # 确保 token 有效（如果需要会自动刷新）
        await token_manager.get_access_token()

        # 获取项目 ID
        project_id = other.get("project") or await token_manager.get_project_id()

        # 如果 token 被刷新，更新数据库
        if token_manager.access_token != access_token:
            from account_manager import update_account_tokens
            # 更新 other 字段，保存过期时间
            other["token_expires_at"] = token_manager.token_expires_at.isoformat() if token_manager.token_expires_at else None
            update_account(account["id"], access_token=token_manager.access_token, other=other)
            logger.info(f"Gemini access token 已更新到数据库")

        # 转换为 Gemini 请求
        gemini_request = convert_claude_to_gemini(
            claude_req,
            project=project_id
        )

        # 获取认证头
        auth_headers = await token_manager.get_auth_headers()

        # 构建完整的请求头
        headers = {
            **auth_headers,
            "Content-Type": "application/json",
            "User-Agent": "antigravity/1.11.3 darwin/arm64",
            "Accept-Encoding": "gzip"
        }

        # API URL
        api_url = f"{other.get('api_endpoint', 'https://daily-cloudcode-pa.sandbox.googleapis.com')}/v1internal:streamGenerateContent?alt=sse"

        async def gemini_byte_stream():
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    logger.info(f"[HTTP] 开始请求 Gemini API: {api_url}")
                    async with client.stream(
                        "POST",
                        api_url,
                        json=gemini_request,
                        headers=headers
                    ) as response:
                        logger.info(f"[HTTP] 收到响应: status_code={response.status_code}")
                        logger.info(f"[HTTP] 响应头: {dict(response.headers)}")

                        # 检测 Gemini API 空响应问题
                        content_length = response.headers.get('content-length', '')
                        if content_length == '0':
                            logger.error("[HTTP] Gemini API 返回空响应 (content-length: 0)")
                            # 生成 SSE 错误事件
                            error_event = 'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"Gemini API 返回空响应，请重试"}}\n\n'
                            yield error_event.encode('utf-8')
                            return

                        if response.status_code != 200:
                            error_text = await response.aread()
                            error_str = error_text.decode() if isinstance(error_text, bytes) else str(error_text)
                            logger.error(f"Gemini API 错误: {response.status_code} {error_str}")

                            # 处理 429 Resource Exhausted 错误
                            if response.status_code == 429:
                                try:
                                    from account_manager import mark_model_exhausted, update_account
                                    from gemini.converter import map_claude_model_to_gemini

                                    # 获取 Gemini 模型名称
                                    gemini_model = map_claude_model_to_gemini(claude_req.model)
                                    logger.info(f"收到 429 错误，正在调用 fetchAvailableModels 获取账号 {account['id']} 的最新配额信息...")

                                    # 调用 fetchAvailableModels 获取最新配额信息
                                    models_data = await token_manager.fetch_available_models(project_id)

                                    # 从 models_data 中提取该模型的配额信息
                                    reset_time = None
                                    remaining_fraction = 0
                                    models = models_data.get("models", {})
                                    if gemini_model in models:
                                        quota_info = models[gemini_model].get("quotaInfo", {})
                                        reset_time = quota_info.get("resetTime")
                                        remaining_fraction = quota_info.get("remainingFraction", 0)

                                    # 如果没有找到 resetTime，使用默认值（1小时后）
                                    if not reset_time:
                                        from datetime import datetime, timedelta, timezone
                                        reset_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace('+00:00', 'Z')
                                        logger.warning(f"未找到模型 {gemini_model} 的 resetTime，使用默认值: {reset_time}")

                                    # 更新账号的 creditsInfo
                                    credits_info = extract_credits_from_models_data(models_data)
                                    other = account.get("other") or {}
                                    if isinstance(other, str):
                                        import json
                                        try:
                                            other = json.loads(other)
                                        except json.JSONDecodeError:
                                            other = {}

                                    other["creditsInfo"] = credits_info
                                    update_account(account['id'], other=other)
                                    logger.info(f"已更新账号 {account['id']} 的配额信息")

                                    # 判断是速率限制还是配额用完
                                    if remaining_fraction > 0.03:
                                        # 配额充足，是速率限制（RPM/TPM）
                                        logger.warning(f"账号 {account['id']} 触发速率限制（RPM/TPM），剩余配额: {remaining_fraction:.2%}")
                                        raise HTTPException(
                                            status_code=429,
                                            detail=f"速率限制：请求过于频繁，请稍后重试（剩余配额: {remaining_fraction:.2%}）"
                                        )
                                    else:
                                        # 配额不足，真的用完了
                                        mark_model_exhausted(account['id'], gemini_model, reset_time)
                                        logger.warning(f"账号 {account['id']} 的模型 {gemini_model} 配额已用完（剩余: {remaining_fraction:.2%}），重置时间: {reset_time}")
                                        raise HTTPException(
                                            status_code=429,
                                            detail=f"配额已用完，重置时间: {reset_time}"
                                        )

                                except HTTPException:
                                    raise
                                except Exception as e:
                                    logger.error(f"处理 429 错误时出错: {e}", exc_info=True)

                            raise HTTPException(
                                status_code=response.status_code,
                                detail=f"Gemini API 错误: {error_str}"
                            )

                        # 返回字节流
                        logger.info("[HTTP] 开始迭代字节流")
                        chunk_count = 0
                        total_bytes = 0
                        async for chunk in response.aiter_bytes():
                            chunk_count += 1
                            if chunk:
                                total_bytes += len(chunk)
                                logger.info(f"[HTTP] Chunk {chunk_count}: {len(chunk)} 字节")
                                yield chunk
                            else:
                                logger.debug(f"[HTTP] Chunk {chunk_count}: 空 chunk")
                        logger.info(f"[HTTP] 字节流结束: 共 {chunk_count} 个 chunk, 总计 {total_bytes} 字节")

                except httpx.RequestError as e:
                    logger.error(f"请求错误: {e}")
                    raise HTTPException(status_code=502, detail=f"上游服务错误: {str(e)}")

        # 返回流式响应
        async def claude_stream():
            async for event in handle_gemini_stream(gemini_byte_stream(), model=claude_req.model):
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
        logger.error(f"处理 Gemini 请求时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


# 账号管理 API 端点
@app.get("/v2/accounts")
async def list_accounts(_: bool = Depends(verify_admin_key)):
    """列出所有账号"""
    accounts = list_all_accounts()
    return JSONResponse(content=accounts)


@app.get("/v2/accounts/{account_id}")
async def get_account_detail(account_id: str, _: bool = Depends(verify_admin_key)):
    """获取账号详情"""
    account = get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="账号不存在")
    return JSONResponse(content=account)


@app.post("/v2/accounts")
async def create_account_endpoint(body: AccountCreate, _: bool = Depends(verify_admin_key)):
    """创建新账号"""
    try:
        account = create_account(
            label=body.label,
            client_id=body.clientId,
            client_secret=body.clientSecret,
            refresh_token=body.refreshToken,
            access_token=body.accessToken,
            other=body.other,
            enabled=body.enabled if body.enabled is not None else True,
            account_type=body.type
        )
        return JSONResponse(content=account)
    except Exception as e:
        logger.error(f"创建账号失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建账号失败: {str(e)}")


@app.patch("/v2/accounts/{account_id}")
async def update_account_endpoint(account_id: str, body: AccountUpdate, _: bool = Depends(verify_admin_key)):
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
async def delete_account_endpoint(account_id: str, _: bool = Depends(verify_admin_key)):
    """删除账号"""
    success = delete_account(account_id)
    if not success:
        raise HTTPException(status_code=404, detail="账号不存在")
    return JSONResponse(content={"deleted": account_id})


@app.post("/v2/accounts/{account_id}/refresh")
async def manual_refresh_endpoint(account_id: str, _: bool = Depends(verify_admin_key)):
    """手动刷新账号 token"""
    try:
        account = get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail="账号不存在")

        account_type = account.get("type", "amazonq")

        if account_type == "gemini":
            # Gemini 账号刷新
            other = account.get("other") or {}
            if isinstance(other, str):
                import json
                try:
                    other = json.loads(other)
                except json.JSONDecodeError:
                    other = {}

            token_manager = GeminiTokenManager(
                client_id=account["clientId"],
                client_secret=account["clientSecret"],
                refresh_token=account["refreshToken"],
                api_endpoint=other.get("api_endpoint", "https://daily-cloudcode-pa.sandbox.googleapis.com")
            )
            await token_manager.refresh_access_token()

            # 更新数据库，保存 access_token 和过期时间
            other["token_expires_at"] = token_manager.token_expires_at.isoformat() if token_manager.token_expires_at else None
            refreshed_account = update_account(
                account_id=account_id,
                access_token=token_manager.access_token,
                other=other
            )
            return JSONResponse(content=refreshed_account)
        else:
            # Amazon Q 账号刷新
            refreshed_account = await refresh_account_token(account)
            return JSONResponse(content=refreshed_account)
    except TokenRefreshError as e:
        logger.error(f"刷新 token 失败: {e}")
        raise HTTPException(status_code=502, detail=f"刷新 token 失败: {str(e)}")
    except Exception as e:
        logger.error(f"刷新 token 失败: {e}")
        raise HTTPException(status_code=500, detail=f"刷新 token 失败: {str(e)}")


@app.post("/v2/accounts/refresh-all")
async def refresh_all_accounts(_: bool = Depends(verify_admin_key)):
    """批量刷新所有 Amazon Q 账号的 token，检测被封禁账号"""
    try:
        # 获取所有 Amazon Q 类型的账号
        all_accounts = list_all_accounts()
        amazonq_accounts = [acc for acc in all_accounts if acc.get('type', 'amazonq') == 'amazonq']

        if not amazonq_accounts:
            return JSONResponse(content={
                "success": True,
                "message": "没有 Amazon Q 账号需要刷新",
                "total": 0,
                "results": []
            })

        results = []
        success_count = 0
        failed_count = 0
        banned_count = 0

        logger.info(f"开始批量刷新 {len(amazonq_accounts)} 个 Amazon Q 账号")

        for account in amazonq_accounts:
            account_id = account.get('id')
            account_label = account.get('label', 'N/A')
            result = {
                "id": account_id,
                "label": account_label,
                "status": "unknown",
                "message": ""
            }

            try:
                # 尝试刷新 token
                refreshed_account = await refresh_account_token(account)
                result["status"] = "success"
                result["message"] = "Token 刷新成功"
                success_count += 1
                logger.info(f"账号 {account_id} ({account_label}) 刷新成功")

            except TokenRefreshError as e:
                error_msg = str(e)
                result["message"] = error_msg

                # 检测是否被封禁
                if "账号已被封禁" in error_msg or "invalid_grant" in error_msg.lower():
                    result["status"] = "banned"
                    banned_count += 1
                    logger.warning(f"账号 {account_id} ({account_label}) 已被封禁")
                else:
                    result["status"] = "failed"
                    failed_count += 1
                    logger.error(f"账号 {account_id} ({account_label}) 刷新失败: {error_msg}")

            except Exception as e:
                result["status"] = "error"
                result["message"] = f"未知错误: {str(e)}"
                failed_count += 1
                logger.error(f"账号 {account_id} ({account_label}) 刷新时发生错误: {e}")

            results.append(result)

        summary = {
            "success": True,
            "message": f"批量刷新完成: 成功 {success_count}, 失败 {failed_count}, 被封禁 {banned_count}",
            "total": len(amazonq_accounts),
            "success_count": success_count,
            "failed_count": failed_count,
            "banned_count": banned_count,
            "results": results
        }

        logger.info(f"批量刷新完成: 总计 {len(amazonq_accounts)}, 成功 {success_count}, 失败 {failed_count}, 被封禁 {banned_count}")
        return JSONResponse(content=summary)

    except Exception as e:
        logger.error(f"批量刷新账号失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量刷新失败: {str(e)}")


@app.get("/v2/accounts/{account_id}/quota")
async def get_account_quota(account_id: str, _: bool = Depends(verify_admin_key)):
    """获取 Gemini 账号配额信息"""
    try:
        account = get_account(account_id)
        if not account:
            raise HTTPException(status_code=404, detail="账号不存在")

        account_type = account.get("type", "amazonq")
        if account_type != "gemini":
            raise HTTPException(status_code=400, detail="只有 Gemini 账号支持配额查询")

        other = account.get("other") or {}
        token_manager = GeminiTokenManager(
            client_id=account["clientId"],
            client_secret=account["clientSecret"],
            refresh_token=account["refreshToken"],
            api_endpoint=other.get("api_endpoint", "https://daily-cloudcode-pa.sandbox.googleapis.com")
        )

        project_id = other.get("project") or await token_manager.get_project_id()
        models_data = await token_manager.fetch_available_models(project_id)

        return JSONResponse(content=models_data)
    except Exception as e:
        logger.error(f"获取配额信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配额信息失败: {str(e)}")


# 管理页面
@app.get("/admin", response_class=FileResponse)
async def admin_page(key: Optional[str] = None):
    """管理页面（需要鉴权）"""
    import os
    from pathlib import Path

    # 获取管理员密钥
    admin_key = os.getenv("ADMIN_KEY")

    # 如果设置了 ADMIN_KEY，则需要验证
    if admin_key:
        if not key or key != admin_key:
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝：需要有效的管理员密钥。请在 URL 中添加 ?key=YOUR_ADMIN_KEY"
            )

    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail="管理页面不存在")
    return FileResponse(str(frontend_path))


# Gemini 投喂站页面
@app.get("/donate", response_class=FileResponse)
async def donate_page():
    """Gemini 投喂站页面"""
    from pathlib import Path
    frontend_path = Path(__file__).parent / "frontend" / "donate.html"
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail="投喂站页面不存在")
    return FileResponse(str(frontend_path))


# OAuth 回调页面
@app.get("/oauth-callback-page", response_class=FileResponse)
async def oauth_callback_page():
    """OAuth 回调页面"""
    from pathlib import Path
    frontend_path = Path(__file__).parent / "frontend" / "oauth-callback-page.html"
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail="回调页面不存在")
    return FileResponse(str(frontend_path))


# Gemini OAuth 回调处理
@app.post("/api/gemini/oauth-callback")
async def gemini_oauth_callback_post(request: Request):
    """处理 Gemini OAuth 回调（POST 请求）"""
    try:
        body = await request.json()
        code = body.get("code")

        if not code:
            raise HTTPException(status_code=400, detail="缺少授权码")

        # 使用固定的 client credentials
        client_id = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
        client_secret = "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"

        # 交换授权码获取 tokens
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": "http://localhost:64312/oauth-callback"
                },
                headers={
                    'x-goog-api-client': 'gl-node/22.18.0',
                    'User-Agent': 'google-api-nodejs-client/10.3.0'
                }
            )

            if response.status_code != 200:
                error_msg = f"Token 交换失败: {response.text}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            tokens = response.json()
            refresh_token = tokens.get('refresh_token')

            if not refresh_token:
                raise HTTPException(status_code=400, detail="未获取到 refresh_token")

        # 测试账号可用性（获取项目 ID）
        token_manager = GeminiTokenManager(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            api_endpoint="https://daily-cloudcode-pa.sandbox.googleapis.com"
        )

        try:
            project_id = await token_manager.get_project_id()
            logger.info(f"账号验证成功，项目 ID: {project_id}")
        except Exception as e:
            error_msg = f"账号验证失败: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # 获取配额信息
        try:
            models_data = await token_manager.fetch_available_models(project_id)
            credits_info = extract_credits_from_models_data(models_data)
            reset_time = extract_reset_time_from_models_data(models_data)
        except Exception as e:
            logger.warning(f"获取配额信息失败: {e}")
            credits_info = {"models": {}, "summary": {"totalModels": 0, "averageRemaining": 0}}
            reset_time = None

        # 自动导入到数据库
        import uuid
        from datetime import datetime

        label = f"Gemini-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        other_data = {
            "project": project_id,
            "api_endpoint": "https://daily-cloudcode-pa.sandbox.googleapis.com",
            "creditsInfo": credits_info,
            "resetTime": reset_time
        }

        account = create_account(
            label=label,
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            access_token=tokens.get('access_token', ''),
            other=other_data,
            enabled=True,
            account_type="gemini"
        )
        logger.info(f"Gemini 账号已添加: {label}")

        return JSONResponse(content={"success": True, "message": "账号添加成功"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理 OAuth 回调失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Gemini OAuth 回调处理（GET 请求，保留兼容性）
@app.get("/api/gemini/oauth-callback")
async def gemini_oauth_callback(code: Optional[str] = None, error: Optional[str] = None):
    """处理 Gemini OAuth 回调"""
    if error:
        logger.error(f"OAuth 授权失败: {error}")
        return JSONResponse(
            status_code=400,
            content={"error": error, "message": "授权失败"}
        )

    if not code:
        raise HTTPException(status_code=400, detail="缺少授权码")

    from fastapi.responses import RedirectResponse
    try:
        # 使用固定的 client credentials
        client_id = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
        client_secret = "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"

        # 交换授权码获取 tokens
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": f"{get_base_url()}/api/gemini/oauth-callback"
                },
                headers={
                    'x-goog-api-client': 'gl-node/22.18.0',
                    'User-Agent': 'google-api-nodejs-client/10.3.0'
                }
            )

            if response.status_code != 200:
                error_msg = f"Token 交换失败: {response.text}"
                logger.error(error_msg)
                from urllib.parse import quote
                return JSONResponse(
                    status_code=302,
                    headers={"Location": f"/donate?error={quote(error_msg)}"}
                )

            tokens = response.json()
            refresh_token = tokens.get('refresh_token')

            if not refresh_token:
                error_msg = "未获取到 refresh_token"
                logger.error(error_msg)
                from urllib.parse import quote
                return JSONResponse(
                    status_code=302,
                    headers={"Location": f"/donate?error={quote(error_msg)}"}
                )

        # 测试账号可用性（获取项目 ID）
        token_manager = GeminiTokenManager(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            api_endpoint="https://daily-cloudcode-pa.sandbox.googleapis.com"
        )

        try:
            project_id = await token_manager.get_project_id()
            logger.info(f"账号验证成功，项目 ID: {project_id}")
        except Exception as e:
            error_msg = f"账号验证失败: {str(e)}"
            logger.error(error_msg)
            from urllib.parse import quote
            return JSONResponse(
                status_code=302,
                headers={"Location": f"/donate?error={quote(error_msg)}"}
            )

        # 获取配额信息
        try:
            models_data = await token_manager.fetch_available_models(project_id)
            credits_info = extract_credits_from_models_data(models_data)
            reset_time = extract_reset_time_from_models_data(models_data)
        except Exception as e:
            logger.warning(f"获取配额信息失败: {e}")
            credits_info = {"models": {}, "summary": {"totalModels": 0, "averageRemaining": 0}}
            reset_time = None

        # 自动导入到数据库
        import uuid
        from datetime import datetime

        label = f"Gemini-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        other_data = {
            "project": project_id,
            "api_endpoint": "https://daily-cloudcode-pa.sandbox.googleapis.com",
            "creditsInfo": credits_info,
            "resetTime": reset_time
        }

        account = create_account(
            label=label,
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            access_token=tokens.get('access_token', ''),
            other=other_data,
            enabled=True,
            account_type="gemini"
        )
        logger.info(f"Gemini 账号已添加: {label}")

        # 重定向回投喂站页面
        return RedirectResponse(url="/donate?success=true", status_code=302)

    except Exception as e:
        logger.error(f"处理 OAuth 回调失败: {e}")
        from urllib.parse import quote
        return RedirectResponse(url=f"/donate?error={quote(str(e))}", status_code=302)


# 获取 Gemini 账号列表和统计信息
@app.get("/api/gemini/accounts")
async def get_gemini_accounts():
    """获取 Gemini 账号列表和统计信息"""
    try:
        accounts = list_enabled_accounts(account_type="gemini")

        # 更新每个账号的配额信息
        updated_accounts = []
        total_credits = 0

        for account in accounts:
            try:
                other = account.get("other") or {}
                if isinstance(other, str):
                    import json
                    try:
                        other = json.loads(other)
                    except json.JSONDecodeError:
                        other = {}

                # 尝试刷新配额信息
                token_manager = GeminiTokenManager(
                    client_id=account.get("clientId", ""),
                    client_secret=account.get("clientSecret", ""),
                    refresh_token=account.get("refreshToken", ""),
                    api_endpoint=other.get("api_endpoint", "https://daily-cloudcode-pa.sandbox.googleapis.com")
                )

                project_id = other.get("project") or await token_manager.get_project_id()
                models_data = await token_manager.fetch_available_models(project_id)

                credits_info = extract_credits_from_models_data(models_data)

                # 更新 other 字段
                other["creditsInfo"] = credits_info
                other["project"] = project_id

                updated_accounts.append({
                    "id": account.get("id", ""),
                    "label": account.get("label", "未命名"),
                    "enabled": account.get("enabled", False),
                    "creditsInfo": credits_info,
                    "projectId": project_id,
                    "created_at": account.get("created_at")
                })

            except Exception as e:
                logger.error(f"更新账号 {account.get('id', 'unknown')} 配额信息失败: {e}")
                other = account.get("other") or {}
                if isinstance(other, str):
                    import json
                    try:
                        other = json.loads(other)
                    except json.JSONDecodeError:
                        other = {}

                updated_accounts.append({
                    "id": account.get("id", ""),
                    "label": account.get("label", "未命名"),
                    "enabled": account.get("enabled", False),
                    "credits": other.get("credits", 0),
                    "resetTime": other.get("resetTime"),
                    "projectId": other.get("project", "N/A"),
                    "created_at": account.get("created_at")
                })

        # 计算每个模型的总配额
        model_totals = {}
        for account in updated_accounts:
            credits_info = account.get("creditsInfo", {})
            models = credits_info.get("models", {})
            for model_id, model_info in models.items():
                if model_info.get("recommended"):
                    if model_id not in model_totals:
                        model_totals[model_id] = {
                            "displayName": model_info.get("displayName", model_id),
                            "totalRemaining": 0,
                            "accountCount": 0
                        }
                    model_totals[model_id]["totalRemaining"] += model_info.get("remainingFraction", 0)
                    model_totals[model_id]["accountCount"] += 1

        # 计算每个模型的平均配额百分比
        for model_id in model_totals:
            avg_fraction = model_totals[model_id]["totalRemaining"] / model_totals[model_id]["accountCount"]
            model_totals[model_id]["averagePercent"] = int(avg_fraction * 100)

        return JSONResponse(content={
            "modelTotals": model_totals,
            "activeCount": len([a for a in accounts if a.get("enabled")]),
            "totalCount": len(accounts),
            "accounts": updated_accounts
        })

    except Exception as e:
        logger.error(f"获取 Gemini 账号列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取账号列表失败: {str(e)}")


def get_base_url() -> str:
    """获取服务器基础 URL"""
    import os
    # 优先使用环境变量
    base_url = os.getenv("BASE_URL")
    if base_url:
        return base_url.rstrip('/')

    # 默认使用 localhost
    port = os.getenv("PORT", "8383")
    return f"http://localhost:{port}"


def extract_credits_from_models_data(models_data: dict) -> dict:
    """从模型数据中提取各个模型的 credits 信息

    返回格式:
    {
        "models": {
            "gemini-3-pro-high": {"remainingFraction": 0.21, "resetTime": "2025-11-20T16:12:51Z"},
            "claude-sonnet-4-5": {"remainingFraction": 0.81, "resetTime": "2025-11-20T16:18:40Z"},
            ...
        },
        "summary": {
            "totalModels": 5,
            "averageRemaining": 0.75
        }
    }
    """
    try:
        models = models_data.get("models", {})
        result = {
            "models": {},
            "summary": {
                "totalModels": 0,
                "averageRemaining": 0
            }
        }

        total_fraction = 0
        count = 0

        for model_id, model_info in models.items():
            quota_info = model_info.get("quotaInfo", {})
            remaining_fraction = quota_info.get("remainingFraction")
            reset_time = quota_info.get("resetTime")

            if remaining_fraction is not None:
                result["models"][model_id] = {
                    "displayName": model_info.get("displayName", model_id),
                    "remainingFraction": remaining_fraction,
                    "remainingPercent": int(remaining_fraction * 100),
                    "resetTime": reset_time,
                    "recommended": model_info.get("recommended", False)
                }
                total_fraction += remaining_fraction
                count += 1

        if count > 0:
            result["summary"]["totalModels"] = count
            result["summary"]["averageRemaining"] = total_fraction / count

        return result
    except Exception as e:
        logger.error(f"提取 credits 失败: {e}")
        return {"models": {}, "summary": {"totalModels": 0, "averageRemaining": 0}}


def extract_reset_time_from_models_data(models_data: dict) -> Optional[str]:
    """从模型数据中提取最早的重置时间

    返回 ISO 8601 格式的时间字符串
    """
    try:
        models = models_data.get("models", {})

        reset_times = []
        for model_id, model_info in models.items():
            quota_info = model_info.get("quotaInfo", {})
            reset_time = quota_info.get("resetTime")
            if reset_time:
                reset_times.append(reset_time)

        # 返回最早的重置时间
        if reset_times:
            return min(reset_times)

        return None
    except Exception as e:
        logger.error(f"提取重置时间失败: {e}")
        return None


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
        system=data.get("system"),
        thinking=data.get("thinking")
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
