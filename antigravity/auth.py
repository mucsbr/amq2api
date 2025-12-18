"""
Antigravity 认证模块
包含 PKCE OAuth 流程、Token 刷新和项目发现
"""

import base64
import hashlib
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlencode

import httpx

from .constants import (
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_CLIENT_SECRET,
    ANTIGRAVITY_REDIRECT_URI,
    ANTIGRAVITY_SCOPES,
    ANTIGRAVITY_ENDPOINTS,
    ANTIGRAVITY_LOAD_ENDPOINTS,
    ANTIGRAVITY_HEADERS,
    GOOGLE_TOKEN_ENDPOINT,
    GOOGLE_USERINFO_ENDPOINT,
    GOOGLE_AUTH_ENDPOINT,
    ANTIGRAVITY_LOAD_PATH,
)

logger = logging.getLogger(__name__)


class AntigravityTokenRefreshError(Exception):
    """Token 刷新错误"""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        description: Optional[str] = None,
        status: int = 0,
    ):
        super().__init__(message)
        self.code = code
        self.description = description
        self.status = status


class AntigravityAuthError(Exception):
    """认证错误"""
    pass


def generate_pkce_pair() -> Tuple[str, str]:
    """
    生成 PKCE verifier 和 challenge

    Returns:
        (verifier, challenge) 元组
    """
    # 生成随机 verifier（32 字节 -> 43 字符 base64url）
    verifier = secrets.token_urlsafe(32)

    # 计算 challenge (SHA256 hash of verifier, base64url encoded)
    challenge_bytes = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).decode().rstrip('=')

    return verifier, challenge


def encode_state(verifier: str, project_id: str = "") -> str:
    """
    编码 OAuth state 参数

    Args:
        verifier: PKCE verifier
        project_id: 可选的项目 ID

    Returns:
        base64url 编码的 state 字符串
    """
    payload = {"verifier": verifier, "projectId": project_id}
    json_bytes = json.dumps(payload).encode('utf-8')
    return base64.urlsafe_b64encode(json_bytes).decode().rstrip('=')


def decode_state(state: str) -> Dict[str, str]:
    """
    解码 OAuth state 参数

    Args:
        state: base64url 编码的 state 字符串

    Returns:
        包含 verifier 和 projectId 的字典
    """
    # 添加 padding
    padded = state + '=' * (4 - len(state) % 4)
    # 替换 URL 安全字符
    normalized = padded.replace('-', '+').replace('_', '/')

    try:
        json_bytes = base64.b64decode(normalized)
        payload = json.loads(json_bytes.decode('utf-8'))

        if not isinstance(payload.get('verifier'), str):
            raise ValueError("Missing PKCE verifier in state")

        return {
            "verifier": payload["verifier"],
            "projectId": payload.get("projectId", ""),
        }
    except Exception as e:
        raise AntigravityAuthError(f"Failed to decode state: {e}")


def build_auth_url(verifier: str, project_id: str = "") -> str:
    """
    构建 Google OAuth 授权 URL

    Args:
        verifier: PKCE verifier
        project_id: 可选的项目 ID

    Returns:
        完整的授权 URL
    """
    # 基于传入的 verifier 计算 challenge
    challenge_bytes = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).decode().rstrip('=')

    params = {
        "client_id": ANTIGRAVITY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
        "scope": " ".join(ANTIGRAVITY_SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": encode_state(verifier, project_id),
        "access_type": "offline",
        "prompt": "consent",
    }

    return f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}"


def generate_auth_url(project_id: str = "") -> Dict[str, str]:
    """
    生成完整的 OAuth 授权信息

    Args:
        project_id: 可选的项目 ID

    Returns:
        包含 url, verifier, state 的字典
    """
    verifier, _ = generate_pkce_pair()
    url = build_auth_url(verifier, project_id)
    state = encode_state(verifier, project_id)

    return {
        "url": url,
        "verifier": verifier,
        "state": state,
        "project_id": project_id,
    }


async def fetch_project_id(access_token: str) -> str:
    """
    通过 loadCodeAssist API 获取项目 ID

    Args:
        access_token: Google OAuth access token

    Returns:
        项目 ID，如果获取失败则返回空字符串
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": ANTIGRAVITY_HEADERS["Client-Metadata"],
    }

    body = {
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    }

    # 尝试所有端点
    endpoints = list(set(ANTIGRAVITY_LOAD_ENDPOINTS + ANTIGRAVITY_ENDPOINTS))
    errors = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for endpoint in endpoints:
            try:
                url = f"{endpoint}{ANTIGRAVITY_LOAD_PATH}"
                response = await client.post(url, headers=headers, json=body)

                if response.status_code != 200:
                    error_text = response.text[:200] if response.text else ""
                    errors.append(f"loadCodeAssist {response.status_code} at {endpoint}: {error_text}")
                    continue

                data = response.json()

                # 尝试提取项目 ID
                if isinstance(data.get("cloudaicompanionProject"), str):
                    return data["cloudaicompanionProject"]

                if isinstance(data.get("cloudaicompanionProject"), dict):
                    project_id = data["cloudaicompanionProject"].get("id")
                    if project_id:
                        return project_id

                errors.append(f"loadCodeAssist missing project id at {endpoint}")

            except Exception as e:
                errors.append(f"loadCodeAssist error at {endpoint}: {str(e)}")

    if errors:
        logger.warning(f"Failed to resolve Antigravity project: {'; '.join(errors)}")

    return ""


async def exchange_code(code: str, state: str) -> Dict[str, Any]:
    """
    用授权码交换 Token

    Args:
        code: OAuth 授权码
        state: OAuth state 参数

    Returns:
        包含 token 信息的字典
    """
    # 解码 state 获取 verifier
    state_data = decode_state(state)
    verifier = state_data["verifier"]
    project_id = state_data.get("projectId", "")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 交换 token
        response = await client.post(
            GOOGLE_TOKEN_ENDPOINT,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "client_id": ANTIGRAVITY_CLIENT_ID,
                "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
                "code_verifier": verifier,
            },
        )

        if response.status_code != 200:
            error_text = response.text
            raise AntigravityAuthError(f"Token exchange failed: {error_text}")

        token_data = response.json()
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        if not refresh_token:
            raise AntigravityAuthError("Missing refresh token in response")

        # 获取用户信息
        email = None
        try:
            user_response = await client.get(
                f"{GOOGLE_USERINFO_ENDPOINT}?alt=json",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if user_response.status_code == 200:
                user_info = user_response.json()
                email = user_info.get("email")
        except Exception as e:
            logger.warning(f"Failed to fetch user info: {e}")

        # 获取项目 ID（如果没有提供）
        effective_project_id = project_id
        if not effective_project_id:
            effective_project_id = await fetch_project_id(access_token)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": expires_in,
            "expires_at": (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
            "email": email,
            "project_id": effective_project_id,
        }


async def refresh_access_token(account: Dict[str, Any]) -> Dict[str, Any]:
    """
    刷新 access token

    Args:
        account: 账号信息字典，需要包含 refreshToken

    Returns:
        更新后的账号信息

    Raises:
        AntigravityTokenRefreshError: 刷新失败时抛出
    """
    refresh_token = account.get("refreshToken")
    if not refresh_token:
        raise AntigravityTokenRefreshError("Missing refresh token", status=400)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GOOGLE_TOKEN_ENDPOINT,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": ANTIGRAVITY_CLIENT_ID,
                "client_secret": ANTIGRAVITY_CLIENT_SECRET,
            },
        )

        if response.status_code != 200:
            error_text = response.text
            error_code = None
            error_desc = None

            try:
                error_data = response.json()
                error_code = error_data.get("error")
                error_desc = error_data.get("error_description")
            except Exception:
                pass

            message = f"Token refresh failed ({response.status_code})"
            if error_code:
                message += f": {error_code}"
            if error_desc:
                message += f" - {error_desc}"

            logger.warning(f"[Antigravity OAuth] {message}")

            raise AntigravityTokenRefreshError(
                message,
                code=error_code,
                description=error_desc or error_text,
                status=response.status_code,
            )

        token_data = response.json()
        access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)
        new_refresh_token = token_data.get("refresh_token", refresh_token)

        # 更新账号信息
        other = account.get("other", {})
        if isinstance(other, str):
            try:
                other = json.loads(other)
            except Exception:
                other = {}

        other["token_expires_at"] = (datetime.now() + timedelta(seconds=expires_in)).isoformat()

        return {
            **account,
            "accessToken": access_token,
            "refreshToken": new_refresh_token,
            "other": other,
        }


def is_token_expired(account: Dict[str, Any], buffer_seconds: int = 60) -> bool:
    """
    检查 token 是否过期

    Args:
        account: 账号信息字典
        buffer_seconds: 提前刷新的缓冲时间（秒）

    Returns:
        True 如果 token 已过期或即将过期
    """
    other = account.get("other", {})
    if isinstance(other, str):
        try:
            other = json.loads(other)
        except Exception:
            return True

    expires_at_str = other.get("token_expires_at")
    if not expires_at_str:
        return True

    try:
        expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
        # 移除时区信息进行比较
        if expires_at.tzinfo:
            expires_at = expires_at.replace(tzinfo=None)
        return datetime.now() >= expires_at - timedelta(seconds=buffer_seconds)
    except Exception:
        return True


async def get_valid_access_token(account: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    获取有效的 access token，如果过期则自动刷新

    Args:
        account: 账号信息字典

    Returns:
        (access_token, updated_account) 元组
    """
    # 检查是否有现有的 access token 且未过期
    access_token = account.get("accessToken")
    if access_token and not is_token_expired(account):
        return access_token, account

    # 需要刷新
    logger.info(f"[Antigravity] Refreshing token for account {account.get('id', 'unknown')[:8]}...")
    updated_account = await refresh_access_token(account)
    return updated_account["accessToken"], updated_account
