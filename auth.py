"""
认证模块
负责 Token 刷新和管理（支持多账号）
"""
import httpx
import logging
import uuid
from typing import Dict, Any, Tuple, Optional
from account_manager import get_random_account, update_account_tokens, update_refresh_status

logger = logging.getLogger(__name__)


class TokenRefreshError(Exception):
    """Token 刷新失败异常"""
    pass


class NoAccountAvailableError(Exception):
    """无可用账号异常"""
    pass


async def refresh_account_token(account: Dict[str, Any]) -> Dict[str, Any]:
    """
    刷新指定账号的 access_token

    Args:
        account: 账号信息字典

    Returns:
        Dict[str, Any]: 更新后的账号信息

    Raises:
        TokenRefreshError: 刷新失败时抛出异常
    """
    account_id = account["id"]

    if not account.get("clientId") or not account.get("clientSecret") or not account.get("refreshToken"):
        logger.error(f"账号 {account_id} 缺少必需的刷新凭证")
        update_refresh_status(account_id, "failed_missing_credentials")
        raise TokenRefreshError("账号缺少 clientId/clientSecret/refreshToken")

    try:
        logger.info(f"开始刷新账号 {account_id} 的 access_token")

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            payload = {
                "grantType": "refresh_token",
                "refreshToken": account["refreshToken"],
                "clientId": account["clientId"],
                "clientSecret": account["clientSecret"]
            }

            headers = {
                "Content-Type": "application/json",
                "User-Agent": "aws-sdk-rust/1.3.9 os/macos lang/rust/1.87.0",
                "X-Amz-User-Agent": "aws-sdk-rust/1.3.9 ua/2.1 api/ssooidc/1.88.0 os/macos lang/rust/1.87.0 m/E app/AmazonQ-For-CLI",
                "Amz-Sdk-Request": "attempt=1; max=3",
                "Amz-Sdk-Invocation-Id": str(uuid.uuid4()),
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br"
            }

            response = await http_client.post(
                "https://oidc.us-east-1.amazonaws.com/token",
                json=payload,
                headers=headers
            )

            response.raise_for_status()
            response_data = response.json()

            new_access_token = response_data.get("accessToken")
            new_refresh_token = response_data.get("refreshToken", account.get("refreshToken"))

            if not new_access_token:
                raise TokenRefreshError("响应中缺少 accessToken")

            # 更新数据库
            updated_account = update_account_tokens(
                account_id,
                new_access_token,
                new_refresh_token,
                "success"
            )

            logger.info(f"账号 {account_id} Token 刷新成功")
            return updated_account

    except httpx.HTTPStatusError as e:
        logger.error(f"账号 {account_id} Token 刷新失败 - HTTP 错误: {e.response.status_code}")
        update_refresh_status(account_id, f"failed_{e.response.status_code}")
        raise TokenRefreshError(f"HTTP 错误: {e.response.status_code}") from e
    except httpx.RequestError as e:
        logger.error(f"账号 {account_id} Token 刷新失败 - 网络错误: {str(e)}")
        update_refresh_status(account_id, "failed_network")
        raise TokenRefreshError(f"网络错误: {str(e)}") from e
    except Exception as e:
        logger.error(f"账号 {account_id} Token 刷新失败 - 未知错误: {str(e)}")
        update_refresh_status(account_id, "failed_unknown")
        raise TokenRefreshError(f"未知错误: {str(e)}") from e


async def get_account_with_token() -> Tuple[Optional[Dict[str, Any]], str]:
    """
    获取一个随机账号及其有效的 access_token
    如果数据库中没有账号，回退到使用 .env 配置（向后兼容）

    Returns:
        Tuple[Optional[Dict[str, Any]], str]: (账号信息或None, access_token)

    Raises:
        NoAccountAvailableError: 无可用账号且 .env 配置不完整
        TokenRefreshError: Token 刷新失败
    """
    account = get_random_account()

    # 如果数据库中有账号，使用多账号模式
    if account:
        access_token = account.get("accessToken")
        token_expired = False

        # 检查 JWT token 是否过期
        if access_token:
            try:
                import base64
                import json
                from datetime import datetime

                parts = access_token.split('.')
                if len(parts) == 3:
                    payload = base64.urlsafe_b64decode(parts[1] + '==')
                    token_data = json.loads(payload)
                    exp = token_data.get('exp')
                    if exp:
                        exp_time = datetime.fromtimestamp(exp)
                        if datetime.now() >= exp_time:
                            token_expired = True
                            logger.info(f"账号 {account['id']} 的 accessToken 已过期")
            except Exception as e:
                logger.warning(f"解析 JWT token 失败: {e}")

        # 如果没有 access_token 或 token 已过期，尝试刷新
        if not access_token or token_expired:
            logger.info(f"账号 {account['id']} 需要刷新 token")
            account = await refresh_account_token(account)
            access_token = account.get("accessToken")

            if not access_token:
                raise TokenRefreshError("刷新后仍无法获取 accessToken")

        return account, access_token

    # 回退到单账号模式（使用 .env 配置）
    logger.info("数据库中没有账号，回退到单账号模式（使用 .env 配置）")
    from config import read_global_config

    try:
        config = await read_global_config()

        # 检查 token 是否过期
        if config.is_token_expired():
            logger.info("Token 已过期，开始刷新")
            await refresh_legacy_token()
            config = await read_global_config()

        if not config.access_token:
            raise NoAccountAvailableError("没有可用账号且 .env 配置不完整")

        return None, config.access_token
    except Exception as e:
        logger.error(f"单账号模式失败: {e}")
        raise NoAccountAvailableError("没有可用账号且 .env 配置不完整") from e


async def refresh_legacy_token() -> bool:
    """
    刷新单账号模式的 token（向后兼容）

    Returns:
        bool: 刷新成功返回 True
    """
    from config import read_global_config, update_global_config

    config = await read_global_config()

    try:
        logger.info("开始刷新单账号模式的 access_token")

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            payload = {
                "grantType": "refresh_token",
                "refreshToken": config.refresh_token,
                "clientId": config.client_id,
                "clientSecret": config.client_secret
            }

            headers = {
                "Content-Type": "application/json",
                "User-Agent": "aws-sdk-rust/1.3.9 os/macos lang/rust/1.87.0",
                "X-Amz-User-Agent": "aws-sdk-rust/1.3.9 ua/2.1 api/ssooidc/1.88.0 os/macos lang/rust/1.87.0 m/E app/AmazonQ-For-CLI",
                "Amz-Sdk-Request": "attempt=1; max=3",
                "Amz-Sdk-Invocation-Id": str(uuid.uuid4()),
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br"
            }

            response = await http_client.post(
                config.token_endpoint,
                json=payload,
                headers=headers
            )

            response.raise_for_status()
            response_data = response.json()

            new_access_token = response_data.get("accessToken")
            new_refresh_token = response_data.get("refreshToken")
            expires_in = response_data.get("expiresIn")

            if not new_access_token:
                raise TokenRefreshError("响应中缺少 accessToken")

            await update_global_config(
                access_token=new_access_token,
                refresh_token=new_refresh_token if new_refresh_token else None,
                expires_in=int(expires_in) if expires_in else 3600
            )

            logger.info("单账号模式 Token 刷新成功")
            return True

    except httpx.HTTPStatusError as e:
        logger.error(f"单账号模式 Token 刷新失败 - HTTP 错误: {e.response.status_code}")
        raise TokenRefreshError(f"HTTP 错误: {e.response.status_code}") from e
    except Exception as e:
        logger.error(f"单账号模式 Token 刷新失败: {str(e)}")
        raise TokenRefreshError(f"刷新失败: {str(e)}") from e


async def get_auth_headers_for_account(account: Dict[str, Any]) -> Dict[str, str]:
    """
    为指定账号获取认证头

    Args:
        account: 账号信息字典

    Returns:
        Dict[str, str]: 认证头

    Raises:
        TokenRefreshError: Token 刷新失败时抛出异常
    """
    access_token = account.get("accessToken")
    token_expired = False

    # 检查 JWT token 是否过期
    if access_token:
        try:
            import base64
            import json
            from datetime import datetime

            parts = access_token.split('.')
            if len(parts) == 3:
                payload = base64.urlsafe_b64decode(parts[1] + '==')
                token_data = json.loads(payload)
                exp = token_data.get('exp')
                if exp:
                    exp_time = datetime.fromtimestamp(exp)
                    if datetime.now() >= exp_time:
                        token_expired = True
                        logger.info(f"账号 {account['id']} 的 accessToken 已过期")
        except Exception as e:
            logger.warning(f"解析 JWT token 失败: {e}")

    # 如果没有 access_token 或 token 已过期，尝试刷新
    if not access_token or token_expired:
        logger.info(f"账号 {account['id']} 需要刷新 token")
        account = await refresh_account_token(account)
        access_token = account.get("accessToken")

        if not access_token:
            raise TokenRefreshError("刷新后仍无法获取 accessToken")

    return {
        "Authorization": f"Bearer {access_token}"
    }


async def get_auth_headers_with_retry() -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
    """
    获取认证头，支持 401/403 重试机制
    支持多账号模式和单账号模式（向后兼容）

    Returns:
        Tuple[Optional[Dict[str, Any]], Dict[str, str]]: (账号信息或None, 认证头)
    """
    account, access_token = await get_account_with_token()

    return account, {
        "Authorization": f"Bearer {access_token}"
    }