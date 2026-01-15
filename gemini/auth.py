"""
Gemini OAuth2 Token 管理模块
"""
import asyncio
import logging
import httpx
from typing import Dict, Optional
from datetime import datetime, timedelta
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# Antigravity API 常量
ANTIGRAVITY_API_USER_AGENT = "antigravity"


class GeminiTokenManager:
    """Gemini Token 管理器"""

    def __init__(self, client_id: str, client_secret: str, refresh_token: str, api_endpoint: str,
                 access_token: Optional[str] = None, token_expires_at: Optional[datetime] = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.api_endpoint = api_endpoint
        self.access_token: Optional[str] = access_token
        self.token_expires_at: Optional[datetime] = token_expires_at
        self.project_id: Optional[str] = None
        self.token_endpoint = "https://oauth2.googleapis.com/token"

    async def get_access_token(self) -> str:
        """获取有效的 access token，如果过期则自动刷新"""
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at - timedelta(minutes=5):
                logger.info("使用缓存的 Gemini access token")
                return self.access_token

        await self.refresh_access_token()
        return self.access_token

    async def refresh_access_token(self) -> None:
        """刷新 access token"""
        logger.info("正在刷新 Gemini access token...")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_endpoint,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": unquote(self.refresh_token)
                },
                timeout=20
            )

            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Token 刷新失败: {response.status_code} {error_text}")
                raise Exception(f"Token 刷新失败: {error_text}")

            token_data = response.json()
            self.access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3599)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

            logger.info(f"Token 刷新成功，有效期至 {self.token_expires_at}")

    def _get_api_headers(self, token: str) -> Dict[str, str]:
        """获取完整的 API 请求头"""
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": ANTIGRAVITY_API_USER_AGENT,
        }

    async def get_project_id(self) -> str:
        """获取 Gemini 项目 ID"""
        if self.project_id:
            return self.project_id

        token = await self.get_access_token()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_endpoint}/v1internal:loadCodeAssist",
                json={
                    "metadata": {
                        "ideType": "ANTIGRAVITY",
                        "platform": "PLATFORM_UNSPECIFIED",
                        "pluginType": "GEMINI"
                    }
                },
                headers=self._get_api_headers(token),
                timeout=30
            )

            if response.status_code != 200:
                error_text = response.text
                logger.error(f"获取项目 ID 失败: {response.status_code} {error_text}")
                raise Exception(f"获取项目 ID 失败: {error_text}")

            data = response.json()
            logger.info(f"loadCodeAssist 响应: {data}")
            self.project_id = data.get("cloudaicompanionProject")

            # 如果没有获取到项目 ID，尝试 onboard
            if not self.project_id:
                logger.info("loadCodeAssist 未返回项目 ID，尝试 onboardUser...")

                # 获取默认 tier ID
                tier_id = "legacy-tier"
                allowed_tiers = data.get("allowedTiers", [])
                for tier in allowed_tiers:
                    if isinstance(tier, dict) and tier.get("isDefault"):
                        tier_id = tier.get("id", tier_id)
                        break

                self.project_id = await self.onboard_user(tier_id)

            if not self.project_id:
                raise Exception("无法从响应中获取项目 ID")

            logger.info(f"获取到项目 ID: {self.project_id}")
            return self.project_id

    async def onboard_user(self, tier_id: str = "legacy-tier") -> Optional[str]:
        """
        用户注册获取项目 ID
        当 loadCodeAssist 未返回 cloudaicompanionProject 时调用
        """
        logger.info(f"正在执行 onboardUser，tier_id: {tier_id}")

        token = await self.get_access_token()
        max_attempts = 5

        request_body = {
            "tierId": tier_id,
            "metadata": {
                "ideType": "ANTIGRAVITY",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI"
            }
        }

        async with httpx.AsyncClient() as client:
            for attempt in range(1, max_attempts + 1):
                logger.debug(f"onboardUser 轮询尝试 {attempt}/{max_attempts}")

                try:
                    response = await client.post(
                        f"{self.api_endpoint}/v1internal:onboardUser",
                        json=request_body,
                        headers=self._get_api_headers(token),
                        timeout=30
                    )

                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"onboardUser 响应: {data}")

                        # 检查操作是否完成
                        if data.get("done"):
                            project_id = None
                            response_data = data.get("response", {})

                            # 尝试从不同格式中提取项目 ID
                            # 格式1: response.cloudaicompanionProject (字符串或对象)
                            cloud_project = response_data.get("cloudaicompanionProject")
                            if isinstance(cloud_project, dict):
                                project_id = cloud_project.get("id", "").strip()
                            elif isinstance(cloud_project, str):
                                project_id = cloud_project.strip()

                            # 格式2: 直接从顶层 data 获取
                            if not project_id:
                                cloud_project = data.get("cloudaicompanionProject")
                                if isinstance(cloud_project, dict):
                                    project_id = cloud_project.get("id", "").strip()
                                elif isinstance(cloud_project, str):
                                    project_id = cloud_project.strip()

                            if project_id:
                                logger.info(f"onboardUser 成功获取项目 ID: {project_id}")
                                return project_id
                            else:
                                logger.error(f"onboardUser 响应中无项目 ID，完整响应: {data}")
                                return None

                        # 未完成，等待后重试
                        logger.info(f"onboardUser 操作未完成，等待 2 秒后重试... (尝试 {attempt}/{max_attempts})")
                        await asyncio.sleep(2)
                        continue

                    else:
                        error_text = response.text[:200] if response.text else "Unknown error"
                        logger.error(f"onboardUser 请求失败: HTTP {response.status_code} - {error_text}")
                        return None

                except httpx.TimeoutException:
                    logger.warning(f"onboardUser 请求超时，尝试 {attempt}/{max_attempts}")
                    if attempt < max_attempts:
                        await asyncio.sleep(2)
                        continue
                    return None
                except Exception as e:
                    logger.error(f"onboardUser 请求异常: {e}")
                    return None

        logger.warning("onboardUser 达到最大尝试次数，未获取到项目 ID")
        return None

    async def get_auth_headers(self) -> Dict[str, str]:
        """获取认证请求头"""
        token = await self.get_access_token()
        return {
            "Authorization": f"Bearer {token}"
        }

    async def fetch_available_models(self, project_id: str) -> Dict:
        """获取可用模型和配额信息"""
        token = await self.get_access_token()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_endpoint}/v1internal:fetchAvailableModels",
                json={"project": project_id},
                headers=self._get_api_headers(token),
                timeout=30
            )

            if response.status_code != 200:
                error_text = response.text
                logger.error(f"获取模型列表失败: {response.status_code} {error_text}")
                raise Exception(f"获取模型列表失败: {error_text}")

            return response.json()