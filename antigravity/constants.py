"""
Antigravity API 常量定义
包含 OAuth 凭证、API 端点和请求头配置
"""

# OAuth 凭证（Google Cloud Code Assist）
ANTIGRAVITY_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
ANTIGRAVITY_CLIENT_SECRET = "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"
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"

# OAuth 作用域
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

# API 端点（故障转移顺序：daily → autopush → prod）
ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_AUTOPUSH = "https://autopush-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"

ANTIGRAVITY_ENDPOINTS = [
    ANTIGRAVITY_ENDPOINT_DAILY,      # 主端点
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,   # 备用 1
    ANTIGRAVITY_ENDPOINT_PROD,       # 备用 2 (生产)
]

# 项目发现端点顺序（prod 优先）
ANTIGRAVITY_LOAD_ENDPOINTS = [
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
]

# 默认项目 ID（用于业务/工作空间账户）
ANTIGRAVITY_DEFAULT_PROJECT_ID = "rising-fact-p41fc"

# 请求头
ANTIGRAVITY_HEADERS = {
    "User-Agent": "antigravity/1.11.5 windows/amd64",
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}

# Google OAuth 端点
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v1/userinfo"
GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"

# API 路径
ANTIGRAVITY_STREAM_PATH = "/v1internal:streamGenerateContent"
ANTIGRAVITY_GENERATE_PATH = "/v1internal:generateContent"
ANTIGRAVITY_LOAD_PATH = "/v1internal:loadCodeAssist"
ANTIGRAVITY_ONBOARD_PATH = "/v1internal:onboardUser"

# 思考模式 Beta 头
ANTHROPIC_THINKING_BETA = "interleaved-thinking-2025-05-14"
