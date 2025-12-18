"""
Antigravity 模块
Google Antigravity (Cloud Code Assist) API 集成
"""

from .constants import (
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_CLIENT_SECRET,
    ANTIGRAVITY_REDIRECT_URI,
    ANTIGRAVITY_SCOPES,
    ANTIGRAVITY_ENDPOINTS,
    ANTIGRAVITY_LOAD_ENDPOINTS,
    ANTIGRAVITY_HEADERS,
    ANTIGRAVITY_DEFAULT_PROJECT_ID,
    GOOGLE_TOKEN_ENDPOINT,
    GOOGLE_USERINFO_ENDPOINT,
    GOOGLE_AUTH_ENDPOINT,
    ANTIGRAVITY_STREAM_PATH,
    ANTIGRAVITY_GENERATE_PATH,
    ANTIGRAVITY_LOAD_PATH,
    ANTHROPIC_THINKING_BETA,
)

from .auth import (
    generate_pkce_pair,
    build_auth_url,
    generate_auth_url,
    exchange_code,
    fetch_project_id,
    refresh_access_token,
    get_valid_access_token,
    is_token_expired,
    AntigravityTokenRefreshError,
    AntigravityAuthError,
)

from .converter import (
    convert_claude_to_antigravity,
    build_antigravity_request_url,
    build_antigravity_headers,
    is_thinking_capable_model,
)

from .handler import (
    handle_antigravity_stream,
    format_sse_event,
)

from .cache import (
    cache_signature,
    get_cached_signature,
    clear_signature_cache,
    get_cache_stats,
)

__all__ = [
    # Constants
    "ANTIGRAVITY_CLIENT_ID",
    "ANTIGRAVITY_CLIENT_SECRET",
    "ANTIGRAVITY_REDIRECT_URI",
    "ANTIGRAVITY_SCOPES",
    "ANTIGRAVITY_ENDPOINTS",
    "ANTIGRAVITY_LOAD_ENDPOINTS",
    "ANTIGRAVITY_HEADERS",
    "ANTIGRAVITY_DEFAULT_PROJECT_ID",
    "GOOGLE_TOKEN_ENDPOINT",
    "GOOGLE_USERINFO_ENDPOINT",
    "GOOGLE_AUTH_ENDPOINT",
    "ANTIGRAVITY_STREAM_PATH",
    "ANTIGRAVITY_GENERATE_PATH",
    "ANTIGRAVITY_LOAD_PATH",
    "ANTHROPIC_THINKING_BETA",
    # Auth
    "generate_pkce_pair",
    "build_auth_url",
    "generate_auth_url",
    "exchange_code",
    "fetch_project_id",
    "refresh_access_token",
    "get_valid_access_token",
    "is_token_expired",
    "AntigravityTokenRefreshError",
    "AntigravityAuthError",
    # Converter
    "convert_claude_to_antigravity",
    "build_antigravity_request_url",
    "build_antigravity_headers",
    "is_thinking_capable_model",
    # Handler
    "handle_antigravity_stream",
    "format_sse_event",
    # Cache
    "cache_signature",
    "get_cached_signature",
    "clear_signature_cache",
    "get_cache_stats",
]
