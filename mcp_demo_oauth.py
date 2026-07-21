"""In-memory OAuth provider for ChatGPT / remote MCP connectors (demo only).

ChatGPT Developer mode often attempts Dynamic Client Registration (DCR).
This provider auto-approves clients and issues tokens so registration succeeds
without Auth0/Google. Not for production — tokens live in process memory.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


class DemoOAuthProvider:
    """Minimal OAuthAuthorizationServerProvider with DCR + auto-approve."""

    def __init__(self, *, token_ttl_seconds: int = 3600 * 8) -> None:
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._codes: dict[str, AuthorizationCode] = {}
        self._access: dict[str, AccessToken] = {}
        self._refresh: dict[str, RefreshToken] = {}
        self._token_ttl = token_ttl_seconds

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        # Demo: skip login UI — immediately issue an authorization code.
        code = secrets.token_urlsafe(32)
        self._codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + 600,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            subject="demo-user",
        )
        return construct_redirect_uri(
            str(params.redirect_uri),
            code=code,
            state=params.state,
        )

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        auth = self._codes.get(authorization_code)
        if auth is None or auth.client_id != client.client_id:
            return None
        if auth.expires_at < time.time():
            self._codes.pop(authorization_code, None)
            return None
        return auth

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        if authorization_code.code not in self._codes:
            raise TokenError(error="invalid_grant", error_description="Unknown code")
        self._codes.pop(authorization_code.code, None)
        return self._issue_tokens(
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
            subject=authorization_code.subject,
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        token = self._refresh.get(refresh_token)
        if token is None or token.client_id != client.client_id:
            return None
        if token.expires_at is not None and token.expires_at < int(time.time()):
            self._refresh.pop(refresh_token, None)
            return None
        return token

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        if refresh_token.token not in self._refresh:
            raise TokenError(error="invalid_grant", error_description="Unknown refresh token")
        self._refresh.pop(refresh_token.token, None)
        use_scopes = scopes or refresh_token.scopes
        return self._issue_tokens(
            client_id=client.client_id,
            scopes=use_scopes,
            resource=None,
            subject=refresh_token.subject,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        access = self._access.get(token)
        if access is None:
            return None
        if access.expires_at is not None and access.expires_at < int(time.time()):
            self._access.pop(token, None)
            return None
        return access

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self._access.pop(token.token, None)
        else:
            self._refresh.pop(token.token, None)

    def _issue_tokens(
        self,
        *,
        client_id: str,
        scopes: list[str],
        resource: str | None,
        subject: str | None,
    ) -> OAuthToken:
        now = int(time.time())
        access_raw = secrets.token_urlsafe(32)
        refresh_raw = secrets.token_urlsafe(32)
        access = AccessToken(
            token=access_raw,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + self._token_ttl,
            resource=resource,
            subject=subject,
        )
        refresh = RefreshToken(
            token=refresh_raw,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + self._token_ttl * 4,
            subject=subject,
        )
        self._access[access_raw] = access
        self._refresh[refresh_raw] = refresh
        return OAuthToken(
            access_token=access_raw,
            token_type="bearer",
            expires_in=self._token_ttl,
            scope=" ".join(scopes) if scopes else None,
            refresh_token=refresh_raw,
        )
