from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.container import Container
from src.schema.auth.api_key_schema import ApiKey

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    return request.app.state.container


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    container: Container = Depends(get_container),
) -> ApiKey:
    """The calling product's key, from `Authorization: Bearer <key>`."""
    return container.auth_service().verify(credentials.credentials if credentials else None)
