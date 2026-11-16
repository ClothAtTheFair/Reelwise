"""Deployment configuration. Set APP_ENV=production only behind a TLS reverse proxy."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def production() -> bool:
    value = os.getenv('APP_ENV', 'development').lower()
    if value not in {'development', 'test', 'production'}:
        raise RuntimeError('APP_ENV must be development, test, or production')
    return value == 'production'


def integer(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f'{name} must be an integer') from exc
    if value < minimum:
        raise RuntimeError(f'{name} must be at least {minimum}')
    return value


def validate_production() -> None:
    if not production():
        return
    # Fail closed: an accidentally exposed public development configuration is unsafe.
    if os.getenv('FRONTEND_ORIGIN', '').rstrip('/').startswith('https://') is False:
        raise RuntimeError('Production requires FRONTEND_ORIGIN=https://your-domain')
    if 'APP_DB_PATH' not in os.environ or 'MODEL_PATH' not in os.environ:
        raise RuntimeError('Production requires absolute APP_DB_PATH and MODEL_PATH')
    for key in ('APP_DB_PATH', 'MODEL_PATH'):
        if not Path(os.environ[key]).is_absolute():
            raise RuntimeError(f'{key} must be an absolute path in production')
    integer('AUTH_TOKEN_TTL_HOURS', 24)
    integer('AUTH_RATE_LIMIT_ATTEMPTS', 8)
    integer('AUTH_RATE_LIMIT_WINDOW_SECONDS', 900)
