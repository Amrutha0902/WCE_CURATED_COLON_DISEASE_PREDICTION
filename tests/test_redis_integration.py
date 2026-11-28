import importlib.util
import os
import re
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("WCE_REDIS_TEST") != "1",
    reason="Requires an isolated Redis integration environment",
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def redis_uri(monkeypatch):
    import redis

    uri = os.environ["WCE_TEST_REDIS_URI"]
    assert uri == "redis://127.0.0.1:6379/15", (
        "Only the dedicated local CI Redis database is allowed"
    )

    connection = redis.Redis.from_url(uri)
    assert connection.ping()

    monkeypatch.setenv("WCE_PRODUCTION", "true")
    monkeypatch.setenv("COOKIE_SECURE", "true")
    monkeypatch.setenv("WCE_ALLOW_SELF_SIGNUP", "false")
    monkeypatch.setenv("RATELIMIT_STORAGE_URI", uri)
    monkeypatch.setenv(
        "FLASK_SECRET_KEY",
        "disposable-redis-integration-test-secret-123456",
    )

    try:
        yield uri
    finally:
        connection.close()


def load_application(monkeypatch):
    name = "wce_redis_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.app.config.update(
        TESTING=False,
        PROPAGATE_EXCEPTIONS=False,
    )

    calls = []

    def reject_credentials(username, password):
        calls.append(username)
        return False

    monkeypatch.setattr(
        module, "verify_doctor_credentials", reject_credentials
    )
    return module, calls


def login_attempt(client, ip):
    options = {
        "base_url": "https://localhost",
        "environ_overrides": {"REMOTE_ADDR": ip},
    }

    page = client.get("/", **options)
    assert page.status_code == 200

    match = re.search(
        rb'name="csrf_token"\s+value="([^"]+)"',
        page.data,
    )
    assert match, "CSRF token missing"

    return client.post(
        "/login-home",
        data={
            "login_username": "absent",
            "login_password": "incorrect",
            "csrf_token": match.group(1).decode(),
        },
        headers={"Referer": "https://localhost/"},
        **options,
    )


def test_shared_redis_rate_limit(redis_uri, monkeypatch):
    first, first_calls = load_application(monkeypatch)
    second, second_calls = load_application(monkeypatch)

    ip = "198.18." + ".".join(
        str(value) for value in uuid.uuid4().bytes[:2]
    )

    client_a = first.app.test_client()
    client_b = second.app.test_client()

    for client in [client_a, client_b, client_a,
                   client_b, client_a]:
        response = login_attempt(client, ip)
        assert response.status_code == 200
        assert b"Invalid username or password" in response.data

    response = login_attempt(client_b, ip)
    assert response.status_code == 429

    assert len(first_calls) + len(second_calls) == 5


def test_redis_unavailable_fails_closed(redis_uri, monkeypatch):
    monkeypatch.setenv(
        "RATELIMIT_STORAGE_URI",
        "redis://127.0.0.1:1/15?socket_connect_timeout=1",
    )

    module, calls = load_application(monkeypatch)

    with module.app.test_client() as client:
        response = login_attempt(client, "198.19.0.1")
        assert response.status_code in (500, 503)

    assert not calls, "Authentication executed without rate limiting"


@pytest.mark.parametrize("variable,value,message", [
    (
        "RATELIMIT_STORAGE_URI", "memory://",
        "Production requires a Redis",
    ),
    (
        "COOKIE_SECURE", "false",
        "Production requires secure session cookies",
    ),
    (
        "WCE_ALLOW_SELF_SIGNUP", "true",
        "Production forbids unverified doctor self-signup",
    ),
])
def test_production_rejects_unsafe_configuration(
    redis_uri, monkeypatch, variable, value, message
):
    monkeypatch.setenv(variable, value)

    with pytest.raises(RuntimeError, match=message):
        load_application(monkeypatch)
