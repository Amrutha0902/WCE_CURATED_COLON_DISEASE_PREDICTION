"""Flask request-level security tests using isolated model/database substitutes.

These tests exercise Flask routes/CSRF/session/rate-limits. They are NOT real MySQL
or neural-network integration tests.
"""
import importlib.util
import io
import re
import sys
import types
from pathlib import Path

import pytest
from werkzeug.security import check_password_hash

ROOT = Path(__file__).resolve().parents[1]


class FakeCursor:
    def __init__(self, users):
        self.users = users
        self.result = None

    def execute(self, query, params=()):
        sql = " ".join(query.split())
        if sql.startswith("SELECT * FROM users WHERE username"):
            self.result = (params[0],) if params[0] in self.users else None
        elif sql.startswith("SELECT password FROM users"):
            entry = self.users.get(params[0])
            self.result = (entry["password"],) if entry and entry["role"] == params[1] else None
        elif sql.startswith("INSERT INTO users"):
            name, hashed, role = params
            self.users[name] = {"password": hashed, "role": role}
            self.result = None
        else:
            raise AssertionError(f"Unexpected SQL in test: {sql[:100]}")

    def fetchone(self):
        return self.result

    def close(self):
        pass


class FakeConnection:
    def __init__(self, users):
        self.users = users

    def cursor(self):
        return FakeCursor(self.users)

    def commit(self):
        pass

    def close(self):
        pass


@pytest.fixture
def application(monkeypatch):
    # Installing entire JAX/Keras/DB/ML model is unnecessary for isolated HTTP
    # tests; stub external integration boundaries before importing the real app.
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-testing-only-secret-not-for-production-12345")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("WCE_ALLOW_SELF_SIGNUP", "true")

    keras = types.ModuleType("keras")
    keras.__path__ = []
    preprocessing = types.ModuleType("keras.preprocessing")
    preprocessing.__path__ = []
    image = types.ModuleType("keras.preprocessing.image")
    image.load_img = lambda *a, **k: None
    image.img_to_array = lambda *a, **k: None
    models = types.ModuleType("keras.models")
    models.load_model = lambda path, **kwargs: object()
    for name, mod in [
        ("keras", keras), ("keras.preprocessing", preprocessing),
        ("keras.preprocessing.image", image), ("keras.models", models)
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    mysql = types.ModuleType("mysql")
    mysql.__path__ = []
    connector = types.ModuleType("mysql.connector")
    class DatabaseError(Exception):
        pass
    connector.Error = DatabaseError
    connector.connect = lambda **kwargs: None
    mysql.connector = connector
    monkeypatch.setitem(sys.modules, "mysql", mysql)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector)

    pandas = types.ModuleType("pandas")
    monkeypatch.setitem(sys.modules, "pandas", pandas)

    spec = importlib.util.spec_from_file_location("wce_test_application", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.app.config.update(TESTING=True, RATELIMIT_ENABLED=True)
    users = {}
    monkeypatch.setattr(module, "get_db_connection", lambda: FakeConnection(users))
    monkeypatch.setattr(module, "account_session_fingerprint", lambda username: ('test-fingerprint' if username in users or username == 'test-doctor' else None))
    return module.app, users


def csrf_token(client, page="/"):
    response = client.get(page)
    assert response.status_code == 200
    match = re.search(rb'name="csrf_token"\s+value="([^"]+)"', response.data)
    assert match, f"CSRF token missing from {page}"
    return match.group(1).decode()


def test_registration_and_login_use_hashes(application):
    app, users = application
    with app.test_client() as client:
        token = csrf_token(client, "/signup-page")
        response = client.post("/signup", data={
            "signup_username": "alice",
            "signup_password": "a-good-password-1234",
            "signup_confirm_password": "a-good-password-1234",
            "csrf_token": token,
        })
        assert response.status_code == 200
        assert "alice" in users
        hashed = users["alice"]["password"]
        assert hashed != "a-good-password-1234"
        assert check_password_hash(hashed, "a-good-password-1234")
        token = csrf_token(client, "/")
        response = client.post("/login-home", data={
            "login_username": "alice",
            "login_password": "a-good-password-1234",
            "csrf_token": token,
        })
        assert response.status_code == 200
        with client.session_transaction() as session:
            assert session["account_fingerprint"] == "test-fingerprint"
            assert session["username"] == "alice"
            assert session["logged_in"] is True


def test_login_rejects_legacy_plaintext_and_wrong_password(application):
    app, users = application
    users["old"] = {"password": "legacy-plaintext-example", "role": "doctor"}
    with app.test_client() as client:
        token = csrf_token(client, "/")
        response = client.post("/login-home", data={
            "login_username": "old",
            "login_password": "legacy-plaintext-example",
            "csrf_token": token,
        })
        assert response.status_code == 200
        assert b"Invalid username or password" in response.data
        with client.session_transaction() as session:
            assert "logged_in" not in session


def test_post_requires_csrf(application):
    app, _users = application
    with app.test_client() as client:
        response = client.post("/login-home", data={
            "login_username": "user", "login_password": "password",
        })
        assert response.status_code == 400


def test_login_is_rate_limited(application):
    app, _users = application
    with app.test_client() as client:
        token = csrf_token(client, "/")
        outcomes = []
        for _ in range(6):
            response = client.post("/login-home", data={
                "login_username": "absent",
                "login_password": "incorrect",
                "csrf_token": token,
            })
            outcomes.append(response.status_code)
        assert outcomes[:5] == [200] * 5
        assert outcomes[5] == 429


@pytest.mark.parametrize("filename,data", [
    ("../../malicious.txt", b"not an image"),
    ("pretend.jpg", b"not actually a jpeg"),
])
def test_upload_rejects_invalid_files(application, filename, data):
    app, _users = application
    with app.test_client() as client:
        with client.session_transaction() as session:
            session["username"] = "test-doctor"
            session["logged_in"] = True
            session["account_fingerprint"] = "test-fingerprint"
        token = csrf_token(client, "/predict")
        response = client.post("/result", data={
            "csrf_token": token,
            "image": (io.BytesIO(data), filename),
        })
        assert response.status_code == 400


def test_doctor_self_signup_is_disabled_by_default(application, monkeypatch):
    app, _users = application
    monkeypatch.delenv("WCE_ALLOW_SELF_SIGNUP", raising=False)
    with app.test_client() as client:
        assert client.get("/signup-page").status_code == 403
        token = csrf_token(client, "/")
        response = client.post("/signup", data={
            "csrf_token": token,
            "signup_username": "unverified",
            "signup_password": "long-password-12345",
            "signup_confirm_password": "long-password-12345",
        })
        assert response.status_code == 403


def test_auth_guard_rejects_session_without_username(application):
    app, _users = application
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["logged_in"] = True
            sess["account_fingerprint"] = "test-fingerprint"
        assert client.get("/analysis").status_code == 200
        assert b"auth" in client.get("/analysis").data.lower()
        assert client.get("/predict").status_code in (302, 303)


def test_analysis_exception_does_not_disclose_internal_details(application, monkeypatch):
    app, _users = application
    module = app.view_functions["analysis"].__wrapped__.__globals__
    class BrokenPandas:
        @staticmethod
        def read_sql(*args, **kwargs):
            raise RuntimeError("private-db-host-and-secret")
    monkeypatch.setitem(module, "pd", BrokenPandas())
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["logged_in"] = True
            sess["username"] = "test-doctor"
            sess["account_fingerprint"] = "test-fingerprint"
        response = client.get("/analysis")
        assert response.status_code == 200
        assert b"private-db-host-and-secret" not in response.data
        assert b"temporarily unavailable" in response.data


def test_report_exception_does_not_disclose_internal_details(application, monkeypatch):
    app, _users = application
    module = app.view_functions["download_report"].__wrapped__.__globals__
    class BrokenPandas:
        @staticmethod
        def read_sql(*args, **kwargs):
            raise RuntimeError("private-db-host-and-secret")
    monkeypatch.setitem(module, "pd", BrokenPandas())
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["logged_in"] = True
            sess["username"] = "test-doctor"
            sess["account_fingerprint"] = "test-fingerprint"
        response = client.get("/download-report")
        assert response.status_code == 500
        assert b"private-db-host-and-secret" not in response.data
        assert b"temporarily unavailable" in response.data


def test_report_excludes_uploaded_image_identifiers_from_query(application, monkeypatch):
    app, _users = application
    module = app.view_functions["download_report"].__wrapped__.__globals__
    captured = []
    class FakeFrame:
        empty = True
    class CapturePandas:
        @staticmethod
        def read_sql(query, connection, params):
            captured.append((query, params))
            return FakeFrame()
    monkeypatch.setitem(module, "pd", CapturePandas())
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["logged_in"] = True
            sess["username"] = "test-doctor"
            sess["account_fingerprint"] = "test-fingerprint"
        response = client.get("/download-report")
        assert response.status_code == 404
    assert captured
    assert "filename" not in captured[0][0].lower()
    assert "image_path" not in captured[0][0].lower()
    assert captured[0][1] == ("test-doctor",)


def test_logout_requires_post_and_csrf(application):
    app, users = application
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'test-doctor'
            sess['account_fingerprint'] = 'test-fingerprint'
        assert client.get('/logout').status_code == 405
        assert client.post('/logout').status_code == 400
        token = csrf_token(client, '/')
        assert client.post('/logout', data={'csrf_token': token}).status_code in (302, 303)
        with client.session_transaction() as sess:
            assert 'logged_in' not in sess


def test_revoked_account_session_cannot_access_reports(application):
    app, users = application
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'revoked-doctor'
            sess['account_fingerprint'] = 'old-fingerprint'
        assert client.get('/download-report').status_code == 401
        with client.session_transaction() as sess:
            assert 'logged_in' not in sess
