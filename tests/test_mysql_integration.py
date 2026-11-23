"""MySQL integration tests against a throwaway CI database.

This tests real parameterized SQL and commits. Keras/model inference and
Pandas remain substituted to avoid conflating DB tests with ML integration.
Run only when WCE_MYSQL_TEST=1 and ONLY against a disposable test database.
"""
import importlib.util
import os
import sys
import types
import uuid
from pathlib import Path

import mysql.connector
import pytest
from werkzeug.security import check_password_hash

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def integration_app():
    if os.getenv("WCE_MYSQL_TEST") != "1":
        pytest.skip("MySQL integration disabled; set WCE_MYSQL_TEST=1 for disposable CI DB")
    dbname = os.environ.get("DB_NAME", "")
    if dbname != "colon_disease_db_test":
        pytest.fail("Refusing tests on non-disposable database: use DB_NAME=colon_disease_db_test")
    os.environ.setdefault("FLASK_SECRET_KEY", "github-ci-test-only-not-production-abcdefghijk")
    os.environ.setdefault("COOKIE_SECURE", "false")

    # Stub expensive ML modules only. mysql.connector remains fully real.
    model_module = types.ModuleType("keras.models")
    model_module.load_model = lambda path, **kwargs: object()
    preprocessing_image = types.ModuleType("keras.preprocessing.image")
    preprocessing_image.load_img = lambda *args, **kw: None
    preprocessing_image.img_to_array = lambda *args, **kw: None
    keras_module = types.ModuleType("keras")
    keras_module.__path__ = []
    preprocessing = types.ModuleType("keras.preprocessing")
    preprocessing.__path__ = []
    for name, module in [
        ("keras", keras_module), ("keras.models", model_module),
        ("keras.preprocessing", preprocessing), ("keras.preprocessing.image", preprocessing_image),
    ]:
        sys.modules[name] = module
    # Pandas is not exercised by these DB-focused tests.
    sys.modules["pandas"] = types.ModuleType("pandas")
    spec = importlib.util.spec_from_file_location("wce_mysql_integration_app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.app.config.update(TESTING=True)
    assert module.get_db_connection() is not None
    return module

@pytest.fixture
def db(integration_app):
    connection = mysql.connector.connect(**integration_app.DB_CONFIG)
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def test_real_mysql_registration_hashing_and_authentication(integration_app, db):
    username = "ci_" + uuid.uuid4().hex[:20]
    password = "correct-test-password-123"
    try:
        assert integration_app.register_user(username, password)
        cur = db.cursor()
        cur.execute("SELECT password, role FROM users WHERE username=%s", (username,))
        row = cur.fetchone()
        cur.close()
        assert row is not None
        assert row[0] != password
        assert check_password_hash(row[0], password)
        assert row[1] == "doctor"
        assert integration_app.verify_doctor_credentials(username, password)
        assert not integration_app.verify_doctor_credentials(username, "incorrect")
        assert not integration_app.verify_doctor_credentials(username + "' OR 1=1 --", password)
    finally:
        db.rollback()
        cur = db.cursor()
        cur.execute("DELETE FROM users WHERE username=%s", (username,))
        db.commit()
        cur.close()


def test_real_mysql_predictions_user_isolation(integration_app, db):
    owner = "ci_" + uuid.uuid4().hex[:20]
    other = "ci_" + uuid.uuid4().hex[:20]
    try:
        assert integration_app.save_prediction_to_db(
            "example.png", "Normal", 0.91, "/temporary/not-a-real-medical-file.png", owner
        )
        cur = db.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions WHERE username=%s", (owner,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM predictions WHERE username=%s", (other,))
        assert cur.fetchone()[0] == 0
        cur.close()
    finally:
        db.rollback()
        cur = db.cursor()
        cur.execute("DELETE FROM predictions WHERE username=%s", (owner,))
        db.commit()
        cur.close()
