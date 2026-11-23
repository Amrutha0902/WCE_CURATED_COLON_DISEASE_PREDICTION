"""Lightweight checks for safety-critical source rules.

These are static regression checks. They do not replace real integration/security tests.
"""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_application_syntax():
    ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))


def test_passwords_are_hashed_not_queried_as_plaintext():
    content = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "generate_password_hash(password)" in content
    assert "check_password_hash(stored_hash, password)" in content
    assert "AND password = %s" not in content


def test_no_seeded_default_account():
    content = (ROOT / "setup_database.sql").read_text(encoding="utf-8")
    assert "INSERT INTO users" not in content


def test_upload_is_not_saved_under_user_supplied_filename():
    content = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "secure_filename(f.filename)" in content
    assert "uuid.uuid4().hex" in content
    assert "f.save(filepath)" in content
    assert "os.path.join(upload_folder, f.filename)" not in content


def test_production_rate_limit_storage_is_redis_only_and_fail_closed():
    content = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "startswith(('redis://', 'rediss://'))" in content
    assert "in_memory_fallback_enabled=False" in content
    assert "swallow_errors=False" in content
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "redis>=5,<7" in requirements


def test_original_image_filename_not_persisted():
    content = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "filename='',  # Never retain" in content
    assert "image_path=''" in content
    assert "MAX_IMAGE_EDGE = 8192" in content


def test_retention_script_requires_explicit_execution_and_confirmation():
    content = (ROOT / "scripts/retention_predictions.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--execute", action="store_true"' in content
    assert 'WCE_RETENTION_DELETE_CONFIRM' in content
    assert 'if args.database != db_name:' in content
    assert 'WHERE prediction_timestamp < %s' in content
