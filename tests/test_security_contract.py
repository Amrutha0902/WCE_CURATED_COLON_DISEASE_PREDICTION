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
