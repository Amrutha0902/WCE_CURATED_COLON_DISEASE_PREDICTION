"""True HTTP -> actual VGG -> disposable MySQL integration.

No patient data is used. This test does not measure clinical accuracy.
"""
import csv
import io
import os
import re
import uuid
from pathlib import Path

import mysql.connector
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def token(client, path):
    response = client.get(path)
    assert response.status_code == 200
    match = re.search(rb'name="csrf_token"\s+value="([^"]+)"', response.data)
    assert match is not None, f"CSRF token missing on {path}"
    return match.group(1).decode()


@pytest.mark.skipif(os.getenv("WCE_E2E_TEST") != "1", reason="Only for disposable CI MySQL")
def test_real_model_http_to_mysql_roundtrip():
    assert os.getenv("DB_NAME") == "colon_disease_db_test", "Refusing non-test database"
    os.environ.setdefault("FLASK_SECRET_KEY", "ci-only-throwaway-signing-key-123456789")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ["WCE_ALLOW_SELF_SIGNUP"] = "true"
    import app as application

    application.app.config.update(TESTING=True)
    username = "e2e_" + uuid.uuid4().hex[:16]
    password = "ci-example-password-123456"
    connection = mysql.connector.connect(**application.DB_CONFIG)
    try:
        with application.app.test_client() as client:
            response = client.post("/signup", data={
                "csrf_token": token(client, "/signup-page"),
                "signup_username": username,
                "signup_password": password,
                "signup_confirm_password": password,
            })
            assert response.status_code == 200
            assert application.verify_doctor_credentials(username, password)

            response = client.post("/login-home", data={
                "csrf_token": token(client, "/"),
                "login_username": username,
                "login_password": password,
            })
            assert response.status_code == 200
            with client.session_transaction() as sess:
                assert sess.get("username") == username

            # Synthetic RGB image is intentionally not a clinical WCE image.
            image = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            response = client.post("/result", data={
                "csrf_token": token(client, "/predict"),
                "image": (buffer, "synthetic.png"),
            }, content_type="multipart/form-data")
            assert response.status_code == 200, response.data[:500]
            assert b"Prediction:" in response.data

            cursor = connection.cursor()
            cursor.execute(
                "SELECT predicted_class, confidence, image_path, filename FROM predictions WHERE username=%s",
                (username,),
            )
            rows = cursor.fetchall()
            cursor.close()
            assert len(rows) == 1
            assert rows[0][0] in ("Normal", "Ulcerative Colitis", "Polyps", "Esophagitis")
            assert 0 <= float(rows[0][1]) <= 1
            # Clean up generated synthetic image after verification.
            assert rows[0][2] in ('', None)
            assert rows[0][3] in ('', None), 'Original patient filename must not persist'
            # Stored metadata no longer contains a filesystem path.
            # Verify analytics renders successfully.
            for date_range in ("till_date", "week"):
                response = client.get(
                    "/analysis", query_string={"range": date_range}
                )
                assert response.status_code == 200
                assert b"temporarily unavailable" not in response.data
                assert b"No prediction data" not in response.data
                assert b"Database connection failed" not in response.data
                assert rows[0][0].encode() in response.data

            # Verify real CSV report generation.
            response = client.get("/download-report")
            assert response.status_code == 200
            assert response.mimetype == "text/csv"
            assert "attachment" in response.headers["Content-Disposition"]

            reader = csv.DictReader(
                io.StringIO(response.data.decode("utf-8"))
            )
            assert reader.fieldnames == [
                "id",
                "predicted_class",
                "confidence",
                "prediction_timestamp",
            ]

            report_rows = list(reader)
            assert len(report_rows) == 1
            assert report_rows[0]["predicted_class"] == rows[0][0]
            assert float(report_rows[0]["confidence"]) == pytest.approx(
                float(rows[0][1])
            )
            assert report_rows[0]["prediction_timestamp"]
            assert "synthetic.png" not in response.data.decode("utf-8")

    finally:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM predictions WHERE username=%s", (username,))
        cursor.execute("DELETE FROM users WHERE username=%s", (username,))
        connection.commit()
        cursor.close()
        connection.close()
