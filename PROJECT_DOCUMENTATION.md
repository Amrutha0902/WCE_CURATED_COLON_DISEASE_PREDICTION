# WCE Colon Disease Prediction - Technical Documentation

## 1. Project Overview

A full-stack research and portfolio application for four-class gastrointestinal image classification, authenticated prediction history, analytics and CSV reporting.

This application is not clinically validated and must not be used for medical diagnosis or patient care.

## 2. Technology Stack

| Component | Technology |
| --- | --- |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Machine learning | TensorFlow, Keras, saved VGG model |
| Image processing | Pillow, NumPy |
| Database | MySQL 8 |
| Analytics | Pandas |
| Authentication | Werkzeug password hashing, Flask sessions |
| Security | Flask-WTF, Flask-Limiter |
| Shared rate limiting | Redis |
| Containerization | Docker, Docker Compose |
| Application server | Gunicorn |
| Testing | pytest |
| CI | GitHub Actions, pip-audit |

Dependency constraints are maintained in requirements.txt.

## 3. System Architecture

Browser
    |
    v
Flask application
    |
    +-- Authentication and session management
    +-- CSRF protection and rate limiting
    +-- Image validation and preprocessing
    +-- TensorFlow/Keras inference
    +-- MySQL persistence
    +-- User-scoped analytics and CSV reporting

Production-mode rate limiting uses Redis.

## 4. Core Application Workflow

1. User authenticates through the login interface.
2. The application validates credentials against stored password hashes.
3. An authenticated user submits an image through a CSRF-protected form.
4. The application validates the image format, dimensions and request size.
5. The image is resized to 224 x 224 RGB and normalized.
6. The saved VGG model generates a four-class output vector.
7. The application maps the highest-scoring output to a class label.
8. Prediction metadata is stored in MySQL under the authenticated username.
9. Temporary image files are removed on successful processing.
10. Users can access their own analytics and CSV reports.

## 5. Model Inference

Model artifact: model_1/Vgg.h5

Runtime: TensorFlow with Keras.

Application class-label order:

- Normal
- Ulcerative Colitis
- Polyps
- Esophagitis

The original training label mapping must be independently verified.

Model output scores are not calibrated clinical probabilities. Synthetic inference testing does not establish diagnostic accuracy.

## 6. Database Architecture

### Users

Stores usernames, password hashes, roles and optional profile information.

### Predictions

Stores predicted class, model output score, timestamp and username.

New predictions do not retain original upload filenames or image paths.

Legacy records may contain sensitive information and require separate review.

Database operations in reviewed application paths use parameterized queries.

## 7. Security Controls

- Salted password hashing.
- CSRF protection for browser POST requests.
- Login rate limiting.
- Authenticated prediction and reporting workflows.
- User-scoped prediction history.
- Account validity checks for protected access.
- POST-only CSRF-protected logout.
- Secure session cookies in production mode.
- Eight-hour permanent-session lifetime.
- Generic client-facing exception responses.
- Basic HTTP security headers.
- Random temporary upload filenames.
- Eight MiB request limit.
- Maximum image size of 20 million pixels.
- Maximum image dimension of 8192 pixels per edge.
- Disabled unverified self-registration by default.

These controls do not constitute a complete penetration test.

## 8. Environment Configuration

Local development:

- WCE_PRODUCTION=false
- RATELIMIT_STORAGE_URI=memory://
- COOKIE_SECURE=false

Production-mode requirements:

- WCE_PRODUCTION=true
- RATELIMIT_STORAGE_URI configured with Redis
- COOKIE_SECURE=true
- WCE_ALLOW_SELF_SIGNUP=false
- Unique Flask signing secret
- Restricted database credentials
- Proper HTTPS infrastructure

Redis errors are not silently bypassed through in-memory fallback.

See .env.example and README.md for configuration details.

## 9. Container Architecture

The Dockerfile defines a Python 3.11 application image with Gunicorn and non-root execution.

Docker Compose provides an isolated localhost demonstration with MySQL and local development settings.

The Compose demonstration is not intended for public deployment.

CI includes image construction, production-mode startup, non-root execution and Redis connectivity checks.

Live execution of the newly added runtime checks remains pending.

## 10. Testing and Quality Assurance

Local engineering verification:

| Test group | Result |
| --- | --- |
| Flask security | 13 passed |
| Static security contracts | 7 passed |
| Real MySQL integration | 2 passed |
| Real VGG inference | 1 passed |
| HTTP-to-MySQL end-to-end | 1 passed |
| Total | 24 passed |
| Redis integration | 5 tests awaiting execution |

Local regression testing was performed using Python 3.12.10 on Windows.

CI targets Python 3.11 on Linux.

## 11. Continuous Integration

The workflow defines seven jobs:

- dependency-audit
- static-security-checks
- mysql-integration
- model-smoke
- full-e2e
- redis-integration
- docker-build

The latest production-readiness changes require successful CI execution before integration is considered verified.

## 12. Repository Structure

- app.py: Flask application.
- model_1/: Saved model artifact.
- templates/: HTML templates.
- static/: Frontend assets.
- tests/: Automated test suites.
- scripts/: Evaluation and administrative utilities.
- migrations/: Database migration scripts.
- requirements.txt: Application dependencies.
- Dockerfile: Application container definition.
- compose.yaml: Local demonstration environment.
- .github/workflows/: CI configuration.
- README.md: Setup and usage instructions.
- SECURITY.md: Security controls, incident history and limitations.

## 13. Outstanding Limitations

- Independent model evaluation and training label verification.
- Verified healthcare account provisioning.
- Production HTTPS and trusted proxy configuration.
- Medical-data retention and privacy governance.
- Isolated inference with hard resource limits.
- Live Redis and Docker runtime verification.
- Production infrastructure security assessment.
- Independent penetration testing.

## 14. Project Status

Locally tested full-stack machine learning portfolio project with security hardening, real database integration, automated regression tests and containerization.

Production-readiness CI validation remains pending.

Not a medical device or clinically validated diagnostic system.

Dataset attribution and evaluation instructions are maintained in README.md.

## Repository cleanup

- Removed the obsolete duplicate Flask application and model.
- Removed 20 legacy academic PDFs and reference templates.
- Preserved the primary Flask application and VGG16 model.
- Preserved all three training notebooks with their original cell sources.
- Removed 64 stored notebook outputs and reset execution counts.
- Verified all 24 local regression tests after cleanup.
- Historical Git copies of removed assets may still exist.
- Live Redis, Docker runtime and dependency vulnerability validation remain pending.
