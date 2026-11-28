# WCE Colon Disease Prediction

**A Flask + TensorFlow/Keras + MySQL demonstration application for four-class image classification, authenticated prediction history, analytics, and CSV reporting.**

> **Research/portfolio demonstration only. Not for clinical diagnosis, treatment decisions, or real patient care.** The saved model has not been independently clinically validated. A successful inference is not evidence of diagnostic accuracy.

## What it does

- Accepts JPG, JPEG, or PNG image uploads from an authenticated session, with an 8 MiB request limit and basic image validation.
- Loads the saved `model_1/Vgg.h5` model through Keras with a **TensorFlow** backend and preprocesses images to 224 × 224 RGB, normalized to [0, 1].
- Displays a predicted class and model output score; stores the result in MySQL with the authenticated username.
- Provides user-scoped analytics with date filters and a downloadable CSV report.
- Uses Flask-WTF CSRF protection, salted Werkzeug password hashes, session cookies, and a login rate limit.

**Application class-label order:** `Normal`, `Ulcerative Colitis`, `Polyps`, `Esophagitis`. **This order is implemented in the code but must still be verified against the original training pipeline/model labels before interpreting predictions.** A model output score must not be described as a calibrated clinical probability.

## Architecture

```text
Browser (Flask templates, Chart.js)
     | login / CSRF-protected upload
     v
Flask application (app.py)
     |-- Auth/session/rate limiting
     |-- Image validation -> Keras/TensorFlow -> saved VGG model
     |-- Parameterized MySQL queries -> users / predictions
     |-- Per-user analytics (Pandas) and CSV export
```

| Layer | Technologies |
| --- | --- |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Backend | Python 3.11, Flask, Flask-WTF, Flask-Limiter, Werkzeug |
| ML inference | TensorFlow 2.21.0, Keras, NumPy, Pillow, saved VGG model |
| Persistence and reporting | MySQL 8, mysql-connector-python, Pandas |
| Infrastructure | Docker, Gunicorn, Redis |
| Quality checks | pytest, GitHub Actions, pip-audit |

The exact dependency constraints are in [requirements.txt](requirements.txt); do not use old JAX installation instructions.

## Local setup (Windows PowerShell)

1. Install **Python 3.11** and MySQL 8. Clone the repository:

   ```powershell
   git clone https://github.com/Amrutha0902/WCE_CURATED_COLON_DISEASE_PREDICTION.git
   cd WCE_CURATED_COLON_DISEASE_PREDICTION
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Initialize a **new, disposable local database** with `setup_database.sql` using a trusted MySQL administrator. The script creates `colon_disease_db` and the `users` and `predictions` tables; it **does not seed a doctor account**. Create a separate least-privilege application database user rather than running Flask as MySQL root. **Do not run the setup or migration scripts against an existing database without reviewing them and backing up first.**

3. Create `.env` locally (never commit it), using `.env.example` as a reference. Configure:

   ```dotenv
   DB_HOST=localhost
   DB_USER=your_restricted_app_user
   DB_PASSWORD=your_unique_database_password
   DB_NAME=colon_disease_db
   FLASK_SECRET_KEY=your_unique_random_signing_secret
   WCE_PRODUCTION=false
   RATELIMIT_STORAGE_URI=memory://
   COOKIE_SECURE=false
   WCE_ALLOW_SELF_SIGNUP=false
   KERAS_BACKEND=tensorflow
   ```

   Generate `FLASK_SECRET_KEY` locally with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. The application loads `.env` through `python-dotenv`. Use `COOKIE_SECURE=false` **only for localhost HTTP development**; use secure cookies with HTTPS in a properly configured deployment.

4. Confirm that `model_1/Vgg.h5` is available, then start the development server:

   ```powershell
   python app.py
   ```

   Open `http://127.0.0.1:5000`. No default credentials are provided. Public doctor signup is **disabled by default**. `WCE_ALLOW_SELF_SIGNUP=true` is **only for an isolated disposable demo**, not real healthcare account provisioning. Existing plaintext-password accounts require a trusted administrative reset; see [SECURITY.md](SECURITY.md).

## Main routes

| Route | Method | Behavior |
| --- | --- | --- |
| `/` | GET | Home and login interface |
| `/login-home` | POST | Rate-limited authentication |
| `/signup-page`, `/signup` | GET, POST | Demo registration, disabled by default |
| `/predict` | GET | Upload page |
| `/result` | POST | Authenticated image inference and MySQL save |
| `/analysis` | GET | Authenticated, user-scoped analytics |
| `/download-report` | GET | Authenticated, user-scoped CSV export |
| `/logout` | POST | CSRF-protected session logout |
| `/healthz` | GET | Basic liveness response; not a DB/model readiness check |

`/result` returns rendered HTML, not a JSON prediction API. New prediction records deliberately store empty filename and image-path fields; uploaded images are deleted after inference on the normal success path. Legacy records may still contain sensitive filenames or paths, and temporary files may remain if deletion fails. Treat database exports and medical images as sensitive.

## Database and migrations

- `users`: username, password hash, role and optional profile fields.
- `predictions`: filename, predicted class, confidence score, timestamp, image path, and username.
- The `predictions.username` field is nullable to accommodate legacy data; **a database foreign-key constraint is not defined by the setup script**.
- Historical predictions without verified ownership must not be reassigned automatically.
- Review `migrations/001_predictions_username.sql` before use on a backed-up legacy database.

## Testing and CI

The production-readiness branch defines a seven-job workflow at [GitHub Actions](https://github.com/Amrutha0902/WCE_CURATED_COLON_DISEASE_PREDICTION/actions):

| Job | What it verifies |
| --- | --- |
| `dependency-audit` | Dependency vulnerability scanning |
| `static-security-checks` | Flask request-level checks with isolated substitutes and source-level assertions |
| `mysql-integration` | Real queries, password hashing and user-scoped inserts against disposable MySQL 8 |
| `model-smoke` | Real saved-model deserialization and synthetic CPU inference with TensorFlow |
| `full-e2e` | Synthetic-image HTTP upload, real VGG inference, login, disposable MySQL persistence and analysis |
| `redis-integration` | Shared rate limits, failure handling and production configuration guards |
| `docker-build` | Compose validation, image build, container startup, non-root execution and Redis connectivity |

Live Redis integration and Docker runtime verification passed in GitHub Actions workflow #114.

**All seven CI jobs passed for commit 92dc472 in workflow #114.** Check the latest [GitHub Actions runs](https://github.com/Amrutha0902/WCE_CURATED_COLON_DISEASE_PREDICTION/actions) for the exact commit and current results. Passing CI does not establish clinical efficacy or constitute a full penetration test.

For local regression testing, run each test group separately:

```powershell
$env:WCE_PRODUCTION = "false"
$env:RATELIMIT_STORAGE_URI = "memory://"
$env:WCE_MYSQL_TEST = "0"
$env:WCE_E2E_TEST = "0"
$env:WCE_REDIS_TEST = "0"

python -m pip check
python -m pytest -q tests/test_flask_security.py tests/test_security_contract.py
python -m pytest -q tests/test_model_smoke.py
```

### Local verification results

- Flask security tests: 13 passed.
- Static security tests: 7 passed.
- Real MySQL integration: 2 passed.
- Real VGG model smoke test: 1 passed.
- Synthetic-image HTTP-to-MySQL E2E: 1 passed.
- Dependency consistency: no broken requirements.
- Redis integration: 5 tests passed in GitHub Actions.

MySQL and E2E tests require explicit opt-in flags and the disposable database colon_disease_db_test. Redis tests require an isolated Redis service. Never run integration tests against production data.

## Docker demo (localhost only)

`Dockerfile` uses Python 3.11, TensorFlow and Gunicorn. `compose.yaml` provides a localhost-bound demo with a disposable MySQL service and enables demo signup. Set unique `DB_PASSWORD`, `MYSQL_ROOT_PASSWORD`, and `FLASK_SECRET_KEY` values locally, then:

```powershell
docker compose config
docker compose up --build
```

Visit `http://127.0.0.1:8000`. **Do not expose this demo to the public Internet or use patient data.** Container runtime validation passed in GitHub Actions workflow #114. Persistent MySQL volumes are not automatically migrated.

## Dataset sources and attribution

The model-training notebooks reference the [Curated Colon Dataset for Deep Learning](https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning), uploaded by **francismon**. Its dataset description asks users to consult the original sources and credits:

- **KVASIR:** K. Pogorelov et al., *KVASIR*, Proceedings of the 8th ACM Multimedia Systems Conference (2017). [DOI: 10.1145/3083187.3083212](https://doi.org/10.1145/3083187.3083212).
- **ETIS-Larib-Polyp DB:** J. Silva et al., *Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer*, International Journal of Computer Assisted Radiology and Surgery 9, 283–293 (2013). [DOI: 10.1007/s11548-013-0926-3](https://doi.org/10.1007/s11548-013-0926-3).
- **Related study cited in the dataset description:** F. J. P. Montalbo, *Diagnosing Gastrointestinal Diseases from Endoscopy Images through a Multi-Fused CNN with Auxiliary Layers, Alpha Dropouts, and a Fusion Residual Block*, Biomedical Signal Processing and Control 76 (2022). [DOI: 10.1016/j.bspc.2022.103683](https://doi.org/10.1016/j.bspc.2022.103683).

**Licensing and provenance:** The supplied dataset description does not establish permission to redistribute individual medical images. All 14 legacy JPGs in the archived Flask `uploads/` directory failed byte-for-byte SHA256 matching against the locally downloaded dataset; this does not rule out transformed copies. The identified legacy uploads were removed from Git tracking and rewritten repository history. Previously downloaded copies and external backups may still exist. Do not commit patient images or identifying data.

## Model evaluation (not yet completed)

The repository includes `scripts/evaluate_vgg.py`. It accepts a labeled CSV manifest with `image_path,label,patient_id` and an image root, and outputs accuracy, macro F1, per-class metrics and a confusion matrix:

```powershell
python scripts/evaluate_vgg.py --manifest "path\to\test_manifest.csv" --image-root "path\to\test_images" --output "evaluation_report.json"
```

**No independently verified model accuracy, sensitivity, specificity or clinical claims are reported.** Before citing results, confirm model class order, data permissions, and patient-disjoint train/validation/test splits. Do not commit identifiable medical images, patient IDs or private reports.

## Security status and limitations

The hardened application improves credential configuration, password hashing, CSRF, login throttling, basic upload validation and user-scoped reporting. It is **not production-ready**. The outstanding issues include:

- Identified sensitive Git history was rewritten and local credentials were rotated. External credential rotation and historical exposure impact remain unverified.
- Legacy plaintext-password accounts need verified reset/re-provisioning; no public doctor credential verification workflow exists.
- Production mode (`WCE_PRODUCTION=true`) requires a Redis `RATELIMIT_STORAGE_URI`, secure cookies, and disabled self-signup. The limiter has no in-memory fallback; a Redis outage may cause request failures rather than bypassing throttling. Development mode may use in-memory storage, which is unsuitable for multi-worker production use.
- Uploads have an 8 MiB request limit and configured image-dimension checks (20 million pixels; 8192 pixels per edge), but these do not guarantee hard CPU/memory/inference limits. Temporary uploads are deleted on the normal completion path; deletion failures, legacy stored images, and retention/access policies still need review.
- Model class mapping, clinical performance, privacy controls, and running-container security require further validation.

See [SECURITY.md](SECURITY.md) and [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for details.

**Status: Locally tested portfolio/research demo. All seven production-readiness CI jobs passed. Not a medical device or clinical decision-support system.**

## Repository cleanup

- Removed the obsolete duplicate Flask application and model.
- Removed 20 legacy academic PDFs and reference templates.
- Preserved the primary Flask application and VGG16 model.
- Preserved all three training notebooks with their original cell sources.
- Removed 64 stored notebook outputs and reset execution counts.
- Verified all 24 local regression tests after cleanup.
- Historical Git copies of removed assets may still exist.
- Live Redis, Docker runtime and dependency vulnerability checks passed in CI.
