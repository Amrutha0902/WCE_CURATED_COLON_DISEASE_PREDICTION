# Security and account migration

## Required incident response

The public Git history previously contained application/database credentials and a default plaintext doctor account. **Treat those values as compromised.** Rotate the database password and Flask signing secret in the actual environment; updating Git cannot revoke them. Review database access, invalidate existing browser sessions, and assess whether exposed records require notification or other incident response. Removing values from the current branch does not remove them from historical commits.

## Legacy accounts

Old registrations saved plaintext passwords. The updated application intentionally does **not** accept those legacy records: existing users must have passwords reset through a trusted administrative process. Do not export or redistribute their existing passwords. Never run an automatic conversion of plaintext data on an untrusted dataset without evaluating the exposure risk. New registrations use Werkzeug's salted password hashing. The users.password column can hold the resulting hashes (VARCHAR(255)).

Before enabling this app publicly, implement a verified account recovery/reset workflow or securely re-provision users and disable old credentials.

## Application security limitations

- CSRF tokens are required for browser POST forms and are covered by Flask request-level tests and the synthetic-image end-to-end CI workflow; this does not establish complete security testing. The login route now has a 5/minute limit with in-memory storage by default; production requires shared persistent rate-limit storage such as Redis, trusted proxy/IP settings, and deployment-specific testing. Do not expose it publicly until full security integration tests are verified.
- New browser sessions are marked Secure by default; for localhost HTTP-only development, set COOKIE_SECURE=false.
- Medical uploads and predictions may be sensitive. Configure retention/access policies and avoid logging private details.
- Upload validation checks image type and size, but a dedicated image processing sandbox, malware checks, and content security controls may be needed for production.
- The saved model has passed synthetic inference smoke tests and end-to-end engineering tests, but its predictive quality and class mapping have **not** been independently evaluated.
- The CI workflow has passed real disposable-MySQL integration and real saved-model end-to-end tests; this does not verify production data, deployment security, or clinical validity.

## Additional operational and authorization risks

- Self-registration creates doctor-role accounts without verifying medical credentials; this is a **demo-only** workflow, not suitable for a healthcare deployment. Doctor access would require identity verification and administrative provisioning.
- Rate limiting uses in-memory state in local development; a production multi-worker environment requires an appropriate shared backend configured using `RATELIMIT_STORAGE_URI`.
- Some legacy routes still expose internal exception details and require error response normalization.
- Request-level tests in `tests/test_flask_security.py` exercise Flask code with model/database substitutes. Real disposable MySQL and TensorFlow model integration have passed CI; deployment security remains unverified.
- Use `migrations/001_predictions_username.sql` only after backup and review on legacy databases. Prediction requests no longer change the database schema.


## Current self-registration policy

Unverified public doctor signup is **disabled by default**. The flag `WCE_ALLOW_SELF_SIGNUP=true` is permitted only for disposable, isolated demonstration environments; it is not identity verification. Before real deployment, build an administrator-approved invitation/provisioning workflow and a secure reset process with out-of-band identity verification. No production account reset has been performed.

## New test and container artifacts

The CI `full-e2e` job uses a disposable MySQL instance and synthetic image to test the actual VGG inference and dashboard workflow. The Dockerfile/Compose setup is for localhost demonstrations only, not a production deployment; it has not been independently built and launched in this audit. The evaluation script requires an independent labeled dataset and does not establish medical efficacy without one.


### Trusted operator-only legacy password reset

A CLI helper `scripts/admin_reset_password.py --username ACCOUNT` prompts for a new password without echoing it and stores a salted hash. It requires the operator to verify identity out of band and explicitly set `WCE_ADMIN_RESET_CONFIRM=I_VERIFIED_IDENTITY`. Use it only from a trusted machine with restricted MySQL access, a backed-up database, and a secure channel for notifying the account holder. This is **not** a self-service recovery system; it does not verify medical qualifications or automatically revoke all existing sessions. No real account was changed by this work.

## Security review update — 23 November 2026

**Review basis:** current `security/wce-hardening-2026` source, its five green GitHub Actions jobs at commit `c3a7ef2`, and the documented prior exposure. This is a code-level review, **not** a penetration test, dependency vulnerability scan, Git-history secret scan, or production deployment assessment. The README was updated after that CI run; the newest documentation-only commit must be checked separately.

### Outstanding findings and required verification

| Priority | Finding | Required action before public/clinical deployment |
| --- | --- | --- |
| Critical | Prior public Git history contained database credentials, a Flask secret, and a default plaintext doctor account. Rotation is not verified. | Rotate real credentials, invalidate sessions as appropriate, review exposure and access logs, and assess incident response. Do not assume deleting current-file secrets fixes history. |
| High | Existing plaintext accounts are intentionally rejected but migration/re-provisioning completion is unverified. | Use verified administrator-led resets and confirm no active plaintext accounts. |
| High | The model class index-to-label mapping and independent patient-disjoint performance are not verified. | Reconcile the original training label mapping, evaluate on a properly separated labeled dataset, and avoid clinical claims. |
| Medium | `app.py` `/analysis` and `/download-report` return exception details in some error responses. | Log internal exceptions server-side; return generic client errors and add regression tests. |
| Medium | Login rate limiting defaults to `memory://`, which is per-process and not suitable for multi-worker production. | Configure shared rate-limit storage, trusted proxy handling and deployment tests. |
| Medium | Uploaded medical images persist on disk; explicit retention/deletion, private data handling, and access controls need review. CSV exports include filenames and image paths. | Define retention and access policy, minimize exported fields, and test privacy boundaries. |
| Medium | `WCE_ALLOW_SELF_SIGNUP=true` creates doctor-role accounts in the localhost Compose demo without identity verification. | Keep disabled outside isolated demos; implement verified account provisioning before real use. |
| Medium | `GET /logout` mutates session state, and the public `/predict` page can render without authentication; inference itself requires login. | Review intended authorization and HTTP method behavior; test before deployment. |
| Medium | Docker CI builds the image and validates Compose syntax but does not run the complete container stack. | Add disposable container-runtime integration testing before claiming deployment readiness. |
| Low | CI actions emit Node.js 20 deprecation warnings for `actions/checkout@v4` and `actions/setup-python@v5`. | Upgrade GitHub Actions versions in a separate reviewed maintenance change. |

### Confirmed engineering evidence

- Static security, disposable MySQL integration, real TensorFlow model smoke, full synthetic-image HTTP-to-MySQL end-to-end, and Docker build jobs all passed in the supplied GitHub Actions screenshot at commit `c3a7ef2`.
- Source uses Werkzeug salted hashes for new accounts, CSRFProtect, parameterized database queries in reviewed paths, random upload filenames, and an 8 MiB request limit.
- The new README removes legacy JAX and default-login instructions and clearly labels the application a research/portfolio demo.

**Merge recommendation:** keep PR #1 as draft until critical credential/legacy-account risks have been addressed or explicitly scoped to a non-deployed demo, and the security review is approved. Do not treat CI success as production or clinical approval.

## Post-audit remediation branch (23 November 2026)

The `security/post-audit-remediation-2026` branch adds basic response headers, an eight-hour permanent-session lifetime for new logins, a 20-million-pixel upload limit, JPEG/PNG format verification, removal of successfully processed image files, and redaction of raw MySQL exception text from application logging. Successful prediction records now store an empty image path. Existing stored images are **not** automatically deleted. These changes require CI and manual validation before merging.

Still open: account revocation/session invalidation, shared rate-limit storage, CSRF-protected POST logout, full medical-data retention governance, isolated image inference, clinical model validation, and production deployment testing. Historical credential rotation was reported locally but deployed credentials and historical access have not been independently audited.


## Follow-up hardening — remediation branch (unverified until CI reruns)

- Production mode (`WCE_PRODUCTION=true`) now requires a `redis://` or `rediss://` `RATELIMIT_STORAGE_URI`. The limiter has no in-memory fallback and does not swallow backend errors. Deploy Redis with authentication, TLS where applicable, restricted networking, and monitoring; test Redis outages in the deployed topology. The application does **not** automatically configure or deploy Redis.
- Dependency scanning is added as a separate `dependency-audit` CI job using pip-audit. Known findings must be triaged and fixed; this scan does not prove the supply chain is secure.
- Image processing additionally rejects images with dimensions over 8192 pixels on either axis. This is a partial resource bound, **not** an inference sandbox or a hard CPU/RAM/time limit. The current Gunicorn two-thread configuration does not guarantee hard cancellation of hung model inference. Use a supervised, isolated inference worker with OS/container memory, CPU, and time limits before public exposure.
- New prediction rows no longer retain original upload filenames or image paths. Previously stored filenames, images, and records are **not** automatically changed.
- `python scripts/retention_predictions.py --days 90 --database YOUR_DB_NAME` previews eligible metadata rows without deletion. A separately authorized operator may use `--execute` only after backup, retention-policy approval, and setting `WCE_RETENTION_DELETE_CONFIRM=DELETE_EXPIRED_PREDICTIONS_FROM_YOUR_DB_NAME`. The script does not delete image files, account records, backups, or historical database rows unless explicitly run in execution mode. The example 90 days is **not** a recommended healthcare retention period; choose a lawful, approved period.
- This is still a **research/portfolio demo**, not a validated clinical device. Original training label mapping, patient-disjoint evaluation, clinician review, identity provisioning, credential incident response, deployed container E2E, and penetration testing remain outstanding.
