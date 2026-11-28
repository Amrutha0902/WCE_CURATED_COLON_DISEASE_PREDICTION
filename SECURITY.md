# Security Policy and Engineering Assessment

## 1. Application Scope

This application is a research and portfolio demonstration.

It is not clinically validated, approved for patient care, or certified for healthcare production deployment.

## 2. Historical Credential Exposure

Previous repository history contained database credentials, a Flask signing secret and a default plaintext account.

Treat all historically exposed credentials as compromised.

Remediation performed:

- Removed hardcoded application credentials.
- Removed seeded default account credentials.
- Rewrote repository history to remove identified sensitive content.
- Local Git history secret scan reported zero findings.
- Local database credentials and Flask signing secret were rotated.
- GitHub Support confirmed cleanup of historical pull-request references.
- Historical pull-request references were subsequently checked and returned HTTP 404.

These actions cannot revoke previously downloaded copies, external clones, forks or backups.

External deployment credentials, historical database access and incident impact remain unverified.

## 3. Implemented Security Controls

| Control | Implementation status |
| --- | --- |
| Password hashing | Implemented and locally tested |
| CSRF protection | Implemented and locally tested |
| Login rate limiting | Local behavior tested |
| Shared Redis rate limiting | Configured; integration tests passed in CI |
| Redis failure handling | Configured without memory fallback |
| Secure session cookies | Required in production mode |
| Session lifetime | Eight hours for permanent sessions |
| Account validity checks | Locally tested |
| POST-only logout | Locally tested |
| User-scoped reporting | Locally tested |
| Generic error responses | Tested for analytics and reporting |
| HTTP security headers | Implemented |
| Upload validation | Implemented and locally tested |
| Temporary image cleanup | Implemented on successful processing |
| Prediction metadata minimization | Implemented |
| Public doctor registration | Disabled by default |
| Non-root Docker execution | Configured; runtime verification passed in CI |
| Dependency vulnerability scanning | Passed in CI workflow #114 |

## 4. Production Configuration

Production mode requires:

- WCE_PRODUCTION=true
- RATELIMIT_STORAGE_URI using redis:// or rediss://
- COOKIE_SECURE=true
- WCE_ALLOW_SELF_SIGNUP=false
- A unique Flask signing secret

The rate limiter does not silently fall back to in-memory storage when Redis is unavailable.

Production infrastructure must additionally provide:

- HTTPS and trusted proxy configuration.
- Restricted database credentials.
- Redis authentication and restricted networking.
- TLS where appropriate.
- Monitoring and secure secret management.
- Database backups and recovery procedures.

The application does not automatically provision these services.

## 5. Account Management

Legacy plaintext passwords are intentionally rejected.

Existing accounts require trusted administrator-led reset or re-provisioning.

The scripts/admin_reset_password.py utility supports an operator-only password reset.

Usage:

    python scripts/admin_reset_password.py --username ACCOUNT

The operator must verify identity out of band and explicitly configure:

    WCE_ADMIN_RESET_CONFIRM=I_VERIFIED_IDENTITY

This utility does not verify medical qualifications or automatically revoke every existing session.

Unverified self-registration is permitted only for isolated disposable demonstrations.

## 6. Medical Image and Data Protection

Implemented controls:

- Eight MiB request limit.
- JPEG and PNG image validation.
- Maximum image size of 20 million pixels.
- Maximum dimension of 8192 pixels per edge.
- Random temporary upload filenames.
- Temporary image removal on successful processing.
- Empty filename and image-path fields for new predictions.
- User-scoped analytics and CSV reporting.

Legacy images, records and backups are not automatically deleted.

Temporary file deletion failures remain possible.

These controls do not guarantee hard inference CPU, memory or execution-time limits.

Do not upload real patient data.

## 7. Data Retention

The retention utility supports previewing eligible prediction metadata:

    python scripts/retention_predictions.py --days 90 --database YOUR_DB_NAME

Execution requires explicit authorization, backup, an approved retention policy and the confirmation variable:

    WCE_RETENTION_DELETE_CONFIRM=DELETE_EXPIRED_PREDICTIONS_FROM_YOUR_DB_NAME

Deletion requires the explicit --execute flag.

The example 90-day period is not a healthcare retention recommendation.

The utility does not automatically remove image files, user accounts or backups.

## 8. Security Verification

Local engineering results:

| Test suite | Result |
| --- | --- |
| Flask security | 13 passed |
| Static security contracts | 7 passed |
| Real MySQL integration | 2 passed |
| Real model smoke test | 1 passed |
| Full HTTP-to-MySQL E2E | 1 passed |
| Total | 24 passed |
| Redis integration | 5 tests passed in CI |

Dependency consistency checks reported no broken requirements.

The latest CI workflow includes dependency auditing, Redis integration and Docker runtime validation.

These new checks must execute successfully before their behavior can be considered verified.

## 9. Outstanding Security and Operational Risks

| Priority | Outstanding requirement |
| --- | --- |
| Critical | Verify external credential rotation and historical exposure impact |
| High | Complete verified account provisioning and legacy account review |
| High | Independently validate model labels and patient-disjoint performance |
| High | Establish medical-data privacy and retention governance |
| High | Isolate inference with hard resource limits |
| High | Monitor Redis availability and validate failure handling in deployment |
| High | Verify container runtime and deployment infrastructure |
| Medium | Configure trusted proxies, HTTPS and operational monitoring |
| Medium | Conduct independent penetration testing |
| Low | Review CI action versions and dependency maintenance |

## 10. Release Status

Local regression verification has passed.

The latest production-readiness branch requires successful GitHub Actions execution and final review before merging.

Passing engineering tests does not establish clinical efficacy, regulatory compliance or complete production security.
